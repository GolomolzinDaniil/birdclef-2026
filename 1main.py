from datetime import datetime, timezone, timedelta

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses
import timm
import logging

from tqdm import tqdm

os.makedirs('log', exist_ok=True)
logging.basicConfig(
    filename='log/1logs.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = setup()

from shared import *

device = torch.device(f'cuda:{local_rank}')

def elapsed():
    return datetime.now(timezone(timedelta(hours=7))).strftime("%H:%M:%S")


batch_size = 64
batch_per_epoch = 100

train_ds = Data1()
train_sampler = DistributedSampler(train_ds)
train_dataset = DataLoader(
    train_ds,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
# train_sampler=BatchSampler(
#     cls_idx=train_ds.cls_idx,
#     num_cls=8,
#     num_samples=8,
#     steps_per_epoch=batch_per_epoch
# )
# train_dataset = DataLoader(
#     train_ds,
#     batch_sampler=train_sampler,
#     num_workers=8,
#     pin_memory=True
# )

valid_ds = Data1(is_train=False)
valid_sampler = DistributedSampler(valid_ds, shuffle=False)
valid_dataset = DataLoader(
    valid_ds,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)


max_epoch = 10
num_epoch = 0
stop_train = False

model = BirdModel(norma=True)
model.to(device)

model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[local_rank])

criterion = losses.ArcFaceLoss(
    num_classes=206,
    embedding_size=256,
).to(device)

optimizer = torch.optim.AdamW(
    params=list(model.parameters()) + list(criterion.parameters()),
    lr=3e-4 
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max', patience=2, factor=.5
)



green = "\033[92m"
yellow = "\033[93m"
reset = "\033[0m"

EPOCH = 100
total_train, total_valid = batch_per_epoch, len(valid_dataset)

best_acc = -float('inf')

if local_rank == 0:
    logging.info('Обучение началось')

try:
    for epoch in range(EPOCH):

        train_sampler.set_epoch(epoch)

        # train
        train_loss = 0.0
        if local_rank == 0:
            print(f'Epoch: [{epoch+1:^2} / {EPOCH}]')
            logging.info(f'Epoch: [{epoch+1:^2} / {EPOCH}] started')

        for xb,yb in tqdm(train_dataset, desc='Train', disable=local_rank != 0):
            
            B, N, C, H, W = xb.shape
            x = xb.view(B*N, C, H, W).to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()

            emb = model(x)
            y = yb.unsqueeze(1).repeat(1, N).view(-1)
            loss = criterion(emb, y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= total_train
            
        model.eval()
        correct_preds = 0
        total_samples = 0
        
        with torch.no_grad():
            for xb,yb in tqdm(valid_dataset, desc='Valid', disable=local_rank != 0):

                B, N, C, H, W = xb.shape
                x = xb.view(B*N, C, H, W).to(device, non_blocking=True)

                emb = F.normalize(model(x).view(B, N, -1).mean(dim=1))
                W_weights = F.normalize(criterion.W)

                similar = emb @ W_weights

                pred = similar.argmax(dim=1)
                yb_cuda = yb.to(device, non_blocking=True)

                correct_preds += (pred == yb_cuda).sum().item()
                total_samples += B

        correct_tensor = torch.tensor(correct_preds, dtype=torch.float32, device=device)
        total_tensor = torch.tensor(total_samples, dtype=torch.float32, device=device)
        
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        valid_acc = correct_tensor.item() / total_tensor.item()

        if local_rank == 0:
            print(f'     acc: {green}{valid_acc:.4f}{reset},   TrainLoss: {yellow}{train_loss:.4f}{reset},   lr: {optimizer.param_groups[0]["lr"]}')
            logging.info(f'Acc: {valid_acc:.4f}, TrainLoss: {train_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]}')

            if valid_acc > best_acc:
                best_acc = valid_acc
                num_epoch = 0
                torch.save({
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "arcface": criterion.state_dict(),
                }, "best_label.pth")
                print(f'     Модель {epoch+1} эпохи сохранена')
                logging.info(f'Модель {epoch+1} эпохи сохранена')
            else:
                num_epoch += 1
                print(f'     Метрика не изменялась')
                if num_epoch >= max_epoch:
                    stop_train = True
                    print(f'     ПЛАТО - ОСТАНОВКА ОБУЧЕНИЯ ')
                    logging.info('Метрика не изменилась. Обучение остановлено')

        # ранняя остановка
        if stop_train:
            break

        model.train()
        scheduler.step(valid_acc)
except Exception as e:
    logging.exception(f'{e}')

dist.destroy_process_group()

