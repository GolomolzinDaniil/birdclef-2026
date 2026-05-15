from datetime import datetime, timezone, timedelta

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import logging
from tqdm import tqdm

from shared import *


os.makedirs('log', exist_ok=True)
logging.basicConfig(
    filename='log/2logs.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

local_rank = setup()
device = torch.device(f'cuda:{local_rank}')


batch_size = 8

train_ds = Data2()
train_sampler = DistributedSampler(train_ds)
train_dataset = DataLoader(
    train_ds,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

valid_ds = Data2(is_train=False)
valid_sampler = DistributedSampler(valid_ds, shuffle=False)
valid_dataset = DataLoader(
    valid_ds,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)


model = BirdModel(
    norma=False,
    emb_size=234
)

checkpoint = torch.load(
    'best_label.pth',
    map_location='cpu'
)

state = checkpoint['model']
state = {
    k:v
    for k,v in state.items()
    if '_linear' not in k
}

model.load_state_dict(state, strict=False)
model.to(device)

model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[local_rank])


criterion = nn.BCEWithLogitsLoss().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max', patience=2, factor=.5
)


green = "\033[92m"
yellow = "\033[93m"
reset = "\033[0m"

EPOCH = 100
total_train, total_valid = len(train_dataset), len(valid_dataset)

best_metric = -float('inf')

max_epoch = 10
num_epoch = 0
stop_train = False

if local_rank == 0:
    logging.info('Обучение началось')

try:
    for epoch in range(EPOCH):

        train_sampler.set_epoch(epoch)

        train_loss = 0.0

        if local_rank == 0:
            print(f'Epoch: [{epoch+1:^2} / {EPOCH}]')
            logging.info(f'Epoch: [{epoch+1:^2} / {EPOCH}] started')


        for xb,yb in tqdm(train_dataset, desc='Train', disable=local_rank != 0):

            B, N, C, H, W = xb.shape

            x = xb.view(B*N, C, H, W).to(device, non_blocking=True)

            y = yb.unsqueeze(1).repeat(1, N, 1).view(B*N, 234).to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(logits, y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= total_train

        model.eval()

        valid_loss = 0.0

        probs_all = []
        targets_all = []

        with torch.no_grad():

            for xb,yb in tqdm(valid_dataset, desc='Valid', disable=local_rank != 0):

                B, N, C, H, W = xb.shape

                x = xb.view(B*N, C, H, W).to(device, non_blocking=True)

                y = yb.unsqueeze(1).repeat(1, N, 1).view(B*N, 234).to(device, non_blocking=True)

                logits = model(x)

                loss = criterion(logits, y)

                valid_loss += loss.item()

                logits = logits.view(B, N, 234).mean(dim=1)

                probs = torch.sigmoid(logits)

                probs_all.append(probs.cpu())
                targets_all.append(yb.cpu())

        valid_loss /= total_valid

        probs_all = torch.cat(probs_all, dim=0)
        targets = torch.cat(targets_all, dim=0)

        world_size = dist.get_world_size()

        gather_probs = [torch.zeros_like(probs_all) for _ in range(world_size)]
        gather_targets = [torch.zeros_like(targets) for _ in range(world_size)]

        dist.all_gather(gather_probs, probs_all)
        dist.all_gather(gather_targets, targets)

        probs_all = torch.cat(gather_probs).numpy()
        targets = torch.cat(gather_targets).numpy()

        best_thr = 0.5
        best_f1 = 0
        thrs = np.linspace(0.1, 0.9, 10)
        for thr in thrs:
            predicts = (probs_all > thr).astype(int)
            f1 = f1_score(targets, predicts, average='macro', zero_division=0)
            if best_f1 < f1:
                best_f1 = f1
                best_thr = thr

        predicts = (probs_all > best_thr).astype(int)

        f1_micro = f1_score(targets, predicts, average='micro', zero_division=0)
        f1_macro = f1_score(targets, predicts, average='macro', zero_division=0)

        if local_rank == 0:

            print(f'     F1-micro: {green}{f1_micro:.4f}{reset},   ValidLoss: {yellow}{valid_loss:.4f}{reset}')
            print(f'     F1-macro: {f1_macro:.4f},   TrainLoss: {train_loss:.4f}')
            print(f'     threshold: {best_thr},   lr: {optimizer.param_groups[0]["lr"]}')

            logging.info(
                f'F1-micro: {f1_micro:.4f}, '
                f'F1-macro: {f1_macro:.4f}, '
                f'TrainLoss: {train_loss:.4f}, '
                f'ValidLoss: {valid_loss:.4f}, '
                f'Threshold: {best_thr:.2f}, '
                f'LR: {optimizer.param_groups[0]["lr"]}'
            )

            if f1_macro > best_metric:
                best_metric = f1_macro
                num_epoch = 0
                torch.save({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'threshold': best_thr,
                }, 'models/restart_stage2.pth')
                torch.save(model.module.state_dict(), 'models/stage2.pth')

                print(f'     Модель {epoch+1} эпохи сохранена')
                logging.info(f'Модель {epoch+1} эпохи сохранена')

            else:
                num_epoch += 1
                print(f'     Метрика не выросла')
                if num_epoch >= max_epoch:
                    stop_train = True
                    print(f'     ПЛАТО - ОСТАНОВКА ОБУЧЕНИЯ')
                    logging.info('Метрика не изменилась. Обучение остановлено')

        if stop_train:
            break

        scheduler.step(f1_macro)
        model.train()

except Exception as e:
    logging.exception(f'{e}')

dist.destroy_process_group()