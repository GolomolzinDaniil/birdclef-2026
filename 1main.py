
import os, sys
from os import listdir
from os.path import join, basename, dirname
from datetime import datetime, timezone, timedelta
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from pytorch_metric_learning import losses
import numpy as np
import numpy.random as npr
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from shared import Data1

device = torch.device('cuda')

def elapsed():
    return datetime.now(timezone(timedelta(hours=7))).strftime("%H:%M:%S")

# os.makedirs('/kaggle/working/cache', exist_ok=True)

batch_size = 8
train_dataset = DataLoader(
    Data1(),
    batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True, persistent_workers=True)
valid_dataset = DataLoader(
    Data1(is_train=False),
    batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True, persistent_workers=True)


model = resnet18(weights=ResNet18_Weights.DEFAULT)

# потому что пока что только 2 спектограммы
old_conv = model.conv1
model.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)

with torch.no_grad():
    model.conv1.weight[:] = old_conv.weight[:, :2]

model.fc = nn.Identity() # type: ignore
model.to(device)


criterion = losses.ArcFaceLoss(
    num_classes=206,
    embedding_size=512,
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
reset = "\033[0m"

EPOCH = 20
total_train, total_valid = len(train_dataset), len(valid_dataset)

best_acc = -float('inf')

# model.train()
for epoch in range(EPOCH):

    # train
    train_loss = 0.0
    for xb,yb in tqdm(train_dataset, desc='Train'):
        
        B, N, C, H, W = xb.shape
        x = xb.view(B*N, C, H, W).to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        y = yb.unsqueeze(1).repeat(1, N).view(-1)

        optimizer.zero_grad()

        emb = F.normalize(model(x))

        loss = criterion(emb, y)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= total_train

    # valid
    model.eval()
    valid_acc = 0.0
    with torch.no_grad():
        for xb,yb in tqdm(valid_dataset, desc='Valid'):

            B, N, C, H, W = xb.shape
            x = xb.view(B*N, C, H, W).to(device, non_blocking=True)

            emb = F.normalize(model(x)).view(B, N, -1).mean(dim=1)
            W = F.normalize(criterion.W)

            similar = emb @ W

            pred = similar.argmax(dim=1)
            yb_cuda = yb.to(device, non_blocking=True)
            acc = (pred == yb_cuda).float().mean()

            valid_acc += acc.item()

    valid_acc /= total_valid

    print(f'Epoch: [ {epoch+1:^2} / {EPOCH} ]   lr: {optimizer.param_groups[0]["lr"]}   [{elapsed()}]')
    print(f'  acc: {green}{valid_acc:.4f}{reset},   TrainLoss: {train_loss:.4f}')

    if valid_acc > best_acc:
        best_acc = valid_acc

        torch.save({
            'model': model.state_dict(),
            'arcface': criterion.state_dict()
        # }, '/kaggle/working/best_params.pth')
        }, 'best_label.pth')
        print(f'  Модель {epoch+1} эпохи сохранена')

    model.train()
    scheduler.step(valid_acc)


# torch.save({
#     'model': model.state_dict(),
#     'arcface': criterion.state_dict()
# # }, '/kaggle/working/best_params.pth')
# }, 'best_params.pth')


# https://www.kaggle.com/code?import=true


