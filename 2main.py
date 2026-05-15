
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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from shared import Data2

device = torch.device('cuda')

def elapsed():
    return datetime.now(timezone(timedelta(hours=7))).strftime("%H:%M:%S")

# os.makedirs('/kaggle/working/cache', exist_ok=True)

batch_size = 4
train_dataset = DataLoader(Data2(), batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
valid_dataset = DataLoader(Data2(is_train=False), batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)


model = resnet18()

old_conv = model.conv1
model.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)

with torch.no_grad():
    model.conv1.weight[:] = old_conv.weight[:, :2]

model.fc = nn.Linear(512, 234)


state = torch.load('best_label.pth', map_location=device)
model.load_state_dict(state['model'], strict=False)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)
# sheduler = torch.she
criterion = nn.BCEWithLogitsLoss().to(device)
model.to(device);


green = "\033[92m"
yellow = "\033[93m"
reset = "\033[0m"

EPOCH = 20
total_train, total_valid = len(train_dataset), len(valid_dataset)

best_metric = -float('inf')

for epoch in range(EPOCH):

    # train
    train_loss = 0.0
    for xb,yb in tqdm(train_loader, desc='Train'):

        B, N, C, H, W = xb.shape

        x = xb.view(B*N, C, H, W).to(device, non_blocking=True)
        y = yb.unsqueeze(1).repeat(1, N, 1).view(B*N, 234)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= total_train

    # valid
    valid_loss = 0.0
    targets, predicts = [], []
    with torch.no_grad():

        for xb,yb in tqdm(valid_loader, desc='Valid'):

            B, N, C, H, W = xb.shape

            x = xb.view(B*N, C, H, W).to(device, non_blocking=True)

            y = yb.unsqueeze(1).repeat(1, N, 1).view(B*N, 234)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)
            valid_loss += loss.item()

            # f1-score
            predict = (torch.sigmoid(logits) > 0.3).float()
            targets.append(y.cpu())
            predicts.append(predict.cpu())

    valid_loss /= total_valid

    targets = torch.cat(targets).numpy()
    predicts = torch.cat(predicts).numpy()

    f1_micro = f1_score(targets, predicts, average='micro', zero_division=0)
    f1_macro = f1_score(targets, predicts, average='macro', zero_division=0)

    print(f'Epoch: [ {epoch+1:^2} / {EPOCH} ]          [ {elapsed()} ]')
    print(f'     F1-micro: {green}{f1_micro:.4f}{reset}, ValidLoss: {yellow}{valid_loss:.4f}{reset}')
    print(f'     F1-macro: {f1_macro:.4f}, TrainLoss: {train_loss:.4f}')


    if f1_macro > best_metric:
        best_metric = f1_macro

        torch.save(model.state_dict(), 'best_multi.pth')
        
        print(f'  Модель {epoch+1} эпохи сохранена')


