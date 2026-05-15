from torch.utils.data import Dataset
import torchaudio.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import soundfile as sf
import torch
import timm
from abc import ABC, abstractmethod
import os
from os import listdir
from os.path import join
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.random as npr
import random
import torch.distributed as dist
from collections import defaultdict

def elapsed():
    return datetime.now(timezone(timedelta(hours=7))).strftime("%H:%M:%S")

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

green = "\033[92m"
yellow = "\033[93m"
reset = "\033[0m"


class Data(Dataset, ABC):

    def __init__(
            self,
            sr: int = 32_000,
            is_train: bool = True
            ):
        self._sr = sr
        self._is_train = is_train
        self._num_classes = 234
        self._samples = []

        self._to_db = T.AmplitudeToDB()
        self._to_mel_low = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=10_000
        )
        self._to_mel_high = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=8_000,
            f_max=16_000
        )

        with open(r'data/sample_submission.csv', 'r', encoding='utf-8') as f:
            clses = f.readline().strip().split(',')[1:]

        self._cls2idx = {cls: idx for idx, cls in enumerate(clses)}
        self._train_cls2idx = {cls: idx for idx, cls in enumerate(sorted(listdir(r'data/train_audio')))}

    def _scaler(self, mel: torch.Tensor):
        log_mel = self._to_db(mel)
        min_ = log_mel.min()
        max_ = log_mel.max()
        if (max_ - min_) == 0:
            return torch.zeros_like(log_mel)
        return (log_mel - min_) / (max_ - min_)

    def _get_sps(self, y):
        with torch.no_grad():
            mel_low = self._to_mel_low(y)
            mel_high = self._to_mel_high(y)

            delta_low = self._scaler(torchaudio.functional.compute_deltas(mel_low))
            delta_high = self._scaler(torchaudio.functional.compute_deltas(mel_high))

        stacked = torch.stack([
            self._scaler(mel_low),
            self._scaler(mel_high),
            delta_low,
            delta_high
        ], dim=0)

        return stacked.float()

    @abstractmethod
    def _make_slides(self, y):
        pass

    def __len__(self):
        return len(self._samples)


class Data1(Data):

    def __init__(
            self,
            sr: int = 32_000,
            duration: int = 5,
            stride: float = 2.5,
            num_segments: int = 5,
            is_train: bool = True
            ):
        super().__init__(sr, is_train)

        self._num_segments = num_segments
        self._train_audio = r'data/train_audio'

        self._segment_len = int(duration * sr)
        self._stride = int(stride * sr)

        clses = sorted(listdir(self._train_audio))
        for cls in clses:

            cls_path = join(self._train_audio, cls)
            cls_audios = listdir(cls_path)

            for audio in cls_audios:
                self._samples.append((
                    join(cls_path, audio),
                    self._train_cls2idx[cls]
                ))

            num_audio = len(cls_audios)

            if num_audio <= 5:
                repeat = 20
            elif num_audio <= 10:
                repeat = 10
            elif num_audio <= 20:
                repeat = 5
            elif num_audio <= 50:
                repeat = 2
            else:
                repeat = 1

            self._samples.extend([
                (join(cls_path, audio), self._train_cls2idx[cls])
                for audio in npr.choice(cls_audios, size=repeat*num_audio, replace=True)
            ])

        train_samples, valid_samples = train_test_split(
            self._samples,
            test_size=0.2,
            random_state=42
        )
        self._samples = train_samples if self._is_train else valid_samples

    def _make_slides(self, y):

        positions = list(range(0, max(1, len(y) - self._segment_len + 1), self._stride))

        if self._is_train:

            if npr.rand() < 0.3:
                y = y * npr.uniform(0.7, 1.3)

            if npr.rand() < 0.3:
                y = y + torch.randn_like(y) * 3e-3

            if npr.rand() < 0.5:
                shift = np.random.randint(0, max(1, len(y)//10))
                y = torch.roll(y, shift)

            y = y.clamp(-1, 1)

            start = npr.choice(positions)

        else:

            start = positions[len(positions)//2]

        segment = y[start:start+self._segment_len]

        if len(segment) < self._segment_len:
            pad = self._segment_len - len(segment)
            segment = F.pad(segment, (0, pad))

        return self._get_sps(segment)

    def __getitem__(self, key):

        path, label = self._samples[key]

        y, _ = sf.read(path, dtype="float32")
        y = torch.from_numpy(y)

        vector = torch.zeros(234)
        vector[label] = 1.0

        return self._make_slides(y), vector.float()


class Data2(Data):

    def __init__(
            self,
            sr: int = 32_000,
            is_train: bool = True
            ):
        super().__init__(sr, is_train)

        self._train_audio = r'data/train_soundscapes'

        duration = 5
        self._segment_len = int(duration * sr)

        with open(r'data/train_soundscapes_labels.csv', 'r', encoding='utf-8') as f:
            next(f)
            for line in f:

                audio, start, end, str_labels = line.split(',')

                vector = self._parsing_labels(str_labels)

                self._samples.append((
                    join(self._train_audio, audio),
                    int(float(start) * sr),
                    vector
                ))

        train_samples, valid_samples = train_test_split(self._samples, test_size=0.2, random_state=42)
        self._samples = train_samples if self._is_train else valid_samples

    def _parsing_labels(self, labels: str):

        labels_ = labels.strip().split(';')
        vector = torch.zeros(234)

        for label in labels_:
            vector[self._cls2idx[label]] = 1

        return vector

    def _make_slides(self, y, start):

        segment = y[start:start+self._segment_len]

        if len(segment) < self._segment_len:
            pad = self._segment_len - len(segment)
            segment = F.pad(segment, (0, pad))

        return self._get_sps(segment)

    def __getitem__(self, key):

        path, start, label = self._samples[key]

        y, _ = sf.read(path, dtype="float32")
        y = torch.from_numpy(y)

        return self._make_slides(y, start), label.float()




embending_size = 256

class BirdModel(nn.Module):

    def __init__(
        self,
        norma: bool,
        embending_size: int
        ):
        super().__init__()

        self._norma = norma
        self.embending_size = embending_size

        self._model = timm.create_model(
            'tf_efficientnet_b0',
            pretrained=True,
            in_chans=4,
            num_classes=234
        )

    def forward(self, x):
        return self._model(x)



# class BirdModel(nn.Module):

#     def __init__(self, norma: bool, embending_size: int):
#         super().__init__()

#         self._norma = norma
#         self.embending_size = embending_size

#         self._model = timm.create_model(
#             'efficientnet_b0',
#             pretrained=True,
#             in_chans=4,
#             num_classes=0
#         )
#         self._linear = nn.Linear(self._model.num_features, embending_size)

#     def forward(self, x):

#         x = self._model(x)
#         x = self._linear(x)

#         return F.normalize(x) if self._norma else x