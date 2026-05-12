from torch.utils.data import Dataset
import torchaudio.transforms as T
import soundfile as sf
import torch
from abc import ABC, abstractmethod
import os
from os import listdir
from os.path import join
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.random as npr





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
            f_max=sr // 2
        )

        with open(r'data/sample_submission.csv', 'r', encoding='utf-8') as f:            
            clses = f.readline().strip().split(',')[1:]

        self._cls2idx = {           # 234
            cls:idx
            for idx,cls in enumerate(clses)
        }

        self._train_cls2idx = {     # 216
            cls: idx
            for idx, cls in enumerate(sorted(listdir(r'data/train_audio')))
        }

    def _scaler(self, mel: torch.Tensor):
        log_mel = self._to_db(mel)
        min_ = log_mel.min()
        max_ = log_mel.max()
        if (max_ - min_) == 0: return torch.zeros_like(log_mel)
        return (log_mel - min_) / (max_ - min_)
    
    def _get_sps(self, y):

        with torch.no_grad():
            mel_low = self._to_mel_low(y)
            mel_high = self._to_mel_high(y)

        stacked = torch.stack([
            self._scaler(mel_low),
            self._scaler(mel_high)
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
            duration: int = 5,      # секунды
            stride: float = 2.5,    # перекрытие в секундах
            num_segments: int = 5,
            is_train: bool = True
            ) -> None:
        super().__init__(sr, is_train)

        self._num_segments = num_segments

        # self._train_audio = r'/kaggle/input/competitions/birdclef-2026/train_audio'
        self._train_audio = r'data/train_audio'
        # self._cache_path = r'/kaggle/working/cache'
        self._cache_path = r'cache'
    
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

        train_samples, valid_samples = train_test_split(
            self._samples,
            test_size=0.2,
            random_state=42
        )
        self._samples = train_samples if self._is_train else valid_samples


    def _make_slides(self, y):

        segments = []
        positions = list(range(0, max(1, len(y) - self._segment_len + 1), self._stride))

        if self._is_train:
            is_replace = len(positions) < self._num_segments
            positions = np.sort(npr.choice(positions, self._num_segments, replace=is_replace))
        else:
            if len(positions) > self._num_segments:
                idxs = np.linspace(0, len(positions)-1, self._num_segments).astype(int)
                positions = [positions[i] for i in idxs]
            else:
                positions = npr.choice( positions, self._num_segments, replace=True)

        for start in positions:
            segment = y[start:start+self._segment_len]

            if len(segment) < self._segment_len:
                pad = self._segment_len - len(segment)
                segment = torch.nn.functional.pad(segment, (0, pad))

            sps = self._get_sps(segment)
            segments.append(sps)

        return torch.stack(segments)
    
    def __getitem__(self, key):
        path, label = self._samples[key]

        y, _ = sf.read(path, dtype="float32")
        y = torch.from_numpy(y)

        if y.ndim > 1:
            y = y.mean(dim=1)
            
        return self._make_slides(y), torch.tensor(label, dtype=torch.long)





class Data2(Data):

    def __init__(
            self,
            sr: int = 32_000,
            is_train: bool = True
            ):
        super().__init__(sr, is_train)

        self._train_audio = r'data/train_soundscapes'
        self._train_labels = r'data/train_soundscapes_labels.csv'

        # hardcode, потому что требуется по 5сек
        duration = 5
        self._segment_len = int(duration * sr)
        # аналогично, поэтому 60 / 5 =
        self._num_segments = 12

        # по записям
        with open(r'data/train_soundscapes_labels.csv', 'r', encoding='utf-8') as f:
            next(f)
            for line in f:

                # [BC2026_Test_<file ID>_<site>_<date>_<time in UTC>.ogg] [start] [end] [label1;label2;label3;...]
                audio, start, end, labels = line.split(',')

                # потому что мульти-классовое
                self._samples.append((
                    join(self._train_audio, audio),     # *Момент в том, что нет привязки ко времени
                    self._parsing_labels(labels)
                ))

        train_samples, valid_samples = train_test_split(
            self._samples,
            test_size=.2,
            random_state=42
        )
        self._samples = train_samples if self._is_train else valid_samples


    def _parsing_labels(self, labels: str):
        '''Создает multi-hot vector длины 234'''

        labels_ = labels.strip().split(';')
        vector = torch.zeros(234)

        for label in labels_:
            vector[self._cls2idx[label]] = 1
        return vector

    def _make_slides(self, y):
        
        segments = []
        for start in range(self._num_segments):
            shift = start * self._segment_len
            segment = y[shift:shift+self._segment_len]
            segments.append(self._get_sps(segment))

        return torch.stack(segments)

    def __getitem__(self, key):
        path, label = self._samples[key]

        y, _ = sf.read(path, dtype="float32")
        y = torch.from_numpy(y)

        if y.ndim > 1:
            y = y.mean(dim=1)
            
        return self._make_slides(y), label.float()

