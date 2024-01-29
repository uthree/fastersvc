import os
import glob
import random

from torchaudio.functional import resample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import pyworld as pw
import numpy as np
from tqdm import tqdm


class WaveFileDirectory(torch.utils.data.Dataset):
    def __init__(self, source_dir_paths=[], length=16000, max_files=-1, sampling_rate=16000):
        super().__init__()
        print("Loading Data")
        self.path_list = []
        self.data = []
        formats = ["mp3", "wav", "ogg"]
        print("Getting paths")
        for dir_path in source_dir_paths:
            for fmt in formats:
                self.path_list += glob.glob(os.path.join(dir_path, f"**/*.{fmt}"), recursive=True)
        if max_files != -1:
            self.path_list = self.path_list[:max_files]
        print("Chunking")
        for path in tqdm(self.path_list):
            tqdm.write(path)
            wf, sr = torchaudio.load(path) # wf.max() = 1 wf.min() = -1
            # Resample
            wf = torchaudio.functional.resample(wf, sr, sampling_rate)
            # Chunk
            waves = torch.split(wf, length, dim=1)
            tqdm.write(f"    Loading {len(waves)} data...")
            for w in waves:
                if w.shape[1] == length:
                    self.data.append(w[0])
        self.length = length
        print(f"Loaded total {len(self.data)} data.")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def compute_f0_dio(wf, sample_rate=8000, segment_size=320, f0_min=20, f0_max=6000):
    if wf.ndim == 1:
        device = wf.device
        signal = wf.detach().cpu().numpy()
        signal = signal.astype(np.double)
        _f0, t = pw.dio(signal, sample_rate, f0_floor=f0_min, f0_ceil=f0_max)
        f0 = pw.stonemask(signal, _f0, t, sample_rate)
        f0 = torch.from_numpy(f0).to(torch.float)
        f0 = f0.to(device)
        f0 = f0.unsqueeze(0).unsqueeze(0)
        f0 = F.interpolate(f0, wf.shape[0] // segment_size, mode='linear')
        f0 = f0.squeeze(0)
        return f0
    elif wf.ndim == 2:
        waves = wf.split(1, dim=0)
        pitchs = [compute_f0_dio(wave[0], sample_rate, segment_size) for wave in waves]
        pitchs = torch.stack(pitchs, dim=0)
        return pitchs


def compute_f0(wf, sample_rate=16000, segment_size=320):
    l = wf.shape[1]
    wf = resample(wf, sample_rate, 8000)
    pitchs = compute_f0_dio(wf, 8000)
    return F.interpolate(pitchs, l // segment_size, mode='linear')


class WaveFileDirectoryWithF0(torch.utils.data.Dataset):
    def __init__(self, source_dir_paths=[], length=16000, max_files=-1, sampling_rate=16000):
        super().__init__()
        print("Loading Data")
        self.path_list = []
        self.data = []
        self.f0 = []
        formats = ["mp3", "wav", "ogg"]
        print("Getting paths")
        for dir_path in source_dir_paths:
            for fmt in formats:
                self.path_list += glob.glob(os.path.join(dir_path, f"**/*.{fmt}"), recursive=True)
        if max_files != -1:
            random.shuffle(self.path_list)
            self.path_list = self.path_list[:max_files]
        print("Chunking")
        for path in tqdm(self.path_list):
            tqdm.write(path)
            wf, sr = torchaudio.load(path) # wf.max() = 1 wf.min() = -1
            # Resample
            wf = torchaudio.functional.resample(wf, sr, sampling_rate)
            # Chunk
            waves = torch.split(wf, length, dim=1)
            tqdm.write(f"    Loading {len(waves)} data...")
            for w in waves:
                if w.shape[1] == length:
                    self.data.append(w[0])
                    self.f0.append(compute_f0(w)[0])
        self.length = length
        print(f"Loaded total {len(self.data)} data.")

    def __getitem__(self, index):
        return self.data[index], self.f0[index]

    def __len__(self):
        return len(self.data)


