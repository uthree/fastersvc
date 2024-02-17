import os
import glob
import random

from torchaudio.functional import resample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from tqdm import tqdm

from .common import compute_f0


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
                w = w.mean(dim=0, keepdim=True)
                if w.shape[1] < length:
                    pad_len = length - w.shape[1]
                    pad = torch.zeros(1, pad_len)
                    w = torch.cat([w, pad], dim=1)
                self.data.append(w[0])
         self.length = length
        print(f"Loaded total {len(self.data)} data.")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class WaveFileDirectoryWithF0(torch.utils.data.Dataset):
    def __init__(self, source_dir_paths=[], length=16000, max_files=-1, sampling_rate=16000, algorithm='harvest'):
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
                    self.f0.append(compute_f0(w, algorithm=algorithm)[0])
        self.length = length
        print(f"Loaded total {len(self.data)} data.")

    def __getitem__(self, index):
        return self.data[index], self.f0[index]

    def __len__(self):
        return len(self.data)


