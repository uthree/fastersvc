import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.functional import resample
import numpy as np
import pyworld as pw


# wave: [BatchSize, 1, Length]
# Output: [BatchSize, 1, Frames]
def spectrogram(wave, n_fft, hop_size):
    window = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft, hop_size, return_complex=True, window=window).abs()
    return spec[:, :, 1:]

# wave: [BatchSize, 1, Length]
# Output: [BatchSize, 1, Frames]
def energy(wave,
           frame_size=320):
    return F.max_pool1d((wave ** 2).unsqueeze(1), frame_size)


# Convert style based kNN
# 
# source: [BatchSize, Channels, Length]
# reference: [BatchSize, Channels, Length]
# Output: [BatchSize, Channels, Length]
def match_features(source, reference, k=4, alpha=0.0):
    input_data = source

    source = source.transpose(1, 2)
    reference = reference.transpose(1, 2)
    source_norm = torch.norm(source, dim=2, keepdim=True)
    reference_norm = torch.norm(reference, dim=2, keepdim=True)
    cos_sims = torch.bmm((source / source_norm), (reference / reference_norm).transpose(1, 2))
    best = torch.topk(cos_sims, k, dim=2)

    result = torch.stack([reference[n][best.indices[n]] for n in range(source.shape[0])], dim=0).mean(dim=2)
    result = result.transpose(1, 2)
    return result * (1-alpha) + input_data * alpha


# Dlilated Causal Convolution
class DCC(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 dilation=1,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, dilation=dilation, groups=groups)
        self.pad_size = (kernel_size - 1) * dilation

    def forward(self, x):
        x = F.pad(x, [self.pad_size, 0], mode='replicate')
        x = self.conv(x)
        return x


class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + self.eps
        x = (x - mu) / sigma
        x = x * self.scale + self.shift
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=1, mlp_mul=1, norm=False, negative_slope=0.1):
        super().__init__()
        self.c1 = DCC(channels, channels, kernel_size, dilation, channels)
        self.norm = ChannelNorm(channels) if norm else nn.Identity()
        self.c2 = nn.Conv1d(channels, channels * mlp_mul, 1)
        self.c3 = nn.Conv1d(channels * mlp_mul, channels, 1)
        self.negative_slope = negative_slope

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = self.c3(x)
        return x + res


def compute_f0_dio(wf, sample_rate=16000, segment_size=320, f0_min=20, f0_max=20000):
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


def compute_f0_harvest(wf, sample_rate=16000, segment_size=320, f0_min=20, f0_max=20000):
    if wf.ndim == 1:
        device = wf.device
        signal = wf.detach().cpu().numpy()
        signal = signal.astype(np.double)
        f0, t = pw.harvest(signal, sample_rate, f0_floor=f0_min, f0_ceil=f0_max)
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


def compute_f0(wf, sample_rate=16000, segment_size=320, algorithm='harvest'):
    l = wf.shape[1]
    wf = resample(wf, sample_rate, 16000)
    if algorithm == 'harvest':
        pitchs = compute_f0_harvest(wf, 16000)
    elif algorithm == 'dio':
        pitchs = compute_f0_dio(wf, 16000)
    return F.interpolate(pitchs, l // segment_size, mode='linear')

