import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ResBlock, spectrogram


class PitchEstimator(nn.Module):
    def __init__(self,
                 n_fft=1920,
                 hop_size=480,
                 internal_channels=256,
                 kernel_size=5,
                 dilations=[1, 3, 5, 7],
                 output_channels=512,
                 f0_min=10
                 ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.output_channels = output_channels
        self.f0_min = 10

        self.input_layer = nn.Conv1d(n_fft // 2 + 1, internal_channels, 1)

        self.output_layer = nn.Conv1d(internal_channels, output_channels, 1)
        self.res_stack = nn.Sequential(
                *[ResBlock(internal_channels, kernel_size, dilation=d, norm=True, mlp_mul=3) for d in dilations])

    @torch.no_grad()
    def estimate(self, wave):
        logits = self.forward(spectrogram(wave, self.n_fft, self.hop_size))
        ids = torch.argmax(logits, dim=1, keepdim=True)
        return self.id2freq(ids)

    def logits(self, wave):
        logits = self.forward(spectrogram(wave, self.n_fft, self.hop_size))
        return logits

    def forward(self, spec):
        x = self.input_layer(spec)
        x = self.res_stack(x)
        x = self.output_layer(x)
        return x

    def freq2id(self, f):
        return torch.round(torch.clamp(48 * torch.log2(f / 10), 0, self.output_channels-1)).to(torch.long)

    def id2freq(self, ids):
        x = ids.to(torch.float)
        x = 10 * (2 ** (x / 48))
        x[x <= self.f0_min] = 0
        return x
