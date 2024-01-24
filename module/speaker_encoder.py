import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ResBlock, spectrogram


class SpeakerEncoder(nn.Module):
    def __init__(self,
                 n_fft=1920,
                 hop_size=480,
                 internal_channels=256,
                 kernel_size=7,
                 dilations=[1, 3, 9, 27],
                 output_channels=256,
                 ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size

        self.input_layer = nn.Sequential(
            nn.Conv1d(n_fft // 2 + 1, 128, 1),
            nn.Conv1d(128, internal_channels, 1))
        self.output_layer = nn.Conv1d(internal_channels, output_channels, 1)

        self.res_stack = nn.Sequential(
                *[ResBlock(internal_channels, kernel_size, dilation=d, norm=True) for d in dilations])

    def forward(self, spec):
        x = self.input_layer(spec)
        x = self.res_stack(x)
        x = x.mean(dim=2, keepdim=True)
        x = self.output_layer(x)
        return x

    def encode(self, wave):
        spec = spectrogram(wave, self.n_fft, self.hop_size)
        return self.forward(spec)
