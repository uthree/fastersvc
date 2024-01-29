import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ResBlock, spectrogram


class ContentEncoder(nn.Module):
    def __init__(self,
                 n_fft=1280,
                 hop_size=320,
                 internal_channels=512,
                 kernel_size=5,
                 dilations=[1, 3, 5, 1],
                 output_channels=512,
                 hubert_dim=768,
                 ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size

        self.input_layer = nn.Conv1d(n_fft // 2 + 1, internal_channels, 1)

        self.res_stack = nn.Sequential(
                *[ResBlock(internal_channels, kernel_size, dilation=d, mlp_mul=2, norm=True) for d in dilations])

        self.output_layer = nn.Conv1d(internal_channels, output_channels, 1, bias=False)
        self.to_hubert = nn.Conv1d(output_channels, hubert_dim, 1, bias=False)

    def forward(self, spec):
        x = self.input_layer(spec)
        x = self.res_stack(x)
        x = self.output_layer(x)
        return x

    def encode(self, wave):
        spec = spectrogram(wave, self.n_fft, self.hop_size)
        return self.forward(spec)
