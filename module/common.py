import torch
import torch.nn as nn
import torch.nn.functional as F


LRELU_SLOPE = 0.1


# wave: [BatchSize, 1, Length]
# Output: [BatchSize, 1, Frames]
def spectrogram(wave, n_fft, hop_size):
    spec = torch.stft(wave, n_fft, hop_size, return_complex=True).abs()
    return spec[:, :, :-1]

# wave: [BatchSize, 1, Length]
# Output: [BatchSize, 1, Frames]
def energy(wave,
           frame_size=480):
    return F.max_pool1d(wave.abs().unsqueeze(1), frame_size * 2, frame_size, frame_size//2)


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
        x = F.pad(x, [self.pad_size, 0])
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
    def __init__(self, channels, kernel_size=7, dilation=1, mlp_mul=1, norm=False):
        super().__init__()
        self.c1 = DCC(channels, channels, kernel_size, dilation, channels)
        self.norm = ChannelNorm(channels) if norm else nn.Identity()
        self.c2 = nn.Conv1d(channels, channels * mlp_mul, 1)
        self.c3 = nn.Conv1d(channels * mlp_mul, channels, 1)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.c3(x)
        return x + res
