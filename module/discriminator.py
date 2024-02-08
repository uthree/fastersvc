import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import weight_norm, spectral_norm


LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class PeriodicDiscriminator(nn.Module):
    def __init__(self,
                 channels=32,
                 period=2,
                 kernel_size=5,
                 stride=3,
                 num_stages=4,
                 groups = [],
                 max_channels=512
                 ):
        super().__init__()
        self.input_layer = weight_norm(
                nn.Conv2d(1, channels, (kernel_size, 1), (stride, 1), padding=get_padding(kernel_size, 1)))
        self.layers = nn.ModuleList([])
        for i in range(num_stages):
            c = min(channels * (4 ** i), max_channels)
            c_next = min(channels * (4 ** (i+1)), max_channels)
            if i == (num_stages - 1):
                self.layers.append(
                        weight_norm(
                            nn.Conv2d(c, c, (kernel_size, 1), (stride, 1), groups=groups[i],
                                      padding=get_padding(kernel_size, 1))))
            else:
                self.layers.append(
                        weight_norm(
                            nn.Conv2d(c, c_next, (kernel_size, 1), (stride, 1), groups=groups[i],
                                      padding=get_padding(kernel_size, 1))))
        c = min(channels * (4 ** (num_stages-1)), max_channels)
        self.final_conv = weight_norm(
                nn.Conv2d(c, c, (5, 1), 1, padding=get_padding(5, 1)))
        self.output_layer = weight_norm(
                nn.Conv2d(c, 1, (3, 1), 1, padding=get_padding(3, 1)))
        self.period = period

    def forward(self, x):
        features = []
        # padding
        if x.shape[1] % self.period != 0:
            pad_len = self.period - (x.shape[1] % self.period)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.view(x.shape[0], self.period, -1)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.input_layer(x)
        for layer in self.layers:
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = layer(x)
            features.append(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.final_conv(x)
        return x, features


class MultiPeriodicDiscriminator(nn.Module):
    def __init__(self,
                 periods=[1, 2, 3, 5, 7, 11, 23],
                 groups=[1, 4, 8, 8, 8, 8],
                 channels=64,
                 kernel_size=5,
                 stride=3,
                 num_stages=5,
                 ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])

        for p in periods:
            self.sub_discriminators.append(
                    PeriodicDiscriminator(channels,
                                          p,
                                          kernel_size,
                                          stride,
                                          num_stages,
                                          groups=groups))

    def forward(self, x):
        logits = []
        features = []
        for sd in self.sub_discriminators:
            l, f = sd(x)
            logits.append(l)
            features += f
        return logits, features


class ResolutionDiscriminator(nn.Module):
    def __init__(self, n_fft=1024, channels=32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.fft_bin = n_fft // 2 + 1
        
        self.layers = nn.ModuleList([
                weight_norm(nn.Conv2d(1, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1)),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1)),
            ])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        features = []
        x = torch.stft(x, self.n_fft, self.hop_length, return_complex=True).abs()
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            features.append(x)
        x = self.conv_post(x)
        return x, features


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, n_ffts=[512, 1024, 2048]):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])
        for n_fft in n_ffts:
            self.sub_discriminators.append(
                    ResolutionDiscriminator(n_fft))

    def forward(self, x):
        logits = []
        features = []
        for sd in self.sub_discriminators:
            l, f = sd(x)
            features += f
            logits.append(l)
        return logits, features


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.MPD = MultiPeriodicDiscriminator()
        self.MRD = MultiResolutionDiscriminator()

    def forward(self, x):
        l1, f1 = self.MPD(x)
        l2, f2 = self.MRD(x)
        return l1 + l2, f1 + f2

