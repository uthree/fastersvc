import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import weight_norm


LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ScaleDiscriminator(nn.Module):
    def __init__(self, scale=1, num_layers=6, channels=32):
        super().__init__()
        self.scale = scale
        self.convs = nn.ModuleList([])
        if scale == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool1d(scale*2, scale)
        self.input_layer = weight_norm(nn.Conv1d(1, channels, 21, 1, 10))
        c = channels
        g = 1
        for i in range(num_layers):
            self.convs.append(weight_norm(nn.Conv1d(c, min(c * 2, 512), 21, 3, 10, groups=g)))
            c = min(c * 2, 512)
            g = min(g * 2, 8)
        self.output_layer = weight_norm(nn.Conv1d(c, 1, 21, 3, 10))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(x)
        x = self.input_layer(x)
        for c in self.convs:
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = c(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.output_layer(x)
        return x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales=[1, 2, 3]):
        super().__init__()
        self.sub_discs = nn.ModuleList([ScaleDiscriminator(s) for s in scales])

    def logits(self, x):
        logits = []
        x = x - x.mean(dim=1, keepdim=True)
        for sd in self.sub_discs:
            logits.append(sd(x))
        return logits


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
        w = torch.hann_window(self.n_fft, device=x.device)
        dtype = x.dtype
        x = x.to(torch.float)
        x = torch.stft(x, self.n_fft, self.hop_length, return_complex=True).abs()
        x = x.to(dtype)
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        return x


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, n_ffts=[512, 1024, 2048]):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])
        for n_fft in n_ffts:
            self.sub_discriminators.append(
                    ResolutionDiscriminator(n_fft))

    def logits(self, x):
        logits = []
        for d in self.sub_discriminators:
            logits.append(d(x))
        return logits


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.msd = MultiScaleDiscriminator()
        self.mrd = MultiResolutionDiscriminator()
    
    def logits(self, x):
        return self.msd.logits(x) + self.mrd.logits(x)

