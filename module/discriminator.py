import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import weight_norm, spectral_norm


LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ScaleDiscriminator(nn.Module):
    def __init__(self, scale=1, num_layers=6, channels=32, norm_f=weight_norm):
        super().__init__()
        self.scale = scale
        self.convs = nn.ModuleList([])
        if scale == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool1d(scale*2, scale)
        self.input_layer = norm_f(nn.Conv1d(1, channels, 21, 1, 10))
        c = channels
        g = 1
        for i in range(num_layers):
            self.convs.append(norm_f(nn.Conv1d(c, min(c * 2, 1024), 41, 3, 20, groups=g)))
            c = min(c * 2, 1024)
            g = min(g * 2, 16)
        self.output_layer = norm_f(nn.Conv1d(c, 1, 21, 3, 10))

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
    def __init__(self, scales=[1, 2, 3], norms=[weight_norm, weight_norm, spectral_norm]):
        super().__init__()
        self.sub_discs = nn.ModuleList([ScaleDiscriminator(s, norm_f=n) for s, n in zip(scales, norms)])

    def logits(self, x):
        logits = []
        x = x - x.mean(dim=1, keepdim=True)
        for sd in self.sub_discs:
            logits.append(sd(x))
        return logits


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.msd = MultiScaleDiscriminator()

    def logits(self, x):
        return self.msd.logits(x)
