import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import weight_norm, spectral_norm


LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ScaleDiscriminator(nn.Module):
    def __init__(self, scale=1, channels=16, num_layers=5, max_channels=512, max_groups=8):
        super().__init__()
        self.pool = nn.AvgPool1d(scale)
        self.input_layer = weight_norm(nn.Conv1d(1, channels, 41, 3, 21))
        self.convs = nn.ModuleList([])
        c = channels
        g = 1
        for _ in range(num_layers):
            self.convs.append(weight_norm(nn.Conv1d(c, min(c*2, max_channels), 41, 3, 21, groups=g)))
            g = min(g * 2, max_groups)
            c = min(c * 2, max_channels)
        self.output_layer = nn.Conv1d(c, 1, 21, 1, 11)

    def forward(self, x):
        feats = []
        x = x.unsqueeze(1)
        x = self.pool(x)
        x = self.input_layer(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        for conv in self.convs:
            x = conv(x)
            feats.append(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.output_layer(x)
        return x, feats


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, scales=[1, 2, 3]):
        super().__init__()
        self.sub_discs = nn.ModuleList([])
        for s in scales:
            self.sub_discs.append(ScaleDiscriminator(s))

    def forward(self, x):
        feats = []
        logits = []
        for sd in self.sub_discs:
            l, f = sd(x)
            feats += f
            logits.append(l)
        return logits, feats


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.msd = MultiscaleDiscriminator()

    def forward(self, x):
        f, l = self.msd(x)
        return f, l
