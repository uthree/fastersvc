import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import weight_norm


LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class ScaleDiscriminator(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.pool = nn.AvgPool1d(scale)
        convs = [
            nn.Conv1d(1, 8, 41, 3, 21),
            nn.Conv1d(8, 16, 41, 3, 21),
            nn.Conv1d(16, 32, 41, 3, 21),
            nn.Conv1d(32, 64, 21, 3, 11, groups=2),
            nn.Conv1d(64, 128, 21, 3, 11, groups=4),
            nn.Conv1d(128, 256, 21, 3, 11, groups=8),
            nn.Conv1d(256, 512, 15, 3, 7, groups=16),
            nn.Conv1d(512, 1, 15, 3, 7)
            ]
        self.convs = nn.ModuleList([weight_norm(c) for c in convs])

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(x)
        feats = []
        for c in self.convs:
            x = c(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feats.append(x)
        return x, feats


class Discriminator(nn.Module):
    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.sub_discs = nn.ModuleList([ScaleDiscriminator(s) for s in scales])
    
    def logits(self, x):
        logits = []
        for sd in self.sub_discs:
            l, _ = sd(x)
            logits.append(l)
        return logits

    def feat_loss(self, fake, real):
        loss = 0
        for sd in self.sub_discs:
            _, feat_f = sd(fake)
            _, feat_r = sd(real)
            for a, b in zip(feat_f, feat_r):
                loss += (a - b).abs().mean()
        return loss

