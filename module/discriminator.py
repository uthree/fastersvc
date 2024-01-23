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


class MultiScaleDiscriminator(nn.Module):
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


class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft, channels):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.channels = channels

        self.layers = nn.ModuleList([
                weight_norm(nn.Conv2d(1, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1)),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1)),
            ])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        window = torch.hann_window(self.n_fft, device=x.device)
        x = torch.stft(x, self.n_fft, self.hop_length, return_complex=True, window=window).abs()
        x = x.unsqueeze(1)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
        return x, feats


class MultiSTFTDiscriminator(nn.Module):
    def __init__(self, n_ffts=[1024, 2048, 4096], channels=32):
        super().__init__()
        self.sub_discs = nn.ModuleList([STFTDiscriminator(s, channels) for s in n_ffts])
    
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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.msd = MultiScaleDiscriminator()
        self.mrd = MultiSTFTDiscriminator()

    def logits(self, x):
        return self.msd.logits(x) + self.mrd.logits(x)

    def feat_loss(self, x, y):
        return self.msd.feat_loss(x, y) + self.mrd.feat_loss(x, y)

