import torch
import torch.nn as nn
import torch.nn.functional as F


def leaky_relu(x):
    return F.leaky_relu(x, 0.1)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = nn.utils.weight_norm if use_spectral_norm == False else nn.utils.spectral_norm
        
        k = kernel_size
        s = stride
        convs = [
                nn.Conv2d(1, 32, (k, 1), (s, 1), (get_padding(5, 1), 0)),
                nn.Conv2d(32, 64, (k, 1), (s, 1), (get_padding(5, 1), 0), groups=2),
                nn.Conv2d(64, 128, (k, 1), (s, 1), (get_padding(5, 1), 0), groups=4),
                nn.Conv2d(128, 256, (k, 1), (s, 1), (get_padding(5, 1), 0), groups=8),
                nn.Conv2d(256, 256, (k, 1), 1, (2, 0), groups=8),
                ]
        self.convs = nn.ModuleList([norm_f(c) for c in convs])
        self.post = norm_f(nn.Conv2d(256, 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        fmap = []
        x = x.unsqueeze(1)

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = leaky_relu(x)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x, fmap


class MultiPeriodicDiscriminator(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.sub_discs = nn.ModuleList([])
        for p in periods:
            self.sub_discs.append(DiscriminatorP(p))

    def forward(self, x):
        feats = []
        logits = []
        for d in self.sub_discs:
            logit, fmap = d(x)
            logits.append(logit)
            feats += fmap
        return logits, feats


class DiscriminatorS(nn.Module):
    def __init__(self, scale=1, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.weight_norm if use_spectral_norm == False else nn.utils.spectral_norm
        
        self.pool = nn.AvgPool1d(scale)
        convs = [
                nn.Conv1d(1, 32, 15, 1, 7),
                nn.Conv1d(32, 64, 41, 2, 20, groups=2),
                nn.Conv1d(64, 128, 41, 2, 20, groups=4),
                nn.Conv1d(128, 256, 41, 2, 20, groups=8),
                nn.Conv1d(256, 256, 41, 2, 20, groups=8),
                nn.Conv1d(256, 256, 41, 2, 20, groups=8),
                nn.Conv1d(256, 256, 41, 2, 20, groups=8),
                ]
        self.convs = nn.ModuleList([norm_f(c) for c in convs])
        self.post = norm_f(nn.Conv1d(256, 1, 3, 1, 1))

    def forward(self, x):
        fmap = []
        x = x.unsqueeze(1)
        x = self.pool(x)
        fmap.append(x)
        for l in self.convs:
            x = l(x)
            x = leaky_relu(x)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.sub_discs = nn.ModuleList([])
        for s in scales:
            self.sub_discs.append(DiscriminatorS(s))

    def forward(self, x):
        feats = []
        logits = []
        for d in self.sub_discs:
            logit, fmap = d(x)
            logits.append(logit)
            feats += fmap
        return logits, feats


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.MPD = MultiPeriodicDiscriminator()
        self.MSD = MultiScaleDiscriminator()

    def forward(self, x):
        mpd_logits, mpd_feats = self.MPD(x)
        msd_logits, msd_feats = self.MSD(x)
        return mpd_logits + msd_logits, mpd_feats + msd_feats
