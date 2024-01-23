import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def safe_log(x):
    return torch.log(x.clamp_min(1e-6))


class MultiScaleSTFTLoss(nn.Module):
    def __init__(
            self,
            scales=[16, 32, 64, 128, 256, 512, 1024, 2048],
            alpha=1.0):
        super().__init__()
        self.scales = scales
        self.alpha = alpha

    def forward(self, x, y):
        loss = 0
        for s in self.scales:
            hop_length = s
            n_fft = s * 4
            x_spec = torch.stft(x, n_fft, hop_length, return_complex=True).abs()
            y_spec = torch.stft(y, n_fft, hop_length, return_complex=True).abs()
            loss += (x_spec - y_spec).abs().mean() + self.alpha * (safe_log(x_spec) - safe_log(y_spec)).abs().mean()
        return loss


class LogMelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate=48000, n_fft=2048, n_mels=128):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft, n_mels=n_mels)

    def forward(self, x, y):
        x = safe_log(self.to_mel(x))
        y = safe_log(self.to_mel(y))
        loss = (x - y).abs().mean()
        return loss
