import torch
import torch.nn as nn
import torch.nn.functional as F

# 1d Adaptive instance normalization for voice style transfer
# based https://arxiv.org/abs/1703.06868
# This technique may be useful for converting recorded audio.

def encode_style(content, eps=1e-6):
    mean = content.mean(dim=2, keepdim=True)
    std = content.std(dim=2, keepdim=True) + eps
    return (mean, std)


def apply_style(content, style, eps=1e-6):
    y_mean, y_std = style
    x_mean = content.mean(dim=2, keepdim=True)
    x_std = content.std(dim=2, keepdim=True) + eps
    z = (content - x_mean) / x_std
    out = z * y_std + y_mean
    return out
