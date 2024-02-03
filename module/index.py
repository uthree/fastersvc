import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import match_features


# kNN based style convertor for onnx exporting
class IndexForOnnx(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = nn.Parameter(index)

    def forward(self, x, metrics='L2'):
        return match_features(x, self.index, metrics=metrics)
