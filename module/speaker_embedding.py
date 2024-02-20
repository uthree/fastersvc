import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEmbedding(nn.Module):
    def __init__(self, num_speakers=8192, spk_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_speakers, spk_dim)

    def forward(self, x): # [Batch] -> [Batch, spk_dim, 1]
        return self.embedding(x).unsqueeze(2)
