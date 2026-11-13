"""Species-only projection and classification."""
import torch
from torch import nn


class SpeciesNet(nn.Module):
    def __init__(self, mode, classes, projection=256, dropout=.2):
        super().__init__()
        self.mode = mode
        if mode not in ('bio', 'dino', 'fusion'):
            raise ValueError(mode)
        if mode != 'dino':
            self.bio = nn.Sequential(nn.Linear(768, projection), nn.LayerNorm(projection))
        if mode != 'bio':
            self.dino = nn.Sequential(nn.Linear(384, projection), nn.LayerNorm(projection))
        self.head = nn.Sequential(nn.Linear(projection * (2 if mode == 'fusion' else 1), 256),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(256, classes))

    def forward(self, bio, dino):
        if self.mode == 'bio':
            features = self.bio(bio)
        elif self.mode == 'dino':
            features = self.dino(dino)
        else:
            features = torch.cat((self.bio(bio), self.dino(dino)), dim=-1)
        return self.head(features)
