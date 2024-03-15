import torch.nn as nn
from layers.einops_modules import RearrangeModule


class Model(nn.Module):
    def __init__(self, D, C, K, dropout=0.1, enable_channel_independence=True):
        super().__init__()
        self.enable_channel_independence = enable_channel_independence

        self.dropout = nn.Dropout(dropout)
        if self.enable_channel_independence:
            self.rearrange = RearrangeModule("(B C) D -> B (C D)", C=C)
            self.linear = nn.Linear(C * D, K)
        else:
            self.linear = nn.Linear(D, K)

    def forward(self, x):  # (B * C, D)
        x = self.dropout(x)
        if self.enable_channel_independence:
            x = self.rearrange(x)  # (B, C * D)
            x = self.linear(x)  # (B, K)
        else:
            x = self.linear(x)  # (B, K)
        return x
