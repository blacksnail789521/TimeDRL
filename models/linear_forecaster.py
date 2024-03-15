import torch.nn as nn
from layers.einops_modules import RearrangeModule


class Model(nn.Module):
    def __init__(self, D, C, T_p, T_out, dropout=0.1, enable_channel_independence=True):
        super().__init__()
        self.enable_channel_independence = enable_channel_independence

        self.T_out = T_out
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout)
        if self.enable_channel_independence:
            self.linear = nn.Linear(T_p * D, T_out)
            self.rearrange = RearrangeModule("(B C) T_out -> B T_out C", C=C)
        else:
            self.linear = nn.Linear(T_p * D, T_out * C)
            self.rearrange = RearrangeModule("B (T_out C) -> B T_out C", C=C)

    def forward(self, x):  # (B * C, T_p, D)
        x = self.flatten(x)  # (B * C, T_p*D)
        x = self.dropout(x)
        x = self.linear(x)  # (B * C, T_out) or (B, T_out * C)
        x = self.rearrange(x)  # (B, T_out, C)
        return x
