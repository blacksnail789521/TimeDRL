import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from collections import OrderedDict


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class PositionalEmbedding_trainable(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create a parameter tensor of size [max_length, d_model]
        pe = torch.randn(max_len, d_model).float()

        # Register it as a parameter that will be updated during training
        self.pe = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        # Just return the first T position embeddings
        return self.pe[None, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, last_dim, d_model, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2  # `same` padding
        self.tokenConv = nn.Conv1d(
            in_channels=last_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, C, d_model):
        super().__init__()

        w = torch.zeros(C, d_model).float()
        w.requires_grad = False

        position = torch.arange(0, C).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(C, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super().__init__()

        minute_size = 4  # 15 minutes
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super().__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(
        self,
        last_dim,  # last dimension of input
        d_model,
        dropout=0.1,
        pos_embed_type="none",
        token_embed_type="linear",
        token_embed_kernel_size=3,
    ):
        super().__init__()

        # Positional embedding (none, learnable, fixed)
        if pos_embed_type == "none":
            self.position_embedding = None
        elif pos_embed_type == "learnable":  # nn.Parameter
            self.position_embedding = PositionalEmbedding_trainable(d_model)
        elif pos_embed_type == "fixed":  # sin/cos
            self.position_embedding = PositionalEmbedding(d_model)
        else:
            raise NotImplementedError

        # Token embedding (linear, conv)
        if token_embed_type == "linear":
            self.token_embedding = nn.Linear(last_dim, d_model, bias=False)
        elif token_embed_type == "conv":
            self.token_embedding = TokenEmbedding(
                last_dim=last_dim, d_model=d_model, kernel_size=token_embed_kernel_size
            )
        else:
            raise NotImplementedError

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):  # (B, T, C)
        # Token embedding
        x = self.token_embedding(x)  # (B, T, D)

        # Position embedding
        if self.position_embedding is not None:
            x = x + self.position_embedding(x)  # (B, T, D)

        return self.dropout(x)


class Patching(nn.Module):
    def __init__(self, patch_len, stride, enable_channel_independence=True):
        super().__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.enable_channel_independence = enable_channel_independence
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

    def forward(self, x):  # (B, T, C)
        x = rearrange(x, "B T C -> B C T")  # (B, C, T)
        x = self.padding_patch_layer(x)  # (B, C, T+S)
        x = x.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # (B, C, T_p, P)
        if self.enable_channel_independence:
            x = rearrange(x, "B C T_p P -> (B C) T_p P")  # (B * C, T_p, P)
        else:
            x = rearrange(x, "B C T_p P -> B T_p (C P)")  # (B, T_p, C * P)

        return x
