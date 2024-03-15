import torch
import torch.nn as nn
from pathlib import Path
from einops import rearrange

from layers.Embed import DataEmbedding, Patching
from layers.RevIN import RevIN
from layers.einops_modules import RearrangeModule
from models._load_encoder import load_encoder
from dataset_loader.augmentation import data_augmentation


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # RevIN (without affine transformation)
        self.revin = RevIN(self.args.C, affine=False)

        # Input layer
        self._set_input_layer()

        # Pretext layer (for predictive and contrastive tasks)
        self._set_pretext_layer()

        # Encoder
        self.encoder = load_encoder(self.args)

        # [CLS] token (we need this no matter what `get_i` is)
        if self.args.enable_channel_independence:
            self.cls_token = nn.Parameter(
                torch.randn(1, self.args.C, self.args.patch_len)
            )
        else:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, self.args.C * self.args.patch_len)
            )

    def _set_input_layer(self):
        self.patching = Patching(
            self.args.patch_len, self.args.stride, self.args.enable_channel_independence
        )  # (B, T_in, C) -> (B * C, T_p, P) (Enable CI) or (B, T_p, C * P) (Disable CI)
        if self.args.enable_channel_independence:
            self.input_layer = DataEmbedding(
                last_dim=self.args.patch_len,
                d_model=self.args.d_model,
                dropout=self.args.dropout,
                pos_embed_type=getattr(self.args, "pos_embed_type", "fixed"),
                token_embed_type=getattr(self.args, "token_embed_type", "linear"),
                token_embed_kernel_size=getattr(
                    self.args, "token_embed_kernel_size", 3
                ),
            )  # (B * C, T_p, P) -> (B * C, T_p, D)
        else:
            self.input_layer = DataEmbedding(
                last_dim=self.args.C * self.args.patch_len,
                d_model=self.args.d_model,
                dropout=self.args.dropout,
                pos_embed_type=getattr(self.args, "pos_embed_type", "fixed"),
                token_embed_type=getattr(self.args, "token_embed_type", "linear"),
                token_embed_kernel_size=getattr(
                    self.args, "token_embed_kernel_size", 3
                ),
            )  # (B, T_p, C * P) -> (B, T_p, D)

    def _set_pretext_layer(self):
        # Predictive task
        if self.args.enable_channel_independence:
            self.predictive_linear = nn.Sequential(
                nn.Dropout(self.args.dropout),
                nn.Linear(self.args.d_model, self.args.patch_len),
            )  # (B * C, T_p, D) -> (B * C, T_p, P)
        else:
            self.predictive_linear = nn.Sequential(
                nn.Dropout(self.args.dropout),
                nn.Linear(self.args.d_model, self.args.C * self.args.patch_len),
            )  # (B, T_p, D) -> (B, T_p, C * P)

        # (For contrastive task) set i_dim based on get_i,
        # and set additional layers if necessary
        if self.args.get_i == "cls":
            assert self.args.i_dim == self.args.d_model
        elif self.args.get_i == "last":
            assert self.args.i_dim == self.args.d_model
        elif self.args.get_i == "gap":
            assert self.args.i_dim == self.args.d_model
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif self.args.get_i == "all":
            assert self.args.i_dim == self.args.T_p * self.args.d_model
            if self.args.enable_channel_independence:
                self.flatten = RearrangeModule(
                    "(B C) T_p D -> (B C) (T_p D)",
                    C=self.args.C,
                    T_p=self.args.T_p,
                    D=self.args.d_model,
                )
            else:
                self.flatten = RearrangeModule(
                    "B T_p D -> B (T_p D)",
                    T_p=self.args.T_p,
                    D=self.args.d_model,
                )

        # Contrastive task
        self.contrastive_predictor = nn.Sequential(
            nn.Linear(self.args.i_dim, self.args.i_dim // 2),
            nn.BatchNorm1d(self.args.i_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.i_dim // 2, self.args.i_dim),
        )  # (B * C, i_dim) -> (B * C, i_dim)

    def forward(self, x):  # (B, T_in, C)
        B, T_in, C = x.shape

        # Instance Normalization
        x = self.revin(x, "norm")  # (B, T_in, C)

        # Create two data (it should be the same if `data_aug` is `none`)
        if self.args.data_aug == "none":
            # dropout randomness
            x_1 = x
            x_2 = x
        else:
            x_1 = data_augmentation(x, self.args.data_aug)
            x_2 = data_augmentation(x, self.args.data_aug)

        # Patching
        x_1 = self.patching(
            x_1
        )  # (B * C, T_p, P) (Enable CI) or (B, T_p, C * P) (Disable CI)
        x_2 = self.patching(x_2)

        # [CLS] token
        if self.args.enable_channel_independence:
            cls_token = self.cls_token.expand(B, -1, -1)  # (B, C, P)
            cls_token = rearrange(cls_token, "B C P -> (B C) 1 P")  # (B * C, 1, P)
            x_1 = torch.cat([cls_token, x_1], dim=1)  # (B * C, T_p + 1, P)
            x_2 = torch.cat([cls_token, x_2], dim=1)
        else:
            cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, C * P)
            x_1 = torch.cat([cls_token, x_1], dim=1)  # (B, T_p + 1, C * P)
            x_2 = torch.cat([cls_token, x_2], dim=1)

        # First pass
        x_1 = self.input_layer(x_1)  # (B * C, T_p + 1, D) or (B, T_p + 1, D)
        z_1 = self.encoder(x_1)  # (B * C, T_p + 1, D) or (B, T_p + 1, D)

        # Second pass
        x_2 = self.input_layer(x_2)  # (B * C, T_p + 1, D) or (B, T_p + 1, D)
        z_2 = self.encoder(x_2)  # (B * C, T_p + 1, D) or (B, T_p + 1, D)

        # Predictive task
        t_1 = z_1[:, 1:, :]  # (B * C, T_p, D) or (B, T_p, D)
        t_2 = z_2[:, 1:, :]  # (B * C, T_p, D) or (B, T_p, D)
        x_pred_1 = self.predictive_linear(t_1)  # (B * C, T_p, P) or (B, T_p, C * P)
        x_pred_2 = self.predictive_linear(t_2)  # (B * C, T_p, P) or (B, T_p, C * P)

        # Contrastive task
        if self.args.get_i == "cls":
            i_1 = z_1[:, 0, :]  # (B * C, D) or (B, D)
            i_2 = z_2[:, 0, :]  # (B * C, D) or (B, D)
        elif self.args.get_i == "last":
            i_1 = t_1[:, -1, :]  # (B * C, D) or (B, D)
            i_2 = t_2[:, -1, :]
        elif self.args.get_i == "gap":
            i_1 = self.gap(t_1.transpose(1, 2)).squeeze(-1)  # (B * C, D) or (B, D)
            i_2 = self.gap(t_2.transpose(1, 2)).squeeze(-1)  # (B * C, D) or (B, D)
        elif self.args.get_i == "all":
            i_1 = self.flatten(t_1)  # (B * C, T_p * D) or (B, T_p * D)
            i_2 = self.flatten(t_2)  # (B * C, T_p * D) or (B, T_p * D)
        else:
            raise NotImplementedError
        i_1_pred = self.contrastive_predictor(i_1)  # (B * C, i_dim) or (B, i_dim)
        i_2_pred = self.contrastive_predictor(i_2)  # (B * C, i_dim) or (B, i_dim)

        return (
            t_1,  # (B * C, T_p, D) or (B, T_p, D)
            t_2,  # (B * C, T_p, D) or (B, T_p, D)
            x_pred_1,  # (B * C, T_p, P) or (B, T_p, C * P)
            x_pred_2,  # (B * C, T_p, P) or (B, T_p, C * P)
            i_1,  # (B * C, i_dim) or (B, i_dim)
            i_2,  # (B * C, i_dim) or (B, i_dim)
            i_1_pred,  # (B * C, i_dim) or (B, i_dim)
            i_2_pred,  # (B * C, i_dim) or (B, i_dim)
        )
