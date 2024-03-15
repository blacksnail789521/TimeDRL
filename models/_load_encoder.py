import torch
import torch.nn as nn
from pathlib import Path
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.einops_modules import RearrangeModule
from einops import rearrange
from tsai.all import ResBlock, TemporalConvNet


def load_transformer(configs, mask_flag=False):
    return Encoder(
        [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(
                        mask_flag=mask_flag,  # Encoder: False, Decoder: True
                        factor=None,  # Unused
                        attention_dropout=configs.dropout,
                        output_attention=False,
                    ),
                    configs.d_model,
                    configs.n_heads,
                ),
                configs.d_model,
                d_ff=None,  # 4 * configs.d_model
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for l in range(configs.n_layers)
        ],
        norm_layer=torch.nn.LayerNorm(configs.d_model),
    ).float()


def load_resnet(configs, kss=[7, 5, 3]):
    class ResNet(nn.Module):
        def __init__(self, d_model, kss=[7, 5, 3]):
            super().__init__()

            self.resblock1 = ResBlock(d_model, d_model, kss=kss)
            self.resblock2 = ResBlock(d_model, d_model, kss=kss)
            self.resblock3 = ResBlock(d_model, d_model, kss=kss)

        def forward(self, x):  # (B, T, C)
            x = x.permute(0, 2, 1)  # (B, C, T)
            x = self.resblock1(x)  # (B, D/2, T)
            x = self.resblock2(x)  # (B, D, T)
            x = self.resblock3(x)  # (B, D, T)
            x = x.permute(0, 2, 1)  # (B, T, D)
            return x

    return ResNet(configs.d_model, kss=kss).float()


def load_lstm(configs, hidden_size=128, num_layers=1, bidirectional=False):
    class BiLSTMWithProjection(nn.Module):
        def __init__(self, C, hidden_size=6, num_layers=1, bidirectional=False):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            # Bidirectional LSTM
            self.lstm = nn.LSTM(
                C,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            # Linear layer to project output back to input feature size
            lienar_input_size = hidden_size * 2 if bidirectional else hidden_size
            self.linear = nn.Linear(lienar_input_size, C)
        def forward(self, x):
            # Forward propagate LSTM
            output, _ = self.lstm(x)
            # Pass the output through the linear layer
            output = self.linear(output)
            return output

    return BiLSTMWithProjection(
        configs.d_model,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
    ).float()


def load_tcn(configs, ks=7):
    return nn.Sequential(
        RearrangeModule("B T D -> B D T"),
        TemporalConvNet(
            c_in=configs.d_model,
            layers=[configs.d_model] * configs.n_layers,
            ks=ks,
            dropout=configs.dropout,
        ),
        RearrangeModule("B D T -> B T D"),
    ).float()


def load_encoder(configs, **kwargs):
    print(f"Loading encoder: {configs.encoder_arch}")
    if configs.encoder_arch == "transformer_encoder":
        return load_transformer(configs, mask_flag=False)
    elif configs.encoder_arch == "transformer_decoder":
        return load_transformer(configs, mask_flag=True)
    elif configs.encoder_arch == "resnet":
        return load_resnet(configs, **kwargs)
    elif configs.encoder_arch == "tcn":
        return load_tcn(configs, **kwargs)
    elif configs.encoder_arch == "lstm":
        return load_lstm(configs, bidirectional=False, **kwargs)
    elif configs.encoder_arch == "bilstm":
        return load_lstm(configs, bidirectional=True, **kwargs)
    else:
        raise ValueError(f"Unknown encoder architecture: {configs.encoder_arch}")
