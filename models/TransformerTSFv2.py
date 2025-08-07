__all__ = ['TransformerTSFv2']

import math
import torch
import argparse
from torch import nn
from torchinfo import summary
import torch.nn.functional as F
import numpy as np
#from utils.logger import LoggerMixin

class Model(nn.Module):
    """
    Implementation by  - Lombardo et al 2022; DOI 10.1088/1361-6560/ac60b7.
    Transformer for respiratory motion prediction utilized in Jeong et al 2022.
    (Paper: https://doi.org/10.1371/journal.pone.0275719
     Code: https://github.com/SangWoonJeong/Respiratory-prediction).
    Underlying Transformer architecture is based on https://arxiv.org/abs/2001.08317 and corresponding implementaion of
    Maclean https://github.com/LiamMaclean216/Pytorch-Transfomer

    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        input_size = 1
        self.layer_dim_val = configs.layer_dim_val

        self.pos = PositionalEncoding(self.layer_dim_val)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.layer_dim_val,
            nhead=configs.n_heads,
            dim_feedforward=self.layer_dim_val,
            dropout=configs.dropout,
            batch_first=True,
            activation=F.elu,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.layer_dim_val,
            nhead=configs.n_heads,
            dim_feedforward=self.layer_dim_val,
            dropout=configs.dropout,
            batch_first=True,
            activation=F.elu,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=configs.num_layers)
        self.enc_input_fc = nn.Linear(input_size, self.layer_dim_val)
        self.dec_input_fc = nn.Linear(input_size, self.layer_dim_val)
        self.out_fc = nn.Linear(self.seq_len * self.layer_dim_val, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expecting x: [batch, seq_len, 1]
        assert x.ndim == 3 and x.shape[2] == 1, f"Expected shape [B, T, 1], got {x.shape}"
        x = self._operations(x)
        return x

    def _operations(self, x: torch.Tensor) -> torch.Tensor:
        e = self.enc_input_fc(x)
        e = self.pos(e)
        e = self.encoder(e)
        d = self.dec_input_fc(x[:, -self.seq_len :])
        d = self.decoder(d, memory=e)
        #print(f'======123123123==========={self.seq_len * self.layer_dim_val}==================')
        #print(f'======123123123==========={np.shape(d)}==================')
        x = self.out_fc(d.flatten(start_dim=1))
        return x


class PositionalEncoding(nn.Module):
    """Taken from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[: x.size(1), :].squeeze(1)
        return x
    
if __name__ == "__main__":
    def args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--layer_dim_val', type=int, default=64)
        parser.add_argument('--seq_len', type=int, default=10)
        parser.add_argument('--pred_len', type=int, default=5)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--n_encoder_layers', type=int, default=2)
        parser.add_argument('--n_heads', type=int, default=4)
        parser.add_argument('--dropout', type=float, default=0.1)
        return parser.parse_args()

    # Instantiate the LSTM model
    args = args()
    print(args)
    model = Model(args)
    device = torch.device("cuda")
    model = model.to(device)
    print(model)