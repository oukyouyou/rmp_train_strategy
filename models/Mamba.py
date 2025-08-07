__all__ = ['Mamba']
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from torchinfo import summary

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.pred_len = args.pred_len
        self.num_layers = args.num_layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=args.d_model,
                d_state=args.d_state,
                d_conv=args.d_conv,
                expand=args.expand,
            )
            for _ in range(self.num_layers)
        ])

        self.proj = nn.Linear(args.input_size, args.d_model)
        self.out_layer = nn.Linear(args.d_model, args.output_dim, bias=False)

    def forecast(self, x):
        x = self.proj(x) # [batch, seq_len, d_model]
        for mamba in self.mamba_layers:
            x = x + mamba(x)
        x_out = self.out_layer(x)
        return x_out

    def forward(self, x: torch.Tensor, **kwargs):
        x_out = self.forecast(x)
        return x_out[:, -self.pred_len:, :]


if __name__ == "__main__":
    def args():
        parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
        parser.add_argument('--num_layers', type=int, default=2, help='Model dimension d_model')
        parser.add_argument('--pred_len', type=int, default=12, help='Model dimension d_model')
        parser.add_argument('--input_size', type=int, default=1, help='Model dimension d_model')
        parser.add_argument('--output_dim', type=int, default=1, help='Model dimension d_model')

        parser.add_argument('--d_model', type=int, default=8, help='Model dimension d_model')
        parser.add_argument('--d_state', type=int, default=16, help='SSM state expansion factor')
        parser.add_argument('--d_conv', type=int, default=4, help='Local convolution width')
        parser.add_argument('--expand', type=int, default=2, help='Block expansion factor')

        args = parser.parse_args()
        return args

    # Instantiate the LSTM model
    args = args()
    print(args)
    model = Model(args)
    device = torch.device("gpu")
    model = model.to(device)
    print(model)