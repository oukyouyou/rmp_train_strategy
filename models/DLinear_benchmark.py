import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MovingAvg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DecompLinear(nn.Module):
    """Decomposition Linear Model.

    Originally published in https://arxiv.org/pdf/2205.13504.pdf.
    Code highly inspired by https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py
    """

    def __init__(self, seq_len, pred_len, individual, enc_in):
        super().__init__()
        self.input_paras = locals()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = SeriesDecomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        inital_x_shape = x.shape
        if x.shape[0] != 1:
            x = x.reshape(-1, x.shape[-1])
            x = x[:, :, None]
        else:
            # re shape x
            x = x.permute((1, 2, 0))
        # x: [batch, seq_len, input_features] -> [seq_len, input_features, batch]
        if x.shape[-1] != 1:
            raise ValueError(f"{x.shape=}; only implemented for batch size = 1")
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                print()
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        # [ batch * seq_len, 1, pred_horizon] -> [batch, seq_len, pre_horizon]
        if inital_x_shape[0] != 1:
            x = x.reshape((inital_x_shape[0], inital_x_shape[1], self.pred_len))
            return x[:, :, -1:]
        return x.permute((1, 0, 2))[:, :, -1:]  # to [Batch, Output length, Channel]
