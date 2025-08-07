import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
import argparse
from ptflops import get_model_complexity_info 

def test_params_flop(model, input_res):
    # input_res = (seq_len, input_size)
    macs, params = get_model_complexity_info(model, input_res, as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model_params, macs, params

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, args):
        super(TimesBlock, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.k = args.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(args.d_model, args.d_ff,
                               num_kernels=args.num_kernels),
            nn.GELU(),
            Inception_Block_V1(args.d_ff, args.d_model,
                               num_kernels=args.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class Model(nn.Module):
    """
    Simplified TimesNet without Embedding and Norm Layer.
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.num_layers = args.num_layers
        self.norm_flag = args.norm_flag
        self.model = nn.ModuleList([TimesBlock(args) for _ in range(args.num_layers)])
        
        self.input_proj = nn.Linear(args.input_size, args.d_model)

        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(args.d_model, args.output_dim, bias=True)

    def forecast(self, x_enc):
        if self.norm_flag:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc.sub(means)
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc.div(stdev)
        enc_out = self.input_proj(x_enc)  # [B, T, C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        for i in range(self.num_layers):
            enc_out = self.model[i](enc_out)  
        
        dec_out = self.projection(enc_out)
        if self.norm_flag:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out.mul(
                    (stdev[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len + self.seq_len, 1)))
            dec_out = dec_out.add(
                    (means[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len + self.seq_len, 1)))

        return dec_out
    
    def forward(self, x):
        dec_out = self.forecast(x)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]



if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(description='TimesNet Config')

        parser.add_argument('--seq_len', type=int, default=640, help='Input sequence length')
        parser.add_argument('--pred_len', type=int, default=12, help='Prediction length')
        parser.add_argument('--input_size', type=int, default=1, help='Input feature dimension')
        parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
        parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
        parser.add_argument('--num_layers', type=int, default=3, help='Number of TimesBlocks')
        parser.add_argument('--top_k', type=int, default=3, help='top_k of fft')
        parser.add_argument('--d_ff', type=int, default=3, help='Feed-Forward Dimension')
        parser.add_argument('--num_kernels', type=int, default=3, help='cnn kernel filters')
        parser.add_argument('--norm_flag', default=True, help='flage of normalization at begining')

        parser.add_argument('--batch_size', type=int, default=32, help='Batch size')




        args = parser.parse_args()
        return args
    
    args = parse_args()
    model = Model(args)
    print("\nModel structure:\n", model)

    # 随机输入测试
    x = torch.randn(args.batch_size, args.seq_len, args.input_size)
    output = model(x)
    print("\nOutput shape:", output.shape)

    # 打印 MACs 和参数量
    model_params, macs, params = test_params_flop(model, (args.seq_len, args.input_size))
    print('\nINFO: Trainable parameter count: {:.5f}M'.format(model_params / 1e6))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))