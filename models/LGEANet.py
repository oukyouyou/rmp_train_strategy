__all__ = ['LGEANet']

# Cell
import torch
import argparse
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
from torch.nn import init
from ptflops import get_model_complexity_info

#from layers.ExternalAttention import ExternalAttention

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(1, hidden_size, num_layers,batch_first=True) # LSTM for each dim in module list, for org or pca data.
    def forward(self, x):
        #x = x.permute(0, 2, 1) 
        # x: [batch_size, sequence_length, input_size]
        # hidden_state = torch.zeros(x.size(0) , self.num_layers, self.hidden_size).to(x.device)
        # cell_state = torch.zeros(x.size(0), self.num_layers,  self.hidden_size).to(x.device)
        #output, _ = self.lstm(x, (hidden_state, cell_state))
        output, (h_n, c_n) = self.lstm(x)
        return output  # Return the entire output sequence

class GlobalTemporalConvolutionModule(nn.Module):
    def __init__(self, input_size, output_size, span, kernel_nums):
        super(GlobalTemporalConvolutionModule, self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, span)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.kernel_nums = kernel_nums
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for convolution [bs, m, T]
        x = self.conv(x)
        x = F.relu(x) # [bs, m]
        x = self.pool(x)
        x = x.repeat(1, 1, self.kernel_nums)  # Repeat the pooled tensor along the k dimension [bs, m, k]
        x = x.permute(0, 2, 1)  # reshape from [bs, m, k] to [bs, k, m]
        return x

class DenseLayer(nn.Module):
    def __init__(self, input_size, pred_len, dims ,DMS_flag):
        super(DenseLayer, self).__init__()
        self.DMS_flag = DMS_flag
        self.pred_len = pred_len
        if self.DMS_flag: # For DMS
            self.fc = nn.Linear(input_size,  pred_len * dims ) # output length * num_directions
        else:        # For IMS (defalut)
            self.fc = nn.Linear(input_size,  dims)
        #self.activation = nn.Softmax()
    
    def forward(self, x):
        batch_size, k, _ = x.size()
        x = x.view(batch_size, -1)  # Reshape to [batch_size, 3m * k]
        output = self.fc(x)
        #output = output.view(batch_size, self.output_size, -1)  # Reshape to [batch_size, self.output_size, num_directions]
        #output = self.activation(output)
        output = torch.sigmoid(output)
        if self.DMS_flag: # For DMS
            output = output.view(batch_size, self.pred_len, -1) 
        return output
    
class AutoregressiveComponent(nn.Module):
    def __init__(self, input_size, pred_len, dims ,DMS_flag):
        super(AutoregressiveComponent, self).__init__()
        self.DMS_flag = DMS_flag
        self.pred_len = pred_len
        if self.DMS_flag: # For DMS
            self.fc = nn.Linear(input_size,  pred_len * dims )
        else:        # For IMS (defalut)
            self.fc = nn.Linear(input_size,  dims)
        #self.activation = nn.Softmax()
    def forward(self, x):
        batch_size, _, _ = x.size()
        x = x.view(batch_size, -1)
        output = self.fc(x)
        #output = output.view(batch_size, self.output_size, -1)  # Reshape to [batch_size, self.output_size, num_directions]
        #output = self.activation(output)
        output = torch.sigmoid(output)
        if self.DMS_flag: # For DMS
            output = output.view(batch_size, self.pred_len, -1) 
        return output

class ExternalAttention(nn.Module):
    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, queries):

        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model
        return out
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        lstm_layers = configs.lstm_layers
        self.hidden_dim_m = configs.hidden_dim_m
        self.MS_flag = configs.MS_flag
        d = 1 if configs.features == 'S' or configs.features == 'MS' else 3
        k = configs.TCN_kernels_nums
        self.lstm_modules = nn.ModuleList([LSTM(d, self.hidden_dim_m,lstm_layers) for _ in range(d)])
        self.global_temporal_convs = nn.ModuleList([GlobalTemporalConvolutionModule(self.hidden_dim_m,self.hidden_dim_m, self.seq_len, k) for _ in range(d)]) 
        self.external_attention = ExternalAttention(self.hidden_dim_m * d, k)
        self.autoregressive = AutoregressiveComponent(self.seq_len * d,  self.pred_len, d, self.MS_flag)
        self.Dense_concat = DenseLayer(self.hidden_dim_m * d * k, self.pred_len ,d, self.MS_flag)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        lstm_outputs = [lstm_module(x[:, :, i].unsqueeze(2)) for i, lstm_module in enumerate(self.lstm_modules)]
        global_temporal_features = [global_temporal_conv(lstm_outputs[i]) for i, global_temporal_conv in enumerate(self.global_temporal_convs)]
        global_temporal_features = torch.cat(global_temporal_features, dim=2)
        attention_output = self.external_attention(global_temporal_features)
        dense_output = self.Dense_concat(attention_output)
        ar_output = self.autoregressive(x)
        output = dense_output
        output = output + ar_output

        # print('Shape of global_temporal_features:',global_temporal_features.size())
        # print('Shape of attention_output:',attention_output.size())
        # print('Shape of dense_output:',dense_output.size())
        # print('Shape of final output:',output.size())
        return output

if __name__ == "__main__":
    def args():
        parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
        # forecasting task
        parser.add_argument('--seq_len', type=int, default=196, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=98, help='start token length')
        parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
        parser.add_argument('--features', type=str, default='M')
        # LGEANet
        parser.add_argument('--hidden_dim_m', type=int, default=36, help='hidden dimension of LSTM (m)')
        parser.add_argument('--TCN_kernels_nums', type=int, default=512, help='external attention module’s hyperparameter S')
        parser.add_argument('--lstm_layers', type=int, default=2, help='num of encoder lstm_layers')
        parser.add_argument('--MS_flag', default= True, help='Flag for multi output or single output')
        args = parser.parse_args()
        return args
    args = args()
    model = Model(args)
    device = torch.device("cpu")
    model = model.to(device)
    print(model)
    input=torch.randn(256,args.seq_len,3)
    print('Shape of input:', input.size())
    output = model(input)
    print('Shape of output:', output.size())
    summary(model=model, input_size=(256,args.seq_len,3), device="cpu")
