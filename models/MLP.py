__all__ = ['MLP']

import argparse
import torch
import torch.nn as nn
from models.DyT import DyT 
from models.Revin import RevIN 
import inspect
from torchinfo import summary
# from utils.tools import test_params_flop

ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'Dyt': DyT,  
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
}

# MLP Model Class
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.use_revin = args.use_revin
        if args.MO_flag:
            self.output_dim = args.pred_len
        else:
            self.output_dim = args.output_dim
        if self.use_revin:
            self.revin = RevIN(args.seq_len)

        if args.activation_function == 'Dyt':
            self.activation_fn = DyT(args.seq_len)
        else:
            self.activation_fn = ACTIVATION_FUNCTIONS.get(args.activation_function, nn.ReLU)()

        self.layers = []
        seq_len = args.seq_len

        if args.output_dim != 1:
            self.layers.append(nn.Linear(args.output_dim, args.hidden_dim))
            self.layers.append(self.activation_fn)
            self.layers.append(nn.Dropout(args.dropout))
            self.layers.append(nn.Flatten(start_dim=1))  # [batch, seq_len, hidden_dim] -> [batch, seq_len * hidden_dim]
            flattened_dim = seq_len * args.hidden_dim  
            for i in range(args.num_layers):
                hidden_dim = args.hidden_dim if i != args.num_layers - 1 else self.output_dim
                self.layers.append(nn.Linear(flattened_dim if i == 0 else hidden_dim, hidden_dim))
                if i != args.num_layers - 1:
                    self.layers.append(self.activation_fn)
                    self.layers.append(nn.Dropout(args.dropout))
                if i == 0:
                    flattened_dim = hidden_dim
        else: # default
            for i in range(args.num_layers):
                hidden_dim = args.hidden_dim if i != args.num_layers - 1 else self.output_dim
                self.layers.append(nn.Linear(seq_len, hidden_dim))
                if i != args.num_layers - 1:
                    self.layers.append(self.activation_fn)
                    self.layers.append(nn.Dropout(args.dropout))
                seq_len = hidden_dim

        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, seq_len, features] → [batch, features, seq_len]
        if self.use_revin:
            x = self.revin(x, mode='norm')
        out = self.network(x)
        if self.use_revin:
            out = self.revin(out, mode='denorm')
        out = out.transpose(1, 2)  # [batch, features, hidden_dim] → [batch, hidden_dim, features]
        return out

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        #print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return model_params, macs, params 
    
if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(description='MLP Model Configuration')
        parser.add_argument('--seq_len', type=int, default=640, help='Input size of the model')
        parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
        parser.add_argument('--MO_flag', default= True, help='Flag for multi-output or single-output')
        # MLP
        parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in MLP')  # 1层、2层、3层等
        parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer size')
        parser.add_argument('--output_dim', type=int, default=1, help='Output size of the model')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
        # 激活函数选择
        parser.add_argument('--use_revin', type=bool, default=False, help='Enable RevIN normalization')
        parser.add_argument('--activation_function', type=str, default='relu', choices=['relu', 'leaky_relu', 'tanh', 'sigmoid', 'Dyt'],
                            help='Activation function to use in the model')
        
        # 训练参数
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        
        # 其他配置
        args = parser.parse_args()

        return args
    
    args = parse_args()
    model = Model(args)
    print(model)
    x = torch.randn(args.batch_size, args.seq_len,args.output_dim)  # 假设batch_size为32, seq_len为64

    model_params, macs, params = test_params_flop(self.model,(batch_x.shape[1],batch_x.shape[2]))
    print('INFO: Trainable parameter count: {:.5f}M'.format(model_params / 1000000.0))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    output = model(x)
    print(output.shape)  # (batch_size, output_dim)
    #summary(model=model, input_size=(256, 196, 1), device="cpu")