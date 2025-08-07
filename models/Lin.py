__all__ = ['Lin']

import torch
import argparse
from torch import nn
from torchinfo import summary

#from utils.logger import LoggerMixin

#from layers.ExternalAttention import ExternalAttention


#     - Lin et al 2019; DOI 10.1088/1361-6560/ab13fa

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.DMS_flag = args.DMS_flag # For IMS and DMS manner. 1 = DMS
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.input_size = args.input_size

        self.lstm = nn.LSTM(
            num_layers=args.num_layers,
            input_size=args.input_size,
            hidden_size=args.hidden_dim,
            dropout=args.dropout,
            batch_first=True,
        )
        # #self.output_dim = 1 if args.features == 'S' or args.features == 'MS' else 3        
        # if self.DMS_flag: # For multipule output (Remained to fix)
        #     self.fc_mid_dim = args.fc_mid_dim
        #     #self.fc = nn.Linear(self.hidden_dim_lin * self.seq_len, self.pred_len * d )
        #     self.fc1 = nn.Linear(self.hidden_dim, self.fc_mid_dim)
        #     self.fc2 = nn.Linear(self.fc_mid_dim, self.output_dim)
        # else:        # For single output (defalut)
        #     self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
    def forward(self, x: torch.Tensor, **kwargs):
        # shape of input x -> # batch, seq_len, input_size
        batch_size, seq_len, input_size = x.size()
        #self.logger.debug(f"{batch_size=}, {seq_len=}, {input_size=} ")
        self.h_c = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device),
        )
        output = torch.zeros((batch_size, self.seq_len, self.output_dim)).to(x.device)

        if self.DMS_flag: # For DMS 
            output, _ = self.lstm(x, self.h_c)
            output = self.fc(output[:, -1:, :]) # Taking only the last time step's output
        else:        # For IMS (defalut)
            predictions = torch.zeros(batch_size, self.args.pred_len, 1).to(x.device)
            current_input = x.clone() 
            h_c = self.h_c if hasattr(self, 'h_c') else None 

            for step in range(self.args.pred_len):
                lstm_out, h_c = self.lstm(current_input,h_c)  # (batch, seq_len, hidden)
                next_pred = self.fc(lstm_out[:, -1, :])  # (batch, 1)
                predictions[:, step] = next_pred

                current_input = torch.cat([
                    current_input[:, 1:, :], 
                    next_pred.unsqueeze(1)  # (batch, seq_len-1, 1)
                ], dim=1)[:, -self.args.seq_len:, :]  # (batch, seq_len+1, 1)
                
            return predictions  # (batch, pred_steps, 1)
        
        return output


if __name__ == "__main__":
    def args():
        parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
        # forecasting task
        parser.add_argument('--seq_len', type=int, default=196, help='input sequence length')
        parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
        parser.add_argument('--features', type=str, default='M')
        parser.add_argument('--dropout', type=float, default=0.1, help='num of encoder lstm_layers')
        parser.add_argument('--DMS_flag', default= False, help='Flag True for DMS or False for IMS')
        # Lin(PureLSTM)
        parser.add_argument('--input_size', type=int, default=1, help='input features of LSTM (m)')
        parser.add_argument('--output_dim', type=int, default=1, help='output dimension of LSTM (m)')
        parser.add_argument('--hidden_dim', type=int, default=15, help='hidden dimension of LSTM (m)')
        parser.add_argument('--num_layers', type=int, default=3, help='num of encoder lstm_layers')

        args = parser.parse_args()
        return args

    # Instantiate the LSTM model
    args = args()
    print(args)
    model = Model(args)
    device = torch.device("cpu")
    model = model.to(device)
    print(model)

    # # Create sample input
    # input_data = torch.randn(256, 196, 1) # for features = M
    # print("Input shape:", input_data.shape)
    # # Forward pass
    # output = model(input_data)
    # print("Output shape:", output.shape)

    # summary(model=model, input_size=(256, 196, 3), device="cpu")



            # for step in range(self.seq_len):
            #     input_per_step = x[:, step : step + 1, :]
            #     lstm_out, self.h_c = self.lstm(input_per_step, self.h_c)
            #     final_hidden_state = lstm_out[:, -1, :].view(batch_size, -1)  # final in the sense of hidden layer of the last layer in the stacked lstm
            #     output_i = self.fc(final_hidden_state)
            #     output[:, step, :] = output_i
