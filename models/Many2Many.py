__all__ = ['Many2Many']

import torch
import argparse
from torch import nn
from torchinfo import summary
from utils.logger import LoggerMixin

# lstm-related
class Model(nn.Module, LoggerMixin):
    """Our LSTM implementation. Code and concept were inspired by.

    - Lin et al 2019; DOI 10.1088/1361-6560/ab13fa
    - Lombardo et al 2022; DOI 10.1088/1361-6560/ac60b7, https://github.com/LMUK-RADONC-PHYS-RES/lstm_centroid_prediction;
    """

    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.input_features = args.input_features
        self.num_layers = args.num_layers
        self.output_dim = args.output_dim
        self.dropout = args.dropout

        self.lstm = nn.LSTM(
            num_layers=self.num_layers,
            input_size=self.input_features,
            hidden_size=self.hidden_dim,
            dropout=self.dropout,
            batch_first=True,
        )
        self.fcl = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input_batch: torch.Tensor, **kwargs):
        # shape of input batch -> # batch, seq_length, input_features
        batch_size, seq_length, input_features = input_batch.size()
        self.logger.debug(f"{batch_size=}, {seq_length=}, {input_features=} ")
        self.h_c = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(input_batch.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(input_batch.device),
        )
        predictions = torch.zeros((batch_size, seq_length, self.output_dim)).to(input_batch.device)
        for step in range(seq_length):
            input_per_step = input_batch[:, step : step + 1, :]
            lstm_out, self.h_c = self.lstm(input_per_step, self.h_c)
            final_hidden_state = lstm_out[:, -1, :].view(
                batch_size, -1
            )  # final in the sense of hidden layer of the last layer in the stacked lstm
            pred_i = self.fcl(final_hidden_state)
            predictions[:, step, :] = pred_i 
        self.logger.debug(f"{predictions.shape=}")
        return predictions


def args():
    parser = argparse.ArgumentParser(description='Many2Many LSTM Model')
    # LSTM model parameters
    parser.add_argument('--seq_len', type=int, default=196, help='Input sequence length')
    parser.add_argument('--input_features', type=int, default=1, help='Number of input features')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--hidden_dim', type=int, default=15, help='Hidden dimension of LSTM')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Get arguments from command line
    args = args()
    print(args)
    
    # Instantiate the Many2Many model
    model = Many2Many(args)
    device = torch.device("cpu")
    model = model.to(device)
    print(model)

    # Create sample input
    input_data = torch.randn(256, args.seq_len, args.input_features)  # Example input shape
    print("Input shape:", input_data.shape)
    
    # Forward pass
    output = model(input_data)
    print("Output shape:", output.shape)

    # Print model summary
    summary(model=model, input_size=(256, args.seq_len, args.input_features), device="cpu")
