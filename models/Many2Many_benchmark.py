__all__ = ['Many2Many']

import torch
import argparse
from torch import nn
from torchinfo import summary

# lstm-related
class Model(nn.Module):
    """Our LSTM implementation. Code and concept were inspired by.

    - Lin et al 2019; DOI 10.1088/1361-6560/ab13fa
    - Lombardo et al 2022; DOI 10.1088/1361-6560/ac60b7, https://github.com/LMUK-RADONC-PHYS-RES/lstm_centroid_prediction;
    """

    def __init__(
        self,
        input_features: int,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_features = input_features
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            num_layers=num_layers,
            input_size=input_features,
            hidden_size=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.fcl = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_batch: torch.Tensor, **kwargs):
        # shape of input batch -> # batch, seq_length, input_features
        batch_size, seq_length, input_features = input_batch.size()
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
        return predictions

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
