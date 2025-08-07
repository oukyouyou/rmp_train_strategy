__all__ = ['WaveletLSTM']

import torch
import torch.nn as nn
import numpy as np
import pywt
import argparse


class ATrousWaveletDecomposition(nn.Module):
    """À trous wavelet decomposition module with fixed output length
    
    Features:
    - Generates odd-length filters for perfect padding
    - Maintains original signal length through symmetric padding
    - Energy normalization across scales
    """
    """À trous wavelet decomposition module with fixed output length"""
    def __init__(self, configs):
        super().__init__()
        self.J = configs.wLSTM_J
        self.wavelet = configs.wavelet
        self.register_buffer('filters', self._create_filter_bank())

    def _create_filter_bank(self):
        wavelet = pywt.Wavelet(self.wavelet)
        dec_lo = np.array(wavelet.dec_lo)

        # Dynamically compute maximum filter length across scales
        max_len = 1
        for j in range(self.J + 1):
            stride = 2 ** j
            current_len = (len(dec_lo) - 1) * stride + 1
            max_len = max(max_len, current_len)

        filters = []
        for j in range(self.J + 1):
            stride = 2 ** j
            extended = np.zeros((len(dec_lo) - 1) * stride + 1)
            extended[::stride] = dec_lo
            h_j = np.cumsum(extended)

            # Pad or trim to max_len
            if len(h_j) < max_len:
                h_j = np.pad(h_j, (0, max_len - len(h_j)), mode='constant')
            else:
                h_j = h_j[:max_len]

            # Energy normalization across scales
            norm = 1 / (np.sqrt(2) ** j)
            filters.append(h_j * norm)

        filters = np.stack(filters, axis=0)
        return torch.tensor(filters, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform à trous wavelet decomposition

        Args:
            x: Tensor of shape (batch_size, seq_len, 1)
        Returns:
            Tensor of shape (batch_size, J+1, seq_len) with wavelet coefficients
        """
        x = x.squeeze(-1)  # (batch, seq_len)
        coeffs = []
        for j in range(self.J + 1):
            # Extract non-zero filter coefficients
            filt = self.filters[j].cpu().numpy()
            valid = filt[filt != 0]
            k = len(valid)
            pad = (k - 1) // 2

            kernel = torch.tensor(valid, dtype=x.dtype).view(1, 1, -1).to(x.device)
            c = nn.functional.conv1d(x.unsqueeze(1), kernel, padding=pad)
            coeffs.append(c.squeeze(1))

        # Truncate to minimum length across scales
        min_len = min(c.shape[1] for c in coeffs)
        coeffs = [c[:, :min_len] for c in coeffs]
        return torch.stack(coeffs, dim=1)


class WaveletLSTM(nn.Module):
    """Wavelet LSTM with attention and residual connections"""
    def __init__(self, configs):
        super().__init__()
        self.J = configs.wLSTM_J
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Wavelet decomposition module
        self.wavelet_decomp = ATrousWaveletDecomposition(configs)

        # Attention mechanism for scale weighting
        self.attention = nn.Sequential(
            nn.Linear(self.J + 1, configs.attention_hidden),
            nn.ReLU(),
            nn.Linear(configs.attention_hidden, self.J + 1),
            nn.Softmax(dim=1)
        )

        # Multi-scale LSTM branches
        self.lstm_branches = nn.ModuleList([
            nn.LSTM(
                input_size=configs.input_features,
                hidden_size=configs.hidden_dim,
                num_layers=configs.num_layers,
                dropout=configs.dropout,
                batch_first=True
            ) for _ in range(self.J + 1)
        ])

        # Fusion network to combine scale outputs
        self.fusion = nn.Sequential(
            nn.Linear(configs.hidden_dim * (self.J + 1), configs.fusion_hidden),
            nn.ReLU(),
            nn.Linear(configs.fusion_hidden, configs.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        coeffs = self.wavelet_decomp(x)                # (batch, J+1, seq_len)
        # Compute attention weights per scale
        weights = self.attention(coeffs.mean(dim=2))  # (batch, J+1)
        # Apply weights to each scale
        weighted = coeffs * weights.unsqueeze(2)      # (batch, J+1, seq_len)

        # Process each scale with its own LSTM
        outs = []
        for j, lstm in enumerate(self.lstm_branches):
            inp = weighted[:, j, :].unsqueeze(-1)      # (batch, seq_len, 1)
            out, _ = lstm(inp)                         # (batch, seq_len, hidden)
            outs.append(out[:, -1, :])                # (batch, hidden)

        # Concatenate scale outputs and fuse
        concat = torch.cat(outs, dim=1)               # (batch, hidden*(J+1))
        return self.fusion(concat)                    # (batch, output_dim)


if __name__ == "__main__":

    # 测试不同配置
    test_cases = [
        {'wavelet': 'sym2', 'J': 3},  # 原始sym2小波
        {'wavelet': 'db4', 'J': 4},    # 更长的小波基
        {'wavelet': 'haar', 'J': 2}    # 奇数长度小波
    ]

    for case in test_cases:
        class Config:
            wLSTM_J = case['J']
            wavelet = case['wavelet']
        
        try:
            decomp = ATrousWaveletDecomposition(Config())
            print(f"Test passed for {case}:")
            print(f"  Filter shape: {decomp.filters.shape}")
        except Exception as e:
            print(f"Test failed for {case}: {str(e)}")

    # Test configuration
    def args():
        parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
        # forecasting task
        parser.add_argument('--seq_len', type=int, default=196, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=98, help='start token length')
        parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
        parser.add_argument('--input_features', type=int, default=1, help='Number of input channels/features')
        parser.add_argument('--output_dim', type=int, default=1,help='Dimension of model output')     
        parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability between LSTM layers')
        parser.add_argument('--MS_flag', default= True, help='Flag True for DMS or False for IMS')

        # Wavelet decomposition parameters
        parser.add_argument('--wLSTM_J',type=int, default= 3, help='Number of decomposition levels (J)')
        parser.add_argument('--wavelet',type=str,default= "sym2", help='Wavelet name for decomposition')

        # Lin(PureLSTM)
        parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of LSTM (m)')
        parser.add_argument('--num_layers', type=int, default=3, help='num of encoder lstm_layers')

        # Attention network parameters
        parser.add_argument('--attention_hidden', type=int, default=32, help='Hidden dimension in attention network')

        # Fusion network parameters
        parser.add_argument('--fusion_hidden', type=int, default=64, help='Hidden dimension in fusion network')

        # Training hyperparameters
        parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type (e.g., "Adam", "SGD")')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
        parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
        parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping threshold')

        args = parser.parse_args()
        return args

    # Instantiate the LSTM model
    args = args()
    print(args)

    model = WaveletLSTM(args)
    
    # Test input
    x = torch.randn(16, args.seq_len, 1)  # (batch, seq_len, features)
    output = model(x)
    print("Test output shape:", output.shape)  # Should be (16, 1)