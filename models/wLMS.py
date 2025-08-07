__all__ = ['wLMS']

import torch
import torch.nn as nn
import numpy as np
import pywt
from scipy.linalg import pinv
from argparse import Namespace

class Model(nn.Module):
    def __init__(self, configs):
        """Wavelet-based Multiscale Autoregressive Predictor (wLMS)
        
        Implements the algorithm from:
        "Prediction of Respiratory Motion with Wavelet-Based Multiscale Autoregression"
        """
        super(Model, self).__init__()
        assert 0 <= configs.wLMS_mu <= 1, "mu must be in [0,1]"
        self.wavelet = configs.wavelet
        self.J = configs.wLMS_J          # Decomposition levels
        self.a = configs.wLMS_a          # Regression depths (initialized in fit)
        self.M = configs.wLMS_M          # History window size
        self.mu = configs.wLMS_mu        # Exponential averaging factor
        self.pred_len = configs.pred_len # Prediction horizon
        self.weights = None              # Model weights
        self.filter_bank = self._create_filter_bank()  # Precomputed filter bank

    def _create_filter_bank(self):
        """Generates à trous wavelet filter bank according to:
        h^{(j)}[k] = ∑_{m=0}^k h[m]·δ_{k,m·2^j} with energy normalization
        """
        wavelet = pywt.Wavelet(self.wavelet)
        h_orig = np.array(wavelet.dec_lo)  # Low-pass filter
        g_orig = np.array(wavelet.dec_hi)  # High-pass filter
        filters = []

        for j in range(self.J):
            # Stride calculation for zero insertion
            stride = 2**j
            # Extended filters with zeros
            h_extended = np.zeros(len(h_orig) * stride)
            g_extended = np.zeros(len(g_orig) * stride)
            h_extended[::stride] = h_orig
            g_extended[::stride] = g_orig
            
            # Cumulative sum (integration)
            h_j = np.cumsum(h_extended)
            g_j = np.cumsum(g_extended)
            
            # Length adjustment for periodization
            target_len = len(h_orig) + (len(h_orig)-1)*(stride-1)
            h_j = h_j[:target_len]
            g_j = g_j[:target_len]
            
            # Energy normalization (√2^j scaling)
            norm_factor = 1 / np.sqrt(2)**j
            filters.append((h_j * norm_factor, g_j * norm_factor))
        
        return filters

    def _a_trous_decompose(self, signal):
        """Performs redundant à trous wavelet decomposition"""
        c_prev = np.array(signal)
        coeffs = []
        for j in range(self.J):
            h_j, g_j = self.filter_bank[j]
            # Convolution with circular padding (same mode)
            c_next = np.convolve(c_prev, h_j, mode='same')
            w_j = np.convolve(c_prev, g_j, mode='same') 
            coeffs.append(w_j)
            c_prev = c_next
        coeffs.append(c_prev)  # Final approximation
        return coeffs

    def _required_input_length(self):
        """Calculates minimum required input length based on a_j"""
        return max([(2**j) * (aj-1) for j, aj in enumerate(self.a)]) + 1 if self.a else 0

    def _get_regression_vector(self, coeffs, t):
        """Constructs regression vector with boundary handling"""
        xt = []
        # Process wavelet bands
        for j, aj in enumerate(self.a[:-1]):
            Wj = coeffs[j]
            for i in range(aj):
                idx = t - (2**j) * i
                xt.append(Wj[idx] if 0 <= idx < len(Wj) else 0.0)
        # Process approximation band
        cJ = coeffs[-1]
        aJ = self.a[-1]
        for i in range(aJ):
            idx = t - (2**self.J) * i
            xt.append(cJ[idx] if 0 <= idx < len(cJ) else 0.0)
        return np.array(xt)

    def _form_B_matrix(self, coeffs, t_start, signal):
        """Builds regression matrix B and target vector s"""
        B = []
        s = []
        valid_range = range(t_start, max(t_start - self.M, -1), -1)
        for t in valid_range:
            if t < 0: continue
            xt = self._get_regression_vector(coeffs, t)
            B.append(xt)
            s.append(signal[t])  # Paper uses original signal values (Sec 2.3)
        return np.array(B), np.array(s)

    def _compute_a_j(self, signal_init):
        """Dynamically computes regression depths per Eq.9"""
        coeffs = self._a_trous_decompose(signal_init)
        energy_total = np.sum((signal_init - coeffs[-1])**2) + 1e-8
        a = []
        for Wj in coeffs[:-1]:
            energy_Wj = np.sum(Wj**2)
            a_j = int(15 * energy_Wj / energy_total)
            a.append(max(1, a_j))
        a.append(2)  # Fixed for approximation band
        return a

    def fit(self, signal):
        # Dynamic a_j initialization
        if self.a is None:
            if len(signal) < 2000:
                raise ValueError("Initial 2000 samples required for a_j calculation")
            self.a = self._compute_a_j(signal[:2000])

        # Input validation
        min_len = self._required_input_length()
        if len(signal) < min_len:
            raise ValueError(f"Need {min_len} samples, got {len(signal)}")

        # Wavelet decomposition
        coeffs = self._a_trous_decompose(signal)
        
        # Matrix construction
        B, s = self._form_B_matrix(coeffs, len(signal)-1, signal)
        
        # Moore-Penrose pseudoinverse (Sec 2.3)
        B_pinv = pinv(B) if B.shape[0] >= B.shape[1] else pinv(B.T @ B) @ B.T
        
        # Weight update with exponential averaging (Eq.8)
        if B.shape[0] >= B.shape[1]:  # 超定方程
            #w_new = np.linalg.pinv(B) @ s
            w_new = np.linalg.pinv(B, rcond=1e-6) @ s
            # w_new = torch.linalg.pinv(B, rcond=1e-6) @ s
            
        else:  # 欠定方程
            w_new = B.T @ np.linalg.pinv(B @ B.T) @ s
            #w_new = B.T @ torch.linalg.pinv(B @ B.T) @ s
        # weight update and controlled by mu
        self.weights = w_new if self.weights is None else \
                    (1 - self.mu) * self.weights + self.mu * w_new

        # w_new = B_pinv @ s
        # self.weights = w_new if self.weights is None else \
        #               (1 - self.mu)*self.weights + self.mu*w_new
        
        # State preservation
        self.last_coeffs = coeffs
        self.last_signal = signal

    def predict_single_step(self):
        """Single-step prediction using current state"""
        t = len(self.last_signal) - 1
        xt = self._get_regression_vector(self.last_coeffs, t)
        return float(np.dot(self.weights, xt))

    def forward(self, x: torch.Tensor):
        """Iterative multi-step prediction (IMS strategy)"""
        #x = x.squeeze(-1).detach().cpu().numpy()
        x = x.squeeze(-1)
        batch_size, seq_len = x.shape
        preds = []
        
        for b in range(batch_size):
            signal = list(x[b])
            pred_b = []
            for _ in range(self.pred_len):
                self.fit(signal)
                pred = self.predict_single_step()
                pred_b.append(pred)
                signal = signal[1:] + [pred]  # Shift-and-append
            preds.append(pred_b)
        
        return torch.tensor(np.array(preds)[..., None], dtype=torch.float32)

    def test_reconstruction(self):
        signal = np.random.randn(1000)
        coeffs = self._a_trous_decompose(signal)
        reconstructed = pywt.waverec(coeffs, self.wavelet)
        error = np.max(np.abs(signal - reconstructed[:len(signal)]))
        assert error < 1e-10, f"Reconstruction failed with error {error}"

if __name__ == '__main__':
    # Paper experiment configuration
    args = Namespace(
        wavelet='sym2',    # Symlet-2 wavelet
        wLMS_J=7,         # Decomposition levels
        wLMS_a=None,      # Dynamic a_j calculation
        wLMS_M=2,         # History window
        wLMS_mu=0.6,      # Learning rate
        pred_len=5         # Prediction steps
    )
    
    model = Model(args)
    x = torch.randn(1, 3000, 1)  # Batch of 16 sequences
    
    # Verify filter bank generation
    for j, (h, g) in enumerate(model.filter_bank):
        #print(f"Level {j} filter lengths: h={len(h)}, g={len(g)}")
        print(f"Level {j}:")
        print(f"  h: {h.tolist()}")
        print(f"  g: {g.tolist()}")
        print(f"  Type: {type(h)}, {type(g)}")
    # Test prediction
    y_pred = model(x)
    # model.test_reconstruction()
    print("Prediction shape:", y_pred.shape)

