import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, activation=F.gelu, bias=True, device="cpu"):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_in
        self.fc1 = nn.Linear(d_in, d_hidden, bias=bias, device=device)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=bias, device=device)
        self.activation = activation
        
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc2.weight)

    def _apply(self, fn):
        super()._apply(fn)
        self.device = next(self.parameters(), torch.empty(0)).device
        return self
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, bias=True, device="cpu"):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_in
        self.fc1 = nn.Linear(d_in, 2 * d_hidden, bias=bias, device=device)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=bias, device=device)
        
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc2.weight)

    def _apply(self, fn):
        super()._apply(fn)
        self.device = next(self.parameters(), torch.empty(0)).device
        return self

    def forward(self, x):
        x, g = torch.chunk(self.fc1(x), 2, dim=-1)
        x = F.silu(g) * x
        x = self.fc2(x)
        return x

class GatedRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5, device="cpu"):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def _apply(self, fn):
        super()._apply(fn)
        self.device = next(self.parameters(), torch.empty(0)).device
        return self
    
    def forward(self, x, z=None):
        if z is not None:
            x = x * F.silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class DyFunc(nn.Module):
    def __init__(self, d_model, func=torch.erf, alpha_0=1.0, shift=True, device="cpu"):
        super().__init__()
        self.func = func # erf or tanh
        self.alpha = nn.Parameter(torch.tensor(alpha_0, device=device))
        self.beta = nn.Parameter(torch.zeros(d_model, device=device))
        self.gamma = nn.Parameter(torch.ones(d_model, device=device))
        self.shift = nn.Parameter(torch.zeros(1, device=device)) if shift else 0.0
    
    def _apply(self, fn):
        super()._apply(fn)
        self.device = next(self.parameters(), torch.empty(0)).device
        return self
    
    def forward(self, x):
        return self.gamma * self.func(self.alpha * x + self.shift) + self.beta

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10_000, device="cpu"):
        super().__init__()
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1)]
