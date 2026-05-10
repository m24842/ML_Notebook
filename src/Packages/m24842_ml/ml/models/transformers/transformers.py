import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from copy import deepcopy
from .attention import *
from ..common import *

class Transformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, ff_dim=None, qk_dim=None,
                 dropout=0.0, causal=True,
                 norm_type="rms", ff_type="swiglu",
                 use_embedding=True, weight_tying=False,
                 ff_bias=False, attn_bias=False,
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim if ff_dim is not None else emb_dim
        self.device = device
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        self.rope_encoding = None
        self.abs_pos_encoding = None
        if pos_encoding == "rope":
            self.rope_encoding = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False).to(device)
        elif pos_encoding == "xpos":
            self.rope_encoding = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False).to(device)
        elif pos_encoding == "sin":
            self.sin_pos_encoding = SinusoidalPositionalEmbedding(emb_dim, max_len=pos_encoding_max_len, device=device)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
            self.abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device)
        
        norm_temp = nn.Identity(emb_dim, device=device)
        if norm_type == "rms":
            norm_temp = nn.RMSNorm(emb_dim, device=device)
        elif norm_type == "layer":
            norm_temp = nn.LayerNorm(emb_dim, device=device)
        elif norm_type == "derf":
            norm_temp = DyFunc(emb_dim, func=torch.erf, device=device)
        elif norm_type == "dyt":
            norm_temp = DyFunc(emb_dim, func=torch.tanh, device=device)
        else:
            raise ValueError("Invalid norm_tyoe")
        
        ff_temp = None
        if ff_type == "mlp":
            ff_temp = MLP(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device)
        elif ff_type == "swiglu":
            ff_temp = SwiGLU(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device)
        else:
            raise ValueError("Invalid ff_type")
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = deepcopy(norm_temp),
                    dropout1 = nn.Dropout(dropout),
                    attention = MultiheadAttention(
                        emb_dim,
                        self.n_heads,
                        qk_dim=qk_dim,
                        bias=attn_bias,
                        batch_first=True,
                        device=device
                    ),
                    norm2 = deepcopy(norm_temp),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = deepcopy(ff_temp),
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.RMSNorm(emb_dim, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
    
    def _apply(self, fn):
        super()._apply(fn)
        self.device = next(self.parameters(), torch.empty(0)).device
        return self

    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        if self.pos_encoding == "abs":
            pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.abs_pos_encoding(pos)
        elif self.pos_encoding == "sin":
            x = x + self.sin_pos_encoding(x)
        for layer in self.layers:
            a_out = layer.attention(layer.norm1(x), rope=self.rope_encoding, causal=self.causal)
            x = x + layer.dropout1(a_out)
            ff_out = layer.feedforward(layer.norm2(x))
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class LinearTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, ff_dim=None,
                 qk_dim=None, dropout=0.0,
                 norm_type="rms", ff_type="swiglu",
                 causal=True, use_embedding=True,
                 weight_tying=False, ff_bias=False, attn_bias=False,
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim if ff_dim is not None else emb_dim
        self.device = device
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        self.rope_encoding = None
        self.abs_pos_encoding = None
        if pos_encoding == "rope":
            self.rope_encoding = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False).to(device)
        elif pos_encoding == "xpos":
            self.rope_encoding = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False).to(device)
        elif pos_encoding == "sin":
            self.sin_pos_encoding = SinusoidalPositionalEmbedding(emb_dim, max_len=pos_encoding_max_len, device=device)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
            self.abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device)
        
        norm_temp = nn.Identity(emb_dim, device=device)
        if norm_type == "rms":
            norm_temp = nn.RMSNorm(emb_dim, device=device)
        elif norm_type == "layer":
            norm_temp = nn.LayerNorm(emb_dim, device=device)
        elif norm_type == "derf":
            norm_temp = DyFunc(emb_dim, func=torch.erf, device=device)
        elif norm_type == "dyt":
            norm_temp = DyFunc(emb_dim, func=torch.tanh, device=device)
        
        ff_temp = None
        if ff_type == "mlp":
            ff_temp = MLP(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device)
        elif ff_type == "swiglu":
            ff_temp = SwiGLU(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device)
        else:
            raise ValueError("Invalid ff_type")
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = deepcopy(norm_temp),
                    dropout1 = nn.Dropout(dropout),
                    attention = LinearAttention(
                        emb_dim,
                        self.n_heads,
                        qk_dim=qk_dim,
                        bias=attn_bias,
                        batch_first=True,
                        device=device
                    ),
                    norm2 = deepcopy(norm_temp),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = deepcopy(ff_temp),
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.RMSNorm(emb_dim, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
    
    def _apply(self, fn):
        super()._apply(fn)
        self.device = next(self.parameters(), torch.empty(0)).device
        return self
    
    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        if self.pos_encoding == "abs":
            pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.abs_pos_encoding(pos)
        elif self.pos_encoding == "sin":
            x = x + self.sin_pos_encoding(x)
        for layer in self.layers:
            a_out = layer.attention(layer.norm1(x), rope=self.rope_encoding, causal=self.causal)
            x = x + layer.dropout1(a_out)
            ff_out = layer.feedforward(layer.norm2(x))
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class OrthoLinearTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, ff_dim=None,
                 qk_dim=None, dropout=0.0,
                 norm_type="rms", ff_type="swiglu",
                 causal=True, use_embedding=True,
                 weight_tying=False, ff_bias=False, attn_bias=False,
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_dim = ff_dim if ff_dim is not None else emb_dim
        self.device = device
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        self.rope_encoding = None
        self.abs_pos_encoding = None
        if pos_encoding == "rope":
            self.rope_encoding = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False).to(device)
        elif pos_encoding == "xpos":
            self.rope_encoding = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False).to(device)
        elif pos_encoding == "sin":
            self.sin_pos_encoding = SinusoidalPositionalEmbedding(emb_dim, max_len=pos_encoding_max_len, device=device)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
            self.abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device)
        
        norm_temp = nn.Identity(emb_dim, device=device)
        if norm_type == "rms":
            norm_temp = nn.RMSNorm(emb_dim, device=device)
        elif norm_type == "layer":
            norm_temp = nn.LayerNorm(emb_dim, device=device)
        elif norm_type == "derf":
            norm_temp = DyFunc(emb_dim, func=torch.erf, device=device)
        elif norm_type == "dyt":
            norm_temp = DyFunc(emb_dim, func=torch.tanh, device=device)
        
        ff_temp = None
        if ff_type == "mlp":
            ff_temp = MLP(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device)
        elif ff_type == "swiglu":
            ff_temp = SwiGLU(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device)
        else:
            raise ValueError("Invalid ff_type")
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = deepcopy(norm_temp),
                    dropout1 = nn.Dropout(dropout),
                    attention = OrthoLinearAttention(
                        emb_dim,
                        self.n_heads,
                        qk_dim=qk_dim,
                        bias=attn_bias,
                        batch_first=True,
                        device=device
                    ),
                    norm2 = deepcopy(norm_temp),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = deepcopy(ff_temp),
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.RMSNorm(emb_dim, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
    
    def _apply(self, fn):
        super()._apply(fn)
        self.device = next(self.parameters(), torch.empty(0)).device
        return self
    
    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        if self.pos_encoding == "abs":
            pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.abs_pos_encoding(pos)
        elif self.pos_encoding == "sin":
            x = x + self.sin_pos_encoding(x)
        for layer in self.layers:
            a_out = layer.attention(layer.norm1(x), rope=self.rope_encoding, causal=self.causal)
            x = x + layer.dropout1(a_out)
            ff_out = layer.feedforward(layer.norm2(x))
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class CompressionTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, ff_dim=None, qk_dim=None,
                 mem_dim=16, dropout=0.0,
                 norm_type="rms", ff_type="swiglu",
                 causal=True, use_embedding=True, weight_tying=False,
                 ff_bias=False, attn_bias=False,
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim if ff_dim is not None else emb_dim
        self.compressed_len = mem_dim
        self.device = device
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        self.rope_encoding = None
        self.abs_pos_encoding = None
        if pos_encoding == "rope":
            self.rope_encoding = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False).to(device)
        elif pos_encoding == "xpos":
            self.rope_encoding = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False).to(device)
        elif pos_encoding == "sin":
            self.sin_pos_encoding = SinusoidalPositionalEmbedding(emb_dim, max_len=pos_encoding_max_len, device=device)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
            self.abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device)
        
        norm_temp = nn.Identity(emb_dim, device=device)
        if norm_type == "rms":
            norm_temp = nn.RMSNorm(emb_dim, device=device)
        elif norm_type == "layer":
            norm_temp = nn.LayerNorm(emb_dim, device=device)
        elif norm_type == "derf":
            norm_temp = DyFunc(emb_dim, func=torch.erf, device=device)
        elif norm_type == "dyt":
            norm_temp = DyFunc(emb_dim, func=torch.tanh, device=device)
        
        ff_temp = None
        if ff_type == "mlp":
            ff_temp = MLP(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device)
        elif ff_type == "swiglu":
            ff_temp = SwiGLU(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device)
        else:
            raise ValueError("Invalid ff_type")
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = deepcopy(norm_temp),
                    dropout1 = nn.Dropout(dropout),
                    attention = CompressionAttention(
                        emb_dim,
                        self.n_heads,
                        qk_dim=qk_dim,
                        compressed_len=self.compressed_len,
                        dropout=dropout,
                        bias=attn_bias,
                        batch_first=True,
                        device=device
                    ),
                    norm2 = deepcopy(norm_temp),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = deepcopy(ff_temp),
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.RMSNorm(emb_dim, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
    
    def _apply(self, fn):
        super()._apply(fn)
        self.device = next(self.parameters(), torch.empty(0)).device
        return self
    
    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        if self.pos_encoding == "abs":
            pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.abs_pos_encoding(pos)
        elif self.pos_encoding == "sin":
            x = x + self.sin_pos_encoding(x)
        for layer in self.layers:
            a_out = layer.attention(layer.norm1(x), rope=self.rope_encoding, causal=self.causal)
            x = x + layer.dropout1(a_out)
            ff_out = layer.feedforward(layer.norm2(x))
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class SlidingWindowTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, ff_dim=None, qk_dim=None,
                 window_len=64, dilate=True, dilation_factor=None,
                 dilation_cap=2**32, use_flex_attn=True, dropout=0.0,
                 causal=True, use_embedding=True, weight_tying=False,
                 ff_bias=False, attn_bias=False,
                 norm_type="rms", ff_type="swiglu",
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim if ff_dim is not None else emb_dim
        self.device = device
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        self.rope_encoding = None
        self.abs_pos_encoding = None
        self.sin_pos_encoding = None
        if pos_encoding == "rope":
            self.rope_encoding = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False).to(device)
        elif pos_encoding == "xpos":
            self.rope_encoding = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False).to(device)
        elif pos_encoding == "sin":
            self.sin_pos_encoding = SinusoidalPositionalEmbedding(emb_dim, max_len=pos_encoding_max_len, device=device)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
            self.abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device)
        
        norm_temp = nn.Identity(emb_dim, device=device)
        if norm_type == "rms":
            norm_temp = nn.RMSNorm(emb_dim, device=device)
        elif norm_type == "layer":
            norm_temp = nn.LayerNorm(emb_dim, device=device)
        elif norm_type == "derf":
            norm_temp = DyFunc(emb_dim, func=torch.erf, device=device)
        elif norm_type == "dyt":
            norm_temp = DyFunc(emb_dim, func=torch.tanh, device=device)
        
        ff_temp = None
        if ff_type == "mlp":
            ff_temp = MLP(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device)
        elif ff_type == "swiglu":
            ff_temp = SwiGLU(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device)
        else:
            raise ValueError("Invalid ff_type")
        
        dilation_factor = window_len if dilation_factor is None else dilation_factor
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = deepcopy(norm_temp),
                    dropout1 = nn.Dropout(dropout),
                    attention = SlidingWindowAttention(
                        emb_dim,
                        self.n_heads,
                        qk_dim=qk_dim,
                        window_len=window_len,
                        use_flex_attn=use_flex_attn,
                        dilation=min(dilation_factor**i, dilation_cap) if dilate else 1,
                        dropout=dropout,
                        bias=attn_bias,
                        batch_first=True,
                        device=device
                    ),
                    norm2 = deepcopy(norm_temp),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = deepcopy(ff_temp),
                )
            ) for i in range(self.n_layers)
        ])
        self.norm_f = nn.RMSNorm(emb_dim, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
        
    def _apply(self, fn):
        super()._apply(fn)
        self.device = next(self.parameters(), torch.empty(0)).device
        return self
    
    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        if self.pos_encoding == "abs":
            pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.abs_pos_encoding(pos)
        elif self.pos_encoding == "sin":
            x = x + self.sin_pos_encoding(x)
        for layer in self.layers:
            a_out = layer.attention(layer.norm1(x), rope=self.rope_encoding, causal=self.causal)
            x = x + layer.dropout1(a_out)
            ff_out = layer.feedforward(layer.norm2(x))
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x
