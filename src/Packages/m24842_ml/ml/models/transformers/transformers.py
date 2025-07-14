import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
import torch.autograd as AG
import math
from einops import rearrange
import opt_einsum
from rotary_embedding_torch import RotaryEmbedding
from .attention import *
from ..common import *

class Transformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, mlp_dim=None, attn_sink=False,
                 dropout=0.0, causal=True, use_embedding=True, weight_tying=False,
                 mlp_bias=True, attn_bias=True,
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False)
        elif pos_encoding == "xpos":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
        else: self.rope = None
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.RMSNorm(emb_dim, device=device),
                    abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device) if pos_encoding == "abs" else None,
                    dropout1 = nn.Dropout(dropout),
                    attention = MultiheadAttention(emb_dim, self.n_heads, attn_sink=attn_sink, bias=attn_bias, batch_first=True, device=device),
                    norm2 = nn.RMSNorm(emb_dim, device=device),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = MLP(emb_dim, self.mlp_dim, emb_dim, bias=mlp_bias, device=device),
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.RMSNorm(emb_dim, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)
        
    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        if self.causal: mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        else: mask = None
        for layer in self.layers:
            x = layer.norm1(x)
            if layer.abs_pos_encoding is not None:
                pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
                x = x + layer.abs_pos_encoding(pos)
            a_out = layer.attention(x, attn_mask=mask, rope=self.rope if self.pos_encoding == "rope" else None)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class LinearTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, mlp_dim=None, attn_sink=False,
                 dropout=0.0, causal=True, use_embedding=True, weight_tying=False,
                 mlp_bias=True, attn_bias=True,
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False)
        elif pos_encoding == "xpos":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
        else: self.rope = None
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.RMSNorm(emb_dim, device=device),
                    dropout1 = nn.Dropout(dropout),
                    abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device) if pos_encoding == "abs" else None,
                    attention = LinearAttention(emb_dim, self.n_heads, attn_sink=attn_sink, bias=attn_bias, device=device),
                    norm2 = nn.RMSNorm(emb_dim, device=device),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = MLP(emb_dim, self.mlp_dim, emb_dim, bias=mlp_bias, device=device),
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.RMSNorm(emb_dim, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)
        
    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        for layer in self.layers:
            x = layer.norm1(x)
            if layer.abs_pos_encoding is not None:
                pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
                x = x + layer.abs_pos_encoding(pos)
            a_out = layer.attention(x, rope=self.rope if self.pos_encoding == "rope" else None, causal=self.causal)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class OrthoLinearTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, mlp_dim=None, attn_sink=False,
                 dropout=0.0, causal=True, use_embedding=True, weight_tying=False,
                 mlp_bias=True, attn_bias=True,
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False)
        elif pos_encoding == "xpos":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
        else: self.rope = None
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.RMSNorm(emb_dim, device=device),
                    dropout1 = nn.Dropout(dropout),
                    abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device) if pos_encoding == "abs" else None,
                    attention = OrthoLinearAttention(emb_dim, self.n_heads, attn_sink=attn_sink, bias=attn_bias, device=device),
                    norm2 = nn.RMSNorm(emb_dim, device=device),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = MLP(emb_dim, self.mlp_dim, emb_dim, bias=mlp_bias, device=device),
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.RMSNorm(emb_dim, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)
        
    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        for layer in self.layers:
            x = layer.norm1(x)
            if layer.abs_pos_encoding is not None:
                pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
                x = x + layer.abs_pos_encoding(pos)
            a_out = layer.attention(x, rope=self.rope if self.pos_encoding == "rope" else None, causal=self.causal)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x + layer.dropout1(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class CompressionTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, mlp_dim=None, mem_dim=16, attn_sink=False, 
                 dropout=0.0, causal=True, use_embedding=True, weight_tying=False,
                 mlp_bias=True, attn_bias=True,
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        self.compressed_len = mem_dim
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False)
        elif pos_encoding == "xpos":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
        else: self.rope = None
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.RMSNorm(emb_dim, device=device),
                    dropout1 = nn.Dropout(dropout),
                    abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device) if pos_encoding == "abs" else None,
                    attention = CompressionAttention(emb_dim, self.n_heads, compressed_len=self.compressed_len, attn_sink=attn_sink, dropout=dropout, bias=attn_bias, batch_first=True, device=device),
                    norm2 = nn.RMSNorm(emb_dim, device=device),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = MLP(emb_dim, self.mlp_dim, emb_dim, bias=mlp_bias, device=device),
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.RMSNorm(emb_dim, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)
        
    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        for layer in self.layers:
            x = layer.norm1(x)
            if layer.abs_pos_encoding is not None:
                pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
                x = x + layer.abs_pos_encoding(pos)
            a_out = layer.attention(x, rope=self.rope if self.pos_encoding == "rope" else None, causal=self.causal)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class SlidingWindowTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, mlp_dim=None, window_len=64, masked_window=True,
                 attn_sink=False, dilate=True, dilation_factor=None, dropout=0.0,
                 causal=True, use_embedding=True, weight_tying=False,
                 mlp_bias=True, attn_bias=True,
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False)
        elif pos_encoding == "xpos":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
        else: self.rope = None
        
        dilation_factor = window_len if dilation_factor is None else dilation_factor
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.RMSNorm(emb_dim, device=device),
                    dropout1 = nn.Dropout(dropout),
                    abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device) if pos_encoding == "abs" else None,
                    attention = SlidingWindowAttention(emb_dim, self.n_heads, window_len=window_len, masked_window=masked_window, attn_sink=attn_sink, dilation=dilation_factor**i if dilate else 1, dropout=dropout, bias=attn_bias, batch_first=True, device=device),
                    norm2 = nn.RMSNorm(emb_dim, device=device),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = MLP(emb_dim, self.mlp_dim, emb_dim, bias=mlp_bias, device=device),
                )
            ) for i in range(self.n_layers)
        ])
        self.norm_f = nn.RMSNorm(emb_dim, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)
    
    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        for layer in self.layers:
            x = layer.norm1(x)
            if layer.abs_pos_encoding is not None:
                pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
                x = x + layer.abs_pos_encoding(pos)
            a_out = layer.attention(x, rope=self.rope if self.pos_encoding == "rope" else None, causal=self.causal)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class ExpandedOrthoLinearTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim, tree_depth, tree_size,
                 n_layers=1, n_heads=1, mlp_dim=None, attn_sink=False,
                 dropout=0.0, causal=True, use_embedding=True, weight_tying=False,
                 mlp_bias=True, attn_bias=True,
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.tree_depth = tree_depth
        self.tree_size = tree_size
        self.causal = causal
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False)
        elif pos_encoding == "xpos":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
        else: self.rope = None
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.RMSNorm(emb_dim, device=device),
                    dropout1 = nn.Dropout(dropout),
                    abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device) if pos_encoding == "abs" else None,
                    attention = ExpandedOrthoLinearAttention(emb_dim, self.n_heads, tree_depth=tree_depth, tree_size=tree_size, attn_sink=attn_sink, bias=attn_bias, device=device),
                    norm2 = nn.RMSNorm(emb_dim, device=device),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = MLP(emb_dim, self.mlp_dim, emb_dim, bias=mlp_bias, device=device),
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.RMSNorm(emb_dim, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)
        
    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        for layer in self.layers:
            x = layer.norm1(x)
            if layer.abs_pos_encoding is not None:
                pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
                x = x + layer.abs_pos_encoding(pos)
            a_out = layer.attention(x, rope=self.rope if self.pos_encoding == "rope" else None, causal=self.causal)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x + layer.dropout1(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class DiffusionTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim, use_embedding=True,
                 n_layers=1, n_heads=1, mlp_dim=None, attn_sink=False,
                 dropout=0.0, mlp_bias=True, attn_bias=True,
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False)
        elif pos_encoding == "xpos":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
        else: self.rope = None
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.RMSNorm(emb_dim, device=device),
                    abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device) if pos_encoding == "abs" else None,
                    dropout1 = nn.Dropout(dropout),
                    attention = MultiheadAttention(emb_dim, self.n_heads, attn_sink=attn_sink, bias=attn_bias, batch_first=True, device=device),
                    norm2 = nn.RMSNorm(emb_dim, device=device),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = MLP(emb_dim, self.mlp_dim, emb_dim, bias=mlp_bias, device=device),
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.RMSNorm(emb_dim, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)
    
    @torch.no_grad()
    def get_noise(self, x, profile_fn=None, t_offset=0, t_range=(-1.0, 0.0), beta_range=(0.0001, 0.02)):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, emb_dim)
            profile_fn: function to apply to profile gaussian noise, defaults to identity
            t_offset: offset to apply to time axis, defaults to 0
            t_range: tuple of (t_min, t_max) for time axis, defaults to (-1.0, 0.0)
            beta_range: tuple of (beta_min, beta_max) for noise schedule, defaults to (0.0001, 0.02)
        """
        if profile_fn is None: profile_fn = lambda x: x
        t_min, t_max = t_range
        beta_min, beta_max = beta_range
        bsz, seq_len = x.shape[:2]
        t_axis = torch.linspace(t_min-t_offset, t_max-t_offset, seq_len, dtype=torch.float32, device=x.device).reshape(1, -1, 1)
        profile = profile_fn(t_axis).clamp(0.0, 1.0)
        betas = profile * (beta_max - beta_min) + beta_min
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=1)
        return betas, alphas, alphas_cumprod
    
    def forward(self, x):
        bsz, seq_len, d_model = x.shape
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        for layer in self.layers:
            x = layer.norm1(x)
            if layer.abs_pos_encoding is not None:
                pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
                x = x + layer.abs_pos_encoding(pos)
            a_out = layer.attention(x, attn_mask=None, rope=self.rope if self.pos_encoding == "rope" else None)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x
    
    def step(self, x):
        return self(x)