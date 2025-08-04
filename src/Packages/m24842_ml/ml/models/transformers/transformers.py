import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
import torch.autograd as AG
from rotary_embedding_torch import RotaryEmbedding
from .attention import *
from ..common import *

class Transformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, ff_dim=None, qk_dim=None,
                 attn_sink=False, dropout=0.0, causal=True,
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
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        self.rope = None
        self.abs_pos_encoding = None
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False)
        elif pos_encoding == "xpos":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
            self.abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.RMSNorm(emb_dim, device=device),
                    dropout1 = nn.Dropout(dropout),
                    attention = MultiheadAttention(
                        emb_dim,
                        self.n_heads,
                        qk_dim=qk_dim,
                        attn_sink=attn_sink,
                        bias=attn_bias,
                        batch_first=True,
                        device=device
                    ),
                    norm2 = nn.RMSNorm(emb_dim, device=device),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = SwiGLU(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device),
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
        if self.abs_pos_encoding is not None:
            pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.abs_pos_encoding(pos)
        for layer in self.layers:
            a_out = layer.attention(layer.norm1(x), rope=self.rope if self.pos_encoding == "rope" else None, causal=self.causal)
            x = x + layer.dropout1(a_out)
            ff_out = layer.feedforward(layer.norm2(x))
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class LinearTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, ff_dim=None,
                 qk_dim=None, attn_sink=False, dropout=0.0,
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
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        self.rope = None
        self.abs_pos_encoding = None
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False)
        elif pos_encoding == "xpos":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
            self.abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.RMSNorm(emb_dim, device=device),
                    dropout1 = nn.Dropout(dropout),
                    attention = LinearAttention(
                        emb_dim,
                        self.n_heads,
                        qk_dim=qk_dim,
                        attn_sink=attn_sink,
                        bias=attn_bias,
                        batch_first=True,
                        device=device
                    ),
                    norm2 = nn.RMSNorm(emb_dim, device=device),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = SwiGLU(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device),
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
        if self.abs_pos_encoding is not None:
            pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.abs_pos_encoding(pos)
        for layer in self.layers:
            a_out = layer.attention(layer.norm1(x), rope=self.rope if self.pos_encoding == "rope" else None, causal=self.causal)
            x = x + layer.dropout1(a_out)
            ff_out = layer.feedforward(layer.norm2(x))
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class OrthoLinearTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, ff_dim=None, attn_sink=False,
                 qk_dim=None, dropout=0.0, causal=True, use_embedding=True,
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
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        self.rope = None
        self.abs_pos_encoding = None
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False)
        elif pos_encoding == "xpos":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
            self.abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.RMSNorm(emb_dim, device=device),
                    dropout1 = nn.Dropout(dropout),
                    attention = OrthoLinearAttention(
                        emb_dim,
                        self.n_heads,
                        qk_dim=qk_dim,
                        attn_sink=attn_sink,
                        bias=attn_bias,
                        batch_first=True,
                        device=device
                    ),
                    norm2 = nn.RMSNorm(emb_dim, device=device),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = SwiGLU(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device),
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
        if self.abs_pos_encoding is not None:
            pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.abs_pos_encoding(pos)
        for layer in self.layers:
            a_out = layer.attention(layer.norm1(x), rope=self.rope if self.pos_encoding == "rope" else None, causal=self.causal)
            x = x + layer.dropout1(a_out)
            ff_out = layer.feedforward(layer.norm2(x))
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class CompressionTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, ff_dim=None, qk_dim=None,
                 mem_dim=16, attn_sink=False, dropout=0.0,
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
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        self.rope = None
        self.abs_pos_encoding = None
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False)
        elif pos_encoding == "xpos":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
            self.abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.RMSNorm(emb_dim, device=device),
                    dropout1 = nn.Dropout(dropout),
                    attention = CompressionAttention(
                        emb_dim,
                        self.n_heads,
                        qk_dim=qk_dim,
                        compressed_len=self.compressed_len,
                        attn_sink=attn_sink,
                        dropout=dropout,
                        bias=attn_bias,
                        batch_first=True,
                        device=device
                    ),
                    norm2 = nn.RMSNorm(emb_dim, device=device),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = SwiGLU(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device),
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
        if self.abs_pos_encoding is not None:
            pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.abs_pos_encoding(pos)
        for layer in self.layers:
            a_out = layer.attention(layer.norm1(x), rope=self.rope if self.pos_encoding == "rope" else None, causal=self.causal)
            x = x + layer.dropout1(a_out)
            ff_out = layer.feedforward(layer.norm2(x))
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class SlidingWindowTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, ff_dim=None, qk_dim=None,
                 window_len=64, dilate=True,dilation_factor=None,
                 use_flex_attn=True, attn_sink=False, dropout=0.0,
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
        
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        self.pos_encoding = pos_encoding
        self.rope = None
        self.abs_pos_encoding = None
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=False, cache_if_possible=False)
        elif pos_encoding == "xpos":
            self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=True, cache_if_possible=False)
        elif pos_encoding == "abs":
            assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
            self.pos_encoding_max_len = pos_encoding_max_len
            self.abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim, device=device)
        
        dilation_factor = window_len if dilation_factor is None else dilation_factor
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.RMSNorm(emb_dim, device=device),
                    dropout1 = nn.Dropout(dropout),
                    attention = SlidingWindowAttention(
                        emb_dim,
                        self.n_heads,
                        qk_dim=qk_dim,
                        window_len=window_len,
                        use_flex_attn=use_flex_attn,
                        attn_sink=attn_sink,
                        dilation=dilation_factor**i if dilate else 1,
                        dropout=dropout,
                        bias=attn_bias,
                        batch_first=True,
                        device=device
                    ),
                    norm2 = nn.RMSNorm(emb_dim, device=device),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = SwiGLU(emb_dim, self.ff_dim, emb_dim, bias=ff_bias, device=device),
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
        if self.abs_pos_encoding is not None:
            pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.abs_pos_encoding(pos)
        for layer in self.layers:
            a_out = layer.attention(layer.norm1(x), rope=self.rope if self.pos_encoding == "rope" else None, causal=self.causal)
            x = x + layer.dropout1(a_out)
            ff_out = layer.feedforward(layer.norm2(x))
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x
