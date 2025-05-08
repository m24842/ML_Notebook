import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
import torch.autograd as AG
import math
import sys
from einops import rearrange
import opt_einsum
from typing import Optional
from rotary_embedding_torch import RotaryEmbedding
from .attention import *
from .rnns import *

class Transformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, mlp_dim=None,
                 dropout=0.0, causal=True, use_embedding=True,
                 mlp_bias=True, attention_bias=True,
                 use_positional_encoding=True, use_xpos=False,
                 device=torch.device('cpu')):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        if use_positional_encoding: self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=use_xpos, cache_if_possible=False)
        else: self.rope = None
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    attention = MultiheadAttention(emb_dim, self.n_heads, bias=attention_bias, batch_first=True),
                    norm2 = nn.LayerNorm(emb_dim),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = nn.Sequential(
                        nn.Linear(emb_dim, self.mlp_dim, bias=mlp_bias),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(self.mlp_dim, emb_dim, bias=mlp_bias)
                    )
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.LayerNorm(emb_dim)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)
        
    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        if self.causal: mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        else: mask = None
        for layer in self.layers:
            x = layer.norm1(x)
            a_out, _ = layer.attention(x, attn_mask=mask, rope=self.rope)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class LinearTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, mlp_dim=None,
                 dropout=0.0, causal=True, use_embedding=True,
                 mlp_bias=True, attention_bias=True,
                 use_positional_encoding=True, use_xpos=False,
                 device=torch.device('cpu')):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        if use_positional_encoding: self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=use_xpos, cache_if_possible=False)
        else: self.rope = None
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    attention = LinearAttention(emb_dim, self.n_heads, bias=attention_bias),
                    norm2 = nn.LayerNorm(emb_dim),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = nn.Sequential(
                        nn.Linear(emb_dim, self.mlp_dim, bias=mlp_bias),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(self.mlp_dim, emb_dim, bias=mlp_bias)
                    )
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.LayerNorm(emb_dim)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)
        
    def forward(self, x):
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        for layer in self.layers:
            x = layer.norm1(x)
            a_out = layer.attention(x, rope=self.rope, causal=self.causal)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class OrthoLinearTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, mlp_dim=None,
                 dropout=0.0, causal=True, use_embedding=True,
                 mlp_bias=True, attention_bias=True,
                 use_positional_encoding=True, use_xpos=False,
                 device=torch.device('cpu')):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        if use_positional_encoding: self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=use_xpos, cache_if_possible=False)
        else: self.rope = None
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    attention = OrthoLinearAttention(emb_dim, self.n_heads, bias=attention_bias),
                    norm2 = nn.LayerNorm(emb_dim),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = nn.Sequential(
                        nn.Linear(emb_dim, self.mlp_dim, bias=mlp_bias),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(self.mlp_dim, emb_dim, bias=mlp_bias)
                    )
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.LayerNorm(emb_dim)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)
        
    def forward(self, x):
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        for layer in self.layers:
            x = layer.norm1(x)
            a_out = layer.attention(x, rope=self.rope, causal=self.causal)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x + layer.dropout1(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class CompressionTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, mlp_dim=None, mem_dim=16,
                 dropout=0.0, causal=True, use_embedding=True,
                 mlp_bias=True, attention_bias=True,
                 use_positional_encoding=True, use_xpos=False,
                 sequential=False, chunk_size=16,
                 device=torch.device('cpu')):
        super().__init__()
        self.sequential = sequential
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        self.compressed_len = mem_dim
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        if use_positional_encoding: self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), use_xpos=use_xpos, cache_if_possible=False)
        else: self.rope = None
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    attention = CompressionAttention(emb_dim, self.n_heads, self.mlp_dim, compressed_len=self.compressed_len, dropout=dropout, bias=attention_bias, batch_first=True, chunk_size=chunk_size),
                    norm2 = nn.LayerNorm(emb_dim),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = nn.Sequential(
                        nn.Linear(emb_dim, self.mlp_dim, bias=mlp_bias),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(self.mlp_dim, emb_dim, bias=mlp_bias)
                    )
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.LayerNorm(emb_dim)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)
        
    def forward(self, x):
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        for layer in self.layers:
            x = layer.norm1(x)
            a_out = layer.attention(x, rope=self.rope, causal=self.causal, sequential=self.sequential)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class Mamba2(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1,
                 use_embedding=True, bidirectional=False,
                 chunk_size=16, device=torch.device('cpu')):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.bidirectional = bidirectional
        self.device = device

        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(input_dim, emb_dim, device=device) if use_embedding else nn.Linear(input_dim, emb_dim, bias=False, device=device),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer_f=Mamba2Block(d_model=emb_dim, n_layers=n_layers, d_state=emb_dim, d_conv=4, expand=2, n_heads=n_heads, chunk_size=chunk_size, device=device),
                                mixer_b=Mamba2Block(d_model=emb_dim, n_layers=n_layers, d_state=emb_dim, d_conv=4, expand=2, n_heads=n_heads, chunk_size=chunk_size, device=device) if bidirectional else None,
                                norm=RMSNorm(emb_dim, device=device),
                            )
                        )
                        for _ in range(n_layers)
                    ]
                ),
                norm_f=RMSNorm(emb_dim, device=device),
            )
        )
        self.out_proj = nn.Linear(
            emb_dim, output_dim, bias=False, device=device
        )
        
        self.to(device)

    def forward(self, x):
        seqlen = x.shape[1]

        x = ((self.input_dim-1)*x).long().squeeze(-1)
        x = self.backbone.embedding(x)
        for i, layer in enumerate(self.backbone.layers):
            y_f = layer.mixer_f(layer.norm(x))
            if self.bidirectional:
                y_b = layer.mixer_b(layer.norm(x.flip(1)))
                x = y_f + y_b + x
            else:
                x = y_f + x

        x = self.backbone.norm_f(x)
        logits = self.out_proj(x)
        return logits[:, :seqlen]

def initialize_model(name, *args, **kwargs):
    model_class = getattr(sys.modules[__name__], name, None)
    return model_class(*args, **kwargs)
