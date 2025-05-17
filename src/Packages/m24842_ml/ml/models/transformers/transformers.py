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
                 n_layers=1, n_heads=1, mlp_dim=None,
                 dropout=0.0, causal=True, use_embedding=True, weight_tying=False,
                 mlp_bias=True, attention_bias=True,
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
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        
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
                    norm1 = nn.LayerNorm(emb_dim),
                    abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim) if pos_encoding == "abs" else None,
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
            a_out, _ = layer.attention(x, attn_mask=mask, rope=self.rope if self.pos_encoding == "rope" else None)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

class LinearTransformer(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, mlp_dim=None,
                 dropout=0.0, causal=True, use_embedding=True, weight_tying=False,
                 mlp_bias=True, attention_bias=True,
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
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        
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
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim) if pos_encoding == "abs" else None,
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
                 n_layers=1, n_heads=1, mlp_dim=None,
                 dropout=0.0, causal=True, use_embedding=True, weight_tying=False,
                 mlp_bias=True, attention_bias=True,
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
        if use_embedding: self.embedding = nn.Embedding(input_dim, emb_dim)
        else: self.embedding = nn.Linear(input_dim, emb_dim, bias=False)
        
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        
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
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim) if pos_encoding == "abs" else None,
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
                 n_layers=1, n_heads=1, mlp_dim=None, mem_dim=16,
                 dropout=0.0, causal=True, use_embedding=True, weight_tying=False,
                 mlp_bias=True, attention_bias=True,
                 pos_encoding=None, pos_encoding_max_len=None,
                 sequential=False, chunk_size=16,
                 device="cpu"):
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
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim) if pos_encoding == "abs" else None,
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
            a_out = layer.attention(x, rope=self.rope if self.pos_encoding == "rope" else None, causal=self.causal, sequential=self.sequential)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x

from ...pc import PCModel, PCLayer
class PCTransformer(PCModel):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, mlp_dim=None,
                 dropout=0.0, causal=True, use_embedding=True, weight_tying=False,
                 mlp_bias=True, attention_bias=True,
                 pos_encoding=None, pos_encoding_max_len=None,
                 device="cpu"):
        class PCTransformerLayer(nn.Module):
            def __init__(self, emb_dim, n_heads, mlp_dim,
                        dropout=0.0, causal=True, use_embedding=True,
                        weight_tying=False, mlp_bias=True,
                        attention_bias=True,
                        pos_encoding=None, pos_encoding_max_len=None, device="cpu"):
                super().__init__()
                self.emb_dim = emb_dim
                self.n_heads = n_heads
                self.mlp_dim = mlp_dim
                self.dropout = dropout
                self.causal = causal
                self.use_embedding = use_embedding
                self.weight_tying = weight_tying
                self.mlp_bias = mlp_bias
                self.attention_bias = attention_bias
                self.pos_encoding = pos_encoding
                self.pos_encoding_max_len = pos_encoding_max_len
                self.attention = MultiheadAttention(emb_dim, n_heads, bias=attention_bias, batch_first=True)
                self.norm1 = nn.LayerNorm(emb_dim)
                self.dropout1 = nn.Dropout(dropout)
                self.norm2 = nn.LayerNorm(emb_dim)
                self.dropout2 = nn.Dropout(dropout)
                self.feedforward = nn.Sequential(
                    nn.Linear(emb_dim, mlp_dim, bias=mlp_bias),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, emb_dim, bias=mlp_bias)
                )
                if pos_encoding == "rope":
                    self.rope = RotaryEmbedding(dim=emb_dim//(2*n_heads), use_xpos=False, cache_if_possible=False)
                elif pos_encoding == "xpos":
                    self.rope = RotaryEmbedding(dim=emb_dim//(2*n_heads), use_xpos=True, cache_if_possible=False)
                elif pos_encoding == "abs":
                    assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
                    self.pos_encoding_max_len = pos_encoding_max_len
                else: self.rope = None
                self.abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim) if pos_encoding == "abs" else None
                
                self.to(device)
            
            def forward(self, x):
                x = self.norm1(x)
                if self.abs_pos_encoding is not None:
                    pos = torch.arange(x.size(1), device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
                    x = x + self.abs_pos_encoding(pos)
                a_out, _ = self.attention(x, attn_mask=None, rope=self.rope if self.pos_encoding == "rope" else None)
                x = x + self.dropout1(a_out)
                x = self.norm2(x)
                ff_out = self.feedforward(x)
                x = x + self.dropout2(ff_out)
                return x

        f_in_proj = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        b_in_proj = nn.Linear(emb_dim, input_dim, bias=False, device=device)
        f_out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        b_out_proj = nn.Linear(output_dim, emb_dim, bias=False, device=device)
        layers = [
                PCLayer(
                    f_in_proj,
                    b_in_proj,
                    f_energy_fn=F.mse_loss,
                    b_energy_fn=F.mse_loss,
                    device=device
                )
            ]+[
            PCLayer(
                PCTransformerLayer(emb_dim, n_heads, mlp_dim,
                                   dropout=dropout, causal=causal, use_embedding=use_embedding,
                                   weight_tying=weight_tying, mlp_bias=mlp_bias,
                                   attention_bias=attention_bias,
                                   pos_encoding=pos_encoding, pos_encoding_max_len=pos_encoding_max_len, device=device),
                PCTransformerLayer(emb_dim, n_heads, mlp_dim,
                                   dropout=dropout, causal=causal, use_embedding=use_embedding,
                                   weight_tying=weight_tying, mlp_bias=mlp_bias,
                                   attention_bias=attention_bias,
                                   pos_encoding=pos_encoding, pos_encoding_max_len=pos_encoding_max_len, device=device),
                f_energy_fn=F.mse_loss,
                b_energy_fn=F.mse_loss,
                device=device
            ) for _ in range(n_layers)
        ]+[
            PCLayer(
                    nn.Sequential(
                        nn.LayerNorm(emb_dim),
                        f_out_proj
                    ),
                    nn.Sequential(
                        b_out_proj,
                        nn.LayerNorm(emb_dim),
                    ),
                    f_energy_fn=F.cross_entropy,
                    b_energy_fn=F.mse_loss,
                    device=device
                )
        ]
        
        nn.init.xavier_uniform_(f_in_proj.weight)
        nn.init.xavier_uniform_(b_in_proj.weight)
        if weight_tying:
            f_out_proj.weight = f_in_proj.weight
            b_out_proj.weight = b_in_proj.weight
        else:
            nn.init.xavier_uniform_(f_out_proj.weight)
            nn.init.xavier_uniform_(b_out_proj.weight)
        
        super().__init__(
            layers=layers,
            max_its=3,
            min_energy=1e-3,
            energy_lr=1e-2,
            energy_optimizer_class=torch.optim.SGD,
            energy_grad_clip_norm=None,
            iterative_train=False,
            state_init_dir="forward", 
            device=device,
        )