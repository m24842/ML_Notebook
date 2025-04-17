import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional
from rotary_embedding_torch import RotaryEmbedding

class OrthoLinearAttention(nn.Module):
    """
    Orthogonal Linear Attention:
    A derivative of linear attention that orthogonalizes queries and keys for each head to reduce crossterm interference.
    
    Interference free capacity scales exponentially with head count by the formula: <head dim>^<head count>.
    An optimal choice for head dimension is 3.
    """
    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.beta = nn.Parameter(torch.zeros(1))
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, x: torch.Tensor, rope: Optional[RotaryEmbedding] = None, causal: bool = True) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, d_model).
            rope (Optional[RotaryEmbedding]): Optional RoPE encoder for rotating queries and keys.

        Returns:
            torch.Tensor: Output sequence of shape (batch_size, seq_len, d_model).
        """
        bsz, seq_len, d_model = x.size()
        q = rearrange(self.q_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b s (h d) -> (b h) s d', h=self.n_heads).contiguous()
        
        if rope:
            q = rope.rotate_queries_or_keys(q).reshape(bsz * self.n_heads, seq_len, self.d_head).contiguous()
            k = rope.rotate_queries_or_keys(k).reshape(bsz * self.n_heads, seq_len, self.d_head).contiguous()
        else:
            q = q.reshape(bsz * self.n_heads, seq_len, self.d_head).contiguous()
            k = k.reshape(bsz * self.n_heads, seq_len, self.d_head).contiguous()
        
        beta = torch.exp(self.beta)
        q = (beta * q).softmax(-1)
        k = (beta * k).softmax(-1)
        
        if causal:
            kv = torch.cumsum(torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2)), dim=1)
            kn = torch.cumsum(k, dim=1)
        else:
            kv = torch.einsum('zsD, zsd -> zDd', k, v).unsqueeze(1)
            kn = k.sum(1, keepdim=True)
        
        out = torch.matmul(q.unsqueeze(-2), kv).squeeze(-2) / (q * kn).sum(-1, keepdim=True)
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.n_heads)
        return self.out_proj(out)

class OrthoLinearTransformer(nn.Module):
    def __init__(self, emb_dim, output_dim, n_layers=1, n_heads=1, mlp_dim=None, vocab_size=10, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), cache_if_possible=False)
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    attention=OrthoLinearAttention(emb_dim, self.n_heads, bias=True),
                    norm2 = nn.LayerNorm(emb_dim),
                    dropout2 = nn.Dropout(dropout),
                    feedforward=nn.Sequential(
                        nn.Linear(emb_dim, self.mlp_dim, bias=True),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(self.mlp_dim, emb_dim, bias=True)
                    )
                )
            ) for _ in range(self.n_layers)
        ])
        self.norm_f = nn.LayerNorm(emb_dim)
        
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer.norm1(x)
            a_out = layer.attention(x, rope=self.rope, causal=True)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x + layer.dropout1(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x[:, -1]