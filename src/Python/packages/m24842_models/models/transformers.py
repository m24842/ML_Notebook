import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Optional
from rotary_embedding_torch import RotaryEmbedding

class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, 
                 add_zero_attn=False, batch_first=False):
        """
        Initialize the MultiheadAttention module.
        
        Args:
            dim: Total dimension of the model
            num_heads: Number of parallel attention heads
            dropout: Dropout probability on attention weights
            bias: Add bias to input projections
            add_bias_kv: Add bias to the key and value sequences
            add_zero_attn: Add a new batch of zeros to the key and value sequences
            dim: Total dimension of the key (default: dim)
            dim: Total dimension of the value (default: dim)
            batch_first: If True, input and output tensors are provided as (batch, seq, feature)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == self.dim, "dim must be divisible by num_heads"
        
        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        
        # Optional bias for key and value
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, dim))
        else:
            self.bias_k = self.bias_v = None
            
        self.add_zero_attn = add_zero_attn
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize projections using Xavier uniform
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
            
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
            
    def forward(self, x, key_padding_mask=None, need_weights=True, 
                attn_mask=None, average_attn_weights=True, rope=None):
        """
        Forward pass for the MultiheadAttention module.
        
        Args:
            query: Query embeddings of shape (seq_len_q, batch_size, dim) or 
                  (batch_size, seq_len_q, dim) if batch_first=True
            key: Key embeddings of shape (seq_len_k, batch_size, dim) or
                 (batch_size, seq_len_k, dim) if batch_first=True
            value: Value embeddings of shape (seq_len_v, batch_size, dim) or
                   (batch_size, seq_len_v, dim) if batch_first=True
            key_padding_mask: If provided, specified padding elements in the key will
                              be ignored by the attention. Shape: (batch_size, seq_len_k)
            need_weights: If True, returns attention weights in addition to attention output
            attn_mask: 2D or 3D mask that prevents attention to certain positions
            average_attn_weights: If True, returns averaged attention weights over heads
            
        Returns:
            attn_output: Attention output of shape (seq_len_q, batch_size, dim) or
                         (batch_size, seq_len_q, dim) if batch_first=True
            attn_output_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)
                                 if need_weights=True, otherwise None
        """
        # Handle batch_first option
        if self.batch_first:
            x = x.transpose(0, 1)
        
        tgt_len, bsz, dim = x.shape
        src_len = x.shape[0]
        
        # Apply linear projections
        q = self.q_proj(x)  # (tgt_len, batch_size, dim)
        k = self.k_proj(x)    # (src_len, batch_size, dim)
        v = self.v_proj(x)  # (src_len, batch_size, dim)
        
        # Handle bias for key and value if present
        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
            src_len += 1
        
        # Add zero attention if requested
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((1, bsz, dim), dtype=k.dtype, device=k.device)], dim=0)
            v = torch.cat([v, torch.zeros((1, bsz, dim), dtype=v.dtype, device=v.device)], dim=0)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        
        # Reshape q, k, v for multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        if rope:
            q = rope.rotate_queries_or_keys(q.reshape(bsz, self.num_heads, tgt_len, self.head_dim)).reshape(bsz * self.num_heads, tgt_len, self.head_dim)
            k = rope.rotate_queries_or_keys(k.reshape(bsz, self.num_heads, src_len, self.head_dim)).reshape(bsz * self.num_heads, src_len, self.head_dim)
        
        # Calculate attention scores
        q = q / math.sqrt(self.head_dim)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # (bsz * num_heads, tgt_len, src_len)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
            attn_output_weights = attn_output_weights + attn_mask
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        # Convert attention weights to probabilities
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        
        # Apply attention weights to values
        attn_output = torch.bmm(attn_output_weights, v)  # (bsz * num_heads, tgt_len, head_dim)
        
        # Reshape output
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, dim)
        attn_output = self.out_proj(attn_output)
        
        # Process attention weights if needed
        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
        else:
            attn_output_weights = None
        
        # Return in the correct format depending on batch_first
        if self.batch_first:
            return attn_output.transpose(0, 1), attn_output_weights
        return attn_output, attn_output_weights

class Transformer(nn.Module):
    def __init__(self, emb_dim, output_dim, n_layers=1, n_heads=1, mlp_dim=None, vocab_size=10, dropout=0.0, causal=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads))
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    attention=MultiheadAttention(emb_dim, self.n_heads, bias=True, batch_first=True),
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
        seq_len = x.size(1)
        x = self.embedding(x)
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
        return x[:, -1]

class LinearMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, x, rope=None, causal=True):
        bsz, seq_len, dim = x.size()
        q = rearrange(self.q_proj(x), 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b s (h d) -> (b h) s d', h=self.num_heads).contiguous()
        
        if rope:
            q = rope.rotate_queries_or_keys(q).reshape(bsz*self.num_heads, seq_len, self.head_dim).contiguous()
            k = rope.rotate_queries_or_keys(k).reshape(bsz*self.num_heads, seq_len, self.head_dim).contiguous()
        
        q = torch.exp(q)
        k = torch.exp(k)
        
        if causal:
            kv = torch.cumsum(torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2)), dim=1)
            k1 = torch.cumsum(k, dim=1)
        else:
            kv = torch.einsum('zsD, zsd -> zDd', k, v).unsqueeze(1)
            k1 = k.sum(dim=1, keepdim=True)
        out = torch.matmul(q.unsqueeze(-2), kv).squeeze(-2) / (q*k1).sum(-1, keepdim=True)
        
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.num_heads)
        return self.out_proj(out)

class LinearTransformer(nn.Module):
    def __init__(self, emb_dim, output_dim, n_layers=1, n_heads=1, mlp_dim=None, vocab_size=10, dropout=0.0, causal=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads))
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    attention=LinearMultiheadAttention(emb_dim, self.n_heads, bias=True),
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
            a_out = layer.attention(x, rope=self.rope, causal=self.causal)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x[:, -1]

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
    def __init__(self, emb_dim, output_dim, n_layers=1, n_heads=1, mlp_dim=None, vocab_size=10, dropout=0.0, causal=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
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
            a_out = layer.attention(x, rope=self.rope, causal=self.causal)
            x = layer.norm2(x + layer.dropout1(a_out))
            ff_out = layer.feedforward(x)
            x = x + layer.dropout1(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x[:, -1]