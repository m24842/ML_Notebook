import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
import torch.autograd as AG
import math
from einops import rearrange
import opt_einsum
from typing import Optional
from rotary_embedding_torch import RotaryEmbedding

# Transformers with RoPE
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, bias=True, add_bias_kv=False, 
                 add_zero_attn=False, batch_first=False):
        """
        Initialize the MultiheadAttention module.
        
        Args:
            d_model: Total dimension of the model
            n_heads: Number of parallel attention heads
            dropout: Dropout probability on attention weights
            bias: Add bias to input projections
            add_bias_kv: Add bias to the key and value sequences
            add_zero_attn: Add a new batch of zeros to the key and value sequences
            d_model: Total dimension of the key (default: d_model)
            d_model: Total dimension of the value (default: d_model)
            batch_first: If True, input and output tensors are provided as (batch, seq, feature)
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.d_head = d_model // n_heads
        
        assert self.d_head * n_heads == self.d_model, "d_model must be divisible by n_heads"
        
        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Optional bias for key and value
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, d_model))
            self.bias_v = nn.Parameter(torch.empty(1, 1, d_model))
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
            query: Query embeddings of shape (seq_len_q, batch_size, d_model) or 
                  (batch_size, seq_len_q, d_model) if batch_first=True
            key: Key embeddings of shape (seq_len_k, batch_size, d_model) or
                 (batch_size, seq_len_k, d_model) if batch_first=True
            value: Value embeddings of shape (seq_len_v, batch_size, d_model) or
                   (batch_size, seq_len_v, d_model) if batch_first=True
            key_padding_mask: If provided, specified padding elements in the key will
                              be ignored by the attention. Shape: (batch_size, seq_len_k)
            need_weights: If True, returns attention weights in addition to attention output
            attn_mask: 2D or 3D mask that prevents attention to certain positions
            average_attn_weights: If True, returns averaged attention weights over heads
            
        Returns:
            attn_output: Attention output of shape (seq_len_q, batch_size, d_model) or
                         (batch_size, seq_len_q, d_model) if batch_first=True
            attn_output_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)
                                 if need_weights=True, otherwise None
        """
        # Handle batch_first option
        if self.batch_first:
            x = x.transpose(0, 1)
        
        tgt_len, bsz, d_model = x.shape
        src_len = x.shape[0]
        
        # Apply linear projections
        q = self.q_proj(x)  # (tgt_len, batch_size, d_model)
        k = self.k_proj(x)  # (src_len, batch_size, d_model)
        v = self.v_proj(x)  # (src_len, batch_size, d_model)
        
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
            k = torch.cat([k, torch.zeros((1, bsz, d_model), dtype=k.dtype, device=k.device)], d_model=0)
            v = torch.cat([v, torch.zeros((1, bsz, d_model), dtype=v.dtype, device=v.device)], d_model=0)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        
        # Reshape q, k, v for multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.n_heads, self.d_head).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.n_heads, self.d_head).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.n_heads, self.d_head).transpose(0, 1)
        
        if rope:
            q = rope.rotate_queries_or_keys(q.reshape(bsz, self.n_heads, tgt_len, self.d_head)).reshape(bsz * self.n_heads, tgt_len, self.d_head)
            k = rope.rotate_queries_or_keys(k.reshape(bsz, self.n_heads, src_len, self.d_head)).reshape(bsz * self.n_heads, src_len, self.d_head)
        
        # Calculate attention scores
        q = q / math.sqrt(self.d_head)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # (bsz * n_heads, tgt_len, src_len)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.repeat(self.n_heads, 1, 1)
            attn_output_weights = attn_output_weights + attn_mask
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.n_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.n_heads, tgt_len, src_len)
        
        # Convert attention weights to probabilities
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        
        # Apply attention weights to values
        attn_output = torch.bmm(attn_output_weights, v)  # (bsz * n_heads, tgt_len, d_head)
        
        # Reshape output
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, d_model)
        attn_output = self.out_proj(attn_output)
        
        # Process attention weights if needed
        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.n_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
        else:
            attn_output_weights = None
        
        # Return in the correct format depending on batch_first
        if self.batch_first:
            return attn_output.transpose(0, 1), attn_output_weights
        return attn_output, attn_output_weights

class Transformer(nn.Module):
    def __init__(self, emb_dim, output_dim, n_layers=1, n_heads=1, mlp_dim=None, vocab_size=1, dropout=0.0, causal=True, use_embedding=True, device=torch.device('cpu')):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(vocab_size, emb_dim)
        else: self.embedding = nn.Linear(vocab_size, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads))
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    attention = MultiheadAttention(emb_dim, self.n_heads, bias=True, batch_first=True),
                    norm2 = nn.LayerNorm(emb_dim),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = nn.Sequential(
                        nn.Linear(emb_dim, self.mlp_dim, bias=True),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(self.mlp_dim, emb_dim, bias=True)
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

class LinearMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
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
    
    def forward(self, x, rope=None, causal=True):
        bsz, seq_len, d_model = x.size()
        q = rearrange(self.q_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b s (h d) -> (b h) s d', h=self.n_heads).contiguous()
        
        if rope:
            q = rope.rotate_queries_or_keys(q).reshape(bsz*self.n_heads, seq_len, self.d_head).contiguous()
            k = rope.rotate_queries_or_keys(k).reshape(bsz*self.n_heads, seq_len, self.d_head).contiguous()
        
        # q = torch.exp(q)
        # k = torch.exp(k)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        if causal:
            kv = torch.cumsum(torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2)), dim=1)
            k1 = torch.cumsum(k, dim=1)
        else:
            kv = torch.einsum('zsD, zsd -> zDd', k, v).unsqueeze(1)
            k1 = k.sum(dim=1, keepdim=True)
        out = torch.matmul(q.unsqueeze(-2), kv).squeeze(-2) / (q*k1).sum(-1, keepdim=True)
        
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.n_heads)
        return self.out_proj(out)

class LinearTransformer(nn.Module):
    def __init__(self, emb_dim, output_dim, n_layers=1, n_heads=1, mlp_dim=None, vocab_size=1, dropout=0.0, causal=True, use_embedding=True, device=torch.device('cpu')):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(vocab_size, emb_dim)
        else: self.embedding = nn.Linear(vocab_size, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads))
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    attention = LinearMultiheadAttention(emb_dim, self.n_heads, bias=True),
                    norm2 = nn.LayerNorm(emb_dim),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = nn.Sequential(
                        nn.Linear(emb_dim, self.mlp_dim, bias=True),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(self.mlp_dim, emb_dim, bias=True)
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

class OrthoLinearAttention(nn.Module):
    """
    Orthogonal Linear Attention:
    A derivative of linear attention that orthogonalizes queries and keys for each head to reduce crossterm interference.
    
    Interference free capacity scales exponentially with head count by the formula: <head d_model>^<head count>.
    An optimal choice for head dimension is 3.
    """
    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.beta = nn.Parameter(torch.zeros(1))
        self.beta._no_weight_decay = True
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
        q = (beta * q).softmax(-1)# * q.norm(dim=-1, keepdim=True)
        k = (beta * k).softmax(-1)# * k.norm(dim=-1, keepdim=True)
        
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
    def __init__(self, emb_dim, output_dim, n_layers=1, n_heads=1, mlp_dim=None, vocab_size=1, dropout=0.0, causal=True, use_embedding=True, device=torch.device('cpu')):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(vocab_size, emb_dim)
        else: self.embedding = nn.Linear(vocab_size, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), cache_if_possible=False)
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    attention = OrthoLinearAttention(emb_dim, self.n_heads, bias=True),
                    norm2 = nn.LayerNorm(emb_dim),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = nn.Sequential(
                        nn.Linear(emb_dim, self.mlp_dim, bias=True),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(self.mlp_dim, emb_dim, bias=True)
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

class CompressionAttention(nn.Module):
    """
    Compression Attention:
    A fixed size memory derivative of softmax attention that compresses input sequences to a fixed length before decompressing back to the original length.
    """
    def __init__(self, d_model, n_heads, mlp_dim, compressed_len, dropout=0.0, bias=True, batch_first=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.d_head = d_model // n_heads
        self.compressed_len = compressed_len
        self.batch_first = batch_first
        self.dropout = dropout
        
        self.q_down = nn.Parameter(torch.empty(compressed_len, d_model))
        self.q_down._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize projections using Xavier uniform
        nn.init.xavier_uniform_(self.q_down)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, x, rope=None, causal=True):
        # Handle batch_first option
        if self.batch_first:
            x = x.transpose(0, 1)
        
        cmprs_len = self.compressed_len
        _, bsz, d_model = x.shape
        src_len = x.shape[0]
        
        q_d = self.q_down.unsqueeze(1).repeat(1, bsz, 1)  # (compressed_len, d_model)
        q_u = self.q_proj(x)  # (src_len, batch_size, d_model)
        k_d = self.k_proj(x)  # (src_len, batch_size, d_model)
        v_d = self.v_proj(x)  # (src_len, batch_size, d_model)
        v_d_k = self.k_proj(v_d)  # (src_len, batch_size, d_model)
        v_d_v = self.v_proj(v_d)  # (src_len, batch_size, d_model)
        
        # Reshape q_d, k_d, v_d for multi-head attention
        q_d = rearrange(q_d, 'c b (h d) -> (b h) c d', h=self.n_heads).contiguous()
        q_u = rearrange(q_u, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        k_d = rearrange(k_d, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        v_d_k = rearrange(v_d_k, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        v_d_v = rearrange(v_d_v, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        v_d_kv = torch.cat([v_d_k, v_d_v], dim=-1)  # (bsz * n_heads, src_len, 2*d_head)
        
        if rope:
            # q_d = rope.rotate_queries_or_keys(q_d.reshape(bsz, self.n_heads, cmprs_len, self.d_head)).reshape(bsz * self.n_heads, cmprs_len, self.d_head)
            k_d = rope.rotate_queries_or_keys(k_d.reshape(bsz, self.n_heads, src_len, self.d_head)).reshape(bsz * self.n_heads, src_len, self.d_head)
        
        ### Downward self attention ###
        q_d = q_d / math.sqrt(self.d_head)
        down_attn_weights = torch.bmm(q_d, k_d.transpose(1, 2))  # (bsz * n_heads, cmprs_len, src_len)
        
        if causal:
            # Manually perform softmax with cumulative sum for causal attention
            down_attn_weights = torch.exp(down_attn_weights - torch.max(down_attn_weights, dim=-1, keepdim=True).values)  # (bsz * n_heads, cmprs_len, src_len)
            down_attn_norm = torch.cumsum(down_attn_weights, dim=-1)  # (bsz * n_heads, cmprs_len, src_len)
            down_attn_weights = F.dropout(down_attn_weights, p=self.dropout, training=self.training)
        else:
            # Convert attention weights to probabilities
            down_attn_weights = F.softmax(down_attn_weights, dim=-1)
            down_attn_weights = F.dropout(down_attn_weights, p=self.dropout, training=self.training)
        
        ### Upward self attention ###
        q_u = q_u / math.sqrt(self.d_head)
        
        if causal:
            # Calculate attention scores for compressed output
            kv_u = torch.cumsum((down_attn_weights.unsqueeze(-1) * v_d_kv.unsqueeze(1)), dim=2) / down_attn_norm.unsqueeze(-1)  # (bsz * n_heads, cmprs_len, 2*d_head)
            k_u, v_u = kv_u.split([self.d_head, self.d_head], dim=-1)  # (bsz * n_heads, cmprs_len, src_len, d_head)
            up_attn_weights = torch.einsum('zsd, zcsd -> zsc', q_u, k_u)  # (bsz * n_heads, src_len, cmprs_len)
            
            # Convert attention weights to probabilities
            up_attn_weights = F.softmax(up_attn_weights, dim=-1)
            up_attn_weights = F.dropout(up_attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention weights to values
            up_attn_output = torch.einsum('zsc, zcsd -> zsd', up_attn_weights, v_u)  # (bsz * n_heads, src_len, d_head)
        else:
            # Calculate attention scores for compressed output
            kv_u = torch.bmm(down_attn_weights, v_d_kv)  # (bsz * n_heads, cmprs_len, 2*d_head)
            k_u, v_u = kv_u.split([self.d_head, self.d_head], dim=-1)  # (bsz * n_heads, cmprs_len, d_head)
            up_attn_weights = torch.bmm(q_u, k_u.transpose(1, 2))  # (bsz * n_heads, src_len, cmprs_len)
            
            # Convert attention weights to probabilities
            up_attn_weights = F.softmax(up_attn_weights, dim=-1)
            up_attn_weights = F.dropout(up_attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention weights to values
            up_attn_output = torch.bmm(up_attn_weights, v_u)  # (bsz * n_heads, src_len, d_head)
        
        # Reshape output
        up_attn_output = up_attn_output.transpose(0, 1).contiguous().view(src_len, bsz, d_model)
        
        # Apply final projection
        up_attn_output = self.out_proj(up_attn_output)
        
        # Return in the correct format depending on batch_first
        if self.batch_first:
            return up_attn_output.transpose(0, 1)
        return up_attn_output

class CompressionTransformer(nn.Module):
    def __init__(self, emb_dim, output_dim, n_layers=1, n_heads=1, mlp_dim=None, compressed_len=64, vocab_size=1, dropout=0.0, causal=True, use_embedding=True, device=torch.device('cpu')):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.causal = causal
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*emb_dim
        self.compressed_len = compressed_len
        self.use_embedding = use_embedding
        if use_embedding: self.embedding = nn.Embedding(vocab_size, emb_dim)
        else: self.embedding = nn.Linear(vocab_size, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False)
        self.rope = RotaryEmbedding(dim=emb_dim//(2*self.n_heads), cache_if_possible=False)
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm1 = nn.LayerNorm(emb_dim),
                    dropout1 = nn.Dropout(dropout),
                    attention = CompressionAttention(emb_dim, self.n_heads, self.mlp_dim, compressed_len=compressed_len, dropout=dropout, batch_first=True),
                    norm2 = nn.LayerNorm(emb_dim),
                    dropout2 = nn.Dropout(dropout),
                    feedforward = nn.Sequential(
                        nn.Linear(emb_dim, self.mlp_dim, bias=True),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(self.mlp_dim, emb_dim, bias=True)
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
            x = x = x + layer.dropout2(ff_out)
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x