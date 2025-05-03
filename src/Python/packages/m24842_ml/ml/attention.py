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

class MultiheadAttention(nn.Module):
    """
    Vanilla Multihead Attention.
    Slight difference: the typical 1/sqrt(d_model) attention score scale is now a per head learnable parameter beta initialized at 1/sqrt(d_model).
    """
    def __init__(self, d_model, n_heads, dropout=0.0, bias=True, add_bias_kv=False, 
                 add_zero_attn=False, batch_first=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.d_head = d_model // n_heads
        
        assert self.d_head * n_heads == self.d_model, "d_model must be divisible by n_heads"
        
        # Linear projections for query, key, and value
        self.beta = nn.Parameter(torch.zeros(self.n_heads))
        self.beta._no_weight_decay = True
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
        nn.init.constant_(self.beta, 0.)
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
        q = q / (math.sqrt(self.d_head) * torch.exp(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1))
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

class LinearAttention(nn.Module):
    """
    Vanilla Linear Attention.
    Kernel function is elu + 1.
    """
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.)
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

class OrthoLinearAttention(nn.Module):
    """
    Orthogonal Linear Attention.
    A derivative of linear attention that orthogonalizes queries and keys for each head to reduce crossterm interference.
    """
    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.beta = nn.Parameter(torch.zeros(self.n_heads))
        self.beta._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.constant_(self.beta, 0.)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.)
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
        
        beta = torch.exp(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1)
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

class CompressionAttention(nn.Module):
    """
    Compression Attention.
    A derivative of softmax attention that compresses input sequences to a fixed length before expanding back to the original length.
    Achieved by two linear with sequence length attention operations.
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
        
        self.q_c = nn.Parameter(torch.empty(compressed_len, d_model))
        self.q_c._no_weight_decay = True
        self.beta = nn.Parameter(torch.zeros(self.n_heads))
        self.beta._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize projections using Xavier uniform
        nn.init.constant_(self.beta, 0.)
        nn.init.xavier_uniform_(self.q_c)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, x, rope=None, causal=True):        
        # Handle batch_first option
        if self.batch_first:
            x = x.transpose(0, 1)
        
        cmprs_len = self.compressed_len
        _, bsz, d_model = x.shape
        src_len = x.shape[0]
        
        q_c = self.q_c.unsqueeze(1).repeat(1, bsz, 1)  # (compressed_len, d_model)
        q_s = self.q_proj(x)  # (src_len, batch_size, d_model)
        k_s = self.k_proj(x)  # (src_len, batch_size, d_model)
        v_s = self.v_proj(x)  # (src_len, batch_size, d_model)
        
        # Reshape for multi-head attention
        q_c = rearrange(q_c, 'c b (h d) -> (b h) c d', h=self.n_heads).contiguous()
        q_s = rearrange(q_s, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        k_s = rearrange(k_s, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        v_s = rearrange(v_s, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        
        if rope:
            # q_c = rope.rotate_queries_or_keys(q_c.reshape(bsz, self.n_heads, cmprs_len, self.d_head)).reshape(bsz * self.n_heads, cmprs_len, self.d_head)
            q_s = rope.rotate_queries_or_keys(q_s.reshape(bsz, self.n_heads, src_len, self.d_head)).reshape(bsz * self.n_heads, src_len, self.d_head)
            k_s = rope.rotate_queries_or_keys(k_s.reshape(bsz, self.n_heads, src_len, self.d_head)).reshape(bsz * self.n_heads, src_len, self.d_head)
        
        kv_s = torch.cat([k_s, v_s], dim=-1)  # (bsz * n_heads, src_len, 2*d_head)
        
        ### Downward self attention ###
        # q_c = q_c / math.sqrt(self.d_head)
        c_attn_weights = torch.bmm(q_c, k_s.transpose(1, 2))  # (bsz * n_heads, cmprs_len, src_len)
        
        if causal:
            # Manually perform softmax with cumulative sum for causal attention
            c_attn_weights = torch.exp(c_attn_weights - torch.max(c_attn_weights, dim=-1, keepdim=True).values)  # (bsz * n_heads, cmprs_len, src_len)
            c_attn_norm = torch.cumsum(c_attn_weights, dim=-1)  # (bsz * n_heads, cmprs_len, src_len)
            c_attn_weights = F.dropout(c_attn_weights, p=self.dropout, training=self.training)
        else:
            # Convert attention weights to probabilities
            c_attn_weights = F.softmax(c_attn_weights, dim=-1)
            c_attn_weights = F.dropout(c_attn_weights, p=self.dropout, training=self.training)
        
        ### Upward self attention ###
        q_s = q_s / (math.sqrt(self.d_head) * torch.exp(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1))
        
        if causal:
            # Calculate attention scores for compressed output
            kv_c = torch.cumsum((c_attn_weights.unsqueeze(-1) * kv_s.unsqueeze(1)), dim=2) / c_attn_norm.unsqueeze(-1)  # (bsz * n_heads, cmprs_len, 2*d_head)
            k_c, v_c = kv_c.split([self.d_head, self.d_head], dim=-1)  # (bsz * n_heads, cmprs_len, src_len, d_head)
            s_attn_weights = torch.einsum('zsd, zcsd -> zsc', q_s, k_c)  # (bsz * n_heads, src_len, cmprs_len)
            
            # Convert attention weights to probabilities
            s_attn_weights = F.softmax(s_attn_weights, dim=-1)
            s_attn_weights = F.dropout(s_attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention weights to values
            s_attn_output = torch.einsum('zsc, zcsd -> zsd', s_attn_weights, v_c)  # (bsz * n_heads, src_len, d_head)
        else:
            # Calculate attention scores for compressed output
            kv_c = torch.bmm(c_attn_weights, kv_s)  # (bsz * n_heads, cmprs_len, 2*d_head)
            k_c, v_c = kv_c.split([self.d_head, self.d_head], dim=-1)  # (bsz * n_heads, cmprs_len, d_head)
            s_attn_weights = torch.bmm(q_s, k_c.transpose(1, 2))  # (bsz * n_heads, src_len, cmprs_len)
            
            # Convert attention weights to probabilities
            s_attn_weights = F.softmax(s_attn_weights, dim=-1)
            s_attn_weights = F.dropout(s_attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention weights to values
            s_attn_output = torch.bmm(s_attn_weights, v_c)  # (bsz * n_heads, src_len, d_head)
        
        # Reshape output
        s_attn_output = s_attn_output.transpose(0, 1).contiguous().view(src_len, bsz, d_model)
        
        # Apply final projection
        s_attn_output = self.out_proj(s_attn_output)
        
        # Return in the correct format depending on batch_first
        if self.batch_first:
            return s_attn_output.transpose(0, 1)
        return s_attn_output
