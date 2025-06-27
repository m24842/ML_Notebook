import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
import math
import warnings
import opt_einsum
from functools import lru_cache
from einops import rearrange
from ..common import *

class MultiheadAttention(nn.Module):
    """
    Vanilla Multihead Attention.
    Slight difference: the typical 1/sqrt(d_model) attention score scale is now a per head learnable parameter beta initialized at 1/sqrt(d_model).
    """
    def __init__(self, d_model, n_heads, dropout=0.0, bias=True,
                 attn_sink=False, batch_first=False, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.d_head = d_model // n_heads
        self.device = device
        
        assert self.d_head * n_heads == self.d_model, "d_model must be divisible by n_heads"
        
        self.beta = nn.Parameter(torch.empty(self.n_heads, device=device))
        self.beta._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        
        self.attn_sink = attn_sink
        
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
        
    def forward(self, x, attn_mask=None, rope=None):
        # Handle batch_first option
        if self.batch_first:
            x = x.transpose(0, 1)
        
        tgt_len, bsz, d_model = x.shape
        src_len = x.shape[0]
        
        # Apply linear projections
        q = self.q_proj(x)  # (tgt_len, batch_size, d_model)
        k = self.k_proj(x)  # (src_len, batch_size, d_model)
        v = self.v_proj(x)  # (src_len, batch_size, d_model)
        
        # Add zero attention if requested
        if self.attn_sink:
            src_len += 1
            k = torch.cat([k, torch.zeros((1, bsz, d_model), dtype=k.dtype, device=k.device)], dim=0)
            v = torch.cat([v, torch.zeros((1, bsz, d_model), dtype=v.dtype, device=v.device)], dim=0)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
        
        # Reshape q, k, v for multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.n_heads, self.d_head).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.n_heads, self.d_head).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.n_heads, self.d_head).transpose(0, 1)
        
        if rope:
            if rope.use_xpos:
                q, k = rope.rotate_queries_and_keys(q.reshape(bsz, self.n_heads, tgt_len, self.d_head), k.reshape(bsz, self.n_heads, src_len, self.d_head))
            else:
                q = rope.rotate_queries_or_keys(q)
                k = rope.rotate_queries_or_keys(k)
        q = q.reshape(bsz * self.n_heads, tgt_len, self.d_head).contiguous()
        k = k.reshape(bsz * self.n_heads, src_len, self.d_head).contiguous()
        
        # Calculate attention scores
        beta = torch.exp(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1)
        # beta = F.softplus(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1)
        q = q / (math.sqrt(self.d_head) * beta)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # (bsz * n_heads, tgt_len, src_len)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.repeat(self.n_heads, 1, 1)
            attn_output_weights = attn_output_weights + attn_mask
        
        # Convert attention weights to probabilities
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        
        # Apply attention weights to values
        attn_output = torch.bmm(attn_output_weights, v)  # (bsz * n_heads, tgt_len, d_head)
        
        # Reshape output
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, d_model)
        attn_output = self.out_proj(attn_output)
        
        # Return in the correct format depending on batch_first
        if self.batch_first:
            return attn_output.transpose(0, 1)
        return attn_output

class LinearAttention(nn.Module):
    """
    Vanilla Linear Attention.
    Kernel function is softplus.
    """
    def __init__(self, d_model, n_heads, bias=True, attn_sink=False, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_sink = attn_sink
        self.device = device
        
        self.beta = nn.Parameter(torch.empty(self.n_heads, device=device))
        self.beta._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        
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
    
    def forward(self, x, rope=None, causal=True):
        bsz, src_len, d_model = x.size()
        tgt_len = src_len
        q = rearrange(self.q_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b s (h d) -> (b h) s d', h=self.n_heads).contiguous()
        
        if self.attn_sink:
            src_len += 1
            k = torch.cat([k, torch.zeros((bsz, self.n_heads, 1, self.d_head), dtype=x.dtype, device=self.device)], dim=2)
            v = torch.cat([v, torch.zeros((bsz*self.n_heads, 1, self.d_head), dtype=x.dtype, device=self.device)], dim=1)
        
        if rope:
            if rope.use_xpos:
                q, k = rope.rotate_queries_and_keys(q, k)
            else:
                q = rope.rotate_queries_or_keys(q)
                k = rope.rotate_queries_or_keys(k)
        q = q.reshape(bsz * self.n_heads, tgt_len, self.d_head).contiguous()
        k = k.reshape(bsz * self.n_heads, src_len, self.d_head).contiguous()
        
        beta = torch.exp(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1)
        # beta = F.softplus(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1)
        q = q / (math.sqrt(self.d_head) * beta)
        k = k / (math.sqrt(self.d_head) * beta)
        
        # q = torch.exp(q)
        # k = torch.exp(k)
        # q = F.elu(q) + 1
        # k = F.elu(k) + 1
        q = F.softplus(q)
        k = F.softplus(k)
        
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
    def __init__(self, d_model, n_heads, bias=True, attn_sink=False, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_sink = attn_sink
        self.device = device
        
        self.beta = nn.Parameter(torch.empty(self.n_heads, device=device))
        self.beta._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        
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
    
    def forward(self, x, rope=None, causal=True):
        bsz, src_len, d_model = x.size()
        tgt_len = src_len
        q = rearrange(self.q_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b s (h d) -> (b h) s d', h=self.n_heads).contiguous()
        
        if self.attn_sink:
            src_len += 1
            k = torch.cat([k, torch.zeros((bsz, self.n_heads, 1, self.d_head), dtype=x.dtype, device=self.device)], dim=2)
            v = torch.cat([v, torch.zeros((bsz*self.n_heads, 1, self.d_head), dtype=x.dtype, device=self.device)], dim=1)
        
        if rope:
            if rope.use_xpos:
                q, k = rope.rotate_queries_and_keys(q, k)
            else:
                q = rope.rotate_queries_or_keys(q)
                k = rope.rotate_queries_or_keys(k)
        q = q.reshape(bsz * self.n_heads, tgt_len, self.d_head).contiguous()
        k = k.reshape(bsz * self.n_heads, src_len, self.d_head).contiguous()
        
        beta = torch.exp(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1)
        # beta = F.softplus(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1)
        q = q * beta
        k = k * beta
        
        q = q.softmax(-1)
        k = k.softmax(-1)
        
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
    def __init__(self, d_model, n_heads, compressed_len, attn_sink=False,
                 dropout=0.0, bias=True, batch_first=False, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.compressed_len = compressed_len
        self.attn_sink = attn_sink
        self.batch_first = batch_first
        self.dropout = dropout
        self.device = device
        
        self.q_c = nn.Parameter(torch.empty((compressed_len, d_model), device=device))
        self.q_c._no_weight_decay = True
        self.beta = nn.Parameter(torch.empty(self.n_heads, device=device))
        self.beta._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, device=device)

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
        if self.batch_first:
            x = x.transpose(0, 1)
        
        cmprs_len = self.compressed_len
        src_len, bsz, d_model = x.shape
        tgt_len = src_len
        
        q_c = self.q_c.unsqueeze(1).repeat(1, bsz, 1)  # (compressed_len, d_model)
        q_s = self.q_proj(x)  # (tgt_len, batch_size, d_model)
        k_s = self.k_proj(x)  # (src_len, batch_size, d_model)
        v_s = self.v_proj(x)  # (src_len, batch_size, d_model)
        
        if self.attn_sink:
            src_len += 1
            k_s = torch.cat([k_s, torch.zeros((1, bsz, d_model), dtype=x.dtype, device=x.device)], dim=0)
            v_s = torch.cat([v_s, torch.zeros((1, bsz, d_model), dtype=x.dtype, device=x.device)], dim=0)
        
        # Reshape for multi-head attention
        q_c = rearrange(q_c, 'c b (h d) -> (b h) c d', h=self.n_heads).contiguous()
        q_s = rearrange(q_s, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        k_s = rearrange(k_s, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        v_s = rearrange(v_s, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        
        if rope:
            if rope.use_xpos:
                q_s, k_s = rope.rotate_queries_and_keys(q_s.reshape(bsz, self.n_heads, tgt_len, self.d_head), k_s.reshape(bsz, self.n_heads, src_len, self.d_head))
            else:
                q_s = rope.rotate_queries_or_keys(q_s.reshape(bsz, self.n_heads, tgt_len, self.d_head))
                k_s = rope.rotate_queries_or_keys(k_s.reshape(bsz, self.n_heads, src_len, self.d_head))
            q_s = q_s.reshape(bsz * self.n_heads, tgt_len, self.d_head).contiguous()
            k_s = k_s.reshape(bsz * self.n_heads, src_len, self.d_head).contiguous()
        
        kv_s = torch.cat([k_s, v_s], dim=-1)  # (bsz * n_heads, src_len, 2*d_head)
        
        ### Compression self attention ###
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
        
        ### Expansion self attention ###
        beta = torch.exp(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1)
        # beta = F.softplus(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1)
        q_s = q_s / (math.sqrt(self.d_head) * beta)
        
        if causal:
            # Calculate attention scores for compressed output
            kv_c = torch.cumsum((c_attn_weights.unsqueeze(-1) * kv_s.unsqueeze(1)), dim=2) / c_attn_norm.unsqueeze(-1)  # (bsz * n_heads, cmprs_len, src_len, 2*d_head)
            k_c, v_c = kv_c.split([self.d_head, self.d_head], dim=-1)  # (bsz * n_heads, cmprs_len, src_len, d_head)
            s_attn_weights = torch.einsum('zsd, zcsd -> zsc', q_s, k_c)  # (bsz * n_heads, tgt_len, cmprs_len)
            
            # Convert attention weights to probabilities
            s_attn_weights = F.softmax(s_attn_weights, dim=-1)
            s_attn_weights = F.dropout(s_attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention weights to values
            s_attn_output = torch.einsum('zsc, zcsd -> zsd', s_attn_weights, v_c)  # (bsz * n_heads, tgt_len, d_head)
        else:
            # Calculate attention scores for compressed output
            kv_c = torch.bmm(c_attn_weights, kv_s)  # (bsz * n_heads, cmprs_len, 2*d_head)
            k_c, v_c = kv_c.split([self.d_head, self.d_head], dim=-1)  # (bsz * n_heads, cmprs_len, d_head)
            s_attn_weights = torch.bmm(q_s, k_c.transpose(1, 2))  # (bsz * n_heads, tgt_len, cmprs_len)
            
            # Convert attention weights to probabilities
            s_attn_weights = F.softmax(s_attn_weights, dim=-1)
            s_attn_weights = F.dropout(s_attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention weights to values
            s_attn_output = torch.bmm(s_attn_weights, v_c)  # (bsz * n_heads, tgt_len, d_head)
        
        # Reshape output
        s_attn_output = s_attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, d_model)
        
        # Apply final projection
        s_attn_output = self.out_proj(s_attn_output)
        
        # Return in the correct format depending on batch_first
        if self.batch_first:
            return s_attn_output.transpose(0, 1)
        return s_attn_output

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention.
    Applies softmax attention over a dilated sliding window of fixed length.
    """
    def __init__(self, d_model, n_heads, window_len, dilation=1,
                 attn_sink=False, dropout=0.0, bias=True, batch_first=False,
                 masked_window=True, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_len = window_len
        self.dilation = dilation
        self.dropout = dropout
        self.batch_first = batch_first
        self.d_head = d_model // n_heads
        self.masked_window = masked_window
        self.attn_sink = attn_sink
        self.device = device
        
        self.beta = nn.Parameter(torch.empty(self.n_heads, device=device))
        self.beta._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        
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
    
    @lru_cache(maxsize=2)
    def causal_windowed_mask(self, seq_len, window_len, dilation=1, to_bias=False):
        idxs = torch.arange(seq_len, device=self.device)
        rows = idxs.unsqueeze(1)
        cols = idxs.unsqueeze(0)
        diff = rows - cols

        allowed = (diff >= 0) & (diff <= dilation * (window_len - 1)) & ((diff % dilation) == 0)
        allowed = allowed.unsqueeze(0)
        
        if not to_bias:
            return allowed.float()

        mask = torch.where(allowed, 0.0, float('-inf'))
        return mask
    
    @lru_cache(maxsize=2)
    def symmetric_windowed_mask(self, seq_len, window_len, dilation=1, to_bias=False):
        idxs = torch.arange(seq_len, device=self.device)
        rows = idxs.unsqueeze(1)
        cols = idxs.unsqueeze(0)

        half = window_len // 2
        diff = rows - cols
        abs_diff = diff.abs()

        allowed = (abs_diff <= dilation * half) & ((diff % dilation) == 0)
        allowed = allowed.unsqueeze(0)
        
        if not to_bias:
            return allowed.float()

        mask = torch.where(allowed, 0.0, float('-inf'))
        return mask

    def shift(self, x, shifts, dims, fill_value=None, shift_min_max=None):
        # dims: (dim_src, dim_shift)
        # shifts: (S,) tensor of ints
        dim_src, dim_shift = dims

        # 1) move dims → [S, L, *rest]
        x2 = x.movedim((dim_src, dim_shift), (0, 1))
        S, L, *rest = x2.shape
        # flat_D = x2.numel() // (S * L)

        if shift_min_max is not None:
            s_min, s_max = shift_min_max
        else:
            if torch.is_tensor(shifts):
                s_min, s_max = shifts.min(), shifts.max()
            else:
                s_min, s_max = min(shifts), max(shifts)
        s_min, s_max = torch.as_tensor(s_min, device=x.device), torch.as_tensor(s_max, device=x.device)
        H_neg = torch.clamp(-s_min, min=0)
        H_pos = torch.clamp( s_max, min=0)
        ext_L = L + H_neg + H_pos

        # 3) allocate & fill
        if fill_value is None:
            padded = x2.new_empty((S, ext_L, *rest))
        else:
            padded = x2.new_full((S, ext_L, *rest), fill_value)

        # 4) build destination indices
        #   pos: (1, L)
        pos = torch.arange(L, device=x.device).view(1, L)
        #   shifts: (S,1)
        sh = shifts.view(S, 1)
        #   dest: (S, L)
        dest = pos + sh + H_neg

        # 5) scatter all at once
        x_flat = x2.reshape(S, L, -1)
        dest_exp = dest.view(S, L, 1).expand(x_flat.shape)
        padded.reshape(S, ext_L, -1).scatter_(1, dest_exp, x_flat)

        # 6) slice & move dims back
        out2 = padded[:, H_neg : H_neg + L]
        return out2.movedim((0,1), dims)
    
    def forward(self, x, rope=None, causal=True):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        src_len, bsz, d_model = x.shape
        
        q = self.q_proj(x)  # (src_len, batch_size, d_model)
        k = self.k_proj(x)  # (src_len, batch_size, d_model)
        v = self.v_proj(x)  # (src_len, batch_size, d_model)
        
        # Reshape for multi-head attention
        q = rearrange(q, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        k = rearrange(k, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        v = rearrange(v, 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        
        if rope:
            if rope.use_xpos:
                q, k = rope.rotate_queries_and_keys(q.reshape(bsz, self.n_heads, src_len, self.d_head), k.reshape(bsz, self.n_heads, src_len, self.d_head))
            else:
                q = rope.rotate_queries_or_keys(q.reshape(bsz, self.n_heads, src_len, self.d_head))
                k = rope.rotate_queries_or_keys(k.reshape(bsz, self.n_heads, src_len, self.d_head))
            q = q.reshape(bsz * self.n_heads, src_len, self.d_head).contiguous()
            k = k.reshape(bsz * self.n_heads, src_len, self.d_head).contiguous()
        
        beta = torch.exp(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1)
        # beta = F.softplus(self.beta).reshape(self.n_heads, 1, 1).repeat(bsz, 1, 1)
        q = q / (math.sqrt(self.d_head) * beta)
        
        if self.masked_window:
            if causal:
                attn_mask = self.causal_windowed_mask(src_len, self.window_len, dilation=self.dilation, to_bias=True)  # (1, src_len, src_len)
            else:
                attn_mask = self.symmetric_windowed_mask(src_len, self.window_len, dilation=self.dilation, to_bias=True)  # (1, src_len, src_len)
            
            if self.attn_sink:
                k = torch.cat([k, torch.zeros((bsz*self.n_heads, 1, self.d_head), dtype=k.dtype, device=k.device)], dim=1)
                v = torch.cat([v, torch.zeros((bsz*self.n_heads, 1, self.d_head), dtype=v.dtype, device=v.device)], dim=1)
                if attn_mask is not None:
                    attn_mask = F.pad(attn_mask, (0, 1))  # (1, src_len, src_len + 1)

            attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # (bsz * n_heads, src_len, src_len)
            attn_output_weights = attn_output_weights + attn_mask  # (bsz * n_heads, src_len, src_len)
            if self.attn_sink:
                sink_weight = torch.zeros((bsz * self.n_heads, src_len, 1), dtype=x.dtype, device=self.device)  # (bsz * n_heads, src_len, 1)
                attn_output_weights = torch.cat([attn_output_weights, sink_weight], dim=-1)  # (bsz * n_heads, src_len, src_len + 1)
            
            # Convert attention weights to probabilities
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
            
            if self.attn_sink: attn_output_weights = attn_output_weights[..., :-1]
            
            # Apply attention weights to values
            attn_output = torch.bmm(attn_output_weights, v)  # (bsz * n_heads, src_len, d_head)
        else:
            if causal:
                pad_amount = min(src_len-1, (self.window_len-1) * self.dilation)
                pad = (0, pad_amount)
            else:
                pad_amount = self.dilation * min((src_len - 1) // self.dilation, self.window_len // 2)
                pad = (pad_amount, pad_amount)

            q = dilated_sliding_window(q, size=src_len, dim=1, stride=self.dilation, dilation=1, pad=pad)
            # q = dilated_sliding_window_nopad(q, size=src_len, dim=1, stride=self.dilation, dilation=1, pad=pad)

            attn_output_weights = torch.einsum('zbsd, zsd -> zbs', q, k)  # (bsz * n_heads, n_bands, src_len)
            
            shifts = torch.arange(-pad[0], pad[1]+1, self.dilation, device=self.device)
            attn_output_weights = self.shift(attn_output_weights, shifts, shift_min_max=(shifts[0], shifts[-1]), dims=(1, 2), fill_value=float('-inf'))
            
            if self.attn_sink:
                sink_weight = torch.zeros((bsz * self.n_heads, 1, src_len), dtype=x.dtype, device=self.device)  # (bsz * n_heads, 1, src_len)
                attn_output_weights = torch.cat([attn_output_weights, sink_weight], dim=1)  # (bsz * n_heads, n_bands+1, src_len)
            
            # Convert attention weights to probabilities
            attn_output_weights = F.softmax(attn_output_weights, dim=1)
            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
            
            if self.attn_sink: attn_output_weights = attn_output_weights[:, :-1]
            
            attn_output_weights = self.shift(attn_output_weights, -shifts, shift_min_max=(-shifts[-1], -shifts[0]), dims=(1, 2), fill_value=0.0)
            
            # Apply attention weights to values
            attn_output = attn_output_weights.unsqueeze(-1) * v.unsqueeze(1)
            
            attn_output = self.shift(attn_output, shifts, shift_min_max=(shifts[0], shifts[-1]), dims=(1, 2), fill_value=0.0)
            
            attn_output = attn_output.sum(dim=1)
        
        # Apply final projection
        attn_output = rearrange(attn_output, '(b h) s d -> s b (h d)', h=self.n_heads)
        attn_output = self.out_proj(attn_output)
        
        # Return in the correct format depending on batch_first
        if self.batch_first:
            return attn_output.transpose(0, 1)
        return attn_output
