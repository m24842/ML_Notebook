import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import math
import warnings
from functools import lru_cache
from einops import rearrange
from ..common import *

class MultiheadAttention(nn.Module):
    """
    Vanilla Softmax Attention.
    """
    def __init__(self, d_model, n_heads, bias=True,
                 qk_dim=None, dropout=0.0,
                 batch_first=False, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.qk_dim = d_model if qk_dim is None else qk_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.d_head = d_model // n_heads
        self.device = device
                
        self.beta = nn.Parameter(torch.empty(n_heads, device=device))
        self.beta._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, self.qk_dim, bias=bias, device=device)
        self.k_proj = nn.Linear(d_model, self.qk_dim, bias=bias, device=device)
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
        
    def forward(self, x, causal=False, rope=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        tgt_len, bsz, d_model = x.shape
        src_len = x.shape[0]
        
        q = self.q_proj(x)  # (tgt_len, batch_size, d_model)
        k = self.k_proj(x)  # (src_len, batch_size, d_model)
        v = self.v_proj(x)  # (src_len, batch_size, d_model)
        
        q = rearrange(q, 's b (h d) -> b h s d', h=self.n_heads).contiguous()  # (bsz, n_heads, tgt_len, d_head)
        k = rearrange(k, 's b (h d) -> b h s d', h=self.n_heads).contiguous()  # (bsz, n_heads, src_len, d_head)
        v = rearrange(v, 's b (h d) -> b h s d', h=self.n_heads).contiguous()  # (bsz, n_heads, src_len, d_head)
        
        if rope:
            if rope.use_xpos:
                q, k = rope.rotate_queries_and_keys(q, k)
            else:
                q = rope.rotate_queries_or_keys(q)
                k = rope.rotate_queries_or_keys(k)
        
        beta = torch.exp(self.beta).reshape(1, self.n_heads, 1, 1)
        q = q / (math.sqrt(self.d_head) * beta)
        
        q = q.flatten(0, 1)
        k = k.flatten(0, 1)
        v = v.flatten(0, 1)
        
        attn_output = F.scaled_dot_product_attention(q, k, v, scale=1.0, dropout_p=self.dropout, is_causal=causal)  # (bsz * n_heads, src_len, d_head)
        
        attn_output = rearrange(attn_output, '(b h) s d -> s b (h d)', h=self.n_heads).contiguous()  # (tgt_len, bsz, d_model)
        attn_output = self.out_proj(attn_output)
        
        if self.batch_first:
            return attn_output.transpose(0, 1)
        return attn_output

class LinearAttention(nn.Module):
    """
    Vanilla Linear Attention.
    """
    def __init__(self, d_model, n_heads, bias=True,
                 qk_dim=None, batch_first=False, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.qk_dim = d_model if qk_dim is None else qk_dim
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.batch_first = batch_first
        self.device = device
        
        self.beta = nn.Parameter(torch.empty(n_heads, device=device))
        self.beta._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, self.qk_dim, bias=bias, device=device)
        self.k_proj = nn.Linear(d_model, self.qk_dim, bias=bias, device=device)
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
    
    def forward(self, x, rope=None, causal=True):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        src_len, bsz, d_model = x.size()
        tgt_len = src_len
        q = rearrange(self.q_proj(x), 's b (h d) -> b h s d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 's b (h d) -> b h s d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 's b (h d) -> (b h) s d', h=self.n_heads).contiguous()
        
        if rope:
            if rope.use_xpos:
                q, k = rope.rotate_queries_and_keys(q, k)
            else:
                q = rope.rotate_queries_or_keys(q)
                k = rope.rotate_queries_or_keys(k)
        
        beta = torch.exp(self.beta).reshape(1, self.n_heads, 1, 1)
        q = q / (math.sqrt(self.d_head) * beta)
        k = k / (math.sqrt(self.d_head) * beta)
        
        q = q.flatten(0, 1).contiguous()
        k = k.flatten(0, 1).contiguous()
        
        q = F.softplus(q)
        k = F.softplus(k)
        
        if causal:
            kv = torch.cumsum(torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2)), dim=1)
            kn = torch.cumsum(k, dim=1)
        else:
            kv = torch.einsum('zsk, zsv -> zkv', k, v).unsqueeze(1)
            kn = k.sum(dim=1, keepdim=True)
        out = torch.matmul(q.unsqueeze(-2), kv).squeeze(-2) / torch.matmul(q.unsqueeze(-2), kn.unsqueeze(-1)).squeeze(-1)
        out = rearrange(out, '(b h) s d -> s b (h d)', h=self.n_heads)
        out = self.out_proj(out)
        
        if self.batch_first:
            out = out.transpose(0, 1)
        return out

class OrthoLinearAttention(nn.Module):
    """
    Orthogonal Linear Attention.
    A derivative of linear attention that orthogonalizes queries and keys for each head to reduce crossterm interference.
    """
    def __init__(self, d_model, n_heads, bias=True,
                 qk_dim=None, batch_first=False, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.qk_dim = d_model if qk_dim is None else qk_dim
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.batch_first = batch_first
        self.device = device
        
        self.beta = nn.Parameter(torch.empty(n_heads, device=device))
        self.beta._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, self.qk_dim, bias=bias, device=device)
        self.k_proj = nn.Linear(d_model, self.qk_dim, bias=bias, device=device)
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
    
    def forward(self, x, rope=None, causal=True):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        src_len, bsz, d_model = x.size()
        tgt_len = src_len
        q = rearrange(self.q_proj(x), 's b (h d) -> b h s d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 's b (h d) -> b h s d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 's b (h d) -> (b h) s d', h=self.n_heads)
        
        if rope:
            if rope.use_xpos:
                q, k = rope.rotate_queries_and_keys(q, k)
            else:
                q = rope.rotate_queries_or_keys(q)
                k = rope.rotate_queries_or_keys(k)
        
        beta = torch.exp(self.beta).reshape(1, self.n_heads, 1, 1)
        q = q * beta
        k = k * beta
        
        q = q.flatten(0, 1).contiguous()
        k = k.flatten(0, 1).contiguous()
        
        q = q.softmax(-1)
        k = k.softmax(-1)
        
        if causal:
            kv = torch.cumsum(torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2)), dim=1)
            kn = torch.cumsum(k, dim=1)
        else:
            kv = torch.einsum('zsk, zsv -> zkv', k, v).unsqueeze(1)
            kn = k.sum(1, keepdim=True)
        out = torch.matmul(q.unsqueeze(-2), kv).squeeze(-2) / torch.matmul(q.unsqueeze(-2), kn.unsqueeze(-1)).squeeze(-1)
        out = rearrange(out, '(b h) s d -> s b (h d)', h=self.n_heads)
        out = self.out_proj(out)
        
        if self.batch_first:
            out = out.transpose(0, 1)
        return out

class CompressionAttention(nn.Module):
    """
    Compression Attention.
    A derivative of softmax attention that compresses input sequences to a fixed length before expanding back to the original length.
    Achieved by two attention operations of linear complexity with respect to sequence length.
    """
    def __init__(self, d_model, n_heads, compressed_len,
                 qk_dim=None, dropout=0.0,
                 bias=True, batch_first=False, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.qk_dim = d_model if qk_dim is None else qk_dim
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.compressed_len = compressed_len
        self.batch_first = batch_first
        self.dropout = dropout
        self.device = device
        
        self.q_c = nn.Parameter(torch.empty((compressed_len, self.qk_dim), device=device))
        self.q_c._no_weight_decay = True
        self.beta = nn.Parameter(torch.empty(n_heads, device=device))
        self.beta._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, self.qk_dim, bias=bias, device=device)
        self.k_proj = nn.Linear(d_model, self.qk_dim, bias=bias, device=device)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, device=device)

        self._reset_parameters()
        
    def _reset_parameters(self):
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
        
        # Reshape for multi-head attention
        q_c = rearrange(q_c, 'c b (h d) -> b h c d', h=self.n_heads).contiguous()
        q_s = rearrange(q_s, 's b (h d) -> b h s d', h=self.n_heads).contiguous()
        k_s = rearrange(k_s, 's b (h d) -> b h s d', h=self.n_heads).contiguous()
        v_s = rearrange(v_s, 's b (h d) -> b h s d', h=self.n_heads).contiguous()
        
        if rope:
            if rope.use_xpos:
                q_s, k_s = rope.rotate_queries_and_keys(q_s, k_s)
            else:
                q_s = rope.rotate_queries_or_keys(q_s)
                k_s = rope.rotate_queries_or_keys(k_s)
        
        beta = torch.exp(self.beta).reshape(1, self.n_heads, 1, 1)
        q_s = q_s / (math.sqrt(self.d_head) * beta)
        
        q_c = q_c.flatten(0, 1)
        q_s = q_s.flatten(0, 1)
        k_s = k_s.flatten(0, 1)
        v_s = v_s.flatten(0, 1)
        
        kv_s = torch.cat([k_s, v_s], dim=-1)  # (bsz * n_heads, src_len, 2*d_head)
        
        if causal:
            # Manually perform softmax with cumulative sum for causal attention
            c_attn_weights = torch.matmul(q_c, k_s.transpose(-2, -1))  # (bsz * n_heads, cmprs_len, src_len)
            c_attn_weights = torch.exp(c_attn_weights - torch.max(c_attn_weights, dim=-1, keepdim=True).values)  # (bsz * n_heads, cmprs_len, src_len)
            c_attn_weights = F.dropout(c_attn_weights, p=self.dropout, training=self.training)
            
            # Generate compressed keys and values
            kv_c = torch.cumsum((c_attn_weights.unsqueeze(-1) * kv_s.unsqueeze(1)), dim=-2) / torch.cumsum(c_attn_weights.unsqueeze(-1), dim=-2)  # (bsz * n_heads, cmprs_len, src_len, 2*d_head)
            
            k_c, v_c = kv_c.transpose(1, 2).split([self.d_head, self.d_head], dim=-1)  # (bsz, n_heads, cmprs_len, src_len, d_head)
            s_attn_output = F.scaled_dot_product_attention(q_s.unsqueeze(-2), k_c, v_c, scale=1.0, dropout_p=self.dropout, is_causal=False).squeeze(-2)  # (bsz * n_heads, tgt_len, d_head)
        else:
            if self.device == "mps":
                c_attn_weights = torch.matmul(q_c, k_s.transpose(-2, -1))  # (bsz * n_heads, cmprs_len, src_len)
                c_attn_weights = F.softmax(c_attn_weights, dim=-1)
                kv_c = torch.matmul(c_attn_weights, kv_s)  # (bsz * n_heads, cmprs_len, src_len)
            else:
                kv_c = F.scaled_dot_product_attention(q_c, k_s, kv_s, scale=1.0, dropout_p=self.dropout, is_causal=False)  # (bsz * n_heads, cmprs_len, 2*d_head)
            k_c, v_c = kv_c.split([self.d_head, self.d_head], dim=-1)  # (bsz * n_heads, cmprs_len, d_head)
            s_attn_output = F.scaled_dot_product_attention(q_s, k_c, v_c, scale=1.0, dropout_p=self.dropout, is_causal=False)  # (bsz * n_heads, tgt_len, d_head)
        
        s_attn_output = rearrange(s_attn_output, '(b h) s d -> s b (h d)', h=self.n_heads).contiguous()  # (tgt_len, bsz, d_model)        
        s_attn_output = self.out_proj(s_attn_output)
        
        if self.batch_first:
            return s_attn_output.transpose(0, 1)
        return s_attn_output

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention.
    Applies softmax attention over a dilated sliding window of fixed length.
    """
    def __init__(self, d_model, n_heads, window_len, dilation=1,
                 qk_dim=None, dropout=0.0, bias=True,
                 batch_first=False, use_flex_attn=True, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.qk_dim = d_model if qk_dim is None else qk_dim
        self.n_heads = n_heads
        self.window_len = window_len
        self.dilation = dilation
        self.dropout = dropout
        self.batch_first = batch_first
        self.d_head = d_model // n_heads
        self.use_flex_attn = use_flex_attn
        self.device = device
        
        self.beta = nn.Parameter(torch.empty(n_heads, device=device))
        self.beta._no_weight_decay = True
        self.q_proj = nn.Linear(d_model, self.qk_dim, bias=bias, device=device)
        self.k_proj = nn.Linear(d_model, self.qk_dim, bias=bias, device=device)
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
    
    @lru_cache(maxsize=2)
    def causal_windowed_block_mask(self, src_len, tgt_len=None):
        if tgt_len is None: tgt_len = src_len
        
        half_span = self.dilation * (self.window_len - 1)
        
        def mask_mod(b, h, q_idx, kv_idx):
            diff = q_idx - kv_idx
            return (diff >= 0) & (diff <= half_span) & ((diff % self.dilation) == 0)

        return create_block_mask(mask_mod, B=None, H=None, Q_LEN=src_len, KV_LEN=tgt_len, BLOCK_SIZE=1, device=self.device)
    
    @lru_cache(maxsize=2)
    def symmetric_windowed_block_mask(self, src_len, tgt_len=None):
        if tgt_len is None: tgt_len = src_len
        
        half = self.window_len // 2
        half_span = self.dilation * half

        def mask_mod(b, h, q_idx, kv_idx):
            diff = q_idx - kv_idx
            return diff.abs().le(half_span) & ((diff % self.dilation) == 0)

        return create_block_mask(mask_mod,  B=None, H=None, Q_LEN=src_len, KV_LEN=tgt_len, BLOCK_SIZE=1, device=self.device)
    
    def forward(self, x, rope=None, causal=True):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        src_len, bsz, d_model = x.shape
        tgt_len = src_len
        
        q = self.q_proj(x)  # (src_len, batch_size, d_model)
        k = self.k_proj(x)  # (src_len, batch_size, d_model)
        v = self.v_proj(x)  # (src_len, batch_size, d_model)
        
        q = rearrange(q, 's b (h d) -> b h s d', h=self.n_heads).contiguous()  # (bsz, n_heads, tgt_len, d_head)
        k = rearrange(k, 's b (h d) -> b h s d', h=self.n_heads).contiguous()  # (bsz, n_heads, src_len, d_head)
        v = rearrange(v, 's b (h d) -> b h s d', h=self.n_heads).contiguous()  # (bsz, n_heads, src_len, d_head)
        
        if rope:
            if rope.use_xpos:
                q, k = rope.rotate_queries_and_keys(q, k)
            else:
                q = rope.rotate_queries_or_keys(q)
                k = rope.rotate_queries_or_keys(k)
        
        beta = torch.exp(self.beta).reshape(1, self.n_heads, 1, 1)
        q = q / (math.sqrt(self.d_head) * beta)
        
        if not self.use_flex_attn:
            q = q.flatten(0, 1)
            k = k.flatten(0, 1)
            v = v.flatten(0, 1)
            
            if causal:
                attn_mask = self.causal_windowed_mask(src_len, self.window_len, dilation=self.dilation, to_bias=True)  # (1, tgt_len, src_len)
            else:
                attn_mask = self.symmetric_windowed_mask(src_len, self.window_len, dilation=self.dilation, to_bias=True)  # (1, tgt_len, src_len)

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0, dropout_p=self.dropout)  # (bsz * n_heads, tgt_len, d_head)
            
            attn_output = rearrange(attn_output, '(b h) s d -> s b (h d)', h=self.n_heads)
        else:
            if causal:
                block_mask = self.causal_windowed_block_mask(src_len, tgt_len)
            else:
                block_mask = self.symmetric_windowed_block_mask(src_len, tgt_len)
            
            attn_device = "cuda" if self.device == "cuda" else "cpu"
            
            q = q.to(attn_device)
            k = k.to(attn_device)
            v = v.to(attn_device)

            block_mask = block_mask.to(attn_device)

            attn_output = flex_attention(q, k, v, block_mask=block_mask, scale=1.0)
            
            attn_output = rearrange(attn_output, 'b h s d -> s b (h d)', h=self.n_heads).to(self.device)
        
        attn_output = self.out_proj(attn_output)
        
        if self.batch_first:
            return attn_output.transpose(0, 1)
        return attn_output
