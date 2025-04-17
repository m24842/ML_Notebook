import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional as AF
import matplotlib.pyplot as plt
from mamba2 import *
from MultiLinear import MultiLinear
import math
from transformers import AutoTokenizer

device = torch.device('mps')
torch.autograd.set_detect_anomaly(True)

class TemporalAutoencoder(nn.Module):
    def __init__(self):
        super(TemporalAutoencoder, self).__init__()
        self.config = Mamba2Config(
            d_model=12,
            d_state=128,
            n_layer=4,
            headdim=12,
            vocab_size=95,
            pad_vocab_size_multiple=1,
        )
        self.ft = Mamba2LMHeadModel(self.config)
        self.bt = Mamba2LMHeadModel(self.config)
    
    def forward(self, x, use_cache=False):
        batch_size, seq_len = x.size()[:2]
        y = torch.zeros((batch_size, seq_len, 95)).to(device)
        if not use_cache:
            self.reset(batch_size)
        
        for i in range(seq_len):
            input_f = x[:, i].clone() if i == 0 else y[:, i-1].clone()
            input_f, self.cache_f = self.ft(input_f, self.cache_f)
            y[:, i] = input_f.squeeze(1)
            input_f = torch.argmax(input_f, dim=-1)
        
        y = y.squeeze(2)
        
        return y
    
    def reverse(self, x, use_cache=False):
        batch_size, seq_len = x.size()[:2]
        y = torch.zeros((batch_size, seq_len, 95)).to(device)
        if not use_cache:
            self.reset(batch_size)
        
        for i in range(seq_len):
            input_b = x[:, i].clone() if i == 0 else y[:, i-1].clone()
            input_b, self.cache_b = self.bt(input_b, self.cache_b)
            y[:, seq_len-1-i] = input_b.squeeze(1)
            input_b = torch.argmax(input_b, dim=-1)
        
        y = y.squeeze(2)
        
        return y
    
    def bidirection(self, x_f, x_b, length, return_state=False):
        batch_size, seq_len = x_f.size()[:2]
        y_f = torch.zeros((batch_size, seq_len+length, 95)).to(device)
        y_b = torch.zeros((batch_size, seq_len+length, 95)).to(device)
        self.reset(batch_size)
        if return_state:
            states_f = torch.stack([
                torch.zeros((batch_size, seq_len+length-1, self.config.nheads, self.config.headdim, self.config.d_state))
                for _ in range(self.config.n_layer)
            ]).to(device)
            states_b = torch.stack([
                torch.zeros((batch_size, seq_len+length-1, self.config.nheads, self.config.headdim, self.config.d_state))
                for _ in range(self.config.n_layer)
            ]).to(device)
        for i in range(seq_len+length):
            if i < seq_len:
                input_f = x_f[:, i].clone()
                input_b = x_b[:, seq_len-1-i].clone()
            input_f, self.cache_f = self.ft(input_f, self.cache_f)
            input_b, self.cache_b = self.bt(input_b, self.cache_b)
            y_f[:, i] = input_f.squeeze(1)
            y_b[:, seq_len+length-1-i] = input_b.squeeze(1)
            input_f = torch.argmax(input_f, dim=-1)
            input_b = torch.argmax(input_b, dim=-1)
            
            if return_state and i < seq_len+length-1:
                for j in range(self.config.n_layer):
                    states_f[j][:, i] = self.cache_f[j].ssm_state
                    states_b[j][:, seq_len+length-2-i] = self.cache_b[j].ssm_state
        
        y_f, y_b = y_f.squeeze(2), y_b.squeeze(2)
        
        if return_state:
            return y_f, y_b, states_f, states_b
        
        return y_f, y_b

    def reset(self, batch_size=1):
        self.cache_f = [
            InferenceCache.alloc(batch_size, self.config, device=device)
            for _ in range(self.config.n_layer)
        ]
        self.cache_b = [
            InferenceCache.alloc(batch_size, self.config, device=device)
            for _ in range(self.config.n_layer)
        ]
        
class TemporalAutoencoderLM(nn.Module):
    def __init__(self):
        super(TemporalAutoencoderLM, self).__init__()
        self.tauto = TemporalAutoencoder()
    
    def forward(self, x, use_cache=False, decode=True):
        x = self.tauto(x, use_cache, decode)
        if decode:
            x = torch.argmax(x, dim=-1)
        return x
    
    def reverse(self, x, use_cache=False, decode=True):
        x = self.tauto.reverse(x, use_cache, decode)
        if decode:
            x = torch.argmax(x, dim=-1)
        return x
    
    def bidirection(self, x_f, x_b, length, decode=False, return_state=False):
        if return_state:
            y_f, y_b, states_f, states_b = self.tauto.bidirection(x_f, x_b, length, return_state)
        else:
            y_f, y_b = self.tauto.bidirection(x_f, x_b, length)
        
        if decode:
            y_f = torch.argmax(y_f, dim=-1)
            y_b = torch.argmax(y_b, dim=-1)
        
        if return_state:
            return y_f, y_b, states_f, states_b
        
        return y_f, y_b
    
    def reset(self, batch_size=1):
        self.tauto.reset(batch_size)
    
class MambaExpert(nn.Module):
    def __init__(self, d_model, d_state, d_conv=4, expand=2):
        super().__init__()
        self.config = Mamba2Config(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            headdim=64,
            chunk_size=10,
            expand=expand,
        )
        self.mamba = Mamba2(self.config, device=device)
        
        self.cache = InferenceCache.alloc(1, self.config, device=device)

    def forward(self, x, use_cache=False):
        x, self.cache = self.mamba(x, self.cache if use_cache else None)
        return x
    
    def step(self, x, use_cache=False):
        x, self.cache = self.mamba.step(x, self.cache if use_cache else None)
        return x

    def reset(self, batch_size=1):
        self.cache = InferenceCache.alloc(batch_size, self.config, device=device)
    
class MoE_MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, num_experts=4, top_k=2):
        """
        d_model: Dimension of input features.
        num_experts: Total number of Mamba experts.
        top_k: Number of experts to activate per input.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Experts (each expert is a Mamba block)
        self.experts = nn.ModuleList([MambaExpert(d_model, d_state) for _ in range(num_experts)])
        
        # Gating network (learned routing)
        self.gate = nn.Linear(d_model, num_experts)  # Each token gets a score for each expert
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5), nonlinearity='linear')
        nn.init.zeros_(self.gate.bias)

    def forward(self, x, use_cache=False):        
        # Compute expert selection probabilities
        logits = self.gate(x)  # Shape: [batch, seq_len, num_experts]
        logits = logits + torch.randn_like(logits) * 1e-2
        scores = F.softmax(logits, dim=-1)  # Normalize scores
        top_k_values, top_k_indices = torch.topk(scores, self.top_k, dim=-1)  # Select top-k experts
        
        # Create a mask to zero out non-selected experts
        expert_mask = torch.zeros_like(scores).scatter(-1, top_k_indices, top_k_values)  # Shape: [batch, seq_len, num_experts]
        
        # Normalize selection weights
        expert_mask = expert_mask / expert_mask.sum(dim=-1, keepdim=True)  # Ensure sum is 1
        
        # Compute expert outputs
        expert_outputs = torch.stack([expert(x, use_cache) for expert in self.experts], dim=-1)  # Shape: [batch, seq_len, d_model, num_experts]
        
        if self.training:
            dropout_prob = 0.1
            mask = (torch.rand(self.num_experts) > dropout_prob).float()
            expert_outputs = expert_outputs * mask.unsqueeze(0).to(device)
        
        # Weighted sum of selected experts
        output = torch.einsum("bse,bsde->bsd", expert_mask, expert_outputs)  # Sum across experts

        return output
    
    def step(self, x, use_cache=False):        
        # Compute expert selection probabilities
        logits = self.gate(x)  # Shape: [batch, seq_len, num_experts]
        logits = logits + torch.randn_like(logits) * 1e-2
        scores = F.softmax(logits, dim=-1)  # Normalize scores
        top_k_values, top_k_indices = torch.topk(scores, self.top_k, dim=-1)  # Select top-k experts
        
        # Create a mask to zero out non-selected experts
        expert_mask = torch.zeros_like(scores).scatter(-1, top_k_indices, top_k_values)  # Shape: [batch, seq_len, num_experts]
        
        # Normalize selection weights
        expert_mask = expert_mask / expert_mask.sum(dim=-1, keepdim=True)  # Ensure sum is 1
        
        # Compute expert outputs
        expert_outputs = torch.stack([expert.step(x, use_cache) for expert in self.experts], dim=-1)  # Shape: [batch, seq_len, d_model, num_experts]
        
        if self.training:
            dropout_prob = 0.1
            mask = (torch.rand(self.num_experts) > dropout_prob).float()
            expert_outputs = expert_outputs * mask.unsqueeze(0).to(device)
        
        # Weighted sum of selected experts
        output = torch.einsum("bse,bsde->bsd", expert_mask, expert_outputs)  # Sum across experts

        return output
    
    def reset(self, batch_size=1):
        for expert in self.experts:
            expert.reset(batch_size)

class MoE_MambaLM(nn.Module):
    def __init__(self, latent_dim, state_dim, num_layers, vocab_size=96, num_experts=4, top_k=2):
        super(MoE_MambaLM, self).__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.vocab_size = vocab_size
        
        self.backbone = nn.ModuleDict(
            dict(
                embedding = nn.Embedding(vocab_size, latent_dim, device=device),
                layers = nn.ModuleList([
                    nn.ModuleDict(
                        dict(
                            norm = RMSNorm(self.latent_dim, device=device),
                            mixer = MoE_MambaLayer(self.latent_dim, self.state_dim, self.num_experts, self.top_k),
                        )
                    ) for _ in range(self.num_layers)
                ]),
                norm_f = RMSNorm(self.latent_dim, device=device),
            )
        )
        
        self.prob_out = nn.Linear(self.latent_dim, self.vocab_size, bias=False, device=device)
    
    def forward(self, x, use_cache=False):
        batch_size, seq_len = x.size()[:2]
        if not use_cache: self.reset(batch_size)
        x_latent = self.backbone.embedding(x.clone().long())
        for layer in self.backbone.layers:
            mixer_out = layer.mixer(layer.norm(x_latent), use_cache)
            x_latent =  x_latent + mixer_out
        x_latent = self.backbone.norm_f(x_latent)
        x = self.prob_out(x_latent)
        return x
    
    def step(self, x, use_cache=False):
        batch_size = x.size(0)
        if not use_cache: self.reset(batch_size)
        x_latent = self.backbone.embedding(x.clone().long())
        for layer in self.backbone.layers:
            mixer_out = layer.mixer(layer.norm(x_latent), use_cache)
            x_latent =  x_latent + mixer_out
        x_latent = self.backbone.norm_f(x_latent)
        x = self.prob_out(x_latent)
        return x
    
    def reset(self, batch_size=1):
        for layer in self.backbone.layers:
            layer.mixer.reset(batch_size)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device=None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class FixedAttention(nn.Module):
    def __init__(self, d_model, n_heads, device):
        super(FixedAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        
        self.fc = nn.Linear(d_model, d_model, bias=False)
        
        nn.init.xavier_uniform_(self.fc.weight)
        
        self.to(device)
        
    def forward(self, q, k, v, attn_mask=None):
        q_exp = torch.exp(q)
        return self.fc(q_exp), attn_mask
    
class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, device):
        super(LinearAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device

        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        
        self.to(device)
    
    def forward(self, q, k, v, attn_mask=None):
        seq_len, batch_size = q.size()[:2]
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        q_exp = torch.exp(q)
        k_exp = torch.exp(k)
        q_head = q_exp.reshape(seq_len, batch_size, self.n_heads, self.d_model//self.n_heads)
        k_head = k_exp.reshape(seq_len, batch_size, self.n_heads, self.d_model//self.n_heads)
        v_head = v.reshape(seq_len, batch_size, self.n_heads, self.d_model//self.n_heads)
        kv_head = torch.einsum('sbnd, sbne -> sbnde', k_head, v_head)
        norm = torch.einsum('sbnd, sbne -> sbnde', k_head, torch.ones_like(v_head))
        out = torch.einsum('sbnd, sbnde -> sbne', q_head, kv_head) / torch.einsum('sbnd, sbnde -> sbne', q_head, norm)
        out = out.reshape(seq_len, batch_size, self.d_model)
        out = out.cumsum(dim=1)
        del q, k, v, q_exp, k_exp, q_head, k_head, v_head, kv_head, norm
        return out, attn_mask
    
class TransformerLM(nn.Module):
    def __init__(self, d_model, nhead, num_layers, vocab_size, device):
        super(TransformerLM, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.backbone = nn.ModuleDict(
            dict(
                embedding = nn.Embedding(vocab_size, d_model, device=device),
                positional_encoding = PositionalEncoding(d_model, dropout=0, device=device),
                layers = nn.ModuleList([
                    nn.ModuleDict(
                        dict(
                            # attention=nn.MultiheadAttention(d_model, nhead),
                            # attention=LinearAttention(d_model, nhead, device),
                            attention=FixedAttention(d_model, nhead, device),
                            norm1=nn.LayerNorm(d_model),
                            feedfoward=nn.Sequential(
                                nn.Linear(d_model, 4*d_model),
                                nn.GELU(),
                                nn.Linear(4*d_model, d_model),
                            ),
                            norm2=nn.LayerNorm(d_model),
                        )
                    ) for _ in range(self.num_layers)
                ]),
                out_proj=nn.Linear(d_model, vocab_size, bias=False),
            )
        )
        
        self.to(device)
    
    def encode(self, x):
        return self.tokenizer(x, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(self.device)
    
    def decode(self, x):
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in x]

    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        x = self.backbone.embedding(x.clone().long())
        x = x + self.backbone.positional_encoding(x.permute(1, 0, 2)).permute(1, 0, 2)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        attn_mask = attn_mask.masked_fill(attn_mask == 1, float('-inf'))
        for layer in self.backbone.layers:
            x_attn = x.permute(1, 0, 2)
            x_attn, _ = layer.attention(x_attn, x_attn, x_attn, attn_mask=attn_mask)
            x_attn = x_attn.permute(1, 0, 2)
            x = layer.norm1(x + x_attn)
            x_ff = layer.feedfoward(layer.norm1(x))
            x = layer.norm2(x + x_ff)
        x = self.backbone.out_proj(x)
        return x