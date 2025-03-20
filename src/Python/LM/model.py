import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional as AF
import matplotlib.pyplot as plt
from mamba2 import *
from MultiLinear import MultiLinear
import math
from AssociativeMemory import AssociativeAttentionBlock

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
    
class LetterAutoencoder(nn.Module):
    def __init__(self, input_dim=95, hidden_dim=64, latent_dim=8):
        super(LetterAutoencoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.SiLU(),
            )
        
        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 8)
        self.logvar_layer = nn.Linear(latent_dim, 8)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1),
            )
        
        self.mu = None
        self.logvar = None
     
    def encode(self, x, reparam=False):
        x = self.encoder(x)
        self.mu, self.logvar = self.mean_layer(x), self.logvar_layer(x)
        if reparam:
            return self.reparameterize()
        return self.mu, self.logvar

    def reparameterize(self):
        epsilon = torch.randn_like(self.logvar).to(device)      
        z = self.mu + self.logvar*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        self.mu, self.logvar = self.encode(x)
        z = self.reparameterize()
        x_recon = self.decode(z)
        return x_recon, self.mu, self.logvar
    
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
            
class AssociativeNet(nn.Module):
    def __init__(self, d_model, d_mem, num_layers, n_heads, vocab_size, retrieval_rate, retrieval_depth, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.retrieval_rate = retrieval_rate
        self.retrieval_depth = retrieval_depth
        self.device = device if device else torch.device('cpu')
        self.vocab_size = vocab_size
        
        self.backbone = nn.ModuleDict(
            dict(
                embedding = nn.Embedding(vocab_size, d_model, device=device),
                layers = nn.ModuleList([
                    nn.ModuleDict(
                        dict(
                            mixer=AssociativeAttentionBlock(d_model, d_mem, retrieval_rate, retrieval_depth, n_heads, device=device),
                            feedfoward=nn.Sequential(
                                nn.Linear(d_model, d_model),
                                nn.SiLU(),
                                nn.Linear(d_model, d_model),
                            ),
                            norm=nn.LayerNorm(d_model),
                        )
                    ) for _ in range(self.num_layers)
                ]),
            )
        )
        
        self.prob_out = nn.Linear(self.d_model, self.vocab_size, bias=False, device=device)
        
        self.to(self.device)
        
    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        x_latent = self.backbone.embedding(x.clone().long())
        for layer in self.backbone.layers:
            x_latent, _ = layer.mixer(x_latent)
            x_latent = x_latent + layer.feedfoward(x_latent)
            x_latent = layer.norm(x_latent[:, :, 0])
        x = self.prob_out(x_latent)
        return x