import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
from mamba2 import Mamba2, Mamba2Config, InferenceCache, Mamba2LMHeadModel
import cv2

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