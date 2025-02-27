import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import LetterAutoencoder, device
import matplotlib.pyplot as plt
import numpy as np

model_path = 'src/Python/TemporalAutoencoderLM/lae.pth'

def char_to_onehot(char):
    idx = ord(char) - 32
    onehot = torch.zeros(95)
    onehot[idx] = 1
    return onehot

def prob_dist_to_char(prob_dist):
    idx = torch.argmax(prob_dist)
    return chr(idx.item() + 32)

def loss_function(x, x_recon, mu, log_var):
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence

def train(net, optimizer, scheduler, epochs):
    print('Training...')
    avgs = []
    letters = np.linspace(32, 126, 95)
    x = torch.stack([char_to_onehot(chr(int(l))) for l in letters], dim=0).to(device)
    for epoch in range(1, epochs+1):
        avg_loss = 0
        for i in range(100):
            optimizer.zero_grad()
            
            outputs, mu, logvar = net(x)
            
            loss = loss_function(x, outputs, mu, logvar)
            
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            accuracy = (outputs.argmax(1) == x.argmax(1)).sum().item()/outputs.size(0)
            
            print(f'Epoch {epoch}, Loss: {loss.item():.5f}, Acc: {accuracy:.5f}, LR: {optimizer.param_groups[0]["lr"]:.5f}')
        
        scheduler.step(avg_loss/(i+1))
        torch.save(net.state_dict(), model_path)
        torch.save(optimizer.state_dict(), 'src/Python/TemporalAutoencoderLM/lae_optimizer.pth')
        torch.save(scheduler.state_dict(), 'src/Python/TemporalAutoencoderLM/lae_scheduler.pth')
        
        # avgs.append(avg_loss/(i+1))
        # plt.clf()
        # plt.plot(avgs)
        # plt.ylim(0, 1.1*max(avgs))
        # plt.pause(1e-1)

net = LetterAutoencoder().to(device)
try:
    net.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
except:
    pass

optimizer = optim.AdamW(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
try:
    optimizer.load_state_dict(torch.load('src/Python/TemporalAutoencoderLM/lae_optimizer.pth', weights_only=True))
except:
    pass
try:
    scheduler.load_state_dict(torch.load('src/Python/TemporalAutoencoderLM/lae_scheduler.pth', weights_only=True))
except:
    pass

# optimizer.param_groups[0]['lr'] = 1e-3

train(net, optimizer, scheduler, epochs=1000)