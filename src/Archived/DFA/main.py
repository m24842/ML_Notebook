import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from dfa import *

mps = torch.device('mps')
cpu = torch.device('cpu')

class DFANet(nn.Module):
    def __init__(self):
        super(DFANet, self).__init__()
        error_dim = 10
        self.in_proj = DFAFullyConnected(784, 256, error_dim, activation='silu')
        self.recurrent = DFAFullyConnected(256, 128, error_dim, activation='silu')
        self.out_proj = DFAFullyConnected(128, 10, error_dim, activation='none', last_layer=True)
    
    def forward(self, x, calc_grad=True):
        x = self.in_proj(x, calc_grad)
        x = self.recurrent(x, calc_grad)
        x = self.out_proj(x, calc_grad)
        return x
    
    def backward(self, global_error):
        self.in_proj.backward(global_error)
        self.recurrent.backward(global_error)
        self.out_proj.backward(global_error)
    
class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(784, 256),
            nn.SiLU(),
        )
        self.recurrent = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(128, 10),
        )
    
    def forward(self, x, calc_grad=True):
        x = self.in_proj(x)
        x = self.recurrent(x)
        x = self.out_proj(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda (lambda x: x.view(-1)),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

dfa = True
bp = not dfa
if dfa: net = DFANet().to(mps)
if bp: net = BPNet().to(mps)
optimizer = optim.AdamW(net.parameters(), lr=1e-3)

size = 784
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(mps)
        
        target = nn.functional.one_hot(target, num_classes=10).float().to(mps)
        optimizer.zero_grad()
        output = net(data, calc_grad=True)
        
        error = output - target
        
        if bp: nn.functional.cross_entropy(output, target, reduction="mean").backward()
        total_loss += nn.functional.cross_entropy(output, target, reduction="mean")
        if dfa: net.backward(error)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {(total_loss.item()/(len(train_loader.dataset))):.4f}")

net.eval()
accuracy = 0
for data, target in test_loader:
    data = data.to(mps)
    target = target.to(mps)
    padding = torch.zeros(data.size(0), size - 784, device=mps)
    data = torch.cat((data, padding), dim=1)
    
    output = net(data)
    pred = torch.argmax(output, dim=1)
    accuracy += (pred == target).sum().item()
accuracy /= len(test_loader.dataset)
print(f"Test Accuracy: {accuracy}")