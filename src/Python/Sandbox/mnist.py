import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mamba2 import Mamba2, Mamba2Config, InferenceCache
import ssl
import numpy as np
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device('mps')

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=128):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sample_dim = 2
        # encoder
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.SiLU(),
            )
        
        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, self.sample_dim)
        self.logvar_layer = nn.Linear(latent_dim, self.sample_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.sample_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.input_dim),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon
        return z

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x_hat = torch.zeros(batch_size, 1, 28, 28).to(device)
        mean = torch.zeros(batch_size, self.sample_dim).to(device)
        logvar = torch.zeros(batch_size, self.sample_dim).to(device)
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

class RFA(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, bottleneck_dim=2):
        super(RFA, self).__init__()
        # Encoder
        self.encoder = nn.ModuleDict(
            dict(
                fc1=nn.Linear(input_dim, hidden_dim),
                act1=nn.SiLU(),
                fc2=nn.Linear(hidden_dim, bottleneck_dim),
                act2=nn.SiLU(),
            )
        )
        
        # Decoder
        self.decoder = nn.ModuleDict(
            dict(
                fc3=nn.Linear(bottleneck_dim, hidden_dim),
                act3=nn.SiLU(),
                fc4=nn.Linear(hidden_dim, input_dim),
                act4=nn.Sigmoid(),
            )
        )
        
        self.rand_feedback = dict(
            fc1=torch.rand(input_dim, input_dim, device=device),
            fc2=torch.rand(hidden_dim, input_dim, device=device),
            fc3=torch.rand(bottleneck_dim, input_dim, device=device),
            fc4=torch.rand(hidden_dim, input_dim, device=device),
        )
        
    def forward(self, x):
        # encoder
        fc1 = self.encoder.act1(self.encoder.fc1(x))
        fc2 = self.encoder.act2(self.encoder.fc2(fc1))
        
        # decoder
        fc3 = self.decoder.act3(self.decoder.fc3(fc2))
        fc4 = self.decoder.act4(self.decoder.fc4(fc3))
        
        return fc4, fc3, fc2, fc1
    
    def decode(self, x):
        fc3 = self.decoder.act3(self.decoder.fc3(x))
        fc4 = self.decoder.act4(self.decoder.fc4(fc3))
        return fc4

    def update_weights(self, x, y, lr=0.01):
        fc4, fc3, fc2, fc1 = self.forward(x)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        fc1 = fc1.unsqueeze(1)
        fc2 = fc2.unsqueeze(1)
        fc3 = fc3.unsqueeze(1)
        fc4 = fc4.unsqueeze(1)
        err = y - fc4
        err_fc4 = (self.rand_feedback['fc4'] @ err.mT) * self.sigmoid_derivative(fc4)
        err_fc3 = (self.rand_feedback['fc3'] @ err.mT) * self.silu_derivative(fc3)
        err_fc2 = (self.rand_feedback['fc2'] @ err.mT) * self.silu_derivative(fc2)
        err_fc1 = (self.rand_feedback['fc1'] @ err.mT) * self.silu_derivative(fc1)

        self.encoder.fc1.weight.data -= (lr * err_fc1 * x.mT).mean(0)
        self.encoder.fc2.weight.data -= (lr * err_fc2 * fc1.mT).mean(0)
        self.decoder.fc3.weight.data -= (lr * err_fc3 * fc2.mT).mean(0)
        self.decoder.fc4.weight.data -= (lr *  err_fc4 * fc3.mT).mean(0)
        
        return err.sum()
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def silu_derivative(self, x):
        sigmoid = 1 / (1 + torch.exp(-x))
        return sigmoid + x * sigmoid * (1 - sigmoid)

def loss_function(x, x_recon, mu, log_var):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence

# Load the training and test datasets
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

net1 = RFA().to(device)
optimizer1 = optim.Adam(net1.parameters(), lr=1e-3)
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.1, patience=2)

net2 = RFA().to(device)
optimizer2 = optim.Adam(net2.parameters(), lr=1e-3)
scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.1, patience=2)

print("Training")
num_epochs = 10
mse = nn.MSELoss()
for epoch in range(num_epochs):
    net1.train()
    net2.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        train_loss += net1.update_weights(data.flatten(1), data.flatten(1), lr=0.01)
        train_loss += net2.update_weights(data.transpose(2, 3).flatten(1), data.transpose(2, 3).flatten(1), lr=0.01)
        # optimizer1.zero_grad()
        # optimizer2.zero_grad()
        # recon_batch1, mu1, logvar1 = net1(data)
        # recon_batch2, mu2, logvar2 = net2(data.transpose(2, 3))
        # recon_batch1 = recon_batch1.view_as(data)
        # recon_batch2 = recon_batch2.view_as(data.transpose(2, 3))
        # loss1 = loss_function(data, recon_batch1, mu1, logvar1) + 1*(mse(mu1, mu2) + mse(logvar1, logvar2))
        # loss1.backward(retain_graph=True)
        # loss2 = loss_function(data.transpose(2, 3), recon_batch2, mu2, logvar2) + 1*(mse(mu1, mu2) + mse(logvar1, logvar2))
        # loss2.backward(retain_graph=True)
        # train_loss += loss1.item() + loss2.item()
        # optimizer1.step()
        # optimizer2.step()
    
    train_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}')
    
    scheduler1.step(train_loss)
    scheduler2.step(train_loss)

# Function to visualize the VAE outputs in the 2D latent space
def visualize_latent_space(model, device, n=20, figsize=15):
    model.eval()
    with torch.no_grad():
        # Create a grid of points in the latent space
        grid_x = np.linspace(-3, 3, n)
        grid_y = np.linspace(-3, 3, n)
        figure = np.zeros((28 * n, 28 * n))
        
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                x_decoded = model.decode(z_sample).cpu().numpy()
                digit = x_decoded[0].reshape(28, 28)
                figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit
        
        plt.figure(figsize=(figsize, figsize))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

# Visualize the VAE outputs in the 2D latent space
visualize_latent_space(net1, device)

visualize_latent_space(net2, device)
    
# net.eval()
# test_loss = 0
# with torch.no_grad():
#     for data, _ in test_loader:
#         data = data.to(device)
#         recon_batch, mu, logvar = net(data)
#         loss = loss_function(data, recon_batch, mu, logvar)
#         test_loss += loss.item()

# test_loss /= len(test_loader.dataset)
# print(f'Test Loss: {test_loss:.4f}')