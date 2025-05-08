import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return x

class Optimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.loss = None
        self.n_params = sum(p.numel() for group in self.param_groups for p in group['params'] if p.requires_grad)
        self.v_max = 1e-3
        self.damp = 0.1
        self.prcnt = 0.01
        self.update_mask = torch.rand(self.n_params) < self.prcnt
        for group in self.param_groups:
            group['velocity'] = []
            for p in group['params']:
                group['velocity'].append((2 * self.v_max * torch.rand_like(p.data, device=p.device)) - self.v_max)

    @torch.no_grad()
    def step(self, loss):
        dloss = loss - self.loss if self.loss is not None else 0
        dloss = torch.tensor(dloss, device=loss.device)
        self.loss = loss
        idx = 0
        for group in self.param_groups:
            lr = group['lr']
            for p, v in zip(group['params'], group['velocity']):
                if not p.requires_grad: continue
                mask = self.update_mask[idx:idx+p.numel()].reshape(p.shape)
                dL = dloss / (torch.linalg.norm(lr * v[mask]) + 1e-8)
                v[mask] -= dL * torch.sign(v[mask])
                v[mask] *= self.damp
                v[mask] = self.v_max*torch.tanh(v[mask] / self.v_max)
                idx += p.numel()
        self.update_mask = torch.rand(self.n_params) < self.prcnt
        idx = 0
        for group in self.param_groups:
            lr = group['lr']
            for p, v in zip(group['params'], group['velocity']):
                if not p.requires_grad: continue
                mask = self.update_mask[idx:idx+p.numel()].reshape(p.shape)
                p.data[mask] -= lr * v[mask]
                idx += p.numel()

@torch.no_grad()
def train(epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            # loss.backward()
            optimizer.step(loss)
            print(loss.item())
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(test_loader):.4f}')

def test():
    model.eval()
    correct = 0
    total_loss = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        data = data.view(data.size(0), -1)
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print(f'Test set: Accuracy: {100. * correct / len(test_loader.dataset):.2f}%, Loss: {total_loss / len(test_loader):.4f}')

transform = transforms.Compose([
    transforms.ToTensor(),
])
train_set = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

model = Net().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = Optimizer(model.parameters(), lr=1e1)

epochs = 10

train(epochs)
test()