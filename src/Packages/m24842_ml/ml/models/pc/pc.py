import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
from contextlib import contextmanager

@contextmanager
def no_param_grad(model):
    requires_grad_states = {}
    for name, param in model.named_parameters():
        requires_grad_states[name] = param.requires_grad
        param.requires_grad = False
    try:
        yield
    finally:
        for name, param in model.named_parameters():
            param.requires_grad = requires_grad_states[name]

class PCLayer(nn.Module):
    def __init__(self, f_module, b_module, f_loss_fn=F.mse_loss, b_loss_fn=F.mse_loss, device=torch.device('cpu')):
        super().__init__()
        self.f_module = f_module
        self.b_module = b_module
        self.f_loss_fn = f_loss_fn
        self.b_loss_fn = b_loss_fn
        self.x_shape = None
        self.y_shape = None
        self.device = device
        self.to(device)
    
    @torch.no_grad()
    def reset(self, x=None, y=None):
        if x is not None:
            self.x_shape = x.shape
            temp = self.f_module(x)
            self.y_shape = temp.shape
            return temp
        elif y is not None:
            self.y_shape = y.shape
            temp = self.b_module(y)
            self.x_shape = temp.shape
            return temp
    
    def forward_energy(self, x, y):
        """
        x -> p_y
        p_y, y -> loss
        """
        assert self.x_shape is not None, "x must be initialized before calling forward_energy"
        p_y = self.f_module(x)
        return self.f_loss_fn(p_y, y)

    def backward_energy(self, x, y):
        """
        p_x <- y
        p_x, x -> loss
        """
        assert self.y_shape is not None, "y must be initialized before calling backward_energy"
        p_x = self.b_module(y)
        return self.b_loss_fn(p_x, x)

class PCModel(nn.Module):
    def __init__(self, layers, max_its=1, min_energy=1e-1, energy_lr=1e-3, energy_optimizer_class=optim.SGD, energy_scheduler=False, device=torch.device('cpu')):
        super().__init__()
        self.layers = layers
        self.cached_x_shape = None
        self.cached_y_shape = None
        self.max_its = max_its
        self.min_energy = min_energy
        self.energy_lr = energy_lr
        self.energy_optimizer_class = energy_optimizer_class
        self.energy_scheduler_class = energy_scheduler
        self.device = device
        self.to(device)
    
    def _reset_layers(self, x=None, y=None):
        if x is not None:
            for layer in self.layers:
                x = layer.reset(x, None)
            return x.shape
        elif y is not None:
            for layer in reversed(self.layers):
                y = layer.reset(None, y)
            return y.shape
    
    def reset(self, x=None, y=None):
        if x is not None:
            self.cached_x_shape = x.shape
            zero_x = torch.zeros_like(x, device=self.device)
            self.cached_y_shape = self._reset_layers(zero_x)
        elif y is not None:
            self.cached_y_shape = y.shape
            zero_y = torch.zeros_like(y, device=self.device)
            self.cached_x_shape = self._reset_layers(zero_y)
    
    @torch.no_grad()
    def forwad_state_init(self, x):
        intr_tensors = []
        for layer in self.layers:
            intr_tensors.append(x.clone().detach().requires_grad_(True))
            x = layer.f_module(x)
        intr_tensors.append(x.clone().detach().requires_grad_(True))
        return intr_tensors
    
    @torch.no_grad()
    def backward_state_init(self, y):
        intr_tensors = []
        for layer in reversed(self.layers):
            intr_tensors.append(y.clone().detach().requires_grad_(True))
            y = layer.b_module(y)
        intr_tensors.append(y.clone().detach().requires_grad_(True))
        return intr_tensors[::-1]
    
    def train_forward(self, x, y, param_optimizer, iterative=False, init_dir="forward"):
        self.reset(x, y)
        # Training
        if init_dir == "forward":
            intr_tensors = self.forwad_state_init(x)
            intr_tensors[-1] = y
        elif init_dir == "backward":
            intr_tensors = self.backward_state_init(y)
            intr_tensors[0] = x
        else:
            raise ValueError("State initialization direction must be either 'forward' or 'backward'")
        energy_optimizer = self.energy_optimizer_class(intr_tensors[1:-1], lr=self.energy_lr)
        energy_scheduler = optim.lr_scheduler.CosineAnnealingLR(energy_optimizer, T_max=self.max_its, eta_min=1e-6) if self.energy_scheduler_class else None
        if iterative:
            # Iterative training:
            # One energy convergence step per parameter update
            for i in range(self.max_its):
                total_energy = 0
                energy_optimizer.zero_grad()
                param_optimizer.zero_grad()
                for j, layer in enumerate(self.layers):
                    f_e = layer.forward_energy(intr_tensors[j], intr_tensors[j+1])
                    b_e = layer.backward_energy(intr_tensors[j], intr_tensors[j+1])
                    total_energy = total_energy + f_e + b_e
                total_energy.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                energy_optimizer.step()
                param_optimizer.step()
                if energy_scheduler is not None: energy_scheduler.step()
                if total_energy.item() < self.min_energy: break
        else:
            # Non-iterative training:
            # Energy convergence before parameter update
            param_optimizer.zero_grad()
            for i in range(self.max_its):
                total_energy = 0
                energy_optimizer.zero_grad()
                for j, layer in enumerate(self.layers):
                    f_e = layer.forward_energy(intr_tensors[j], intr_tensors[j+1])
                    b_e = layer.backward_energy(intr_tensors[j], intr_tensors[j+1])
                    total_energy = total_energy + f_e + b_e
                total_energy.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                energy_optimizer.step()
                if energy_scheduler is not None: energy_scheduler.step()
                if total_energy.item() < self.min_energy: break
            param_optimizer.step()
        return total_energy
    
    def forward(self, x=None, y=None):
        assert not (x is None and y is None), "Either x or y must be provided"
        self.reset(x, y)

        if x is not None:
            # Forward inference
            intr_tensors = self.forwad_state_init(x)
            energy_optimizer = self.energy_optimizer_class(intr_tensors[1:], lr=self.energy_lr) # Optimizing outputs only
            energy_scheduler = optim.lr_scheduler.CosineAnnealingLR(energy_optimizer, T_max=self.max_its, eta_min=1e-6) if self.energy_scheduler_class else None
            for i in range(self.max_its):
                total_energy = 0
                energy_optimizer.zero_grad()
                for j, layer in enumerate(self.layers):
                    f_e = layer.forward_energy(intr_tensors[j], intr_tensors[j+1])
                    b_e = layer.backward_energy(intr_tensors[j], intr_tensors[j+1])
                    total_energy = total_energy + f_e + b_e
                total_energy.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                energy_optimizer.step()
                if energy_scheduler is not None: energy_scheduler.step()
                if total_energy.item() < self.min_energy: break
            return intr_tensors[-1]
        elif y is not None:
            # Backward inference
            intr_tensors = self.backward_state_init(y)
            energy_optimizer = self.energy_optimizer_class(intr_tensors[:-1], lr=self.energy_lr) # Optimizing inputs only
            energy_scheduler = optim.lr_scheduler.CosineAnnealingLR(energy_optimizer, T_max=self.max_its, eta_min=1e-6) if self.energy_scheduler_class else None
            for i in range(self.max_its):
                total_energy = 0
                energy_optimizer.zero_grad()
                for j, layer in enumerate(self.layers):
                    f_e = layer.forward_energy(intr_tensors[j], intr_tensors[j+1])
                    b_e = layer.backward_energy(intr_tensors[j], intr_tensors[j+1])
                    total_energy = total_energy + f_e + b_e
                total_energy.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                energy_optimizer.step()
                if energy_scheduler is not None: energy_scheduler.step()
                if total_energy.item() < self.min_energy: break
            return intr_tensors[0]

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cpu')

transform = transforms.Compose([transforms.ToTensor(),])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

def cross_entropy(y, target):
    return F.cross_entropy(y, target.argmax(dim=1))

model = PCModel(
    layers=nn.ModuleList([
        PCLayer(nn.Sequential(nn.Linear(784, 128, bias=False), nn.ReLU(), nn.Linear(128, 128, bias=False)), nn.Sequential(nn.Linear(128, 128, bias=False), nn.ReLU(), nn.Linear(128, 784, bias=False)), device=device),
        PCLayer(nn.Linear(128, 10, bias=False), nn.Linear(10, 128, bias=False), f_loss_fn=cross_entropy, b_loss_fn=F.mse_loss, device=device)
    ]),
    max_its=10,
    min_energy=1e-3,
    energy_lr=1e-1,
    energy_optimizer_class=optim.SGD,
    energy_scheduler=True,
    device=device
)

param_optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)

def train():
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.flatten(1).to(device)
        target = target.to(device)
        target = F.one_hot(target, num_classes=10).float()
        train_energy = model.train_forward(data, target, param_optimizer, iterative=True, init_dir="forward")
        if batch_idx % 100 == 0 and batch_idx > 0:
            with no_param_grad(model):
                y = model(data)
                acc = (y.argmax(dim=1) == target.argmax(dim=1)).float().mean()
                loss = cross_entropy(y, target)
                print(f"Train epoch: {epoch}, batch: {batch_idx}", train_energy.item(), loss.item(), 100*acc.item())

def test():
    model.eval()
    with no_param_grad(model):
        total_loss = 0
        correct = 0
        for data, target in test_loader:
            data = data.flatten(1).to(device)
            target = target.to(device)
            y = model(data)
            loss = F.cross_entropy(y, target)
            total_loss += loss.item()
            correct += (y.argmax(dim=1) == target).sum().item()

    print("Test Loss:", total_loss / len(test_loader))
    print("Test Accuracy:", 100 * correct / len(test_loader.dataset))

epochs = 10
for epoch in range(epochs):
    train()
    test()