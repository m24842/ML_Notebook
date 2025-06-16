import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
from contextlib import nullcontext, contextmanager

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
    def __init__(self, f_module, b_module, f_energy_fn=F.mse_loss, b_energy_fn=F.mse_loss, device=torch.device('cpu')):
        super().__init__()
        self.f_module = f_module
        self.b_module = b_module
        self.f_energy_fn = f_energy_fn
        self.b_energy_fn = b_energy_fn
        self.device = device
        self.to(device)
    
    @torch.no_grad()
    def get_io_shapes(self, x=None, y=None):
        if x is not None: return self.f_module(x)
        elif y is not None: return self.b_module(y)
        else: raise ValueError("Either x or y must be provided")
    
    def forward_energy(self, x, y):
        """
        x -> p_y
        p_y, y -> loss
        """
        p_y = self.f_module(x)
        return self.f_energy_fn(p_y, y)

    def backward_energy(self, x, y):
        """
        p_x <- y
        p_x, x -> loss
        """
        p_x = self.b_module(y)
        return self.b_energy_fn(p_x, x)

class PCModel(nn.Module):
    def __init__(self, layers, max_its=1, min_energy=1e-1, energy_lr=1e-3, energy_optimizer_class=optim.SGD, energy_grad_clip_norm=None, iterative_train=False, state_init_dir="forward", device=torch.device('cpu')):
        super().__init__()
        assert isinstance(layers, (list, nn.ModuleList)), "Layers must be a list or torch.nn.ModuleList"
        for layer in layers: assert isinstance(layer, PCLayer), "All layers must be of type PCLayer"
        self.layers = nn.ModuleList(layers) if type(layers) is list else layers
        self.cached_x_shape = None
        self.cached_y_shape = None
        self.max_its = max_its
        self.min_energy = min_energy
        self.energy_lr = energy_lr
        self.energy_optimizer_class = energy_optimizer_class
        self.energy_grad_clip_norm = energy_grad_clip_norm
        self.iterative_train = iterative_train
        self.state_init_dir = state_init_dir
        self.device = device
        self.is_pc_model = True
        self.to(device)
    
    def _zero_param_grads(self):
        for layer in self.layers:
            for param in layer.parameters():
                if param.grad is not None:
                    param.grad.zero_()
    
    def _normalize_param_grads(self):
        for layer in self.layers:
            for param in layer.parameters():
                if param.grad is not None:
                    param.grad.div_(self.max_its)
    
    @torch.no_grad()
    def get_io_shapes(self, x=None, y=None):
        if x is not None:
            for layer in self.layers:
                x = layer.get_io_shapes(x, None)
            return x.shape
        elif y is not None:
            for layer in reversed(self.layers):
                y = layer.get_io_shapes(None, y)
            return y.shape
    
    @torch.no_grad()
    def update_io_shapes(self, x=None, y=None):
        if (x is not None and self.cached_x_shape == x.shape) or (y is not None and self.cached_y_shape == y.shape): return
        if x is not None:
            self.cached_x_shape = x.shape
            zero_x = torch.zeros_like(x, device=self.device)
            self.cached_y_shape = self.get_io_shapes(zero_x)
        elif y is not None:
            self.cached_y_shape = y.shape
            zero_y = torch.zeros_like(y, device=self.device)
            self.cached_x_shape = self.get_io_shapes(zero_y)
    
    @torch.no_grad()
    def forwad_state_init(self, x):
        state_tensors = []
        for layer in self.layers:
            state_tensors.append(x.clone().detach().requires_grad_(True))
            x = layer.f_module(x)
        state_tensors.append(x.clone().detach().requires_grad_(True))
        return state_tensors
    
    @torch.no_grad()
    def backward_state_init(self, y):
        state_tensors = []
        for layer in reversed(self.layers):
            state_tensors.append(y.clone().detach().requires_grad_(True))
            y = layer.b_module(y)
        state_tensors.append(y.clone().detach().requires_grad_(True))
        return state_tensors[::-1]

    def compute_model_energy(self, state_tensors):
        energy = torch.stack([
            layer.forward_energy(state_tensors[j], state_tensors[j+1])
            for j, layer in enumerate(self.layers)
        ]+[
            layer.backward_energy(state_tensors[j], state_tensors[j+1])
            for j, layer in enumerate(self.layers)
        ]).sum()
        return energy
    
    def train_forward(self, x, y, scaler=None):
        self.update_io_shapes(x, y)
        
        # Amortized state initialization
        if self.state_init_dir == "forward":
            state_tensors = self.forwad_state_init(x)
            state_tensors[-1] = y
        elif self.state_init_dir == "backward":
            state_tensors = self.backward_state_init(y)
            state_tensors[0] = x
        else:
            raise ValueError("State initialization direction must be either 'forward' or 'backward'")
        
        energy_optimizer = self.energy_optimizer_class(state_tensors[1:-1], lr=self.energy_lr)
        
        if self.iterative_train:
            # Iterative training:
            # One energy convergence step per parameter gradient calculation
            for i in range(self.max_its):
                energy_optimizer.zero_grad()
                
                energy = self.compute_model_energy(state_tensors)
                
                if scaler is not None: scaler.scale(energy).backward()
                else: energy.backward()
                if self.energy_grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(state_tensors, max_norm=self.energy_grad_clip_norm)
                
                energy_optimizer.step()
                
                # if energy.item() < self.min_energy: break
            self._normalize_param_grads() # Normalize gradients by number of iterations
        else:
            # Non-iterative training:
            # Energy convergence before parameter gradient calculation
            for i in range(self.max_its):
                if i == self.max_its - 1: self._zero_param_grads() # Only calculate param grads on final iteration
                energy_optimizer.zero_grad()
                
                energy = self.compute_model_energy(state_tensors)
                
                if scaler is not None: scaler.scale(energy).backward()
                else: energy.backward()
                if self.energy_grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(state_tensors, max_norm=self.energy_grad_clip_norm)
                
                energy_optimizer.step()
                
                # if energy.item() < self.min_energy: break
        return energy
    
    def forward(self, x=None, y=None):
        assert not (x is None and y is None), "Either x or y must be provided"
        
        self.update_io_shapes(x, y)

        with nullcontext() if self.training else no_param_grad(self):
            if x is not None:
                # Forward inference
                state_tensors = self.forwad_state_init(x)
                
                energy_optimizer = self.energy_optimizer_class(state_tensors[1:], lr=self.energy_lr) # Exclude input
                for i in range(self.max_its):
                    energy_optimizer.zero_grad()
                    
                    energy = self.compute_model_energy(state_tensors)
                    
                    energy.backward()
                    if self.energy_grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(state_tensors, max_norm=self.energy_grad_clip_norm)
                    
                    energy_optimizer.step()
                    
                    if energy.item() < self.min_energy: break
                return state_tensors[-1]
            elif y is not None:
                # Backward inference
                state_tensors = self.backward_state_init(y)
                
                energy_optimizer = self.energy_optimizer_class(state_tensors[:-1], lr=self.energy_lr) # Exclude output
                for i in range(self.max_its):
                    energy_optimizer.zero_grad()
                    
                    energy = self.compute_model_energy(state_tensors)
                    
                    energy.backward()
                    if self.energy_grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(state_tensors, max_norm=self.energy_grad_clip_norm)
                    
                    energy_optimizer.step()
                    
                    if energy.item() < self.min_energy: break
                return state_tensors[0]

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# device = torch.device('cpu')

# transform = transforms.Compose([transforms.ToTensor(),])
# train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# model = PCModel(
#     layers=[
#         PCLayer(nn.Sequential(nn.Linear(784, 16, bias=False), nn.SiLU(), nn.Linear(16, 1024, bias=False)), nn.Sequential(nn.Linear(1024, 16, bias=False), nn.SiLU(), nn.Linear(16, 784, bias=False)), device=device),
#         PCLayer(nn.Sequential(nn.Linear(1024, 16, bias=False), nn.SiLU(), nn.Linear(16, 1024, bias=False)), nn.Sequential(nn.Linear(1024, 16, bias=False), nn.SiLU(), nn.Linear(16, 1024, bias=False)), device=device),
#         # PCLayer(nn.Sequential(nn.Linear(1024, 16, bias=False), nn.SiLU(), nn.Linear(16, 1024, bias=False)), nn.Sequential(nn.Linear(1024, 16, bias=False), nn.SiLU(), nn.Linear(16, 1024, bias=False)), device=device),
#         # PCLayer(nn.Sequential(nn.Linear(1024, 16, bias=False), nn.SiLU(), nn.Linear(16, 1024, bias=False)), nn.Sequential(nn.Linear(1024, 16, bias=False), nn.SiLU(), nn.Linear(16, 1024, bias=False)), device=device),
#         # PCLayer(nn.Sequential(nn.Linear(1024, 16, bias=False), nn.SiLU(), nn.Linear(16, 1024, bias=False)), nn.Sequential(nn.Linear(1024, 16, bias=False), nn.SiLU(), nn.Linear(16, 1024, bias=False)), device=device),
#         PCLayer(nn.Linear(1024, 10, bias=False), nn.Linear(10, 1024, bias=False), f_energy_fn=F.cross_entropy, b_energy_fn=F.mse_loss, device=device),
#     ],
#     max_its=3,
#     min_energy=1e-3,
#     energy_lr=1e-2,
#     energy_optimizer_class=optim.SGD,
#     device=device
# )

# param_optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)

# def train():
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data = data.flatten(1).to(device)
#         target = target.to(device)
#         target = F.one_hot(target, num_classes=10).float()
#         param_optimizer.zero_grad()
#         train_energy = model.train_forward(data, target)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         param_optimizer.step()
#         if batch_idx % 100 == 0 and batch_idx > 0:
#             with no_param_grad(model):
#                 y = model(data)
#                 acc = (y.argmax(dim=1) == target.argmax(dim=1)).float().mean()
#                 loss = F.cross_entropy(y, target)
#                 print(f"Train epoch: {epoch}, batch: {batch_idx}", train_energy.item(), loss.item(), 100*acc.item())

# def test():
#     model.eval()
#     total_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         data = data.flatten(1).to(device)
#         target = target.to(device)
#         y = model(data)
#         loss = F.cross_entropy(y, target)
#         total_loss += loss.item()
#         correct += (y.argmax(dim=1) == target).sum().item()

#     print("Test Loss:", total_loss / len(test_loader))
#     print("Test Accuracy:", 100 * correct / len(test_loader.dataset))

# epochs = 10
# for epoch in range(epochs):
#     train()
#     test()