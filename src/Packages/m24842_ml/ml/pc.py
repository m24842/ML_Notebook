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

# from .models.transformers import MultiheadAttention
# class PCTransformer(PCModel):
#     def __init__(self, emb_dim, input_dim, output_dim,
#                  n_layers=1, n_heads=1, mlp_dim=None,
#                  dropout=0.0, causal=True, use_embedding=True, weight_tying=False,
#                  mlp_bias=True, attention_bias=True,
#                  pos_encoding=None, pos_encoding_max_len=None,
#                  device="cpu"):
#         class PCTransformerLayer(nn.Module):
#             def __init__(self, emb_dim, n_heads, mlp_dim,
#                         dropout=0.0, causal=True, use_embedding=True,
#                         weight_tying=False, mlp_bias=True,
#                         attention_bias=True,
#                         pos_encoding=None, pos_encoding_max_len=None, device="cpu"):
#                 super().__init__()
#                 self.emb_dim = emb_dim
#                 self.n_heads = n_heads
#                 self.mlp_dim = mlp_dim
#                 self.dropout = dropout
#                 self.causal = causal
#                 self.use_embedding = use_embedding
#                 self.weight_tying = weight_tying
#                 self.mlp_bias = mlp_bias
#                 self.attention_bias = attention_bias
#                 self.pos_encoding = pos_encoding
#                 self.pos_encoding_max_len = pos_encoding_max_len
#                 self.attention = MultiheadAttention(emb_dim, n_heads, bias=attention_bias, batch_first=True)
#                 self.norm1 = nn.LayerNorm(emb_dim)
#                 self.dropout1 = nn.Dropout(dropout)
#                 self.norm2 = nn.LayerNorm(emb_dim)
#                 self.dropout2 = nn.Dropout(dropout)
#                 self.feedforward = nn.Sequential(
#                     nn.Linear(emb_dim, mlp_dim, bias=mlp_bias),
#                     nn.ReLU(),
#                     nn.Dropout(dropout),
#                     nn.Linear(mlp_dim, emb_dim, bias=mlp_bias)
#                 )
#                 if pos_encoding == "rope":
#                     self.rope = RotaryEmbedding(dim=emb_dim//(2*n_heads), use_xpos=False, cache_if_possible=False)
#                 elif pos_encoding == "xpos":
#                     self.rope = RotaryEmbedding(dim=emb_dim//(2*n_heads), use_xpos=True, cache_if_possible=False)
#                 elif pos_encoding == "abs":
#                     assert pos_encoding_max_len is not None, "pos_encoding_max_len must be provided for absolute positional encoding"
#                     self.pos_encoding_max_len = pos_encoding_max_len
#                 else: self.rope = None
#                 self.abs_pos_encoding = nn.Embedding(pos_encoding_max_len, emb_dim) if pos_encoding == "abs" else None
                
#                 self.to(device)
            
#             def forward(self, x):
#                 x = self.norm1(x)
#                 if self.abs_pos_encoding is not None:
#                     pos = torch.arange(x.size(1), device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
#                     x = x + self.abs_pos_encoding(pos)
#                 a_out, _ = self.attention(x, attn_mask=None, rope=self.rope if self.pos_encoding == "rope" else None)
#                 x = x + self.dropout1(a_out)
#                 x = self.norm2(x)
#                 ff_out = self.feedforward(x)
#                 x = x + self.dropout2(ff_out)
#                 return x

#         f_in_proj = nn.Linear(input_dim, emb_dim, bias=False, device=device)
#         b_in_proj = nn.Linear(emb_dim, input_dim, bias=False, device=device)
#         f_out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
#         b_out_proj = nn.Linear(output_dim, emb_dim, bias=False, device=device)
#         layers = [
#                 PCLayer(
#                     f_in_proj,
#                     b_in_proj,
#                     f_energy_fn=F.mse_loss,
#                     b_energy_fn=F.mse_loss,
#                     device=device
#                 )
#             ]+[
#             PCLayer(
#                 PCTransformerLayer(emb_dim, n_heads, mlp_dim,
#                                    dropout=dropout, causal=causal, use_embedding=use_embedding,
#                                    weight_tying=weight_tying, mlp_bias=mlp_bias,
#                                    attention_bias=attention_bias,
#                                    pos_encoding=pos_encoding, pos_encoding_max_len=pos_encoding_max_len, device=device),
#                 PCTransformerLayer(emb_dim, n_heads, mlp_dim,
#                                    dropout=dropout, causal=causal, use_embedding=use_embedding,
#                                    weight_tying=weight_tying, mlp_bias=mlp_bias,
#                                    attention_bias=attention_bias,
#                                    pos_encoding=pos_encoding, pos_encoding_max_len=pos_encoding_max_len, device=device),
#                 f_energy_fn=F.mse_loss,
#                 b_energy_fn=F.mse_loss,
#                 device=device
#             ) for _ in range(n_layers)
#         ]+[
#             PCLayer(
#                     nn.Sequential(
#                         nn.LayerNorm(emb_dim),
#                         f_out_proj
#                     ),
#                     nn.Sequential(
#                         b_out_proj,
#                         nn.LayerNorm(emb_dim),
#                     ),
#                     f_energy_fn=F.cross_entropy,
#                     b_energy_fn=F.mse_loss,
#                     device=device
#                 )
#         ]
        
#         nn.init.xavier_uniform_(f_in_proj.weight)
#         nn.init.xavier_uniform_(b_in_proj.weight)
#         if weight_tying:
#             f_out_proj.weight = f_in_proj.weight
#             b_out_proj.weight = b_in_proj.weight
#         else:
#             nn.init.xavier_uniform_(f_out_proj.weight)
#             nn.init.xavier_uniform_(b_out_proj.weight)
        
#         super().__init__(
#             layers=layers,
#             max_its=3,
#             min_energy=1e-3,
#             energy_lr=1e-2,
#             energy_optimizer_class=torch.optim.SGD,
#             energy_grad_clip_norm=None,
#             iterative_train=False,
#             state_init_dir="forward", 
#             device=device,
#         )

# def pc_train_epoch(epoch, train_loader, model, optimizer, loss_fn, acc_fn, data_fn=default_data_fn,
#                 scheduler=None, device="cpu",
#                 output_dir="", model_name=None, val_loader=None,
#                 wandb_logging=True, wandb_metrics=["acc", "loss"],
#                 grad_clip_norm=None, accumulation_steps=0,
#                 mixed_precision=False,
#                 checkpoint_freq=None, val_freq=None, info_freq=None):
#     # Default model name
#     if model_name is None: model_name = model.__class__.__name__
#     model.train()
#     train_loss = 0
#     train_acc = 0
#     iterable = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False, bar_format='{desc}: [{n_fmt}/{total_fmt}] {percentage:.0f}%|{bar}| [{rate_fmt}] {postfix}')
#     scaler = GradScaler(device=device) if mixed_precision else None
#     optimizer.zero_grad()
#     for batch_idx, (data, target) in enumerate(iterable):
#         with autocast(device_type=device) if mixed_precision else nullcontext():
#             # Train pass
#             data = data.to(device)
#             target = target.to(device)
#             data, target = data_fn(data, target)
#             output = model.train_forward(data, target)
        
#         if (batch_idx + 1) % (accumulation_steps + 1) == 0:
#             if grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
#             optimizer.step() if not mixed_precision else scaler.step(optimizer)
#             optimizer.zero_grad()
#             if scheduler: scheduler.step()
#             if mixed_precision: scaler.update()
            
#             model.eval()
#             output = model(data)
#             model.train()
            
#             # Accuracy
#             with torch.no_grad():
#                 accuracy = acc_fn(output.clone(), target.clone())
#                 train_acc += accuracy
            
#             # Loss
#             loss = loss_fn(output, target)
#             train_loss += loss.item()
            
#             # WandB logging
#             if wandb_logging:
#                 log_data = {}
#                 if "acc" in wandb_metrics: log_data["train/acc"] = accuracy
#                 if "loss" in wandb_metrics: log_data["train/loss"] = loss.item()
#                 if "ppl" in wandb_metrics: log_data["train/ppl"] = math.exp(loss.item())
#                 if "lr" in wandb_metrics: log_data["misc/lr"] = scheduler.get_last_lr()[0]
#                 if "seq_len" in wandb_metrics: log_data["misc/seq_len"] = train_loader.dataset.len
#                 wandb.log(log_data)
        
#         # Post info
#         if info_freq and batch_idx % info_freq == 0 and batch_idx != 0:
#             tqdm.write(f'Train Epoch {epoch}: [{batch_idx}/{len(train_loader)}] LR: {scheduler.get_last_lr()[0]:.1e}, Loss: {loss.item():.4f}, Acc: {accuracy:.2f}%')
        
#         # Checkpoint
#         if checkpoint_freq and batch_idx % checkpoint_freq == 0 and batch_idx != 0:
#             checkpoint(model_name, output_dir, model, optimizer, scheduler)
        
#         # Validation
#         if val_freq and batch_idx % val_freq == 0 and batch_idx != 0:
#             if val_loader: val_epoch(model, val_loader, loss_fn, acc_fn, device=device, wandb_logging=wandb_logging, wandb_metrics=wandb_metrics)
#             model.train()
    
#     # Account for last accumulated batch
#     if (batch_idx + 1) % (accumulation_steps + 1) != 0:
#         if grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
#         optimizer.step() if not mixed_precision else scaler.step(optimizer)
#         optimizer.zero_grad()
#         if scheduler: scheduler.step()
#         if mixed_precision: scaler.update()
        
#         model.eval()
#         output = model(data)
#         model.train()
        
#         # Accuracy
#         with torch.no_grad():
#             accuracy = acc_fn(output.clone(), target.clone())
#             train_acc += accuracy
        
#         # Loss
#         loss = loss_fn(output, target)
#         train_loss += loss.item()
        
#         # WandB logging
#         if wandb_logging:
#             log_data = {}
#             if "acc" in wandb_metrics: log_data["train/acc"] = accuracy
#             if "loss" in wandb_metrics: log_data["train/loss"] = loss.item()
#             if "ppl" in wandb_metrics: log_data["train/ppl"] = math.exp(loss.item())
#             if "lr" in wandb_metrics: log_data["misc/lr"] = scheduler.get_last_lr()[0]
#             if "seq_len" in wandb_metrics: log_data["misc/seq_len"] = train_loader.dataset.len
#             wandb.log(log_data)
    
#     # Step sequence length if applicable
#     if hasattr(train_loader.dataset, "step"):
#         train_loader.dataset.step()
    
#     train_loss /= len(train_loader) / (accumulation_steps + 1)
#     train_acc /= len(train_loader) / (accumulation_steps + 1)
    
#     return train_loss, train_acc