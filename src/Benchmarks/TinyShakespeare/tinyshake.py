import torch
import torch.nn.functional as F
from ml.utils import get_available_device
from ml.train import *

CONFIG_PATH = "src/Benchmarks/TinyShakespeare/configs.yaml"

device = get_available_device()

bitmaps = torch.load("data/char-8.pt", map_location=device)

def loss_fn(data, output, target):
    return F.cross_entropy(output.transpose(1, 2), target, ignore_index=0)

def log_fn(loss, output, data, target):
    acc = 100 * (output.argmax(dim=-1) == target).float().mean()
    ppl = torch.exp(loss)
    return [
        Metric(name="acc", value=acc, reset_value=0.0, batch_avg=True),
        Metric(name="ppl", value=ppl, reset_value=1.0, batch_avg=True),
    ]

@torch.no_grad()
def data_fn(data, target, model, dataset):
    bsz, seq_len = data.shape[:2]
    
    data_p = bitmaps[data-2].flatten(-2)
    
    betas, alphas, alphas_cumprod = model.get_noise(data_p, profile_fn=torch.sigmoid, t_range=(-16, 4), beta_range=(0.0001, 0.05))
    noise = torch.randn_like(data_p)
    data_p_noisy = torch.sqrt(alphas_cumprod) * data_p + torch.sqrt(1 - alphas_cumprod) * noise
    return data_p_noisy, noise

def diffusion_log_fn(loss, output, data, target):
    return []

def diffusion_loss_fn(data, output, target):
    return F.mse_loss(output, target)

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, [loss_fn, diffusion_loss_fn], [log_fn, diffusion_log_fn], data_fn=data_fn, device=device)