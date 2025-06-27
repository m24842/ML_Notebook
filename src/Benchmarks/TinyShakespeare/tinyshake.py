import torch
import torch.nn.functional as F
from ml.utils import get_available_device
from ml.train import *

CONFIG_PATH = "src/Benchmarks/TinyShakespeare/configs.yaml"

device = get_available_device()

bitmaps = torch.load("data/char-8.pt", map_location=device)

def loss_fn(data, output, target):
    return F.mse_loss(output, target)
    output = (data - torch.sqrt(1 - alphas_cumprod) * output) / torch.sqrt(alphas_cumprod)
    return F.cross_entropy(output.transpose(1, 2), target, ignore_index=0)

def log_fn(loss, output, data, target):
    return []
    noise, target, alphas_cumprod = target
    output = (data - torch.sqrt(1 - alphas_cumprod) * output) / torch.sqrt(alphas_cumprod)
    acc = 100 * (output.argmax(dim=-1) == target).float().mean()
    ppl = torch.exp(loss)
    return [
        Metric(name="acc", value=acc, reset_value=0.0, batch_avg=True),
        Metric(name="ppl", value=ppl, reset_value=1.0, batch_avg=True),
    ]

# @torch.no_grad()
def data_fn(data, target, model, dataset):
    bsz, seq_len = data.shape[:2]
    # data_p = torch.zeros((bsz, seq_len, model.input_dim), device=device)
    
    # batch_idx = torch.arange(bsz).unsqueeze(1).expand(bsz, seq_len)
    # seq_len_idx = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len)
    # data_p[batch_idx, seq_len_idx, data] = 1.0
    
    data_p = bitmaps[data-2].flatten(-2)
    
    betas, alphas, alphas_cumprod = model.get_noise(data_p, profile_fn=lambda x: x, t_range=(0, 1), beta_range=(0.0001, 0.05))
    noise = torch.randn_like(data_p)
    data_p_noisy = torch.sqrt(alphas_cumprod) * data_p + torch.sqrt(1 - alphas_cumprod) * noise
    return data_p_noisy, noise

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, loss_fn, log_fn, data_fn=data_fn, device=device)