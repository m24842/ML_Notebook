import math
import torch
import torch.nn.functional as F
from ml.utils import get_available_device
from ml.train import *

CONFIG_PATH = "src/Benchmarks/TinyShakespeare/configs.yaml"

device = get_available_device()

def loss_fn(output, target):
    # return F.mse_loss(output, target)
    # return F.cross_entropy(output.transpose(1, 2), target.softmax(-1).transpose(1, 2))
    return F.cross_entropy(output.transpose(1, 2), target, ignore_index=0)

def log_fn(loss, output, data, target):
    # acc = 100 * (output.argmax(dim=-1) == target).sum().item() / target.numel()
    ppl = torch.exp(loss)
    return [
        # Metric(name="acc", value=acc, reset_value=0.0, batch_avg=True),
        Metric(name="ppl", value=ppl, reset_value=1.0, batch_avg=True),
    ]

def data_fn(data, target, model, dataset):
    bsz, seq_len = data.shape[:2]
    data_p = torch.zeros((bsz, seq_len, model.input_dim), device=device)
    target_p = torch.zeros((bsz, seq_len, model.input_dim), device=device)
    
    batch_idx = torch.arange(bsz).unsqueeze(1).expand(bsz, seq_len)
    seq_len_idx = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len)
    data_p[batch_idx, seq_len_idx, data] = 1.0
    target_p[batch_idx, seq_len_idx, target] = 1.0
    
    noise = model.get_noise(data_p, profile_fn=torch.sigmoid, scale=8/seq_len, offset=4.0)
    data_p = data_p + noise
    target_p = target_p + noise
    return data_p, target_p

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, loss_fn, log_fn, data_fn=data_fn, device=device)