import torch
import torch.nn.functional as F
from ml.utils import get_available_device
from ml.train import *

CONFIG_PATH = "src/Benchmarks/WikiText/configs.yaml"

device = get_available_device()

def loss_fn(data, output, target):
    return F.cross_entropy(output.transpose(1, 2), target, ignore_index=-100)

def log_fn(loss, output, data, target):
    mask = target != -100
    acc = 100 * ((output.argmax(dim=-1) == target) & mask).sum().item() / mask.sum().item()
    ppl = torch.exp(loss)
    return [
        Metric(name="acc", value=acc, reset_value=0.0, batch_avg=True),
        Metric(name="ppl", value=ppl, reset_value=1.0, batch_avg=True),
    ]

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, loss_fn, log_fn, device=device)