import torch
import torch.nn.functional as F
from ml.utils import get_available_device
from ml.train import *

CONFIG_PATH = "src/Benchmarks/IMDb/configs.yaml"

device = get_available_device()

def loss_fn(data, output, target):
    return F.cross_entropy(output[:, 0], target)

def log_fn(loss, output, data, target):
    pred = output[:, 0].argmax(dim=1)
    return 100 * (pred == target).sum().item() / len(target)

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, loss_fn, log_fn, device=device)