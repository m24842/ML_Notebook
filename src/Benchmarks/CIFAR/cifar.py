import torch
import torch.nn.functional as F
from ml.utils import get_available_device
from ml.train import *

CONFIG_PATH = "src/Benchmarks/CIFAR/configs.yaml"

device = get_available_device()

def loss_fn(data, output, target):
    return F.cross_entropy(output[:, -1], target)

def log_fn(loss, output, data, target):
    pred = output[:, -1].argmax(dim=1)
    return 100 * (pred == target).sum().item() / len(target)

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, loss_fn, acc_fn, device=device)