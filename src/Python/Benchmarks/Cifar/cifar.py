import torch
import torch.nn.functional as F
from ml.utils import get_available_device
from ml.train import train_from_config_file

CONFIG_PATH = "src/Python/Benchmarks/Cifar/configs.yaml"

device = get_available_device()

def loss_fn(output, target):
    return F.cross_entropy(output[:, -1], target)

def acc_fn(output, target):
    pred = output[:, -1].argmax(dim=1)
    return (pred == target).sum().item()

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, loss_fn, acc_fn, device)