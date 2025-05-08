import torch
import torch.nn.functional as F
from ml.utils import get_available_device
from ml.train import train_from_config_file

CONFIG_PATH = "src/Benchmarks/TinyShakespeare/configs.yaml"

device = get_available_device()

def loss_fn(output, target):
    return F.cross_entropy(output.transpose(1, 2), target, ignore_index=0)

def acc_fn(output, target):
    return (output.argmax(dim=-1) == target).sum().item()

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, loss_fn, acc_fn, device)