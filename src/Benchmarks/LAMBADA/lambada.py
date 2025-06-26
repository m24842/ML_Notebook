import torch
import torch.nn.functional as F
from ml.utils import get_available_device
from ml.train import *

CONFIG_PATH = "src/Benchmarks/LAMBADA/configs.yaml"

device = get_available_device()

def loss_fn(data, output, target):
    return F.cross_entropy(output.transpose(1, 2), target, ignore_index=-100)

def acc_fn(output, target):
    correct = (output.argmax(dim=-1) == target)[target != -100]
    return 100 * correct.sum().item() / len(correct)

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, loss_fn, acc_fn, device=device)