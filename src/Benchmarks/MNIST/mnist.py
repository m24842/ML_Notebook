import torch
import torch.nn.functional as F
from ml.utils import get_available_device
from ml.train import *

CONFIG_PATH = "src/Benchmarks/MNIST/configs.yaml"

device = get_available_device()

def loss_fn(output, target):
    return F.cross_entropy(output[:, -1], target)

def log_fn(loss, output, data, target):
    pred = output[:, -1].argmax(dim=1)
    acc = 100 * (pred == target).sum().item() / len(target)
    return [
        Metric(name="acc", value=acc, reset_value=0.0, batch_avg=True),
    ]

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, loss_fn, log_fn, device=device)