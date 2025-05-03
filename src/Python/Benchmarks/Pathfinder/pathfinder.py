import torch
import torch.nn.functional as F
from ml.train import train_from_config_file

CONFIG_PATH = "src/Python/Benchmarks/Pathfinder/configs.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def loss_fn(output, target):
    return F.cross_entropy(output[:, -1], target)

def acc_fn(output, target):
    pred = output[:, -1].argmax(dim=1)
    return (pred == target).sum().item()

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, loss_fn, acc_fn, device)