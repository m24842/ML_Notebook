import torch
import torch.nn.functional as F
from ml.train import train_from_config_file

CONFIG_PATH = "src/Python/Benchmarks/TinyShakespeare/configs.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def loss_fn(output, target):
    return F.cross_entropy(output.transpose(1, 2), target)

def acc_fn(output, target):
    return (output.argmax(dim=-1) == target).sum().item()

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, loss_fn, acc_fn, device)