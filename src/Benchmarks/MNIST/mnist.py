import torch
import torch.nn.functional as F
from ml.utils import get_available_device
from ml.train import train_from_config_file

CONFIG_PATH = "src/Benchmarks/MNIST/configs.yaml"

device = get_available_device()

def data_fn(data, target):
    target = F.one_hot(target, num_classes=10).float().unsqueeze(1).repeat(1, data.shape[1], 1)
    return data, target

def loss_fn(output, target):
    # return F.cross_entropy(output[:, -1], target)
    return F.cross_entropy(output[:, -1], target[:, -1])

def acc_fn(output, target):
    # pred = output[:, -1].argmax(dim=1)
    # return 100 * (pred == target).sum().item() / len(target)
    pred = output[:, -1].argmax(dim=1)
    return 100 * (pred == target[:, -1].argmax(dim=1)).sum().item() / len(target)

if __name__ == "__main__":
    train_from_config_file(CONFIG_PATH, loss_fn, acc_fn, data_fn, device)