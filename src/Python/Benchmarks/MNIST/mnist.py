import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from argparse import ArgumentParser
import os
import time
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.transformers import *

DATA_DIR = "data"
OUTPUT_DIR = "src/Python/Benchmarks/MNIST/mnist_models"
LOG_PATH = "src/Python/Benchmarks/MNIST/experiments.log"
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format='%(asctime)s - %(message)s')

device = torch.device("mps")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, data_loader, optimizer, criterion, scheduler, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc=f"Train Epoch {epoch}", leave=False, bar_format='{desc}: [{n_fmt}/{total_fmt}] {percentage:.0f}%|{bar}| [{rate_fmt}] {postfix}')):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        accuracy = (output.argmax(dim=-1) == target).sum().item()
        correct = accuracy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if batch_idx % 100 == 0 and batch_idx != 0:
            tqdm.write(f'Train Epoch {epoch}: [{batch_idx}/{len(data_loader)}] LR: {scheduler.get_last_lr()[0]:.1e}, Loss: {loss.item():.4f}, Acc: {100. * accuracy / len(data):.0f}%')
            checkpoint(model, optimizer, scheduler)
    return total_loss / len(data_loader), 100 * correct / len(data_loader.dataset)

@ torch.no_grad()
def test(model, data_loader, criterion):
    print()
    model.eval()
    test_loss = 0
    correct = 0
    start = time.time()
    for data, target in tqdm(data_loader, desc=f"Test Epoch", leave=False, bar_format='\033[92m{desc}: [{n_fmt}/{total_fmt}] {percentage:.0f}%|{bar}| [{rate_fmt}] {postfix}\033[0m'):
        data = data.to(device).squeeze(1)
        target = target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    total_time = time.time() - start
    test_loss /= len(data_loader)
    print(f'\033[92mTest Epoch: Loss: {test_loss:.4f}, Acc: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.0f}%), Elapsed: {total_time:.3f}s\033[0m\n')
    return test_loss, 100 * correct / len(data_loader.dataset)

def log_info(model_name, args, train_accuracies, test_accuracies):
    logging.info(model_name)
    logging.info(f"Total params: {count_parameters(model):,}")
    logging.info('Hyperparams:\n' + '\n'.join([f'\t{key}: {value}' for key, value in vars(args).items()]))
    logging.info("Train accuracies: " + ', '.join(str(round(acc, 2)) for acc in train_accuracies))
    logging.info("Test accuracies: " + ', '.join(str(round(acc, 2)) for acc in test_accuracies))

def checkpoint(model, optimizer, scheduler):
    model_name = model.__class__.__name__
    model_dir = f'{OUTPUT_DIR}/{model_name}'
    model_path = f'{model_dir}/{model_name}.pt'
    optimizer_path = f'{model_dir}/{model_name}_opt.pt'
    scheduler_path = f'{model_dir}/{model_name}_sch.pt'
    
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(scheduler.state_dict(), scheduler_path)

def arg_parse():
    parser = ArgumentParser()
    # parser.add_argument("--model", type=str, default="Transformer")
    parser.add_argument("--seed", type=int, default=3333)
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--mlp_dim", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--total_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    return parser.parse_args()

if __name__ == "__main__":
    try:
        print("\033[?25l\033c", end="", flush=True)
        
        args = arg_parse()
        
        torch.manual_seed(args.seed)
        bsz = args.bsz
        emb_dim = args.emb_dim
        n_classes = args.n_classes
        n_layers = args.n_layers
        n_heads = emb_dim//8 if args.n_heads is None else args.n_heads
        mlp_dim = 2*emb_dim if args.mlp_dim is None else args.mlp_dim
        vocab_size = args.vocab_size
        dropout = args.dropout
        
        dim1 = 28
        dim2 = 28
        
        torch.manual_seed(3333)
        random_permutation = torch.randperm(dim1*dim2).reshape(dim1, dim2)
        
        # Load dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((dim1, dim2)),
            # transforms.Lambda(lambda x: x.view(-1)[random_permutation].view(dim1, dim2)),
        ])
        train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
        # train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        # test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=False)

        # model = Transformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, vocab_size, dropout).to(device)
        # model = LinearTransformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, vocab_size, dropout).to(device)
        model = OrthoLinearTransformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, vocab_size, dropout).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        warmup_steps = args.warmup_epochs * len(train_loader)
        total_steps = args.total_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        model_name = model.__class__.__name__
        model_dir = f'{OUTPUT_DIR}/{model_name}'
        model_path = f'{model_dir}/{model_name}.pt'
        optimizer_path = f'{model_dir}/{model_name}_opt.pt'
        scheduler_path = f'{model_dir}/{model_name}_sch.pt'
        
        if not os.path.exists(model_dir): os.makedirs(model_dir)

        try:
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True, map_location=device))
            scheduler.load_state_dict(torch.load(scheduler_path, weights_only=True, map_location=device))
        except:
            pass
        
        print(f'\033[1m{model_name}\033[0m')
        print(f'\033[4mTotal params: {count_parameters(model):,}\033[0m\n')
        
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        for epoch in range(1, args.total_epochs + 1):
            train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, scheduler, epoch)
            test_loss, test_accuracy = test(model, test_loader, criterion)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            
            checkpoint(model, optimizer, scheduler)
            
        log_info(model_name, args, train_accuracies, test_accuracies)
        
        plt.figure()

        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.ylim(0, None)
        plt.title('Losses')
        plt.legend()

        # Plot accuracies
        plt.subplot(2, 1, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.ylim(0, 100)
        plt.title('Accuracies')
        plt.legend()

        plt.tight_layout()
        plt.show()
    finally:
        print("\033[?25h", end='', flush=True)