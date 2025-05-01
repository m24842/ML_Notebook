import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import argparse
import wandb
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.transformers import *
from models.utils import *

ENTITY = os.getenv("WANDB_ENTITY")
DATA_DIR = "data"
OUTPUT_DIR = "src/Python/Benchmarks/MNIST/models"
LOG_PATH = "src/Python/Benchmarks/MNIST/experiments.log"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train(model, data_loader, optimizer, criterion, scheduler, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc=f"Train Epoch {epoch}", leave=False, bar_format='{desc}: [{n_fmt}/{total_fmt}] {percentage:.0f}%|{bar}| [{rate_fmt}] {postfix}')):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)[:, -1]
        loss = criterion(output, target)
        total_loss += loss.item()
        accuracy = (output.argmax(dim=-1) == target).sum().item()
        correct += accuracy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        wandb.log({
            "train/acc": 100. * accuracy / len(data),
            "train/loss": loss,
            "misc/lr": scheduler.get_last_lr()[0],
        })
        if batch_idx % 100 == 0 and batch_idx != 0:
            tqdm.write(f'Train Epoch {epoch}: [{batch_idx}/{len(data_loader)}] LR: {scheduler.get_last_lr()[0]:.1e}, Loss: {loss.item():.4f}, Acc: {100. * accuracy / len(data):.0f}%')
            checkpoint(model_name, OUTPUT_DIR, model, optimizer, scheduler)
    return total_loss / len(data_loader), 100. * correct / len(data_loader.dataset)

@ torch.no_grad()
def test(model, data_loader, criterion):
    print()
    model.eval()
    test_loss = 0
    correct = 0
    start = time.time()
    for data, target in tqdm(data_loader, desc=f"Test Epoch", leave=False, bar_format='\033[92m{desc}: [{n_fmt}/{total_fmt}] {percentage:.0f}%|{bar}| [{rate_fmt}] {postfix}\033[0m'):
        data = data.to(device)
        target = target.to(device)
        output = model(data)[:, -1]
        test_loss += criterion(output, target).item()
        correct += output.argmax(dim=-1).eq(target).sum().item()

    total_time = time.time() - start
    test_loss /= len(data_loader)
    wandb.log({
        "test/acc": 100 * correct / len(data_loader.dataset),
        "test/loss": test_loss,
    })
    print(f'\033[92mTest Epoch: Loss: {test_loss:.4f}, Acc: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.0f}%), Elapsed: {total_time:.3f}s\033[0m\n')
    return test_loss, 100 * correct / len(data_loader.dataset)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permuted", type=bool, default=False)
    parser.add_argument("--img_dim", type=int, default=28)
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--mem_dim", type=int, default=64)
    parser.add_argument("--causal", type=bool, default=False)
    parser.add_argument("--vocab_size", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--total_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    return parser.parse_args()

if __name__ == "__main__":
    try:
        print("\033[?25l\033c", end="", flush=True)
        
        args = arg_parse()
        
        torch.manual_seed(args.seed)
        img_dim = args.img_dim
        bsz = args.bsz
        emb_dim = args.emb_dim
        n_classes = args.n_classes
        n_layers = args.n_layers
        n_heads = args.n_heads
        mlp_dim = args.mlp_dim
        mem_dim = args.mem_dim
        causal = args.causal
        vocab_size = args.vocab_size
        dropout = args.dropout
                
        # Load dataset
        T = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * max(1, vocab_size-1)),
            transforms.Lambda(lambda x: x.view(-1, 1))
        ]
        if args.permuted:
            random_permutation = torch.randperm(img_dim**2)
            T.append(transforms.Lambda(lambda x: x.view(3, -1)[:, random_permutation].permute(0, 2, 1).view(-1, 1)))
        transform = transforms.Compose(T)
        train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
        # train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        # test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        # train_dataset = datasets.EMNIST(root=DATA_DIR, train=True, download=True, transform=transform, split='byclass')
        # test_dataset = datasets.EMNIST(root=DATA_DIR, train=False, download=True, transform=transform, split='byclass')

        train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=False)

        # model = Transformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, vocab_size, dropout, causal, use_embedding=False, device=device)
        model = LinearTransformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, vocab_size, dropout, causal, use_embedding=False, device=device)
        # model = OrthoLinearTransformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, vocab_size, dropout, causal, use_embedding=False, device=device)
        # model = CompressionTransformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, mem_dim, vocab_size, dropout, causal, use_embedding=False, device=device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(apply_weight_decay(model, args.weight_decay), lr=args.lr, weight_decay=args.weight_decay)
        warmup_steps = args.warmup_epochs * len(train_loader)
        total_steps = args.total_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        model_name = model.__class__.__name__
        model, optimizer, scheduler = load_checkpoint(model_name, OUTPUT_DIR, model, optimizer, scheduler, device=device)
        
        if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
        
        benchmark_name = train_dataset.__class__.__name__
        args = vars(args)
        args["benchmark"] = benchmark_name
        args = argparse.Namespace(**args)
        print(f'\033[1m{benchmark_name} Benchmark\033[0m')
        print(f'\033[1m{model_name}\033[0m')
        print(f'\033[4mTotal params: {count_parameters(model):,}\033[0m\n')
        
        wandb.init(
            settings=wandb.Settings(silent=True),
            entity=ENTITY,
            project="Machine Learning",
            name=f"{model_name}",
            config=args,
        )
        
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
            
            checkpoint(model_name, OUTPUT_DIR, model, optimizer, scheduler)
        
        log_info(LOG_PATH, model, model_name, args, train_accuracies, test_accuracies)
        wandb.finish()
        
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
        if not wandb.run._is_finished: wandb.Api().run(f'{ENTITY}/Machine Learning/{wandb.run.id}').delete()
        print("\033[?25h", end='', flush=True)