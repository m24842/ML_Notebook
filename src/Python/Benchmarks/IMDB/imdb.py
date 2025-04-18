import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from argparse import ArgumentParser
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import *

OUTPUT_DIR = "src/Python/Benchmarks/IMDB/imdb_models"

device = torch.device("mps")

class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=None):
        self.max_len = 1000 if max_len is None else max_len
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized = torch.tensor(tokenizer(item['text'])['input_ids'], dtype=torch.long)
        target = torch.tensor(item['label'], dtype=torch.long)
        padded_tokenized = torch.nn.functional.pad(tokenized, (0, self.max_len - tokenized.size(0)), value=0)
        return padded_tokenized, target

# Get the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training loop
def train(model, data_loader, optimizer, criterion, scheduler, epoch):
    model.train()
    total_loss = 0
    iterable = tqdm(data_loader, desc=f"Train Epoch {epoch}", leave=False, bar_format='{desc}: [{n_fmt}/{total_fmt}] {percentage:.0f}%|{bar}| [{rate_fmt}] {postfix}')
    for batch_idx, (data, target) in enumerate(iterable):
        data = data.to(device).squeeze(1)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        accuracy = (output.argmax(dim=-1) == target).sum().item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if batch_idx % 500 == 0 and batch_idx != 0:
            test(model, val_loader, criterion, is_val=True)
            model.train()
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            torch.save(scheduler.state_dict(), scheduler_path)
        if batch_idx % 100 == 0 and batch_idx != 0:
            tqdm.write(f'Train Epoch {epoch}: [{batch_idx}/{len(data_loader)}] LR: {scheduler.get_last_lr()[0]:.1e}, Loss: {loss.item():.4f}, Acc: {100. * accuracy / len(data):.0f}%')
    return total_loss / len(data_loader)

# Testing loop
@ torch.no_grad()
def test(model, data_loader, criterion, is_val=False):
    model.eval()
    test_loss = 0
    correct = 0
    start = time.time()
    if not is_val:
        tqdm.write("")
        iterable = tqdm(data_loader, desc=f"Test Epoch", leave=False, bar_format='\033[92m{desc}: [{n_fmt}/{total_fmt}] {percentage:.0f}%|{bar}| [{rate_fmt}] {postfix}\033[0m')
    else:
        iterable = data_loader
    for data, target in iterable:
        data = data.to(device).squeeze(1)
        target = target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    total_time = time.time() - start
    test_loss /= len(data_loader)
    
    if not is_val: tqdm.write(f'\033[92mTest Epoch: Loss: {test_loss:.4f}, Acc: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.0f}%), Elapsed: {total_time:.3f}s\033[0m\n')
    else: tqdm.write(f'\033[93mVal Epoch: Loss: {test_loss:.4f}, Acc: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.0f}%)\033[0m')
    
    return test_loss, 100 * correct / len(data_loader.dataset)

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Transformer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--mlp_dim", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--causal", type=bool, default=False)
    parser.add_argument("--vocab_size", type=int, default=256000)
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
        max_len = args.max_len
        causal = args.causal
        vocab_size = args.vocab_size
        dropout = args.dropout
        
        imbd = load_dataset('imdb')
        train_data, val_data = imbd['train'].train_test_split(test_size=0.1, shuffle=True, seed=args.seed).values()
        test_data = imbd['test']
        
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        train_set = IMDBDataset(train_data, tokenizer, max_len=max_len)
        val_set = IMDBDataset(val_data, tokenizer, max_len=max_len)
        test_set = IMDBDataset(test_data, tokenizer, max_len=max_len)
        
        train_loader = DataLoader(train_set, batch_size=bsz, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=bsz, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=bsz, shuffle=False)
        
        # model = Transformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, vocab_size, dropout, causal)
        # model = LinearTransformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, vocab_size, dropout, causal)
        model = OrthoLinearTransformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, vocab_size, dropout, causal)
        
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        warmup_steps = args.warmup_epochs * len(train_loader)
        total_steps = args.total_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        model_dir = f'{OUTPUT_DIR}/{model.__class__.__name__}'
        model_path = f'{model_dir}/{model.__class__.__name__}.pt'
        optimizer_path = f'{model_dir}/{model.__class__.__name__}_opt.pt'
        scheduler_path = f'{model_dir}/{model.__class__.__name__}_sch.pt'
        
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        
        try:
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True, map_location=device))
            scheduler.load_state_dict(torch.load(scheduler_path, weights_only=True, map_location=device))
        except:
            pass
        
        print(f'\033[1m{model.__class__.__name__}\033[0m')
        print(f'\033[4mTotal params: {count_parameters(model):,}\033[0m\n')
        
        train_losses = []
        test_losses = []
        test_accuracies = []
        for epoch in range(1, args.total_epochs + 1):
            train_loss = train(model, train_loader, optimizer, criterion, scheduler, epoch)
            test_loss, test_accuracy = test(model, test_loader, criterion)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            torch.save(scheduler.state_dict(), scheduler_path)
        
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
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.ylim(0, 100)
        plt.title('Accuracies')
        plt.legend()

        plt.tight_layout()
        plt.show()
    finally:
        print("\033[?25h", end="", flush=True)