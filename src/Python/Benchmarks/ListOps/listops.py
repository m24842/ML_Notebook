import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from argparse import ArgumentParser
import time
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.transformers import *
from models.utils import *

DATA_DIR = "data/listops-1000"
OUTPUT_DIR = "src/Python/Benchmarks/ListOps/listops_models"
LOG_PATH = "src/Python/Benchmarks/ListOps/experiments.log"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def tokenize_listops(data):
    token_map = {
        "[CLS]": 0,
        "[PAD]": 1,
        "[MAX": 2,
        "[MIN": 3,
        "[MED": 4,
        "[SUM_MOD": 5,
        "[SM": 6,
        "[AVG": 7,
        "[MAJ": 8,
        "]": 9,
        "(": 10,
        ")": 11,
        **{str(i): i + 12 for i in range(10)}
    }

    tokens = ["[CLS]"] + data["Source"].split()
    try:
        tokenized = [token_map[token] for token in tokens]
    except KeyError as e:
        raise ValueError(f"Unexpected token: {e.args[0]}")
    
    tokenized = torch.tensor(tokenized, dtype=torch.long)
    target = torch.tensor(data["Target"], dtype=torch.long)
    return tokenized, target

class ListOpsDataset(Dataset):
    def __init__(self, data, tokenizer, min_len=1, max_len=1000, warmup_epochs=0):
        if warmup_epochs < 1:
            self.min_len = max_len
        else:
            self.min_len = min_len
        self.max_len = max_len
        self.len = self.min_len
        self.step_size = (self.max_len - self.min_len) // (warmup_epochs + 1)
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        tokenized, target = self.tokenizer(item)
        padded_tokenized = torch.nn.functional.pad(tokenized, (0, self.len - tokenized.size(0)), value=1)  # Pad with [PAD] token (1)
        return padded_tokenized, target

    def step(self):
        if self.len + self.step_size <= self.max_len:
            self.len += self.step_size
        else:
            self.len = self.max_len

def train(model, data_loader, optimizer, criterion, scheduler, epoch):
    model.train()
    total_loss = 0
    correct = 0
    iterable = tqdm(data_loader, desc=f"Train Epoch {epoch}", leave=False, bar_format='{desc}: [{n_fmt}/{total_fmt}] {percentage:.0f}%|{bar}| [{rate_fmt}] {postfix}')
    for batch_idx, (data, target) in enumerate(iterable):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)[:, 0]
        loss = criterion(output, target)
        total_loss += loss.item()
        accuracy = (output.argmax(dim=-1) == target).sum().item()
        correct += accuracy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if batch_idx % 100 == 0 and batch_idx != 0:
            tqdm.write(f'Train Epoch {epoch}: [{batch_idx}/{len(data_loader)}] LR: {scheduler.get_last_lr()[0]:.1e}, Loss: {loss.item():.4f}, Acc: {100. * accuracy / len(data):.0f}%')
        if batch_idx % 500 == 0 and batch_idx != 0:
            checkpoint(model_name, OUTPUT_DIR, model, optimizer, scheduler)
            test(model, val_loader, criterion, is_val=True)
            model.train()
    train_set.step()
    return total_loss / len(data_loader), 100 * correct / len(data_loader.dataset)

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
        output = model(data)[:, 0]
        test_loss += criterion(output, target).item()
        correct += output.argmax(dim=-1).eq(target).sum().item()

    total_time = time.time() - start
    test_loss /= len(data_loader)
    
    if not is_val: tqdm.write(f'\033[92mTest Epoch: Loss: {test_loss:.4f}, Acc: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.0f}%), Elapsed: {total_time:.3f}s\033[0m\n')
    else: tqdm.write(f'\033[93mVal Epoch: Loss: {test_loss:.4f}, Acc: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.0f}%)\033[0m')
    
    return test_loss, 100 * correct / len(data_loader.dataset)

def arg_parse():
    parser = ArgumentParser()
    # parser.add_argument("--model", type=str, default="Transformer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--mem_dim", type=int, default=256)
    parser.add_argument("--min_len", type=int, default=1024)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--causal", type=bool, default=False)
    parser.add_argument("--vocab_size", type=int, default=22)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--total_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
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
        n_heads = args.n_heads
        mlp_dim = args.mlp_dim
        mem_dim = args.mem_dim
        min_len = args.min_len
        max_len = args.max_len
        causal = args.causal
        vocab_size = args.vocab_size
        dropout = args.dropout
        
        train_df = pd.read_csv(f"{DATA_DIR}/basic_train.tsv", sep="\t")
        val_df = pd.read_csv(f"{DATA_DIR}/basic_val.tsv", sep="\t")
        test_df = pd.read_csv(f"{DATA_DIR}/basic_test.tsv", sep="\t")
        
        train_set = ListOpsDataset(train_df, tokenize_listops, min_len, max_len, warmup_epochs=args.warmup_epochs)
        val_set = ListOpsDataset(val_df, tokenize_listops, min_len, max_len)
        test_set = ListOpsDataset(test_df, tokenize_listops, min_len, max_len)
        
        train_loader = DataLoader(train_set, batch_size=bsz, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=bsz, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=bsz, shuffle=False)
        
        # model = Transformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, vocab_size, dropout, causal, device=device)
        model = LinearTransformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, vocab_size, dropout, causal, device=device)
        # model = OrthoLinearTransformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, vocab_size, dropout, causal, device=device)
        # model = CompressionTransformer(emb_dim, n_classes, n_layers, n_heads, mlp_dim, mem_dim, vocab_size, dropout, causal, device=device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(apply_weight_decay(model, args.weight_decay), lr=args.lr, weight_decay=args.weight_decay)
        warmup_steps = args.warmup_epochs * len(train_loader)
        total_steps = args.total_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        model_name = model.__class__.__name__
        model, optimizer, scheduler = load_checkpoint(model_name, OUTPUT_DIR, model, optimizer, scheduler, device=device)
        
        model = allocate_dynamic_memory(model, bsz, min_len, max_len, device=device)
        
        print('\033[1mListOps Benchmark\033[0m')
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
                        
            checkpoint(model_name, OUTPUT_DIR, model, optimizer, scheduler)
        
        log_info(LOG_PATH, "ListOps", model, model_name, args, train_accuracies, test_accuracies)
        
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
        print("\033[?25h", end="", flush=True)