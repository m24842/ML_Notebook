import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import random
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer
from datasets import load_dataset

class SequentialMNIST(datasets.MNIST):
    def __init__(self, root, train, download=True, permuted=False):
        self.transform = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1, 1))
        ]
        if permuted:
            random_permutation = torch.randperm(28 * 28)
            self.transform.append(transforms.Lambda(lambda x: x.view(-1)[random_permutation].view(-1, 1)))
        self.transform = transforms.Compose(self.transform)
        super().__init__(root, train=train, download=download, transform=self.transform)

class SequentialEMNIST(datasets.EMNIST):
    def __init__(self, root, train, split, download=True, permuted=False):
        self.transform = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1, 1))
        ]
        if permuted:
            random_permutation = torch.randperm(28 * 28)
            self.transform.append(transforms.Lambda(lambda x: x.view(-1)[random_permutation].view(-1, 1)))
        self.transform = transforms.Compose(self.transform)
        super().__init__(root, train=train, download=download, split=split, transform=self.transform)

class SequentialFashionMNIST(datasets.FashionMNIST):
    def __init__(self, root, train, download=True, permuted=False):
        self.transform = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1, 1))
        ]
        if permuted:
            random_permutation = torch.randperm(28 * 28)
            self.transform.append(transforms.Lambda(lambda x: x.view(-1)[random_permutation].view(-1, 1)))
        self.transform = transforms.Compose(self.transform)
        super().__init__(root, train=train, download=download, transform=self.transform)

class SequentialCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train, download=True, permuted=False):
        self.transform = [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1, 1))
        ]
        if permuted:
            random_permutation = torch.randperm(32 * 32)
            self.transform.append(transforms.Lambda(lambda x: x.view(-1)[random_permutation].view(-1, 1)))
        self.transform = transforms.Compose(self.transform)
        super().__init__(root, train=train, download=download, transform=self.transform)

class SequentialCIFAR100(datasets.CIFAR100):
    def __init__(self, root, train, download=True, permuted=False):
        self.transform = [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1, 1))
        ]
        if permuted:
            random_permutation = torch.randperm(32 * 32)
            self.transform.append(transforms.Lambda(lambda x: x.view(-1)[random_permutation].view(-1, 1)))
        self.transform = transforms.Compose(self.transform)
        super().__init__(root, train=train, download=download, transform=self.transform)

class Pathfinder(Dataset):
    def __init__(self, root, dim, train, subset="curv_baseline", split_idx=180, permuted=False):
        self.transform = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1, 1))
        ]
        if permuted:
            random_permutation = torch.randperm(dim**2)
            self.transform.append(transforms.Lambda(lambda x: x.view(-1)[random_permutation].view(-1, 1)))
        self.transform = transforms.Compose(self.transform)
        self.data = []
        
        min_idx = 0 if train else split_idx
        max_idx = split_idx if train else 200
        dataset_root = os.path.join(root, f"pathfinder{dim}", subset)
        metadata_root = os.path.join(root, f"pathfinder{dim}", subset, 'metadata')
        for i in range(min_idx, max_idx):
            with open(os.path.join(metadata_root, f"{i}.npy"), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    img_rel_path = parts[0:2]
                    label = int(parts[3])
                    img_path = os.path.join(dataset_root, *img_rel_path)
                    self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = None
        while image is None:
            img_path, label = self.data[idx]
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                idx = (idx + 1) % len(self.data)
        if self.transform:
            image = self.transform(image)
        return image, label

class ListOps(Dataset):
    def __init__(self, root, split, min_len=1, max_len=1000, warmup_epochs=0, balance=False):
        """
        splits: ["train", "val", "test"]
        """
        if warmup_epochs < 1:
            self.min_len = max_len
        else:
            self.min_len = min_len
        self.max_len = max_len
        self.len = self.min_len
        self.step_size = (self.max_len - self.min_len) // (warmup_epochs + 1)
        self.data = pd.read_csv(f"{root}/listops/basic_{split}.tsv", sep="\t")
        
        if balance: self._balance_data()
    
    def _balance_data(self):
        grouped = defaultdict(list)
        for idx, item in self.data.iterrows():
            grouped[item["Target"]].append(item)

        min_class_size = min(len(items) for items in grouped.values())

        balanced_items = []
        for items in grouped.values():
            balanced_items.extend(items[:min_class_size])

        self.data = pd.DataFrame(balanced_items)
        
    def tokenizer(self, data):
        token_map = {
            "CLS": 0,
            "PAD": 1,
            "[MAX": 2,
            "[MIN": 3,
            "[MED": 4,
            "[SM": 5,
            "]": 6,
            **{str(i): i + 7 for i in range(10)}
        }

        src = data["Source"].translate({ ord("("): None, ord(")"): None })
        tokens = ["CLS"] + src.split()
        try:
            tokenized = [token_map[token] for token in tokens]
        except KeyError as e:
            raise ValueError(f"Unexpected token: {e.args[0]}")
        
        tokenized = torch.tensor(tokenized, dtype=torch.long)
        target = torch.tensor(data["Target"], dtype=torch.long)
        return tokenized, target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        tokenized, target = self.tokenizer(item)
        padded_tokenized = torch.nn.functional.pad(tokenized, (0, self.len - tokenized.size(0)), value=1)  # Pad with PAD token (1)
        return padded_tokenized, target

    def step(self):
        if self.len + self.step_size <= self.max_len:
            self.len += self.step_size
        else:
            self.len = self.max_len
    
    def seq_len_range(self):
        return self.min_len, self.max_len

class IMDb(Dataset):
    def __init__(self, train, min_len=1, max_len=1000, warmup_epochs=0):
        if warmup_epochs < 1:
            self.min_len = max_len
        else:
            self.min_len = min_len
        self.max_len = max_len
        self.len = self.min_len
        self.step_size = (self.max_len - self.min_len) // (warmup_epochs + 1)
        self.data = load_dataset('imdb')['train' if train else 'test']
        
    def tokenizer(self, text):
        """
        Tokenizes the input text for IMDb dataset.
        0: CLS
        1: PAD
        """
        return [0] + [2 + ord(c) for c in text]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized = torch.tensor(self.tokenizer(item['text']), dtype=torch.long)
        target = torch.tensor(item['label'], dtype=torch.long)
        padded_tokenized = torch.nn.functional.pad(tokenized, (0, self.len - tokenized.size(0)), value=1)  # Pad with PAD token (1)
        return padded_tokenized, target
    
    def step(self):
        if self.len + self.step_size <= self.max_len:
            self.len += self.step_size
        else:
            self.len = self.max_len
    
    def seq_len_range(self):
        return self.min_len, self.max_len

class TinyShakespeare(Dataset):
    def __init__(self, train, tokenizer, min_len=1, max_len=1000, warmup_epochs=0):
        if warmup_epochs < 1:
            self.min_len = max_len
        else:
            self.min_len = min_len
        self.max_len = max_len
        self.len = self.min_len
        self.step_size = (self.max_len - self.min_len) // (warmup_epochs + 1)
        self.data = load_dataset('tiny_shakespeare')['train' if train else 'test']['text'][0].split(' ')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    def __len__(self):
        return len(self.data) // 8
    
    def __getitem__(self, idx):
        start_idx = torch.randint(0, len(self.data) - self.len - 1, (1,)).item()
        end_idx = start_idx + self.len
        item = ' '.join(self.data[start_idx:end_idx])
        tokenized = torch.tensor(self.tokenizer(item)['input_ids'], dtype=torch.long)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        padded_tokenized = torch.nn.functional.pad(tokenized, (0, self.len - tokenized.size(0)), value=pad_token_id)
        return padded_tokenized[:-1], padded_tokenized[1:]
    
    def step(self):
        if self.len + self.step_size <= self.max_len:
            self.len += self.step_size
        else:
            self.len = self.max_len
    
    def seq_len_range(self):
        return self.min_len, self.max_len

class LAMBADA(Dataset):
    def __init__(self, split, tokenizer, min_len=1, max_len=1000, warmup_epochs=0):
        """
        splits: ["train", "validation", "test"]
        """
        if warmup_epochs < 1:
            self.min_len = max_len
        else:
            self.min_len = min_len
        self.max_len = max_len
        self.len = self.min_len
        self.step_size = (self.max_len - self.min_len) // (warmup_epochs + 1)
        self.data = load_dataset('lambada', split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # ignore token id = -100
        item = self.data[idx]['text'].strip().split()
        context = self.tokenizer(' '.join(item[:-1]), add_special_tokens=False)['input_ids']
        label = self.tokenizer(" " + item[-1], add_special_tokens=False)['input_ids']
        full_context = torch.tensor(context + label[:-1], dtype=torch.long)
        full_label = torch.tensor([-100] * len(context[1:]) + label, dtype=torch.long)
        full_context = full_context[-self.len:]
        full_label = full_label[-self.len:]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        padded_context = torch.nn.functional.pad(full_context, (0, self.len - full_context.size(0)), value=pad_token_id)
        padded_label = torch.nn.functional.pad(full_label, (0, self.len - full_label.size(0)), value=-100)
        return padded_context, padded_label
    
    def step(self):
        if self.len + self.step_size <= self.max_len:
            self.len += self.step_size
        else:
            self.len = self.max_len
    
    def seq_len_range(self):
        return self.min_len, self.max_len

class ThePile(Dataset):
    def __init__(self, split, tokenizer, min_len=1, max_len=1000, warmup_epochs=0):
        """
        splits: ["train", "validation", "test"]
        """
        if warmup_epochs < 1:
            self.min_len = max_len
        else:
            self.min_len = min_len
        self.max_len = max_len
        self.len = self.min_len
        self.step_size = (self.max_len - self.min_len) // (warmup_epochs + 1)
        self.data = load_dataset('monology/pile-uncopyrighted', split=split, streaming=split=="train")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]['text']
        tokenized = torch.tensor(self.tokenizer(item, add_special_tokens=False)['input_ids'], dtype=torch.long)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        padded_tokenized = torch.nn.functional.pad(tokenized, (0, self.len - tokenized.size(0)), value=pad_token_id)
        return padded_tokenized[:-1], padded_tokenized[1:]
    
    def step(self):
        if self.len + self.step_size <= self.max_len:
            self.len += self.step_size
        else:
            self.len = self.max_len
    
    def seq_len_range(self):
        return self.min_len, self.max_len

def initialize_dataset(name, *args, **kwargs):
    dataset_class = getattr(sys.modules[__name__], name, None)
    return dataset_class(*args, **kwargs)
