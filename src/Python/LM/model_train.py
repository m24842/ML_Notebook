import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import tqdm
import math
import sys
sys.path.append('src/Python/MultiLinear')
sys.path.append('src/Python/AssociativeMemory')
from model import *

model_path = 'src/Python/LM/associative.pth'
optimizer_path = 'src/Python/LM/associative_optimizer.pth'
scheduler_path = 'src/Python/LM/associative_scheduler.pth'

def text_to_binary(text):
    binary_tensor = torch.tensor([int(bit) for c in text for bit in f"{ord(c):08b}"], dtype=torch.long)
    return binary_tensor

def binary_to_text(binary_tensor):
    text = ''.join([chr(int(''.join(map(str, binary_tensor[i:i+8].tolist())), 2)) for i in range(0, len(binary_tensor), 8)])
    return text

def char_to_index(char):
    return ord(char) - 32

def text_to_indices(text):
    return torch.tensor([char_to_index(c) for c in text])

def char_to_onehot(char, vocab_size=256):
    try:
        idx = ord(char) - 32
        onehot = torch.zeros(vocab_size)
        onehot[idx] = 1
    except:
        print(f"Invalid char: {char}")
        onehot = torch.zeros(vocab_size)
    return onehot

def prob_dist_to_char(prob_dist):
    idx = torch.argmax(prob_dist)
    return chr(idx.item() + 32)

def text_to_tensor(text, vocab_size=256):
    return torch.stack([char_to_onehot(c, vocab_size) for c in text], dim=0)

def indices_to_onehot(indices, vocab_size=256):
    onehot = torch.zeros((indices.size(0), indices.size(1), 95))
    indices[indices > vocab_size-1] = 0
    onehot[:, torch.arange(indices.size(1)), indices] = 1
    return onehot.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(net, optimizer, scheduler, dataloader, epochs):
    print('Training...')
    total_length = 100
    avgs = []
    for epoch in range(1, epochs+1):
        train_dataset = np.random.choice(dataset["train"]["text"], size=4, replace=False).tolist()
        dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        avg_loss = 0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = [text[:total_length] for text in data]
            x_tensors = []
            for text in data:
                values = text_to_indices(text)
                target_size = total_length
                # values = text_to_binary(text)
                # target_size = 8 * total_length
                values = values[:target_size]
                padding_size = target_size - values.size(0)
                padded_values = torch.cat([values, torch.zeros(padding_size, dtype=torch.long)])
                x_tensors.append(padded_values)

            x = torch.stack(x_tensors, dim=0).to(device)
            x_diff = x.clone()
            # x_diff[:, 1:] = x_diff[:, 1:] - x_diff[:, :-1] + 1

            try:
                net.reset()
            except:
                pass
            outputs, _ = net(x[:, :-1])
            # outputs = net(x[:, :-1])
            
            loss = 0
            loss += F.cross_entropy(outputs.transpose(1, 2), x[:, 1:], reduction='mean')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            avg_loss += loss.item()/math.prod(x.size()[:2])
            
            print(f'Epoch {epoch}, Batch {(i+1):}, Loss: {loss.item()/math.prod(x.size()[:2]):.5e}, LR: {optimizer.param_groups[0]["lr"]:.0e}')
            
            if True:#epoch % 2 == 0 and i == 0:
                print(''.join([prob_dist_to_char(outputs[0, k]) for k in range(outputs.size(1))]))
                # print(binary_to_text(torch.argmax(outputs[0, 7:, :2], dim=-1).cpu().detach()).encode('unicode_escape').decode())
                print(data[0][1:].encode('unicode_escape').decode())
                print()
            
        scheduler.step(avg_loss/(i+1))
        lr_ratio = optimizer.param_groups[0]['lr'] / initial_lr
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = initial_weight_decay * lr_ratio
        torch.save(net.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        torch.save(scheduler.state_dict(), scheduler_path)
        
        avgs.append(avg_loss/(i+1))
        plt.clf()
        plt.plot(avgs)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.pause(1e-1)

@torch.no_grad()
def test(net, dataloader):
    net.eval()
    print('Testing...')
    context_length = 100
    total_length = 200
    for i, data in enumerate(dataloader):
        x_tensors = []
        for text in data:
            values = text_to_indices(text)
            target_size = total_length
            # values = text_to_binary(text)
            # target_size = 8 * total_length
            values = values[:target_size]
            padding_size = target_size - values.size(0)
            padded_values = torch.cat([values, torch.zeros(padding_size, dtype=torch.long)])
            x_tensors.append(padded_values)

        x = torch.stack(x_tensors, dim=0).to(device)
    
        try:
            net.reset(x.size(0))
        except:
            pass
        # outputs = []
        # out = x[:, 0:1]
        # for t in tqdm.tqdm(range(min(total_length, x.size(1))), desc="Processing", leave=False):
        #     out = net.step(x[:, t:t+1] if t < context_length else out, use_cache=True)
        #     outputs.append(out.detach().cpu())
        #     out = torch.argmax(out, dim=-1)
        # outputs = torch.stack(outputs, dim=1).squeeze(2)
        # First process the context then generate the rest
        context_x = x[:, :8*context_length]
        
        # Option 1: Use step-by-step generation
        outputs = []
        out = context_x[:, 0:1]
        net_state = [InferenceCache.alloc(x.size(0), net.args, device=device) for _ in range(net.args.n_layer)]
        
        # Process context tokens first to build state
        for t in tqdm.tqdm(range(min(8*context_length, context_x.size(1))), desc="Processing Context"):
            logits, net_state = net(context_x[:, t:t+1], net_state)
            out = torch.argmax(logits, dim=-1)
            outputs.append(out.detach().cpu())
        
        # Then generate new tokens
        for t in tqdm.tqdm(range(8*total_length - 8*context_length), desc="Generating"):
            logits, net_state = net(out, net_state)
            out = torch.argmax(logits, dim=-1)
            outputs.append(out.detach().cpu())
            
        outputs = torch.cat(outputs, dim=1)

        for j in range(outputs.size(0)):
            # print(''.join([prob_dist_to_char(outputs[j, k]) for k in range(outputs.size(1))]))
            print(binary_to_text(outputs[0]).encode('unicode_escape').decode())
            print()

net = MoE_MambaLM(latent_dim=64, state_dim=64, num_experts=8, top_k=2, num_layers=4, vocab_size=2).to(device)
mamba_config = Mamba2Config(
    d_model=256,
    d_state=256,
    d_conv=4,
    n_layer=4,
    vocab_size=95,
    pad_vocab_size_multiple=5,
    chunk_size=99,
)
net = Mamba2LMHeadModel(mamba_config, device)
# mamba_config = Mamba2Config(
#     d_model=64,
#     d_state=128,
#     d_conv=4,
#     n_layer=24,
#     vocab_size=2,
#     pad_vocab_size_multiple=1,
#     chunk_size=799,
# )
# net = Mamba2LMHeadModel(mamba_config, device)
# net = AssociativeNet(256, 128, 4, 4, 95, 2, 4, device=device)
try:
    net.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
except:
    pass

initial_lr = 5e-4
initial_weight_decay = 1e-4
optimizer = optim.AdamW(net.parameters(), lr=initial_lr, weight_decay=initial_weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
try:
    optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
except:
    pass
try:
    scheduler.load_state_dict(torch.load(scheduler_path, weights_only=True))
except:
    pass

dataset = load_dataset("imdb")
train_dataset = np.random.choice(dataset["train"]["text"], size=128, replace=False).tolist()
test_dataset = np.random.choice(dataset["test"]["text"], size=1, replace=False).tolist()

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

optimizer.param_groups[0]['lr'] = initial_lr

print(f'Model Parameters: {count_parameters(net):,}')

train(net, optimizer, scheduler, train_dataloader, epochs=1000)

test(net, test_dataloader)