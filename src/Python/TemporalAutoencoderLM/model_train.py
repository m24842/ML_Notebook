import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TemporalAutoencoderLM, device
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

model_path = 'src/Python/TemporalAutoencoderLM/tautolm.pth'
optimizer_path = 'src/Python/TemporalAutoencoderLM/tautolm_optimizer.pth'
scheduler_path = 'src/Python/TemporalAutoencoderLM/tautolm_scheduler.pth'

def char_to_index(char):
    return ord(char) - 32

def text_to_indices(text):
    return torch.tensor([char_to_index(c) for c in text])

def char_to_onehot(char):
    try:
        idx = ord(char) - 32
        onehot = torch.zeros(95)
        onehot[idx] = 1
    except:
        print(f"Invalid char: {char}")
        onehot = torch.zeros(95)
    return onehot

def prob_dist_to_char(prob_dist):
    idx = torch.argmax(prob_dist)
    return chr(idx.item() + 32)

def text_to_tensor(text):
    return torch.stack([char_to_onehot(c) for c in text], dim=0)

def indices_to_onehot(indices):
    onehot = torch.zeros((indices.size(0), indices.size(1), 95))
    indices[indices > 94] = 0
    onehot[:, torch.arange(indices.size(1)), indices] = 1
    return onehot.to(device)

def train(net, optimizer, scheduler, dataloader, epochs):
    print('Training...')
    context_length = 99
    total_length = 100
    avgs = []
    for epoch in range(1, epochs+1):
        avg_loss = 0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = [text[:total_length] for text in data]
            x = torch.stack([torch.cat([text_to_indices(text), torch.zeros(total_length - len(text))]) for text in data], dim=0).to(device).to(torch.long)
            x.unsqueeze_(2)
            try:
                net.reset()
            except:
                pass
            outputs_f, outputs_b, states_f, states_b = net.bidirection(
                x_f=x[:, :context_length],
                x_b=x[:, -context_length:],
                length=x.size(1)-context_length,
                # decode=True,
                return_state=True,
            )
            states_f = states_f.reshape(states_f.size(1), states_f.size(2), states_f.size(0), states_f.size(3), states_f.size(4), states_f.size(5))
            states_b = states_b.reshape(states_b.size(1), states_b.size(2), states_b.size(0), states_b.size(3), states_b.size(4), states_b.size(5))
            
            loss = 0
            loss += nn.functional.binary_cross_entropy(outputs_f[:, -1], indices_to_onehot(x)[:, -1], reduction='mean')
            loss += nn.functional.binary_cross_entropy(outputs_b[:, 0], indices_to_onehot(x)[:, 0], reduction='mean')
            loss += nn.functional.smooth_l1_loss(states_f[: 40:60], states_b[: 40:60], reduction='mean')
            
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            print(f'Epoch {epoch}, Loss: {loss.item():.5f}, LR: {optimizer.param_groups[0]["lr"]:.5f}')
            
            # # Differences
            # with torch.no_grad():
            #     outputs_diff = torch.abs(outputs_f + outputs_b - 2*indices_to_onehot(x)).flatten(2).mean(0)
            #     outputs_diff_magnitudes = outputs_diff.sum(1).cpu().detach().numpy()
            #     plt.clf()
            #     plt.plot(outputs_diff_magnitudes)
                
            #     states_diff = torch.abs(states_f - states_b).flatten(2).mean(0)
            #     states_diff_magnitudes = states_diff.sum(1).cpu().detach().numpy()
            #     plt.plot(states_diff_magnitudes)
                
            #     # plt.ylim(0, max(outputs_diff_magnitudes))
            #     plt.ylim(0, max(np.concatenate((outputs_diff_magnitudes, states_diff_magnitudes))))
            #     plt.pause(1e-1)
                
            if epoch % 2 == 0 and i == 0:
                outputs_f_decoded = outputs_f
                outputs_b_decoded = outputs_b
                print(''.join([prob_dist_to_char(outputs_f_decoded[0, k]) for k in range(outputs_f_decoded.size(1))]))
                print(''.join([prob_dist_to_char(outputs_b_decoded[0, k]) for k in range(outputs_b_decoded.size(1))]))
                print(data[0])
                print()
            
        # scheduler.step(avg_loss/(i+1))
        torch.save(net.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        torch.save(scheduler.state_dict(), scheduler_path)
        
        avgs.append(avg_loss/(i+1))
        plt.clf()
        plt.plot(avgs)
        plt.ylim(0, 1.1*max(avgs))
        plt.pause(1e-1)

def test(net, dataloader):
    print('Testing...')
    context_length = 10
    for i, data in enumerate(dataloader):
        x = torch.stack([torch.cat([text_to_indices(text)]) for text in data], dim=0).to(device)
        
        try:
            net.reset()
        except:
            pass
        outputs = []
        out = x[:, 0:1]
        for t in range(x.size(1)):
            out = net(x[:, t:t+1] if t < context_length else out, use_cache=True)
            outputs.append(out.detach().cpu())
        outputs = torch.stack(outputs, dim=1).squeeze(2)

        for j in range(outputs.size(0)):
            print(''.join([prob_dist_to_char(outputs[j, k]) for k in range(outputs.size(1))]))
            print()

net = TemporalAutoencoderLM().to(device)
try:
    net.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
except:
    pass

optimizer = optim.AdamW(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
try:
    optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
except:
    pass
try:
    scheduler.load_state_dict(torch.load(scheduler_path, weights_only=True))
except:
    pass

dataset = load_dataset("imdb")
train_dataset = np.random.choice(dataset["train"]["text"], size=16, replace=False).tolist()
test_dataset = np.random.choice(dataset["test"]["text"], size=1, replace=False).tolist()

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

optimizer.param_groups[0]['lr'] = 1e-3

train(net, optimizer, scheduler, train_dataloader, epochs=100)

test(net, test_dataloader)