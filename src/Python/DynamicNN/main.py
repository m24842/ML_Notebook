import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, norm, processLength):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size + hidden_size, input_size + hidden_size)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.norm = norm
        self.processLength = processLength
        
    def forward(self, x):
        for _ in range(self.processLength):
            x = self.leaky_relu(self.fc(x))
        return x

    def forwardBatch(self, x):
        outputs = torch.zeros((x.size(0), self.input_size + self.hidden_size))
        curr = torch.zeros((self.input_size + self.hidden_size), requires_grad=True)
        for t in range(x.size(0)):
            for _ in range(self.processLength):
                curr_input = torch.cat((x[t], curr[self.input_size:]), dim=0)
                curr = self.leaky_relu(self.fc(curr_input))
            outputs[t] = curr
        return outputs

    def backward(self, outputs, targets):
        """Backward pass calculates the loss and updates the parameters."""
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    
    def loadExisting(self, path):
        try:
            self.load_state_dict(torch.load(path))
            print("Model loaded successfully!")
        except:
            print("Model not found!")
            pass
    
    def save(self, path):
        torch.save(self.state_dict(), path)