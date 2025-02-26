import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

mps = torch.device('mps')

class PolarLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolarLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = nn.Parameter(torch.ones(1, device=mps))
        self.rotations = nn.Parameter(torch.randn(output_dim - 1, device=mps))
    
    def forward(self, r, theta):
        r, theta = r.to(mps), theta.to(mps)
        r = r * self.scale
        if theta.size(1) < self.output_dim - 1:
            padding = torch.zeros((theta.size(0), self.output_dim - 1 - theta.size(1)), device=mps)
            theta = torch.cat((theta, padding), dim=1)
        theta = theta[:, :self.output_dim-1] + self.rotations
        return r, theta

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = PolarLinear(784, 1024)
        self.fc2 = PolarLinear(1024, 4096)
        self.fc3 = PolarLinear(4096, 10)
        
    def forward(self, x):
        r, theta = self.cartesian_to_polar(x)
        r, theta = self.fc1(r, theta)
        r, theta = F.relu(r), F.relu(theta)
        r, theta = self.fc2(r, theta)
        r, theta = F.relu(r), F.relu(theta)
        r, theta = self.fc3(r, theta)
        x = self.polar_to_cartesian(r, theta)
        x = F.softmax(x, dim=1)
        return x
    
    def cartesian_to_polar(self, cartesian):
        cartesian = torch.tensor(cartesian, dtype=torch.float32) if not isinstance(cartesian, torch.Tensor) else cartesian
        batch_size, n = cartesian.shape

        # Compute the radius (Euclidean norm)
        r = torch.norm(cartesian, dim=1, keepdim=True)  # Shape: (batch_size, 1)

        # Compute angles
        thetas = []
        for k in range(n - 1):
            denominator = torch.norm(cartesian[:, k:], dim=1, keepdim=True)
            theta = torch.acos(torch.clamp(cartesian[:, k:k+1] / denominator, -1.0, 1.0))  # Clamp for numerical stability
            thetas.append(theta)

        # Special case for the last angle (azimuthal angle)
        eps = 1e-7  # Small constant to avoid NaNs
        thetas[-1] = torch.atan2(cartesian[:, 1:2] + eps, cartesian[:, 0:1] + eps)


        return r.squeeze(1), torch.cat(thetas, dim=1)  # r: (batch_size,), thetas: (batch_size, n-1)

    def polar_to_cartesian(self, r, thetas):
        r = r.unsqueeze(1)  # Ensure r has shape (batch_size, 1)
        batch_size, n_minus_1 = thetas.shape
        n = n_minus_1 + 1

        # Initialize Cartesian coordinates
        cartesian = torch.zeros((batch_size, n), dtype=torch.float32)

        # Compute x1 and x2
        sin_prod = torch.cumprod(torch.sin(thetas[:, :-1]), dim=1) if n > 2 else torch.ones_like(r)
        cartesian[:, 0] = r[:, 0] * torch.cos(thetas[:, -1]) * sin_prod[:, -1] if n > 2 else r[:, 0] * torch.cos(thetas[:, -1])
        cartesian[:, 1] = r[:, 0] * torch.sin(thetas[:, -1]) * sin_prod[:, -1] if n > 2 else r[:, 0] * torch.sin(thetas[:, -1])

        # Compute intermediate x_k
        for k in range(2, n - 1):
            sin_prod_k = torch.cumprod(torch.sin(thetas[:, :n-k]), dim=1)[:, -1]
            cartesian[:, k] = r[:, 0] * torch.cos(thetas[:, n-k-1]) * sin_prod_k

        # Compute x_n
        cartesian[:, -1] = r[:, 0] * torch.cos(thetas[:, 0])

        return cartesian
    
# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize the network and optimizer
net = Net().to(mps)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-5)

# Training loop
for epoch in range(10):  # 10 epochs
    net.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.view(inputs.size(0), -1).to(mps), labels.to(mps)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs.to(mps), labels.to(mps))
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}')

print('Finished Training')

# Testing loop
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.view(inputs.size(0), -1).to(mps), labels.to(mps)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test set: {100 * correct / total:.2f}%')
