import cv2
import torch
from dfa import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from mamba2 import *

torch.autograd.set_detect_anomaly(True)

def generate_next_frame(width, height, x, y, vx, vy, circle_radius=10, colored=True):
    # Create a blank frame.
    if colored:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        circle_color = (255, 255, 255)  # White in BGR.
    else:
        frame = np.zeros((height, width), dtype=np.uint8)
        circle_color = 255  # White in grayscale.
    
    # Draw the circle. OpenCV's cv2.circle works with both grayscale and color images.
    center = (int(round(x)), int(round(y)))
    cv2.circle(frame, center, circle_radius, circle_color, -1)
    
    # Update the circle's position.
    x += vx
    y += vy
    
    # Bounce off the left/right borders.
    if x - circle_radius < 0 or x + circle_radius > width:
        vx = -vx
        x = max(circle_radius, min(x, width - circle_radius))
        
    # Bounce off the top/bottom borders.
    if y - circle_radius < 0 or y + circle_radius > height:
        vy = -vy
        y = max(circle_radius, min(y, height - circle_radius))
    
    return torch.from_numpy(frame).float() / 255.0, x, y, vx, vy

def generate_bouncing_circle_frames(width, height, num_frames, circle_radius=10, colored=True):
    # Initialize a random starting position, ensuring the circle is fully inside the frame.
    x = np.random.uniform(circle_radius, width - circle_radius)
    y = np.random.uniform(circle_radius, height - circle_radius)
    
    # Initialize a random velocity.
    vx = np.random.uniform(-5, 5)
    vy = np.random.uniform(-5, 5)
    
    frames = []

    for _ in range(num_frames):
        # Create a blank frame.
        if colored:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            circle_color = (255, 255, 255)  # White in BGR.
        else:
            frame = np.zeros((height, width), dtype=np.uint8)
            circle_color = 255  # White in grayscale.
        
        # Draw the circle. OpenCV's cv2.circle works with both grayscale and color images.
        center = (int(round(x)), int(round(y)))
        cv2.circle(frame, center, circle_radius, circle_color, -1)
        frames.append(frame)
        
        # Update the circle's position.
        x += vx
        y += vy
        
        # Bounce off the left/right borders.
        if x - circle_radius < 0 or x + circle_radius > width:
            vx = -vx
            x = max(circle_radius, min(x, width - circle_radius))
            
        # Bounce off the top/bottom borders.
        if y - circle_radius < 0 or y + circle_radius > height:
            vy = -vy
            y = max(circle_radius, min(y, height - circle_radius))
    
    # Stack the list of frames into a single NumPy array (tensor)
    tensor = np.stack(frames, axis=0) / 255.0
    tensor = torch.from_numpy(tensor).float()
    return tensor

def generate_bouncing_circle_dataset(num_samples, width, height, num_frames, circle_radius=10, colored=True):
    dataset = []
    for _ in range(num_samples):
        frames = generate_bouncing_circle_frames(width, height, num_frames, circle_radius, colored)
        dataset.append(frames)
    return torch.stack(dataset)

def generate_bouncing_circle_dataloader(num_samples, width, height, num_frames, batch_size=64, circle_radius=10, colored=True):
    dataset = generate_bouncing_circle_dataset(num_samples, width, height, num_frames, circle_radius, colored)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta=1):
        super(BetaVAE, self).__init__()
        self.beta = beta  # Controls the KL divergence weight

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(256, latent_dim)  # Log variance

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Output in range [0,1]
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Reparameterization trick

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bvae = BetaVAE(4096, 256, beta=1)
        self.config = Mamba2Config(
            d_model=256,
            d_state=1024,
            headdim=64,
            n_layer=4,
        )
        self.state = None
        self.mamba = nn.ModuleDict(
            dict(
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba2(self.config, device=mps),
                                norm1=RMSNorm(self.config.d_model, device=mps),
                                feedforward=nn.Sequential(
                                    nn.Linear(self.config.d_model, self.config.d_model),
                                    nn.SiLU(),
                                    nn.Linear(self.config.d_model, self.config.d_model),
                                    nn.SiLU(),
                                ),
                                norm2=RMSNorm(self.config.d_model, device=mps),
                            )
                        )
                        for _ in range(self.config.n_layer)
                    ]
                ),
                norm_f=RMSNorm(self.config.d_model, device=mps),
            )
        )
        
    def forward(self, x):
        if self.state is None: self.state = [InferenceCache.alloc(x.size(0), self.config, device=mps) for _ in range(self.config.n_layer)]
        outputs = torch.zeros_like(x)
        bvae_loss = 0
        for i in range(x.shape[1]):
            mu, logvar = self.bvae.encode(x[:, i])
            in_proj = self.bvae.reparameterize(mu, logvar)
            bvae_loss += self.bvae.loss_function(self.bvae.decode(in_proj), x[:, i], mu, logvar)[0]
            in_proj = in_proj.unsqueeze(1)
            for j, layer in enumerate(self.mamba.layers):
                out, self.state[i] = layer.mixer.step(layer.norm1(in_proj), self.state[j])
                # in_proj = in_proj + out
                out = layer.feedforward(layer.norm2(out))
                in_proj = in_proj + out
            in_proj = self.mamba.norm_f(in_proj)
            in_proj = in_proj.squeeze(1)
            outputs[:, i] = self.bvae.decode(in_proj).squeeze(0)
        return outputs, bvae_loss
    
    def reset(self):
        self.state = None
    
    def detach(self):
        if self.state:
            for layer in self.state:
                layer.ssm_state = layer.ssm_state.detach()
                layer.conv_state = layer.conv_state.detach()

color = False
batch_size = 16
width, height = 64, 64
num_frames = 500

net = Net().to(mps)
optimizer = optim.AdamW(net.parameters(), lr=1e-3)

# epochs = 30
# for epoch in range(epochs):
#     total_loss = 0
#     for batch_idx, data in enumerate(data_loader):
#         print(f"Batch {batch_idx}", end="\r")
#         data = data.flatten(2).to(mps)
#         inputs = data[:, :-1]
#         targets = data[:, 1:]
#         optimizer.zero_grad()
#         net.reset()
#         output = net(inputs)
#         loss = nn.functional.mse_loss(output, targets)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch}, Loss: {total_loss/(batch_idx+1)}")

total_loss = 0
x = np.random.uniform(10, width - 10)
y = np.random.uniform(10, height - 10)
vx = np.random.uniform(-5, 5)
vy = np.random.uniform(-5, 5)
frame, x, y, vx, vy = generate_next_frame(width, height, x, y, vx, vy, circle_radius=10, colored=color)
for i in range(num_frames):
    optimizer.zero_grad()
    net.detach()
    pred, bvae_loss = net(frame.flatten(0).unsqueeze(0).unsqueeze(0).to(mps))
    frame, x, y, vx, vy = generate_next_frame(width, height, x, y, vx, vy, circle_radius=10, colored=color)
    loss = nn.functional.mse_loss(pred.squeeze(0).squeeze(0), frame.flatten(0).to(mps), reduction='sum')
    loss += bvae_loss
    loss.backward()
    optimizer.step()
    total_loss = 0.9 * total_loss + 0.1 * loss.item()
    print(f"Frame: {i} Loss: {loss.item()}")

net.reset()
num_gen_frames = 500
data = generate_bouncing_circle_frames(64, 64, num_gen_frames, circle_radius=10, colored=color).flatten(1).to(mps)
generated_frames = torch.zeros((num_gen_frames, 4096), device=mps)
for i in range(num_gen_frames):
    with torch.inference_mode():
        if i < 100:
            generated_frames[i], _ = net(data[i].unsqueeze(0).unsqueeze(0).to(mps))
        else:
            generated_frames[i], _ = net(generated_frames[i-1].unsqueeze(0).unsqueeze(0))

generated_frames = generated_frames.squeeze().view(-1, 64, 64).cpu().numpy() * 255
generated_frames = generated_frames.astype(np.uint8)

if not color:
    generated_frames = np.expand_dims(generated_frames, axis=-1)

out = cv2.VideoWriter('src/Python/DFA/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (512, 512), isColor=color)
for i in range(generated_frames.shape[0]):
    frame = cv2.resize(generated_frames[i], (512, 512), interpolation=cv2.INTER_NEAREST)
    out.write(frame)
out.release()