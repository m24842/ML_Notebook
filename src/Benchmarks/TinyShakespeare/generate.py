import torch
import torch.nn.functional as F
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from ml.utils import *
from ml.models import initialize_model
from ml.datasets import initialize_dataset

device = "cpu"#get_available_device()

# model_name = "Transformer"
model_name = "DiffusionTransformer"

if model_name == "DiffusionTransformer":
    model = initialize_model(
        name=model_name,
        emb_dim=128,
        mlp_dim=128,
        n_heads=4,
        n_layers=6,
        input_dim=50277,
        output_dim=50277,
        attn_sink=False,
        mlp_bias=False,
        attn_bias=False,
        dropout=0.0,
        pos_encoding="rope",
        device=device,
    )
else:
    model = initialize_model(
        name=model_name,
        emb_dim=128,
        mlp_dim=128,
        n_heads=4,
        n_layers=6,
        input_dim=50277,
        output_dim=50277,
        attn_sink=False,
        mlp_bias=False,
        attn_bias=False,
        dropout=0.0,
        pos_encoding="xpos",
        device=device,
        causal=True,
        use_embedding=True,
    )

model = load_checkpoint(
    model_name=model_name,
    checkpoint_dir="src/Benchmarks/TinyShakespeare/checkpoints",
    model=model,
    device=device
)
model.eval()

dataset = initialize_dataset("TinyShakespeare", train=True, tokenizer="EleutherAI/gpt-neox-20b", min_len=256, max_len=256)

tokenizer = dataset.tokenizer

def indices_to_text(indices):
    return tokenizer.decode(indices, skip_special_tokens=True)

def text_to_indices(text):
    return tokenizer.encode(text, add_special_tokens=False)

sample, target = dataset[random.randint(0, len(dataset)-1)]

sample = sample.unsqueeze(0).to(device)

seq_len = sample.shape[1]
if model_name == "DiffusionTransformer":
    sample_p = torch.zeros((1, seq_len, model.input_dim), device=device)
    seq_len_idx = torch.arange(seq_len).unsqueeze(0).expand(1, seq_len)
    sample_p[0, seq_len_idx, sample] = 1.0
    betas, alphas, alphas_cumprod = model.get_noise(sample_p, profile_fn=torch.sigmoid, t_range=(-16, 4), beta_range=(0.0001, 0.05))
    noise = torch.randn_like(sample_p)
    sample_p = torch.sqrt(alphas_cumprod) * sample_p + torch.sqrt(1-alphas_cumprod) * noise
    # plt.plot((torch.sqrt(betas) * noise)[0].max(-1).values.cpu().numpy(), label='noise')
    # plt.plot((torch.sqrt(betas) * noise)[0].min(-1).values.cpu().numpy(), label='noise')
    
    # plt.plot(sample[0].cpu().numpy(), label='clean')
    # plt.plot(sample_p[0].argmax(dim=-1).cpu().numpy(), label='noisy')
    # plt.legend()
    # plt.show()
    # exit()

with torch.no_grad():
    if model_name == "DiffusionTransformer":
        generated = []
        for i in tqdm(range(seq_len), leave=False):
            pred_noise = model(sample_p)
            
            generated.append(sample_p[0, i].argmax().item())
            tqdm.write("\033c"+indices_to_text(generated))
            group_size = 500
            multiple = math.ceil(model.input_dim / group_size)
            grouped_sample_p = F.pad(sample_p, (0, group_size*multiple - model.input_dim), value=0.0)
            grouped_sample_p = grouped_sample_p.reshape(1, seq_len, group_size, -1).max(-2).values#.sum(dim=-2)
            plt.clf()
            plt.imshow(grouped_sample_p[0].cpu().numpy().T, aspect='auto', origin='lower')
            # plt.plot(sample_p[0].argmax(dim=-1).cpu().numpy(), label='noisy')
            plt.pause(0.01)
            
            betas, alphas, alphas_cumprod = model.get_noise(sample_p, profile_fn=torch.sigmoid, t_range=(-16, 4), beta_range=(0.0001, 0.05))
            noise = torch.randn_like(sample_p)
            sample_p = (1.0 / torch.sqrt(alphas)) * (sample_p - (betas / torch.sqrt(1.0 - alphas_cumprod)) * pred_noise) + torch.sqrt(betas) * noise
            noise_token = torch.randn((1, 1, model.input_dim), device=device) * torch.sqrt(1-alphas_cumprod[:, -1])
            sample_p = torch.cat((sample_p[:, 1:], noise_token), dim=1)
    else:
        generated = model(sample)[0].argmax(dim=-1)
        print(indices_to_text(generated))