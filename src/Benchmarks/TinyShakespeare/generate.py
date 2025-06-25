import torch
import torch.nn.functional as F
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ml.utils import *
from ml.models import initialize_model
from ml.datasets import initialize_dataset

plt.figure(figsize=(12, 6))

device = "cpu"#get_available_device()

# model_name = "Transformer"
model_name = "DiffusionTransformer"

if model_name == "DiffusionTransformer":
    model = initialize_model(
        name=model_name,
        emb_dim=256,
        mlp_dim=256,
        n_heads=4,
        n_layers=6,
        input_dim=256,
        output_dim=256,
        attn_sink=False,
        mlp_bias=False,
        attn_bias=False,
        dropout=0.0,
        pos_encoding="rope",
        use_embedding=False,
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

tokenizer = "char" if model_name == "DiffusionTransformer" else "EleutherAI/gpt-neox-20b"
dataset = initialize_dataset("TinyShakespeare", train=True, tokenizer=tokenizer, max_len=256)

tokenizer = dataset.tokenizer

def indices_to_text(indices):
    if model_name == "DiffusionTransformer":
        return "".join([chr(max(0, i-2)) for i in indices])
    else:
        return tokenizer.decode(indices, skip_special_tokens=True)

def text_to_indices(text):
    if model_name == "DiffusionTransformer":
        return tokenizer(text)
    else:
        return tokenizer.encode(text, add_special_tokens=False)

sample, target = dataset[random.randint(0, len(dataset)-1)]

sample = sample.unsqueeze(0).to(device)

seq_len = sample.shape[1]
if model_name == "DiffusionTransformer":
    sample_p = torch.zeros((1, seq_len, model.input_dim), device=device)
    # target_p = torch.zeros((bsz, seq_len, model.input_dim), device=device)
    
    seq_len_idx = torch.arange(seq_len).unsqueeze(0)
    sample_p[0, seq_len_idx, sample] = 1.0
    betas, alphas, alphas_cumprod = model.get_noise(sample_p, profile_fn=torch.sigmoid, t_range=(-16, 4), beta_range=(0.0001, 0.05))
    noise = torch.randn_like(sample_p)
    sample = torch.sqrt(alphas_cumprod) * sample_p + torch.sqrt(1-alphas_cumprod) * noise
    # plt.plot((torch.sqrt(betas))[0].max(-1).values.cpu().numpy(), label='noise')
    # plt.plot((torch.sqrt(1-alphas_cumprod))[0].min(-1).values.cpu().numpy(), label='noise')
    
    # plt.plot(sample[0].cpu().numpy(), label='clean')
    # plt.plot(sample[0].argmax(dim=-1).cpu().numpy(), label='noisy')
    # plt.legend()
    # plt.show()
    # exit()

with torch.no_grad():
    if model_name == "DiffusionTransformer":
        generated = []
        for i in tqdm(range(seq_len), leave=False):
            pred_noise = model(sample)
            
            betas, alphas, alphas_cumprod = model.get_noise(sample, profile_fn=torch.sigmoid, t_range=(-16, 4), beta_range=(0.0001, 0.05))
            sample = (1.0 / torch.sqrt(alphas)) * (sample - (betas / torch.sqrt(1.0 - alphas_cumprod)) * pred_noise)
            sample = sample.softmax(dim=-1)
            
            generated.append(sample[0, 0].argmax(dim=-1).cpu().numpy())
            text = indices_to_text(generated)
            tqdm.write("\033c"+text)
            
            group_size = 1
            multiple = math.ceil(model.input_dim / group_size)
            grouped_sample = F.pad(sample, (0, group_size*multiple - model.input_dim), value=0.0)
            grouped_sample = grouped_sample.reshape(1, seq_len, group_size, -1).max(dim=-2).values
            
            # grouped_pred_noise = F.pad(pred_noise, (0, group_size*multiple - model.input_dim), value=0.0)
            # grouped_pred_noise = grouped_pred_noise.reshape(1, seq_len, group_size, -1).max(dim=-2).values
            plt.clf()
            plt.subplot(1, 3, (1, 2))
            plt.title("Token Visualization")
            plt.xlabel('Time')
            plt.ylabel('Token Dimension')
            bound = grouped_sample[0].abs().max().item()
            plt.imshow(grouped_sample[0].cpu().numpy().T, aspect='auto', origin='lower', cmap="hot")#, vmin=0, vmax=bound)
            plt.colorbar()
            
            plt.subplot(1, 3, 3)
            plt.title("Generated Text")
            plt.axis('off')
            plt.text(0.0, 1.0, text, ha='left', va='top', fontsize=10, wrap=True, clip_on=True)
            plt.tight_layout()
            plt.pause(0.01)
            
            noise = torch.randn_like(sample)
            sample += torch.sqrt(betas) * noise
            noise_token = torch.randn((1, 1, model.input_dim), device=device) * torch.sqrt(1-alphas_cumprod[:, -1])
            sample = torch.cat((sample[:, 1:], noise_token), dim=1)
    else:
        sample = model(sample)
        print("\033c"+indices_to_text(sample[0].argmax(-1)))
        group_size = 500
        multiple = math.ceil(model.input_dim / group_size)
        grouped_sample = F.pad(sample, (0, group_size*multiple - model.input_dim), value=0.0)
        grouped_sample = grouped_sample.reshape(1, seq_len, group_size, -1).max(dim=-2).values
        plt.imshow(grouped_sample[0].softmax(-1).cpu().numpy().T, aspect='auto', origin='lower', cmap="hot")
        plt.colorbar()
        plt.show()