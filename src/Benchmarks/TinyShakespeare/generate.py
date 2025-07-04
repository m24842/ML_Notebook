import torch
import torch.nn.functional as F
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from ml.utils import *
from ml.models import initialize_model
from ml.datasets import initialize_dataset

torch.set_grad_enabled(False)

plt.figure(figsize=(16, 6))

device = "cpu"#get_available_device()
bitmaps = torch.load("data/char-8.pt", map_location=device)

# model_name = "Transformer"
model_name = "DiffusionTransformer"

if model_name == "DiffusionTransformer":
    model = initialize_model(
        name=model_name,
        emb_dim=256,
        mlp_dim=256,
        n_heads=4,
        n_layers=6,
        input_dim=64,
        output_dim=64,
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
        input_dim=256,
        output_dim=256,
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

tokenizer_type = "char"
# tokenizer_type = "EleutherAI/gpt-neox-20b"
dataset = initialize_dataset("TinyShakespeare", train=True, tokenizer=tokenizer_type, max_len=256)

tokenizer = dataset.tokenizer

def indices_to_text(indices):
    if tokenizer_type == "char":
        return "".join([chr(max(0, i-2)) for i in indices])
    else:
        return tokenizer.decode(indices, skip_special_tokens=True)

def text_to_indices(text):
    if tokenizer_type == "char":
        return tokenizer(text)
    else:
        return tokenizer.encode(text, add_special_tokens=False)

sample, target = dataset[random.randint(0, len(dataset)-1)]

sample = sample.to(device)

seq_len = sample.shape[0]
if model_name == "DiffusionTransformer":
    # sample_p = torch.zeros((1, seq_len, model.input_dim), device=device)
    # target_p = torch.zeros((bsz, seq_len, model.input_dim), device=device)
    
    # seq_len_idx = torch.arange(seq_len).unsqueeze(0)
    # sample_p[0, seq_len_idx, sample] = 1.0
    sample_p = bitmaps[sample-2].unsqueeze(0).flatten(-2)
    betas, alphas, alphas_cumprod = model.get_noise(sample_p, profile_fn=torch.sigmoid, t_range=(-16, 4), beta_range=(0.0001, 0.05))
    noise = torch.randn_like(sample_p)
    sample_p = torch.sqrt(alphas_cumprod) * sample_p + torch.sqrt(1-alphas_cumprod) * noise
    sample_p_orig = sample_p.clone()
    # plt.plot((torch.sqrt(betas))[0].max(-1).values.cpu().numpy(), label='noise')
    # plt.plot((torch.sqrt(1-alphas_cumprod))[0].min(-1).values.cpu().numpy(), label='noise')
    
    # plt.plot(sample.cpu().numpy(), label='clean')
    # plt.plot(sample_p[0].argmax(dim=-1).cpu().numpy(), label='noisy')
    # plt.legend()
    # plt.show()
    # exit()

if model_name == "DiffusionTransformer":
    # generated = torch.zeros_like(sample)
    generated = []
    n_generate = 1*seq_len
    for i in tqdm(range(n_generate), leave=False):
        output = model.step(sample_p)
        
        pred_noise = output #(sample_p - torch.sqrt(alphas_cumprod) * output) / torch.sqrt(1 - alphas_cumprod)
        sample_p = (1.0 / torch.sqrt(alphas)) * (sample_p - (betas / torch.sqrt(1.0 - alphas_cumprod)) * pred_noise)
        generated.append(sample_p[:, 0:1])
        
        # generated[i] = sample_p[0, 0].argmax(-1)
        # text = indices_to_text(generated[:i+1])
        # target_text = indices_to_text(sample[:i+1])
        # acc = 100 * (generated[:i+1] == sample[:i+1]).float().mean()
        # mask = (generated[:i+1] == sample[:i+1]).tolist()
        # text_colored = "".join([f"{c}" if m else f"\033[91m{c}\033[0m" for c, m in zip(text, mask)])
        # # tqdm.write(f"{(100 * (sample_p[0].argmax(-1) == sample).float().mean()).item()}")
        # # tqdm.write(indices_to_text(sample_p[0].argmax(-1)))
        # # exit()
        # tqdm.write("\033c"+text_colored)
        
        # group_size = 1
        # multiple = math.ceil(model.input_dim / group_size)
        # grouped_sample_p = F.pad(sample_p, (0, group_size*multiple - model.input_dim), value=0.0)
        # grouped_sample_p = grouped_sample_p.reshape(1, seq_len, group_size, -1).max(-2).values
        
        # # grouped_pred_noise = F.pad(pred_noise, (0, group_size*multiple - model.input_dim), value=0.0)
        # # grouped_pred_noise = grouped_pred_noise.reshape(1, seq_len, group_size, -1).max(-2).values
        # plt.clf()
        # plt.suptitle(f"Step: {i+1}/{seq_len} | Accuracy: {acc:.2f}%")
        # plt.subplot(1, 4, (1, 2))
        # plt.title("Token Visualization")
        # plt.xlabel('Time')
        # plt.ylabel('Token Dimension')
        # bound = grouped_sample_p[0].abs().max().item()
        # plt.imshow(grouped_sample_p[0].cpu().numpy().T, aspect='auto', origin='lower', cmap="hot", vmin=0, vmax=bound)
        # plt.colorbar()
        
        # plt.subplot(1, 4, 3)
        # plt.title(f"Generated Text")
        # plt.axis('off')
        # plt.text(0.0, 1.0, text, ha='left', va='top', fontsize=10, wrap=True, clip_on=True)
        
        # plt.subplot(1, 4, 4)
        # plt.title("Target Text")
        # plt.axis('off')
        # plt.text(0.0, 1.0, target_text, ha='left', va='top', fontsize=10, wrap=True, clip_on=True)
        
        # plt.tight_layout()
        # plt.pause(0.01)
        
        # plt.clf()
        # grid_shape = (8, 32)
        # img = sample_p[0].reshape(*grid_shape, 8, 8).permute(0, 2, 1, 3).reshape(grid_shape[0] * 8, grid_shape[1] * 8)
        # plt.imshow(img.cpu().numpy(), cmap="hot", interpolation="nearest", vmin=0, vmax=1)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.pause(0.01)
        
        noise = torch.randn_like(sample_p)
        sample_p += torch.sqrt(betas) * noise
        noise_token = torch.randn((1, 1, model.input_dim), device=device)# * torch.sqrt(1-alphas_cumprod[:, -1])
        sample_p = torch.cat((sample_p[:, 1:], noise_token), dim=1)
    
    generated = torch.cat(generated, dim=1)
    plt.clf()
    grid_shape = ((seq_len+n_generate)//32, 32)
    generated = torch.cat((generated, sample_p), dim=1)
    img = generated[0].reshape(*grid_shape, 8, 8).permute(0, 2, 1, 3).reshape(grid_shape[0] * 8, grid_shape[1] * 8)
    plt.imshow(img.cpu().numpy(), cmap="hot", interpolation="nearest", vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
else:
    sample = model(sample.unsqueeze(0)).softmax(-1)
    print("\033c"+indices_to_text(sample[0].argmax(-1)))
    group_size = 1
    multiple = math.ceil(model.input_dim / group_size)
    grouped_sample = F.pad(sample, (0, group_size*multiple - model.input_dim), value=0.0)
    grouped_sample = grouped_sample.reshape(1, seq_len, group_size, -1).max(-2).values
    plt.imshow(grouped_sample[0].cpu().numpy().T, aspect='auto', origin='lower', cmap="hot")
    plt.colorbar()
plt.show()