import time

import torch
from transformers import AutoTokenizer

from mamba2 import Mamba2LMHeadModel

device = torch.device('mps')

model = Mamba2LMHeadModel.from_pretrained("state-spaces/mamba2-130m", device=device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id

generation_config = dict(
    max_new_length=10,
    temperature=1.0,
    top_k=30,
    top_p=1.0,
)

def generate(prompt: str, seed: int = 0, show_perf: bool = True):
    """Generate streaming completion"""
    torch.manual_seed(seed)

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)[0]
    print(prompt, end="")

    start = time.process_time()
    n_generated = 0
    for i, (token_id, _hidden_state) in enumerate(model.generate(input_ids, **generation_config)):
        token = tokenizer.decode([token_id])
        if i == 0:
            now = time.process_time()
            prompt_eval_elapsed, start = now - start, now
        else:
            n_generated += 1
        print(token, end="", flush=True)
    if show_perf:
        elapsed = time.process_time() - start
        print('\n\n---')
        print(f'Prompt eval | tokens: {input_ids.shape[0]} | elapsed: {prompt_eval_elapsed:.2f}s | tok/s: {input_ids.shape[0] / prompt_eval_elapsed:.2f}')
        print(f'Generation | tokens: {n_generated} | elapsed: {elapsed:.2f}s | tok/s: {n_generated / elapsed:.2f}')
        
generate("Mamba is a new state space model architecture")