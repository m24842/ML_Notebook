import os
import sys
import termios
import logging
import signal
import requests
import wandb
import torch
import torch.nn as nn

class NoEcho:
    _og_attrs = None

    @classmethod
    def disable_echo(cls):
        if not sys.stdin.isatty(): return
        fd = sys.stdin.fileno()
        cls._og_attrs = termios.tcgetattr(fd)
        new_attrs = termios.tcgetattr(fd)
        new_attrs[3] = new_attrs[3] & ~termios.ECHO  # Disable ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, new_attrs)

    @classmethod
    def enable_echo(cls):
        if cls._og_attrs is None: return
        if not sys.stdin.isatty(): return
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSADRAIN, cls._og_attrs)

class NoKeyboardInterrupt:
    def __enter__(self):
        self._orig_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self._orig_handler)

def online(timeout=3):
    try:
        _ = requests.get("https://www.google.com", timeout=timeout)
        return True
    except requests.RequestException:
        return False

def cleanup_wandb(entity, project):
    with NoKeyboardInterrupt():
        if wandb.run and not wandb.run._is_finished:
            run_id = wandb.run.id
            wandb.finish()
            delete_run = input("Delete WandB run? (y/n): ").strip().lower()
            if delete_run == "y": wandb.Api().run(f'{entity}/{project}/{run_id}').delete()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_info(log_path, model, model_name, configs, train_accuracies=None, test_accuracies=None):
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m-%d-%Y %H:%M')
    log_message = (
        f"{model_name}\n"
        + f"Total params: {count_parameters(model):,}\n"
        + f"Hyperparams:\n"
        + '\n'.join([f'\t{key}: {value}' for key, value in configs.items()]) + '\n'
        + (f"Train accuracies:\n" if train_accuracies else "")
        + (f"\t{', '.join(f'{acc:.2f}' for acc in train_accuracies)}\n" if train_accuracies else "")
        + (f"Test accuracies:\n" if test_accuracies else "")
        + (f"\t{', '.join(f'{acc:.2f}' for acc in test_accuracies)}" if test_accuracies else "")
    )
    logging.info(log_message)

def get_available_device():
    return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def apply_weight_decay(model, weight_decay, exclude=["bias", "norm"]):
    """
    Disable weight decay for specified parameters.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if getattr(param, '_no_weight_decay', False) or any(nd in name.lower() for nd in exclude):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

def checkpoint(model_name, output_dir, model, optimizer=None, scheduler=None):
    model_dir = f'{output_dir}/{model_name}'
    model_path = f'{model_dir}/{model_name}.pt'
    optimizer_path = f'{model_dir}/{model_name}_opt.pt'
    scheduler_path = f'{model_dir}/{model_name}_sch.pt'
    
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    torch.save(model.state_dict(), model_path)
    if optimizer: torch.save(optimizer.state_dict(), optimizer_path)
    if scheduler: torch.save(scheduler.state_dict(), scheduler_path)

def load_checkpoint(model_name, output_dir, model, optimizer=None, scheduler=None, device=torch.device('cpu')):
    model_dir = f'{output_dir}/{model_name}'
    model_path = f'{model_dir}/{model_name}.pt'
    optimizer_path = f'{model_dir}/{model_name}_opt.pt'
    scheduler_path = f'{model_dir}/{model_name}_sch.pt'
    
    try:
        state_dict = torch.load(model_path, weights_only=True, map_location=device)
        if "_orig_mod." in list(state_dict.keys())[0]:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        if optimizer: optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True, map_location=device))
        if scheduler: scheduler.load_state_dict(torch.load(scheduler_path, weights_only=True, map_location=device))
        print(f'\033[92mResuming from checkpoint\033[0m')
    except:
        print(f'\033[91mStarting from scratch\033[0m')
    
    if model and not optimizer and not scheduler:
        output = model
    else:
        output = (model,)
        if optimizer: output += (optimizer,)
        if scheduler: output += (scheduler,)
    return output

def allocate_dynamic_memory(model, bsz, min_len, max_len, device=torch.device('cpu')):
    """
    Allocate dynamic memory on the specified device.
    """
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    temp = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
    torch._dynamo.mark_dynamic(temp, 1, min=min_len, max=max_len)
    model = torch.compile(model, dynamic=True, backend="eager")
    with torch.no_grad(): model(temp)
    return model

def try_to_float(dictionary):
    """
    Convert string values to float if possible.
    """
    for key, value in dictionary.items():
        if type(value) is str:
            try: dictionary[key] = float(value)
            except: pass
    return dictionary