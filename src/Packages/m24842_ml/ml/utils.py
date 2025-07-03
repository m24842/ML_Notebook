import os
import sys
import termios
import signal
import requests
import wandb
import torch

class InvalidLossBackoff:
    def __init__(self, max_backoffs=10, mode="consecutive"):
        """
        modes: ["consecutive", "cumulative"]
            * consecutive: triggered if loss is invalid for max_backoffs consecutive steps
            * cumulative: triggered if loss is invalid for max_backoffs steps in total
        """
        self.mode = mode
        self.max_backoffs = max_backoffs
        self.backoff_count = 0
    
    def step(self, loss):
        invalid_loss = False
        if self.mode == "consecutive":
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                invalid_loss = True
                self.backoff_count += 1
            else:
                self.backoff_count = 0
        elif self.mode == "cumulative":
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                invalid_loss = True
                self.backoff_count += 1
        
        if self.backoff_count >= self.max_backoffs:
            raise ValueError("Invalid loss backoff limit reached.")

        return invalid_loss

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

def report_bad_params(model):
    any_bad_params = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            any_bad_params = True
            print(f"Parameter {name}:\n{param.data}\n")
        elif torch.isinf(param).any():
            any_bad_params = True
            print(f"Parameter {name}:\n{param.data}\n")
    return any_bad_params

def get_available_device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

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

def checkpoint(model_name, checkpoint_dir, model, optimizer=None, scheduler=None):
    model_dir = f'{checkpoint_dir}/{model_name}'
    model_path = f'{model_dir}/{model_name}.pt'
    optimizer_path = f'{model_dir}/{model_name}_opt.pt'
    scheduler_path = f'{model_dir}/{model_name}_sch.pt'
    
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    torch.save(model.state_dict(), model_path)
    if optimizer: torch.save(optimizer.state_dict(), optimizer_path)
    if scheduler: torch.save(scheduler.state_dict(), scheduler_path)

def load_checkpoint(model_name, checkpoint_dir, model, optimizer=None, scheduler=None, device="cpu"):
    model_dir = f'{checkpoint_dir}/{model_name}'
    model_path = f'{model_dir}/{model_name}.pt'
    optimizer_path = f'{model_dir}/{model_name}_opt.pt'
    scheduler_path = f'{model_dir}/{model_name}_sch.pt'
    
    try:
        state_dict = torch.load(model_path, weights_only=True, map_location=device)
        if "_orig_mod." in list(state_dict.keys())[0]:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f'\033[92mResuming from checkpoint\033[0m')
    except:
        print(f'\033[91mStarting from scratch\033[0m')
    
    if optimizer:
        try:
            optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True, map_location=device))
        except:
            print(f'\033[91mFailed to load optimizer state dict\033[0m')
    if scheduler:
        try:
            scheduler.load_state_dict(torch.load(scheduler_path, weights_only=True, map_location=device))
        except:
            print(f'\033[91mFailed to load scheduler state dict\033[0m')
    
    if model and not optimizer and not scheduler:
        output = model
    else:
        output = (model,)
        if optimizer: output += (optimizer,)
        if scheduler: output += (scheduler,)
    return output

def allocate_dynamic_memory(model, bsz, min_len, max_len, device="cpu"):
    """
    Allocate dynamic memory on the specified device.
    """
    input_dim = getattr(model, 'input_dim', 1)
    use_embedding = getattr(model, 'use_embedding', input_dim == 1)
    if use_embedding and input_dim == 1:
        shape = (bsz, max_len)
    else:
        shape = (bsz, max_len, input_dim)
    temp = torch.zeros(shape, device=device)
    
    torch._dynamo.mark_dynamic(temp, 1, min=min_len, max=max_len)
    
    backends = ["inductor", "aot_eager", "eager"]
    for backend in backends:
        try:
            compiled_model = torch.compile(model, dynamic=True, backend=backend)
            with torch.no_grad(): compiled_model(temp)
            model = compiled_model
            break
        except:
            pass
    
    return model

def compile_model(model, input_shape, device="cpu"):
    """
    Allocate dynamic memory on the specified device.
    """
    temp = torch.zeros(input_shape, device=device)
    backends = ["inductor", "aot_eager", "eager"]
    for backend in backends:
        try:
            compiled_model = torch.compile(model, dynamic=True, backend=backend)
            with torch.no_grad(): compiled_model(temp)
            model = compiled_model
            break
        except:
            pass
    
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
