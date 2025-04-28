import os
import logging
import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_info(log_path, benchmark, model, model_name, args, train_accuracies=None, test_accuracies=None):
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m-%d-%Y %H:%M')
    log_message = (
        (f"{benchmark} - {model_name}\n" if benchmark else f"{model_name}\n")
        + f"Total params: {count_parameters(model):,}\n"
        + f"Hyperparams:\n"
        + '\n'.join([f'\t{key}: {value}' for key, value in vars(args).items()]) + '\n'
        + (f"Train accuracies:\n" if train_accuracies else "")
        + (f"\t{', '.join(f'{acc:.2f}' for acc in train_accuracies)}\n" if train_accuracies else "")
        + (f"Test accuracies:\n" if test_accuracies else "")
        + (f"\t{', '.join(f'{acc:.2f}' for acc in test_accuracies)}" if test_accuracies else "")
    )
    logging.info(log_message)

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
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
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