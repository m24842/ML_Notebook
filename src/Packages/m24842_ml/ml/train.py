import os
import sys
import time
import math
import yaml
import copy
import torch
import wandb
import traceback
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from .utils import *
from .models import initialize_model
from .schedulers import initialize_scheduler
from .datasets import initialize_dataset

def default_data_fn(data, target):
    return data, target

def train_epoch(epoch, train_loader, model, optimizer, loss_fn, acc_fn, data_fn=default_data_fn,
                scheduler=None, device="cpu", completed_steps=0, train_steps=None,
                output_dir="", model_name=None, val_loader=None,
                wandb_logging=True, wandb_metrics=["acc", "loss"],
                grad_clip_norm=None, accumulation_steps=0,
                mixed_precision=False,
                checkpoint_freq=None, val_freq=None, info_freq=None):
    # Default model name
    if model_name is None: model_name = model.__class__.__name__
    model.train()
    train_loss = 0
    train_acc = 0
    accumulated_batch_loss = 0
    accumulated_batch_acc = 0
    iterable = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False, bar_format='{desc}: [{n_fmt}/{total_fmt}] {percentage:.0f}%|{bar}| [{rate_fmt}] {postfix}')
    scaler = GradScaler(device=device) if mixed_precision else None
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(iterable):
        with autocast(device_type=device) if mixed_precision else nullcontext():
            # Forward pass
            data = data.to(device)
            target = target.to(device)
            data, target = data_fn(data, target)
            output = model(data)
            
            # Accuracy
            with torch.no_grad():
                accuracy = acc_fn(output.clone(), target.clone())
                accumulated_batch_acc += accuracy
                train_acc += accuracy
            
            # Loss
            loss = loss_fn(output, target)
            if torch.isnan(loss).any():
                report_bad_params(model)
                raise ValueError(f"Loss is NaN: Epoch {epoch}, Batch {batch_idx}")
            if torch.isinf(loss).any():
                report_bad_params(model)
                raise ValueError(f"Loss is Inf: Epoch {epoch}, Batch {batch_idx}")
            batch_loss = loss.item()
            accumulated_batch_loss += batch_loss
            train_loss += loss.item()
        
        # Backward pass and gradient accumulation if applicable
        loss = loss / (accumulation_steps + 1)
        loss.backward() if not mixed_precision else scaler.scale(loss).backward()
        
        if (batch_idx + 1) % (accumulation_steps + 1) == 0:
            if grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step() if not mixed_precision else scaler.step(optimizer)
            optimizer.zero_grad()
            if scheduler: scheduler.step()
            if mixed_precision: scaler.update()
            
            # WandB logging
            log_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
            accumulated_batch_loss /= (accumulation_steps + 1)
            accumulated_batch_acc /= (accumulation_steps + 1)
            if wandb_logging:
                log_data = {}
                if "acc" in wandb_metrics: log_data["train/acc"] = accumulated_batch_acc
                if "loss" in wandb_metrics: log_data["train/loss"] = accumulated_batch_loss
                if "ppl" in wandb_metrics: log_data["train/ppl"] = math.exp(accumulated_batch_loss)
                if "lr" in wandb_metrics: log_data["misc/lr"] = log_lr
                if "seq_len" in wandb_metrics: log_data["misc/seq_len"] = train_loader.dataset.len
                wandb.log(log_data)
            accumulated_batch_loss = 0
            accumulated_batch_acc = 0
            
            # Post info
            if info_freq and completed_steps % info_freq == 0 and completed_steps > 0:
                tqdm.write(f'Train Epoch {epoch}: [{batch_idx}/{len(train_loader)}] LR: {log_lr:.1e}, Loss: {batch_loss:.4f}, Acc: {accuracy:.2f}%')
            
            # Checkpoint
            if checkpoint_freq and completed_steps % checkpoint_freq == 0 and completed_steps > 0:
                checkpoint(model_name, output_dir, model, optimizer, scheduler)
            
            # Validation
            if val_freq and completed_steps % val_freq == 0 and completed_steps > 0:
                if val_loader: val_epoch(model, val_loader, loss_fn, acc_fn, device=device, wandb_logging=wandb_logging, wandb_metrics=wandb_metrics)
                model.train()
            
            completed_steps += 1
        
        if train_steps is not None and completed_steps >= train_steps: break
    
    # Account for last accumulated batch
    if (batch_idx + 1) % (accumulation_steps + 1) != 0:
        completed_steps += 1
        if grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step() if not mixed_precision else scaler.step(optimizer)
        optimizer.zero_grad()
        if scheduler: scheduler.step()
        if mixed_precision: scaler.update()
        
        # WandB logging
        log_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
        accumulated_batch_loss /= (batch_idx % (accumulation_steps + 1)) + 1
        accumulated_batch_acc /= (batch_idx % (accumulation_steps + 1)) + 1
        if wandb_logging:
            log_data = {}
            if "acc" in wandb_metrics: log_data["train/acc"] = accumulated_batch_acc
            if "loss" in wandb_metrics: log_data["train/loss"] = accumulated_batch_loss
            if "ppl" in wandb_metrics: log_data["train/ppl"] = math.exp(accumulated_batch_loss)
            if "lr" in wandb_metrics: log_data["misc/lr"] = log_lr
            if "seq_len" in wandb_metrics: log_data["misc/seq_len"] = train_loader.dataset.len
            wandb.log(log_data)
    
    # Step sequence length if applicable
    if hasattr(train_loader.dataset, "step"):
        train_loader.dataset.step()
    
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    
    return train_loss, train_acc, completed_steps

@ torch.no_grad()
def val_epoch(model, val_loader, loss_fn, acc_fn, data_fn=default_data_fn,
              device="cpu",
              wandb_logging=True, wandb_metrics=["acc", "loss"],):
    model.eval()
    val_loss = 0
    val_acc = 0
    start = time.time()
    iterable = val_loader
    for data, target in iterable:
        data = data.to(device)
        target = target.to(device)
        data, target = data_fn(data, target)
        output = model(data)
        val_loss += loss_fn(output, target).item()
        val_acc += acc_fn(output, target)

    total_time = time.time() - start
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    
    tqdm.write(f'\033[93mVal Epoch: Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Elapsed: {total_time:.3f}s\033[0m')
    if wandb_logging:
        log_data = {}
        if "acc" in wandb_metrics: log_data["val/acc"] = val_acc
        if "loss" in wandb_metrics: log_data["val/loss"] = val_loss
        if "ppl" in wandb_metrics: log_data["val/ppl"] = math.exp(val_loss)
        wandb.log(log_data)
    
    return val_loss, val_acc

@ torch.no_grad()
def test_epoch(model, test_loader, loss_fn, acc_fn, data_fn=default_data_fn,
               device="cpu",
               wandb_logging=True, wandb_metrics=["acc", "loss"],):
    model.eval()
    test_loss = 0
    test_acc = 0
    start = time.time()
    tqdm.write("")
    iterable = tqdm(test_loader, desc=f"Test Epoch", leave=False, bar_format='\033[92m{desc}: [{n_fmt}/{total_fmt}] {percentage:.0f}%|{bar}| [{rate_fmt}] {postfix}\033[0m')
    for data, target in iterable:
        data = data.to(device)
        target = target.to(device)
        data, target = data_fn(data, target)
        output = model(data)
        test_loss += loss_fn(output, target).item()
        test_acc += acc_fn(output, target)

    total_time = time.time() - start
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    
    tqdm.write(f'\033[92mTest Epoch: Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, Elapsed: {total_time:.3f}s\033[0m\n')
    if wandb_logging:
        log_data = {}
        if "acc" in wandb_metrics: log_data["test/acc"] = test_acc
        if "loss" in wandb_metrics: log_data["test/loss"] = test_loss
        if "ppl" in wandb_metrics: log_data["test/ppl"] = math.exp(test_loss)
        wandb.log(log_data)
        
    return test_loss, test_acc

def train(epochs, train_steps, benchmark_name, model, train_loader, optimizer, loss_fn, acc_fn, data_fn=default_data_fn,
          scheduler=None, device="cpu",
          train_config=None, mixed_precision=False,
          output_dir="", model_name=None,
          val_loader=None, test_loader=None,
          local_log_path=None,
          wandb_logging=True, wandb_entity=None, wandb_project=None,
          wandb_metrics=["acc", "loss"],
          grad_clip_norm=None, accumulation_steps=0,
          checkpoint_freq=None, val_freq=None, info_freq=None):
    try:
        sys.stdout.write("\033[?25l")
        
        NoEcho.disable_echo()
        
        # Default model name
        if model_name is None: model_name = model.__class__.__name__
        
        print(f'\033[1m{benchmark_name} Benchmark\033[0m')
        print(f'\033[1m{model_name}\033[0m')
        print(f'\033[4mTotal params: {count_parameters(model):,}\033[0m\n')
        
        # WandB Initialization
        if wandb_logging:
            assert wandb_entity is not None, "WandB entity is required for logging."
            assert wandb_project is not None, "WandB project is required for logging."
            wandb.init(
                settings=wandb.Settings(silent=True),
                mode="online" if online() else "offline",
                entity=wandb_entity,
                project=wandb_project,
                name=model_name,
                config=train_config,
            )
        
        # Use multiple GPUs if available
        if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
        
        # Allocate dynamic memory if applicable for dataset
        if hasattr(train_loader.dataset, "seq_len_range"):
            min_len, max_len = train_loader.dataset.seq_len_range()
            model = allocate_dynamic_memory(model, train_loader.batch_size, min_len, max_len, device)
        
        # Metrics
        train_losses = []
        train_accuracies = []
        if test_loader:
            test_losses = []
            test_accuracies = []
        else:
            test_losses = None
            test_accuracies = None
        
        # Train loop
        if epochs is not None:
            for epoch in range(1, epochs + 1):
                # Train epoch
                train_loss, train_acc, _ = train_epoch(
                    epoch=epoch, train_loader=train_loader, val_loader=val_loader,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    loss_fn=loss_fn, acc_fn=acc_fn, data_fn=data_fn, device=device,
                    output_dir=output_dir, model_name=model_name,
                    wandb_logging=wandb_logging, wandb_metrics=wandb_metrics,
                    grad_clip_norm=grad_clip_norm, accumulation_steps=accumulation_steps,
                    mixed_precision=mixed_precision,
                    checkpoint_freq=checkpoint_freq, val_freq=val_freq, info_freq=info_freq
                )
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                
                # Test epoch
                if test_loader:
                    test_loss, test_acc = test_epoch(
                        model, test_loader, loss_fn, acc_fn,
                        device=device,
                        wandb_logging=wandb_logging, wandb_metrics=wandb_metrics
                    )
                    test_losses.append(test_loss)
                    test_accuracies.append(test_acc)
                
                # Model checkpoint
                checkpoint(model_name=model_name, output_dir=output_dir, model=model, optimizer=optimizer, scheduler=scheduler)
        
        elif train_steps is not None:
            completed_steps = 0
            epoch = 1
            while completed_steps < train_steps:
                train_loss, train_acc, completed_steps = train_epoch(
                    epoch=epoch, completed_steps=completed_steps, train_steps=train_steps, train_loader=train_loader, val_loader=val_loader,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    loss_fn=loss_fn, acc_fn=acc_fn, data_fn=data_fn, device=device,
                    output_dir=output_dir, model_name=model_name,
                    wandb_logging=wandb_logging, wandb_metrics=wandb_metrics,
                    grad_clip_norm=grad_clip_norm, accumulation_steps=accumulation_steps,
                    mixed_precision=mixed_precision,
                    checkpoint_freq=checkpoint_freq, val_freq=val_freq, info_freq=info_freq
                )
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                
                # Test epoch
                if test_loader:
                    test_loss, test_acc = test_epoch(
                        model, test_loader, loss_fn, acc_fn,
                        device=device,
                        wandb_logging=wandb_logging, wandb_metrics=wandb_metrics
                    )
                    test_losses.append(test_loss)
                    test_accuracies.append(test_acc)
                
                # Model checkpoint
                checkpoint(model_name=model_name, output_dir=output_dir, model=model, optimizer=optimizer, scheduler=scheduler)
                
                epoch += 1
        
        # Final logging
        if local_log_path: log_info(log_path=local_log_path, model=model, model_name=model_name, configs=train_config, train_accuracies=train_accuracies, test_accuracies=test_accuracies)
        if wandb_logging: wandb.finish()
    
    except KeyboardInterrupt as e: raise e
    except Exception as e:
        if wandb_logging: wandb.finish()
        raise e
    finally:
        NoEcho.enable_echo()
        if wandb_logging: cleanup_wandb(wandb_entity, wandb_project)
        sys.stdout.write("\033[?25h")

def train_from_config_file(yaml_path, loss_fn, acc_fn, data_fn=default_data_fn, device="cpu"):
    """
    **Config file options:**
        global:
            benchmark_name: Name of the benchmark.
            output_dir: Directory to save model checkpoints.
            
            dataset:
                name: Class name of the dataset to use.
                splits: Dictionary of dataset splits with their configurations. (e.g., "train", "val", "test")
            
            logging (optional):
                info_freq (default: 100): Frequency of CLI logging training information. No CLI logging if unspecified.
                local_log_path: Path to save local logs.
                wandb: WandB logging configurations.
                    entity: WandB entity name.
                    project: WandB project name.
                    metrics (default: ["acc", "loss"]): List of metrics to log. (e.g., ["acc", "loss", "ppl", "lr", "seq_len"]).
            
            val_freq (default: 500): Frequency of validation during training. No validation if unspecified.
            checkpoint_freq (default: 500): Frequency of saving model checkpoints. No checkpointing if unspecified.
        
        experiments:
            **List item format:**
                general:
                    seed (default: 0): Random seed for reproducibility.
                    batch_size (default: 32): Batch size for training.
                    accumulation_steps (default: 0): Number of batches to accumulate gradients for.
                    train_steps (optional): Number of training steps. **(Mutually exclusive with epochs)**
                    epochs (optional): Number of epochs to train. **(Mutually exclusive with train_steps)**
                    grad_clip_norm (optional): Gradient clipping norm. No clipping if unspecified.
                    load_checkpoint (default: False): Whether to attempt loading model from checkpoint.
                    mixed_precision (default: False): Whether to use mixed precision for training.
                    num_workers (default: 0): Number of workers for data loading.
                
                model:
                    name: Model class name.
                    Model arguments...
                
                optimizer:
                    name: Optimizer class name (e.g., "SGD", "Adam", "AdamW").
                    exclude_weight_decay (default ["bias", "norm"]): List of parameter names to exclude from weight decay. Provide empty list to not exclude any parameters.
                    Optimizer arguments...
                
                scheduler (optional):
                    name: Scheduler class name (e.g., "ConstantLR", "LinearLR", "CosineAnnealingLR").
                    Scheduler arguments...
    
    Args:
        yaml_path (str): Path to YAML configuration file.
        loss_fn (Callable): A function to compute model loss. Args: (model_output, target). Returns: loss.
        acc_fn (Callable): A function to compute model accuracy. Args: (model_output, target). Returns: accuracy.
        data_fn (Callable, optional): A function to augment data and target before passing into model. Args: (data, target). Returns: (data, target).
        device (str, optional): Device to run training on. Defaults to cpu.
    """
    os.system('clear')
    
    with open(yaml_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    # Extract global training configurations
    global_config = configs.get("global")
    benchmark_name = global_config.get("benchmark_name")
    output_dir = global_config.get("output_dir")
    
    # Initialize datasets
    dataset_config = global_config.get("dataset")
    dataset_name = dataset_config.get("name")
    dataset_splits = dataset_config.get("splits", {})
    if "train" in dataset_splits:
        dataset_args = dataset_splits["train"]
        train_dataset = initialize_dataset(dataset_name, **dataset_args)
    val_dataset = None
    if "val" in dataset_splits:
        dataset_args = dataset_splits["val"]
        val_dataset = initialize_dataset(dataset_name, **dataset_args)
    test_dataset = None
    if "test" in dataset_splits:
        dataset_args = dataset_splits["test"]
        test_dataset = initialize_dataset(dataset_name, **dataset_args)
    
    # Get logging configurations
    logging_config = global_config.get("logging", {})
    info_freq = logging_config.get("info_freq", 100)
    local_log_path = logging_config.get("local_log_path")
    wandb_config = logging_config.get("wandb", {})
    wandb_logging = wandb_config is not None
    wandb_entity = wandb_config.get("entity")
    wandb_project = wandb_config.get("project")
    wandb_metrics = wandb_config.get("metrics", ["acc", "loss"])
    
    # Get checkpointing configurations
    val_freq = global_config.get("val_freq", 500)
    checkpoint_freq = global_config.get("checkpoint_freq", 500)
    
    # Run all experiments
    successful_count = 0
    experiments = configs.get("experiments")
    for i, experiment in enumerate(experiments):
        print(f'\033[1mRunning Experiment [{i + 1}/{len(experiments)}]\033[0m\n')
        
        # Reset dataset sequence lengths if applicable
        if hasattr(train_dataset, "reset"): train_dataset.reset()
        if hasattr(val_dataset, "reset"): val_dataset.reset()
        if hasattr(test_dataset, "reset"): test_dataset.reset()
        
        general_config = copy.deepcopy(experiment.get("general"))
        general_config = try_to_float(general_config)
        
        # Set seed
        seed = general_config.get("seed", 0)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        mixed_precision = general_config.get("mixed_precision", False) and device=="cuda"
        
        # Initialize dataloaders
        batch_size = general_config.get("batch_size", 32)
        epochs = general_config.get("epochs", None)
        train_steps = general_config.get("train_steps", None)
        assert not (train_steps is None and epochs is None), "Either train_steps or epochs must be specified."
        assert not (train_steps is not None and epochs is not None), "Only one of train_steps or epochs can be specified."
        grad_clip_norm = general_config.get("grad_clip_norm", None)
        accumulation_steps = general_config.get("accumulation_steps", 0)
        num_workers = general_config.get("num_workers", 0)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=(num_workers>0), pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
        val_loader = None
        test_loader = None
        if val_dataset: val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=(num_workers>0), pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
        if test_dataset: test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=(num_workers>0), pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
        
        model_config = copy.deepcopy(experiment.get("model"))
        model_name = model_config.pop("name")
        model_config = try_to_float(model_config)
        model_args = model_config.copy()
        model_args["device"] = device
        
        # Initialize model
        model = initialize_model(model_name, **model_args)
        
        # Initialize optimizer
        def initialize_optimizer(name, *args, **kwargs):
            optimizer_class = getattr(sys.modules["torch.optim"], name, None)
            return optimizer_class(*args, **kwargs)
        optimizer_config = copy.deepcopy(experiment.get("optimizer"))
        optimizer_name = optimizer_config.pop("name")
        optimizer_config = try_to_float(optimizer_config)
        weight_decay = float(optimizer_config.get("weight_decay", 0.0))
        exclude_weight_decay = optimizer_config.pop("exclude_weight_decay", None)
        apply_weight_decay_args = dict(
            model=model,
            weight_decay=weight_decay,
        )
        if exclude_weight_decay is not None: apply_weight_decay_args["exclude"] = exclude_weight_decay
        optimizer_config["params"] = apply_weight_decay(**apply_weight_decay_args)
        optimizer = initialize_optimizer(optimizer_name, **optimizer_config)
        
        # Initialize scheduler if specified
        scheduler = None
        scheduler_config = copy.deepcopy(experiment.get("scheduler", {}))
        if scheduler_config:
            scheduler_name = scheduler_config.pop("name")
            scheduler_config = try_to_float(scheduler_config)
            scheduler_config["optimizer"] = optimizer
            scheduler = initialize_scheduler(scheduler_name, **scheduler_config)
        
        # Load model from checkpoint if specified
        load_from_checkpoint = general_config.get("load_checkpoint", False)
        if load_from_checkpoint:
            model, optimizer, scheduler = load_checkpoint(
                model_name=model_name, output_dir=output_dir,
                model=model, optimizer=optimizer, scheduler=scheduler,
                device=device
            )
        else: print(f'\033[91mStarting from scratch\033[0m')
        
        # Collect all training configurations for logging
        train_config = model_config.copy()
        train_config.update({
            "benchmark": benchmark_name,
            "model": model_name,
            "seed": seed,
            "bsz": batch_size,
            "accumulation_steps": accumulation_steps,
            "lr": optimizer_config.get("lr"),
            "weight_decay": weight_decay,
            "grad_clip_norm": grad_clip_norm,
            "permuted": dataset_splits["train"].get("permuted"),
            "min_len": train_loader.dataset.min_len if hasattr(train_loader.dataset, "min_len") else None,
            "max_len": train_loader.dataset.max_len if hasattr(train_loader.dataset, "max_len") else None,
        })
        
        # Train the model
        successful = True
        try:
            train(
                epochs=epochs, train_steps=train_steps, benchmark_name=benchmark_name, model_name=model_name,
                model=model, optimizer=optimizer, scheduler=scheduler,
                loss_fn=loss_fn, acc_fn=acc_fn, data_fn=data_fn,
                train_config=train_config, mixed_precision=mixed_precision,
                output_dir=output_dir,
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                local_log_path=local_log_path,
                wandb_logging=wandb_logging, wandb_entity=wandb_entity, wandb_project=wandb_project,
                wandb_metrics=wandb_metrics,
                grad_clip_norm=grad_clip_norm, accumulation_steps=accumulation_steps,
                checkpoint_freq=checkpoint_freq, val_freq=val_freq, info_freq=info_freq,
                device=device,
            )
        except KeyboardInterrupt:
            successful = False
            terminate = input('Terminate all experiments? (y/n): ').strip().lower()
            if terminate == 'y': break
        except Exception as e:
            successful = False
            traceback_str = traceback.format_exc()
            print(f'\033[91mExperiment [{i + 1}/{len(experiments)}] failed with error:\n{traceback_str}\033[0m\n')
        if successful:
            successful_count += 1
            print(f'\033[92mExperiment [{i + 1}/{len(experiments)}] completed successfully\033[0m\n')
        
        del general_config, model_config, optimizer_config, scheduler_config, train_config
    
    print(f'\033[1m{successful_count}/{len(experiments)} experiments completed successfully\033[0m')