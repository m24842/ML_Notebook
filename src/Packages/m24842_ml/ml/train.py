import os
import sys
import time
import math
import yaml
import copy
import torch
import wandb
import warnings
import traceback
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from dataclasses import dataclass
from .utils import *
from .models import initialize_model
from .schedulers import initialize_scheduler
from .datasets import initialize_dataset

# === Metric Logging ===

@dataclass
class Metric:
    prefix: str = None
    name: str = ""
    value: float = 0.0
    reset_value: float = 0.0
    batch_avg: bool = True
    
    def __eq__(self, metric):
        if isinstance(metric, Metric):
            return self.name == metric.name
        return False

    def __str__(self, prefixed_name=False):
        if prefixed_name:
            return f"{self.get_prefixed_name()}: {self.value:,.4g}"
        return f"{self.name}: {self.value:,.4g}"
    
    def accumulate(self, value):
        if self.batch_avg:
            self.value += value
        else:
            self.value = value
    
    def rescale(self, count):
        if self.batch_avg:
            self.value /= count
    
    def reset(self):
        self.value = self.reset_value
    
    def get_prefixed_name(self, prefix=""):
        prefix = self.prefix if self.prefix is not None else prefix
        return prefix + self.name

class MetricCollection:
    def __init__(self, metrics=None):
        if metrics is None:
            self.metrics = {}
        elif isinstance(metrics, Metric):
            self.metrics = {metrics.name: metrics}
        elif isinstance(metrics, list):
            self.metrics = {metric.name: metric for metric in metrics if isinstance(metric, Metric)}
        elif isinstance(metrics, dict):
            self.metrics = {name: metric for name, metric in metrics.items() if isinstance(metric, Metric)}
        else:
            raise TypeError(f"Expected Metric or list of Metrics, got {type(metrics)}")
    
    def __getitem__(self, name):
        return self.metrics.get(name, None)
    
    def __len__(self):
        return len(self.metrics)
    
    def __iter__(self):
        return iter(self.metrics.values())
    
    def __str__(self):
        return ", ".join(str(metric) for metric in sorted(self.metrics.values(), key=lambda m: str(m)))
    
    def __add__(self, other):
        if isinstance(other, Metric):
            return MetricCollection(metrics={**self.metrics, other.name: other})
        elif isinstance(other, MetricCollection):
            return MetricCollection(metrics={**self.metrics, **other.metrics})
        else:
            raise TypeError(f"Expected Metric or MetricCollection, got {type(other)}")
    
    def append(self, metric):
        self.add_metric(metric)
    
    def add_metric(self, metric):
        if metric is None: return
        if isinstance(metric, Metric):
            self.metrics[metric.name] = metric
        else:
            raise TypeError(f"Expected Metric or list of Metrics, got {type(metric)}")
    
    def add_metrics(self, metrics):
        if metrics is None: return
        for metric in metrics:
            self.add_metric(metric)
    
    def accumulate_metrics(self, new_metrics):
        if new_metrics is None: return
        if isinstance(new_metrics, Metric): new_metrics = [new_metrics]
        for metric in new_metrics:
            if self.metrics.get(metric.name) is not None:
                self.metrics[metric.name].accumulate(metric.value)
            else:
                self.add_metric(metric)

    def rescale_metrics(self, count):
        for metric in self.metrics.values():
            metric.rescale(count)

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()
    
    def to_dict(self, prefix=""):
        return {metric.get_prefixed_name(prefix): metric.value for metric in self.metrics.values()}

def default_log_fn(output, data, target):
    return []

# === Data Augmentation ===

def default_data_fn(data, target, model, dataset):
    return data, target

# === Training ===

def train_epoch(epoch, train_loader, model, optimizer, loss_fn, log_fn=default_log_fn, data_fn=default_data_fn,
                scheduler=None, device="cpu", completed_steps=0, train_steps=None,
                checkpoint_dir="", model_name=None, val_loader=None,
                wandb_logging=False, wandb_metrics=["acc", "loss"],
                grad_clip_norm=None, accumulation_steps=1,
                mixed_precision=False, loss_backoff=InvalidLossBackoff(10, "consecutive"),
                checkpoint_freq=None, val_freq=None, info_freq=None):
    # Default model name
    if model_name is None: model_name = model.__class__.__name__
    model.train()
    accumulated_batch_metrics = MetricCollection()
    iterable = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False, bar_format='{desc}: [{n_fmt}/{total_fmt}] {percentage:.0f}%|{bar}| [{rate_fmt}] {postfix}')
    scaler = GradScaler(device=device) if mixed_precision else None
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(iterable):
        with autocast(device_type=device) if mixed_precision else nullcontext():
            # Forward pass
            data = data.to(device)
            target = target.to(device)
            data, target = data_fn(data, target, model=model, dataset=train_loader.dataset)
            
            output = model(data)
            
            # Loss
            loss = loss_fn(data, output, target)
            
            # Check for invalid loss and param values
            if loss_backoff.step(loss):
                any_bad_params = report_bad_params(model)
                warnings.warn(f"Detected Invalid Loss: Epoch {epoch}, Batch {batch_idx}", RuntimeWarning)
                if any_bad_params: raise RuntimeError("Invalid values detected in model parameters.")
            
            # Accumulate metrics
            with torch.no_grad():
                new_metrics = log_fn(
                    loss=loss,
                    output=output,
                    data=data,
                    target=target,
                )
                new_metrics.append(Metric(name="loss", value=loss.item(), reset_value=0.0, batch_avg=True))
                accumulated_batch_metrics.accumulate_metrics(new_metrics)
        
        # Backward pass and gradient accumulation if applicable
        loss = loss / accumulation_steps
        loss.backward() if not mixed_precision else scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            if grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step() if not mixed_precision else scaler.step(optimizer)
            if mixed_precision: scaler.update()
            optimizer.zero_grad()
            if scheduler: scheduler.step()
            
            # WandB logging
            accumulated_batch_metrics.rescale_metrics(accumulation_steps)
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            accumulated_batch_metrics.add_metric(
                Metric(name="lr", prefix="misc/", value=lr, batch_avg=False),
            )
            if hasattr(train_loader.dataset, "len"):
                accumulated_batch_metrics.add_metric(
                    Metric(name="seq_len", prefix="misc/", value=train_loader.dataset.len, reset_value=0.0, batch_avg=False)
                )
            if wandb_logging:
                log_data = accumulated_batch_metrics.to_dict(prefix="train/")
                wandb.log(log_data)
            
            # Post info
            if info_freq is not None and completed_steps % info_freq == 0 and completed_steps > 0:
                tqdm.write(f'Train Epoch {epoch}: [{batch_idx}/{len(train_loader)}] {accumulated_batch_metrics}')
            
            # Checkpoint
            if checkpoint_freq is not None and completed_steps % checkpoint_freq == 0 and completed_steps > 0:
                checkpoint(model_name, checkpoint_dir, model, optimizer, scheduler)
            
            # Validation
            if val_loader is not None and val_freq is not None and completed_steps % val_freq == 0 and completed_steps > 0:
                val_epoch(model, val_loader, loss_fn=loss_fn, log_fn=log_fn, data_fn=data_fn, device=device, wandb_logging=wandb_logging, wandb_metrics=wandb_metrics)
                model.train()
            
            accumulated_batch_metrics.reset_metrics()
            completed_steps += 1
        
        if train_steps is not None and completed_steps >= train_steps: break
    
    # Account for last accumulated batch
    if (batch_idx + 1) % accumulation_steps != 0:
        completed_steps += 1
        if grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step() if not mixed_precision else scaler.step(optimizer)
        if mixed_precision: scaler.update()
        optimizer.zero_grad()
        if scheduler: scheduler.step()
        
        # WandB logging
        accumulated_batch_metrics.rescale_metrics((batch_idx % accumulation_steps) + 1)
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
        accumulated_batch_metrics.add_metric(
            Metric(name="lr", prefix="misc/", value=lr, batch_avg=False),
        )
        if hasattr(train_loader.dataset, "len"):
            accumulated_batch_metrics.add_metric(
                Metric(name="seq_len", prefix="misc/", value=train_loader.dataset.len, reset_value=0.0, batch_avg=False)
            )
        if wandb_logging:
            wandb.log(accumulated_batch_metrics.to_dict(prefix="train/"))
    
    # Step sequence length if applicable
    if hasattr(train_loader.dataset, "step"):
        train_loader.dataset.step()
    
    return completed_steps

@ torch.no_grad()
def val_epoch(model, val_loader, loss_fn, log_fn=default_log_fn, data_fn=default_data_fn,
              device="cpu",
              wandb_logging=False,):
    model.eval()
    val_metrics = MetricCollection()
    start = time.time()
    iterable = val_loader
    for data, target in iterable:
        data = data.to(device)
        target = target.to(device)
        data, target = data_fn(data, target, model=model, dataset=val_loader.dataset)
        output = model(data)
        loss = loss_fn(data, output, target)
        new_metrics = log_fn(
            loss=loss,
            output=output,
            data=data,
            target=target,
        )
        new_metrics.append(Metric(name="loss", value=loss, reset_value=0.0, batch_avg=True))
        val_metrics.accumulate_metrics(new_metrics)

    total_time = time.time() - start
    val_metrics.rescale_metrics(len(val_loader))
    
    tqdm.write(f'\033[93mVal Epoch: {val_metrics}, Elapsed: {total_time:.3f}s\033[0m')
    if wandb_logging:
        wandb.log(val_metrics.to_dict(prefix="val/"))

@ torch.no_grad()
def test_epoch(model, test_loader, loss_fn, log_fn=default_log_fn, data_fn=default_data_fn,
               device="cpu",
               wandb_logging=False,):
    model.eval()
    test_metrics = MetricCollection()
    start = time.time()
    tqdm.write("")
    iterable = tqdm(test_loader, desc=f"Test Epoch", leave=False, bar_format='\033[92m{desc}: [{n_fmt}/{total_fmt}] {percentage:.0f}%|{bar}| [{rate_fmt}] {postfix}\033[0m')
    for data, target in iterable:
        data = data.to(device)
        target = target.to(device)
        data, target = data_fn(data, target, model=model, dataset=test_loader.dataset)
        output = model(data)
        loss = loss_fn(data, output, target)
        new_metrics = log_fn(
            loss=loss,
            output=output,
            data=data,
            target=target,
        )
        new_metrics.append(Metric(name="loss", value=loss, reset_value=0.0, batch_avg=True))
        test_metrics.accumulate_metrics(new_metrics)

    total_time = time.time() - start
    test_metrics.rescale_metrics(len(test_loader))
    
    tqdm.write(f'\033[92mTest Epoch: {test_metrics}, Elapsed: {total_time:.3f}s\033[0m\n')
    if wandb_logging:
        wandb.log(test_metrics.to_dict(prefix="test/"))

def train(epochs, train_steps, benchmark_name, model, train_loader, optimizer, loss_fn, log_fn=default_log_fn, data_fn=default_data_fn,
          scheduler=None, device="cpu",
          train_config=None, mixed_precision=False,
          checkpoint_dir="", model_name=None,
          val_loader=None, test_loader=None,
          wandb_logging=True, wandb_entity=None, wandb_project=None,
          grad_clip_norm=None, accumulation_steps=1,
          loss_backoff=InvalidLossBackoff(10, "consecutive"),
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
        else:
            # Compile the model for faster training
            model = compile_model(model, train_loader.dataset[0][0].shape, device)
        
        # Train loop
        if epochs is not None:
            for epoch in range(1, epochs + 1):
                # Train epoch
                train_epoch(
                    epoch=epoch, train_loader=train_loader, val_loader=val_loader,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    loss_fn=loss_fn, log_fn=log_fn, data_fn=data_fn, device=device,
                    checkpoint_dir=checkpoint_dir, model_name=model_name,
                    wandb_logging=wandb_logging,
                    grad_clip_norm=grad_clip_norm, accumulation_steps=accumulation_steps,
                    mixed_precision=mixed_precision, loss_backoff=loss_backoff,
                    checkpoint_freq=checkpoint_freq, val_freq=val_freq, info_freq=info_freq
                )
                
                # Test epoch
                if test_loader:
                    test_epoch(
                        model=model, test_loader=test_loader, loss_fn=loss_fn, log_fn=log_fn, data_fn=data_fn,
                        device=device,
                        wandb_logging=wandb_logging,
                    )
                
                # Model checkpoint
                checkpoint(model_name=model_name, checkpoint_dir=checkpoint_dir, model=model, optimizer=optimizer, scheduler=scheduler)
        
        elif train_steps is not None:
            completed_steps = 0
            epoch = 1
            while completed_steps < train_steps:
                completed_steps = train_epoch(
                    epoch=epoch, completed_steps=completed_steps, train_steps=train_steps, train_loader=train_loader, val_loader=val_loader,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    loss_fn=loss_fn, log_fn=log_fn, data_fn=data_fn, device=device,
                    checkpoint_dir=checkpoint_dir, model_name=model_name,
                    wandb_logging=wandb_logging,
                    grad_clip_norm=grad_clip_norm, accumulation_steps=accumulation_steps,
                    mixed_precision=mixed_precision, loss_backoff=loss_backoff,
                    checkpoint_freq=checkpoint_freq, val_freq=val_freq, info_freq=info_freq
                )
                
                # Test epoch
                if test_loader:
                    test_epoch(
                        model=model, test_loader=test_loader, loss_fn=loss_fn, log_fn=log_fn, data_fn=data_fn,
                        device=device,
                        wandb_logging=wandb_logging,
                    )
                
                # Model checkpoint
                checkpoint(model_name=model_name, checkpoint_dir=checkpoint_dir, model=model, optimizer=optimizer, scheduler=scheduler)
                
                epoch += 1
        
        # Finalize logging
        if wandb_logging: wandb.finish()
    
    except KeyboardInterrupt as e: raise e
    except Exception as e:
        if wandb_logging: wandb.finish()
        raise e
    finally:
        NoEcho.enable_echo()
        if wandb_logging: cleanup_wandb(wandb_entity, wandb_project)
        sys.stdout.write("\033[?25h")

def train_from_config_file(yaml_path, loss_fn, log_fn=default_log_fn, data_fn=default_data_fn, device="cpu"):
    """
    **Config file options:**
        `global`:
            `benchmark_name`: Name of the benchmark.
            
            `checkpoint_dir`: Directory to save model checkpoints.
            
            `dataset`:
                `name`: Class name of the dataset to use.

                `splits`: Dictionary of dataset splits with their configurations. (e.g., "train", "val", "test")
            
            `logging` (optional):
                `info_freq` (default: 100): Frequency of CLI logging training information. No CLI logging if unspecified.
                
                `wandb`: WandB logging configurations.
                    `entity`: WandB entity name.
                
                    `project`: WandB project name.
                    
            `val_freq` (default: 500): Frequency of validation during training. Ignored in no validation dataset is provided.
            
            `checkpoint_freq` (default: 500): Frequency of saving model checkpoints. No checkpointing if set to None.
        
        `experiments`:
            **List item format:**
                `general`:
                    `seed` (default: 0): Random seed for reproducibility.
                    
                    `batch_size` (default: 32): Batch size for training.
                    
                    `accumulation_steps` (default: 1): Number of batches to accumulate gradients for.
                    
                    `train_steps` (optional): Number of training steps. **(Mutually exclusive with epochs)**
                    
                    `epochs` (optional): Number of epochs to train. **(Mutually exclusive with train_steps)**
                    
                    `use_loss_fn` (default: 0):
                        If loss_fn is a list, this specifies which function to use from the list.
                        0 means use the first function in the list.
                    
                    `use_log_fn` (default: 0):
                        If log_fn is a list, this specifies which function to use from the list.
                        -1 means use default_log_fn which only logs the loss.
                    
                    `use_data_fn` (default: -1):
                        If data_fn is a list, this specifies which function to use from the list.
                        -1 means use default_data_fn which doesn't modify the data.
                    
                    `grad_clip_norm` (optional): Gradient clipping norm. No clipping if unspecified.
                    
                    `loss_backoff_count` (default: 10): Number of invalid loss backoffs before stopping training.
                    
                    `loss_backoff_type` (default: consecutive): Type of invalid loss backoff.
                    
                    `load_checkpoint` (default: False): Whether to attempt loading model from checkpoint.
                    
                    `mixed_precision` (default: False): Whether to use mixed precision for training.
                    
                    `num_workers` (default: 0): Number of workers for data loading.
                
                `model`:
                    `name`: Model class name.
                    
                    Model arguments...
                
                `optimizer`:
                    `name`: Optimizer class name (e.g., "SGD", "Adam", "AdamW").
                    
                    `exclude_weight_decay` (default ["bias", "norm"]):
                        List of parameter names to exclude from weight decay.
                        Provide empty list to not exclude any parameters.
                    
                    Optimizer arguments...
                
                `scheduler` (optional):
                    `name`: Scheduler class name (e.g., "ConstantLR", "LinearLR", "CosineAnnealingLR").
                    
                    Scheduler arguments...
    
    Args:
        yaml_path (str):
            Path to YAML configuration file.

        loss_fn (Callable):
            Args: (model_output, target)
            Returns: loss
            A function or list of functions that each compute model loss.
            If a list, `use_log_fn` in experiment general config specifies which function to use.
            If a single function, it will be treated as a list with a single item.
            At least one loss function must be provided.
            Defaults to the first function in the list if provided.
        
        log_fn (Callable, optional):
            Args: (loss, output, data, target)
            Returns: list of Metric objects or MetricCollection
            A function or list of functions that each log arbitrary metrics during training.
            If a list, `use_log_fn` in experiment general config specifies which function to use.
            If a single function, it will be treated as a list with a single item.
            Defaults to a function that logs nothing.
            **Note**: loss, lr, and sequence length (if applicable) are logged automatically.
        
        data_fn (Callable, optional):
            Args: (data, target, model, dataset)
            Returns: (data, target)
            A function or list of functions that each augment data and target before passing into model.
            If a list, `use_data_fn` in experiment general config specifies which function to use.
            If a single function, it will be treated as a list with a single item.
            Defaults to a function that does not modify the data.
        
        device (str, optional):
            Device to run training on. Defaults to cpu.
    """
    os.system('clear')
    
    if not isinstance(loss_fn, list): loss_fn = [loss_fn]
    if not isinstance(log_fn, list): log_fn = [log_fn]
    if not isinstance(data_fn, list): data_fn = [data_fn]
    
    with open(yaml_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    # Extract global training configurations
    global_config = configs.get("global")
    benchmark_name = global_config.get("benchmark_name")
    checkpoint_dir = global_config.get("checkpoint_dir")
    
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
    wandb_config = logging_config.get("wandb", {})
    wandb_logging = wandb_config is not None
    wandb_entity = wandb_config.get("entity")
    wandb_project = wandb_config.get("project")
    
    # Get checkpointing configurations
    val_freq = global_config.get("val_freq", 500)
    checkpoint_freq = global_config.get("checkpoint_freq", 500)
    
    # Run all experiments
    successful_count = 0
    experiments = configs.get("experiments")
    for i, experiment in enumerate(experiments):
        successful = True
        try:
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
            
            batch_size = general_config.get("batch_size", 32)
            accumulation_steps = general_config.get("accumulation_steps", 1)
            epochs = general_config.get("epochs", None)
            train_steps = general_config.get("train_steps", None)
            assert not (train_steps is None and epochs is None), "Either train_steps or epochs must be specified."
            assert not (train_steps is not None and epochs is not None), "Only one of train_steps or epochs can be specified."
            
            grad_clip_norm = general_config.get("grad_clip_norm", None)
            mixed_precision = general_config.get("mixed_precision", False) and device=="cuda"
            
            loss_backoff_count = general_config.get("loss_backoff_count", 10)
            loss_backoff_type = general_config.get("loss_backoff_type", "consecutive")
            loss_backoff = InvalidLossBackoff(loss_backoff_count, loss_backoff_type)
            
            # Initialize dataloaders
            num_workers = general_config.get("num_workers", 0)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=(num_workers>0), pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
            val_loader = None
            test_loader = None
            if val_dataset: val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=(num_workers>0), pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
            if test_dataset: test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=(num_workers>0), pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
            
            # Choose loss function
            loss_fn_index = general_config.get("use_loss_fn", 0)
            assert 0 <= loss_fn_index < len(loss_fn), f"Invalid loss_fn index: {loss_fn_index}. Must be between 0 and {len(loss_fn) - 1}."
            experiment_loss_fn = loss_fn[loss_fn_index]
            
            # Choose logging function
            log_fn_index = general_config.get("use_log_fn", 0)
            if log_fn_index == -1:
                experiment_log_fn = default_log_fn
            else:
                assert 0 <= log_fn_index < len(log_fn), f"Invalid log_fn index: {log_fn_index}. Must be between 0 and {len(log_fn) - 1}."
                experiment_log_fn = log_fn[log_fn_index]
            
            # Choose data function
            data_fn_index = general_config.get("use_data_fn", -1)
            if data_fn_index == -1:
                experiment_data_fn = default_data_fn
            else:
                assert 0 <= data_fn_index < len(data_fn), f"Invalid data_fn index: {data_fn_index}. Must be between 0 and {len(data_fn) - 1}."
                experiment_data_fn = data_fn[data_fn_index]
            
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
                    model_name=model_name, checkpoint_dir=checkpoint_dir,
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
            train(
                epochs=epochs, train_steps=train_steps, benchmark_name=benchmark_name, model_name=model_name,
                model=model, optimizer=optimizer, scheduler=scheduler,
                loss_fn=experiment_loss_fn, log_fn=experiment_log_fn, data_fn=experiment_data_fn,
                train_config=train_config, mixed_precision=mixed_precision,
                checkpoint_dir=checkpoint_dir,
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                wandb_logging=wandb_logging, wandb_entity=wandb_entity, wandb_project=wandb_project,
                grad_clip_norm=grad_clip_norm, accumulation_steps=accumulation_steps,
                loss_backoff=loss_backoff,
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