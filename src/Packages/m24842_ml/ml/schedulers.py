import sys
import math
from torch.optim.lr_scheduler import *
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingLRWithWarmup(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, eta_min=0.0, last_epoch=-1):
        self.warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        if current_step < self.warmup_steps:
            return [
                base_lr * current_step / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            progress = (current_step - self.warmup_steps) / (self.num_training_steps - self.warmup_steps)
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1.0 + math.cos(2 * math.pi * self.num_cycles * progress))
                for base_lr in self.base_lrs
            ]

def initialize_scheduler(name, *args, **kwargs):
    scheduler_class = getattr(sys.modules[__name__], name, None)
    return scheduler_class(*args, **kwargs)