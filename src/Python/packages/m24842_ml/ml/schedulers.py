import sys
from torch.optim.lr_scheduler import *
from transformers import get_cosine_schedule_with_warmup

class CosineAnnealingLRWithWarmup():
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch
        )

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

def initialize_scheduler(name, *args, **kwargs):
    scheduler_class = getattr(sys.modules[__name__], name, None)
    return scheduler_class(*args, **kwargs)