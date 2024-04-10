import math
import importlib
from datetime import datetime
from omegaconf.dictconfig import DictConfig
from torch.optim.lr_scheduler import LambdaLR


def get_timestamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, extra_kwargs=dict()):
    config_dict = dict(config)
    if not "target" in config_dict:
        if config_dict == '__is_first_stage__':
            return None
        elif config_dict == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    target_kwargs = dict(config_dict.get('kwargs', dict()))

    for k, v in target_kwargs.items():
        if isinstance(v, DictConfig) and 'target' in v.keys():
            target_kwargs[k] = instantiate_from_config(v)
    target_kwargs.update(extra_kwargs)
    return get_obj_from_str(config_dict["target"])(**target_kwargs)


def get_cosine_schedule_with_warmup(
    optimizer,  num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
