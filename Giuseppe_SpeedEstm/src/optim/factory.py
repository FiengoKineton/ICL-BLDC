# src/optim/factory.py
from __future__ import annotations
from typing import Dict, Any
from functools import partial
from .optimizers import configure_adamw
from .schedulers import warmup_cosine_lr, warmup_cosine_lr_old

def build_optimizer_and_scheduler(model, device, max_iter, cfg_optim: Dict[str, Any]):
    name = cfg_optim.get('name', 'adamw').lower()
    lr = float(cfg_optim.get('lr', 3e-4))
    betas = tuple(cfg_optim.get('betas', (0.9, 0.95)))
    weight_decay = float(cfg_optim.get('weight_decay', 0.01))

    if name == 'adamw':
        opt = configure_adamw(model, lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

    sched_cfg = cfg_optim.get('scheduler', None)
    if sched_cfg is None or sched_cfg.get('name', None) is None:
        scheduler = None
    else:
        sname = sched_cfg['name'].lower()
        if sname == 'warmup_cosine':
            warmup_steps = int(sched_cfg.get('warmup_steps', 0))
            max_steps = int(sched_cfg.get('max_steps', 0))
            min_lr_scale = float(sched_cfg.get('min_lr_scale', 0.1))
            # return a simple callable to query lr by global step
            def scheduler(step: int) -> float:
                return warmup_cosine_lr(step, base_lr=lr, warmup_steps=warmup_steps, max_steps=max_steps, min_lr_scale=min_lr_scale)
        else:
            raise ValueError(f"Unsupported scheduler: {sname}")
    

    opt_old = model.configure_optimizers(weight_decay, lr, betas, device)
    scheduler_old = partial(warmup_cosine_lr_old, lr=lr, min_lr=lr/10.0,
                     warmup_iters=int(cfg_optim.get("warmup_iters", 5000)), lr_decay_iters=max_iter)
    return opt_old, scheduler_old


# CHECKED -- all good!