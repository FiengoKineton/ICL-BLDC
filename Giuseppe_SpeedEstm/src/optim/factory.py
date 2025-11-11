# src/optim/factory.py
from __future__ import annotations
from typing import Dict, Any, Tuple
from .optimizers import configure_adamw
from .schedulers import warmup_cosine_lr

def build_optimizer_and_scheduler(model, cfg_optim: Dict[str, Any]):
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
    return opt, scheduler
