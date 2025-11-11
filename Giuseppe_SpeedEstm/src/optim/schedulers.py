# src/optim/schedulers.py
from __future__ import annotations
import math
from typing import Callable, Dict

def warmup_cosine_lr(step: int, *, base_lr: float, warmup_steps: int, max_steps: int, min_lr_scale: float = 0.1) -> float:
    """Warmup + Cosine decay scheduler (per-step API).
    Mirrors the behavior used in the original training scripts.
    Args:
        step: current step (0-based)
        base_lr: initial LR peak after warmup
        warmup_steps: linear warmup length
        max_steps: total steps
        min_lr_scale: final lr = base_lr * min_lr_scale
    Returns:
        float: learning rate for this step
    """
    if max_steps <= 0:
        return base_lr
    if step < warmup_steps and warmup_steps > 0:
        return base_lr * (step + 1) / warmup_steps
    # Cosine schedule from base_lr down to base_lr * min_lr_scale
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_scale + (1.0 - min_lr_scale) * cosine)
