# src/optim/schedulers.py
from __future__ import annotations
import math


def warmup_cosine_lr_old(iter, lr, min_lr, warmup_iters, lr_decay_iters):
    # LR schedule:
    # 1) Warmup: lr * (iter / warmup_iters), for iter < warmup_iters
    # 2) Cosine decay from lr → min_lr over [warmup_iters, lr_decay_iters]
    #    lr(iter) = min_lr + 0.5*(lr - min_lr)*(1 + cos(pi * progress))
    # 3) Clamp to min_lr after lr_decay_iters

    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return lr * iter / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)


def warmup_cosine_lr(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    max_steps: int,
    min_lr_scale: float = 0.1,
) -> float:
    """
    Exact port of the old warmup_cosine_lr, just with renamed arguments:
      step          -> iter
      base_lr       -> lr
      warmup_steps  -> warmup_iters
      max_steps     -> lr_decay_iters
      min_lr        -> base_lr * min_lr_scale
    """
    lr = base_lr
    min_lr = base_lr * min_lr_scale
    warmup_iters = warmup_steps
    lr_decay_iters = max_steps

    # 1) linear warmup for warmup_iters steps
    if step < warmup_iters:
        return lr * step / warmup_iters

    # 2) if step > lr_decay_iters, return min learning rate
    if step > lr_decay_iters:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    if lr_decay_iters == warmup_iters:
        # degenerate case: no decay range; old code would divide by 0,
        # so we just return min_lr here to stay well-defined
        return min_lr

    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def __warmup_cosine_lr(step: int, *, base_lr: float, warmup_steps: int, max_steps: int, min_lr_scale: float = 0.1) -> float:
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


# CHECKED -- all good but could have called the fnc directly from transformer_zero_step.py