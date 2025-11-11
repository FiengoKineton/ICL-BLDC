# src/optim/optimizers.py
from __future__ import annotations
from typing import Iterable, Tuple
import torch
import torch.nn as nn

def configure_adamw(model: nn.Module, *, lr: float, betas: Tuple[float, float] = (0.9, 0.95), weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """AdamW with decoupled weight decay, excluding bias/LayerNorm from decay.
    Mirrors the logic commonly used in GPT-like models.
    """
    decay, no_decay = set(), set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
    # special cases for top-level parameters not in any module
    for pn, p in model.named_parameters():
        if '.' not in pn:  # e.g., positional embeddings
            no_decay.add(pn) if pn.endswith('bias') else decay.add(pn)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
    return optimizer
