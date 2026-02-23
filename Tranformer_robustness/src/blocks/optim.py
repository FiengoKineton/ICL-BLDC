# src/blocks/optim.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn

@dataclass
class OptimConfig:
    name: str = "adamw"                 # "adamw" or "sgd"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    momentum: float = 0.9               # for SGD
    nesterov: bool = False              # for SGD

def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p is not None]

    if cfg.name.lower() == "adamw":
        decay, no_decay = [], []
        for p in params:
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)

    if cfg.name.lower() == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum,
            nesterov=cfg.nesterov, weight_decay=cfg.weight_decay
        )

    raise ValueError(f"Unknown optimizer: {cfg.name}")