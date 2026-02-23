# src/blocks/train.py
from __future__ import annotations
from typing import Dict, Optional, Iterable, Tuple
import torch
import torch.nn as nn

from src.blocks.ar import ar_rollout
from src.blocks.losses import mse_loss

def grad_global_norm(model: nn.Module) -> torch.Tensor:
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        if p is None or p.grad is None:
            continue
        total = total + p.grad.detach().pow(2).sum()
    return torch.sqrt(total)

def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_u: torch.Tensor,
    batch_y: torch.Tensor,
    *,
    feedback_channel: int = 4,
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    yhat = ar_rollout(batch_u, model, feedback_channel=feedback_channel)
    loss = mse_loss(yhat, batch_y)

    loss.backward()
    gnorm = grad_global_norm(model)

    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

    optimizer.step()

    return {"loss": loss.detach(), "grad_norm": gnorm.detach()}

@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch_u: torch.Tensor,
    batch_y: torch.Tensor,
    *,
    feedback_channel: int = 4,
) -> Dict[str, torch.Tensor]:
    model.eval()
    yhat = ar_rollout(batch_u, model, feedback_channel=feedback_channel)
    loss = mse_loss(yhat, batch_y)
    return {"loss": loss.detach()}

def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]],
    *,
    device: torch.device,
    epochs: int,
    feedback_channel: int = 4,
) -> Dict[str, list]:
    history = {"train_loss": [], "train_gnorm": [], "val_loss": []}

    model.to(device)

    for _ep in range(epochs):
        # train
        tr_losses, tr_gnorms = [], []
        for u, y in train_loader:
            u, y = u.to(device), y.to(device)
            stats = train_step(model, optimizer, u, y, feedback_channel=feedback_channel)
            tr_losses.append(stats["loss"].item())
            tr_gnorms.append(stats["grad_norm"].item())
        history["train_loss"].append(sum(tr_losses) / max(1, len(tr_losses)))
        history["train_gnorm"].append(sum(tr_gnorms) / max(1, len(tr_gnorms)))

        # val
        if val_loader is not None:
            vl_losses = []
            for u, y in val_loader:
                u, y = u.to(device), y.to(device)
                stats = eval_step(model, u, y, feedback_channel=feedback_channel)
                vl_losses.append(stats["loss"].item())
            history["val_loss"].append(sum(vl_losses) / max(1, len(vl_losses)))

    return history