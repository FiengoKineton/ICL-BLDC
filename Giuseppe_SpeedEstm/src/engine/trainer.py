from __future__ import annotations
from typing import Dict, Any, List
import torch
from torch.utils.data import DataLoader
from ..losses import build_loss
from .evaluator import rollout
from .metrics import mse

def train_loop(model, optimizer, scheduler_fn, train_loader: DataLoader, val_loader: DataLoader, cfg: Dict[str, Any], device: torch.device):
    loss_fn = build_loss(cfg["loss"])
    history: List[Dict[str, float]] = []
    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            B,H,C = x.shape
            # rollout within training (no teacher forcing)
            last = y[:, :1]
            y_hat = []
            for t in range(H):
                x_t = x.clone()
                x_t[:, :, -1] = torch.cat([last.squeeze(-1), torch.zeros(B, H-1, device=x.device)], dim=1)[:, t]
                out = model(x_t)
                pred_t = out[:, t:t+1]
                y_hat.append(pred_t)
                last = pred_t
            y_hat = torch.cat(y_hat, dim=1)
            losses = loss_fn(y_hat, y)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if cfg["train"].get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
            if scheduler_fn:
                for pg in optimizer.param_groups:
                    pg["lr"] = scheduler_fn(step)
            step += 1
        # validation
        val_losses = {}
        with torch.no_grad():
            for batch in val_loader:
                y_hat = rollout(model, batch, device)
                l = loss_fn(y_hat, batch["y"].to(device))
                for k, v in l.items():
                    val_losses[k] = val_losses.get(k, 0.0) + float(v.item())
            for k in val_losses:
                val_losses[k] /= max(1, len(val_loader))
        rec = {"epoch": epoch, "train_total": float(losses["total"].item())}
        rec.update({f"val_{k}": v for k, v in val_losses.items()})
        history.append(rec)
    return history
