from __future__ import annotations
from typing import Dict, Any, List
import torch
from torch.utils.data import DataLoader
from ..losses import build_loss
from .evaluator import rollout
from .metrics import mse


def train_loop(
    model,
    optimizer,
    scheduler_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
):
    loss_fn = build_loss(cfg["loss"])
    history: List[Dict[str, float]] = []
    step = 0

    show_print: bool = bool(cfg.get("show_print", False))

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(device)   # expected (B, H, C)
            y = batch["y"].to(device)   # expected (B, H)
            B, H, C = x.shape

            if show_print and epoch == 0 and batch_idx == 0:
                print("\n=== DEBUG BATCH (FIRST ONLY) ===")
                print(f"epoch: {epoch}, batch_idx: {batch_idx}")
                print("x.shape:", x.shape)
                print("y.shape:", y.shape)
                if x.ndim != 3:
                    print(">>> PROBLEM: x is not 3D (B, H, C).")
                if y.ndim != 2:
                    print(">>> PROBLEM: y is not 2D (B, H).")

            # autoregressive rollout during training
            last = y[:, 0:1]            # (B, 1), initial omega

            if show_print and epoch == 0 and batch_idx == 0:
                print("last.shape (init):", last.shape)
                t_dbg = 0
                x_t_dbg = x.clone()
                print("x_t_dbg[:, t_dbg, -1].shape (target slice):",
                      x_t_dbg[:, t_dbg, -1].shape)
                print("last.squeeze(-1).shape (init):", last.squeeze(-1).shape)
                try:
                    x_t_dbg[:, t_dbg, -1] = last.squeeze(-1)
                    print("Assignment x_t[:, t, -1] = last.squeeze(-1) **WORKED** (init)")
                except Exception as e:
                    print("Assignment x_t[:, t, -1] = last.squeeze(-1) FAILED (init):", repr(e))

            y_hat_steps = []

            for t in range(H):
                x_t = x.clone()                        # (B, H, C)
                x_t[:, t, -1] = last.squeeze(-1)       # (B,) → matches slice (B,)

                out = model(x_t)
                # Normalize model output shape to (B, H)
                if out.ndim == 3 and out.shape[-1] == 1:
                    out = out.squeeze(-1)              # (B, H, 1) → (B, H)
                elif out.ndim != 2:
                    raise RuntimeError(
                        f"Unexpected model output shape {tuple(out.shape)}; "
                        "expected (B, H) or (B, H, 1)."
                    )

                if show_print and epoch == 0 and batch_idx == 0 and t == 0:
                    print("out.shape after normalization:", out.shape)

                pred_t = out[:, t:t+1]                 # (B, 1)
                y_hat_steps.append(pred_t)
                last = pred_t                          # keep (B, 1)

            y_hat = torch.cat(y_hat_steps, dim=1)      # (B, H)

            losses = loss_fn(y_hat, y)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()

            if cfg["train"].get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg["train"]["grad_clip"],
                )

            optimizer.step()
            if scheduler_fn:
                for pg in optimizer.param_groups:
                    pg["lr"] = scheduler_fn(step)
            step += 1

        # validation
        val_losses: Dict[str, float] = {}
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

        if show_print:
            print(
                f"[epoch {epoch}] train_total={rec['train_total']:.4e} "
                + " ".join(f"{k}={v:.4e}" for k, v in val_losses.items())
            )

    return history
