from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable
import torch, copy, time
from torch.utils.data import DataLoader
from ..losses import build_loss
from .evaluator import rollout



def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_fn: Optional[Callable[[int], float]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
):
    """
    Main loop: runs multiple epochs, calling `train` and `validate`.
    Uses per-step scheduler_fn(step) if provided.
    """
    # if you use a custom loss builder:
    # loss_fn = build_loss(cfg["loss"])
    # otherwise just pass torch.nn.MSELoss() etc. into cfg
    loss_cfg = cfg.get("loss", None)
    if loss_cfg is None:
        loss_fn = torch.nn.MSELoss()
    else:
        # overwrite with your own builder if you have it
        # loss_fn = build_loss(loss_cfg)
        loss_fn = torch.nn.MSELoss()

    history: List[Dict[str, float]] = []
    best_history: List[Dict[str, float]] = []
    step = 0
    epochs = cfg["train"]["epochs"]
    patience = cfg["train"]["patience"]
    show_print: bool = bool(cfg.get("show_print", True))

    best_val_loss = float("inf")
    best_state_dict = best_opt_dict = None
    best_epoch = -1

    epoch_time = time_start = time.time()
    no_improve = 0

    for epoch in range(epochs):
        train_loss, step = train(
            model=model,
            dataloader=train_loader,
            criterion=loss_fn,
            optimizer=optimizer,
            device=device,
            scheduler_fn=scheduler_fn,
            step=step,
        )

        val_loss = validate(
            model=model,
            dataloader=val_loader,
            criterion=loss_fn,
            device=device,
        )

        epoch_time = time.time() - epoch_time

        rec = {
            "epoch": epoch,
            "train_total": float(train_loss),
            "val_total": float(val_loss),
            "time": epoch_time,
        }
        history.append(rec)

        # --- track best model by validation loss ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # IMPORTANT: clone the weights, do NOT just keep a reference
            best_state_dict = copy.deepcopy(model.state_dict())
            best_opt_dict = copy.deepcopy(optimizer.state_dict())
            best_history = copy.deepcopy(history)
            best_time = epoch_time

            no_improve = 0
        else: 
            no_improve += 1

        if show_print:
            print(
                f"[{epoch:05d}] \n"
                f"\ttime={epoch_time:.4e}, no_improve={no_improve:03d} / {patience:05d} \n"
                f"\ttrain={train_loss:.4e} | "
                f"val={val_loss:.4e} | "
                f"best={best_val_loss:.4e} ({best_epoch:03d}) | "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

        if no_improve >= patience:
            print(f"Early stopping at iter {epoch} (patience={patience})")
            break

    train_time = time.time() - time_start
    return history, (best_state_dict, best_opt_dict, best_val_loss, best_epoch, best_history, best_time), train_time


def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler_fn: Optional[Callable[[int], float]] = None,
    step: int = 0,
):
    """
    One epoch of autoregressive training over windows of length H.
    Ported from your OLD `train`:

      - Input: batch_u: (B,H,5), batch_y: (B,H,1) or (B,H)
      - ω̂_0 := 0
      - For t = 0..H-1:
          - zero last_omega channel everywhere
          - inject ω̂_{t-1} at time t
          - feed strictly-causal prefix [:t+1] to the model
          - read ŷ_t = model(... )[:, -1, 0]
      - Loss: MSE(y, ŷ) over full window.
    """

    torch.autograd.set_detect_anomaly(True)
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Support both dict-style {"x","y"} and tuple-style (x,y)
        if isinstance(batch, dict):
            batch_u, batch_y = batch["x"], batch["y"]
        else:
            batch_u, batch_y = batch

        batch_u = batch_u.to(device)  # (B,H,5)
        batch_y = batch_y.to(device)  # (B,H) or (B,H,1)

        if batch_y.ndim == 2:
            batch_y = batch_y.unsqueeze(-1)  # (B,H,1) like old code

        B, H, C = batch_u.shape
        assert C == 5, "Expected 5 channels: [ia, ib, va, vb, last_omega]"

        optimizer.zero_grad(set_to_none=True)

        # Copy inputs and zero last_omega channel
        batch_u_copy = batch_u.clone()
        batch_u_copy[:, :, 4] = 0.0

        # ω̂_0 = 0 for all sequences
        last_predictions = torch.zeros(B, device=device, requires_grad=True)
        batch_y_pred_list = []

        for t in range(H):
            # Clone and inject last prediction at time t in channel 4
            batch_u_step = batch_u_copy.clone()
            batch_u_step[:, t, 4] = last_predictions

            # Strictly-causal prefix
            batch_u_tmp = batch_u_step[:, :t+1, :]  # (B, t+1, 5)

            # Forward pass: take last time position
            out = model(batch_u_tmp)[:, -1, :]  # (B, d)
            last_predictions = out.view(-1)     # (B,)

            batch_y_pred_list.append(last_predictions.unsqueeze(1))  # (B,1)

        # Concatenate predictions over time: (B,H,1)
        batch_y_pred = torch.cat(batch_y_pred_list, dim=1).unsqueeze(-1)

        loss = criterion(batch_y, batch_y_pred)
        loss.backward()
        optimizer.step()

        # Per-step LR scheduler, if provided
        if scheduler_fn is not None:
            lr_val = scheduler_fn(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_val
        step += 1

        running_loss += float(loss.item())

        # Optional: debug missing grads
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(f"Warning: No gradient computed for {name}")

    return running_loss / max(1, len(dataloader)), step


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
):
    """
    One epoch of autoregressive validation.
    Ported from your OLD `validate`:

      - Same autoregressive rollout as training (no teacher forcing).
      - No gradients.
      - Returns mean validation loss.
    """

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                batch_u, batch_y = batch["x"], batch["y"]
            else:
                batch_u, batch_y = batch

            batch_u = batch_u.to(device)  # (B,H,5)
            batch_y = batch_y.to(device)  # (B,H) or (B,H,1)

            if batch_y.ndim == 2:
                batch_y = batch_y.unsqueeze(-1)  # (B,H,1)

            B, H, C = batch_u.shape
            assert C == 5, "Expected 5 channels: [ia, ib, va, vb, last_omega]"

            batch_y_pred = torch.zeros_like(batch_y)

            # Zero the last_omega channel initially
            batch_u_copy = batch_u.clone()
            batch_u_copy[:, :, 4] = 0.0

            last_predictions = torch.zeros(B, device=device)

            for t in range(H):
                batch_u_step = batch_u_copy.clone()
                batch_u_step[:, t, 4] = last_predictions
                batch_u_tmp = batch_u_step[:, :t+1, :]

                last_predictions = model(batch_u_tmp)[:, -1, :].view(-1)
                batch_y_pred[:, t, 0] = last_predictions

            loss = criterion(batch_y, batch_y_pred)
            running_loss += float(loss.item())

    return running_loss / max(1, len(dataloader))




def __train_loop(
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


# CHECKED -- all good!