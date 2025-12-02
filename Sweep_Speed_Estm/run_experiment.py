# run_experiment.py

import time
from pathlib import Path
from functools import partial
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from transformer_zerostep import GPTConfig, GPT, warmup_cosine_lr
from engine import train, validate


def build_device(cfg_compute: Dict[str, Any]):
    torch.set_num_threads(cfg_compute.get("threads", 16))

    no_cuda = cfg_compute.get("no_cuda", False)
    cuda_device = cfg_compute.get("cuda_device", "cuda:0")

    use_cuda = (not no_cuda) and torch.cuda.is_available()
    device_name = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = "cuda" if "cuda" in device_name else "cpu"

    if device_type == "cuda":
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
        print(f"[device] CUDA available: {torch.cuda.is_available()}")
        print(f"[device] Current device: {torch.cuda.current_device()}")

    print(f"[device] Using device: {device_name}")
    return device, device_type


def build_model(cfg_model: Dict[str, Any],
                cfg_data: Dict[str, Any],
                device,
                device_type: str):
    model_args = dict(
        n_layer=cfg_model["n_layer"],
        n_head=cfg_model["n_head"],
        n_embd=cfg_model["n_embd"],
        n_x=cfg_data["nx"],
        n_y=cfg_data["ny"],
        n_u=cfg_data["nu"],
        block_size=cfg_data["seq_len"],
        bias=cfg_model.get("bias", False),
        dropout=cfg_model.get("dropout", 0.0),
    )

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    if torch.cuda.device_count() > 1 and device_type == "cuda":
        print("[model] Using DataParallel on all GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    if cfg_model.get("compile", False):
        model = torch.compile(model)

    return model, model_args


def configure_optimizer(model, cfg_training: Dict[str, Any], device_type: str):
    if isinstance(model, nn.DataParallel):
        optim_target = model.module
    else:
        optim_target = model

    optimizer = optim_target.configure_optimizers(
        cfg_training.get("weight_decay", 0.0),
        cfg_training["lr"],
        (0.9, 0.95),
        device_type
    )
    return optimizer


def run_single_experiment(
    cfg: Dict[str, Any],
    train_ds,
    val_ds,
    test_ds,
    run_dir: Path,
) -> float:
    """
    Core function: trains one model, saves checkpoints + history.

    Returns:
        best_val_loss (float)
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- unpack config blocks ---
    cfg_exp = cfg["experiment"]
    cfg_data = cfg["data"]
    cfg_model = deepcopy(cfg["model"])
    cfg_training = deepcopy(cfg["training"])
    cfg_compute = deepcopy(cfg["compute"])
    cfg_logging = deepcopy(cfg["logging"])

    # Slightly stupid thing: attach compile flag to model
    cfg_model["compile"] = cfg_compute.get("compile", False)

    # Seed
    seed = cfg_exp.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Derived parameters
    cfg_data["seq_len"] = cfg_data.get("seq_len", 10)
    cfg_training["eval_batch_size"] = cfg_training.get("batch_size")
    cfg_training["lr_decay_iters"] = cfg_training["max_iters"]
    cfg_training["min_lr"] = cfg_training["lr"] / 10.0
    cfg_training["decay_lr"] = not cfg_training.get("fixed_lr", False)
    smooth = cfg_training["smoothness"]
    R_smooth = cfg_training["R"] if smooth else None

    # DataLoaders (dataset already prepared)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg_training["batch_size"],
        pin_memory=True,
        shuffle=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg_training["eval_batch_size"],
        pin_memory=True,
        shuffle=True,
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=cfg_training["eval_batch_size"],
        pin_memory=True,
        shuffle=True,
    )

    # Device + model
    device, device_type = build_device(cfg_compute)
    model, model_args = build_model(cfg_model, cfg_data, device, device_type)
    optimizer = configure_optimizer(model, cfg_training, device_type)

    # Criterion
    criterion = torch.nn.MSELoss()

    # LR schedule
    get_lr = partial(
        warmup_cosine_lr,
        lr=cfg_training["lr"],
        min_lr=cfg_training["min_lr"],
        warmup_iters=cfg_training["warmup_iters"],
        lr_decay_iters=cfg_training["lr_decay_iters"],
    )

    print(f"[run] Running experiment in {run_dir}")
    print(f"[run] seq_len={cfg_data['seq_len']}, "
          f"max_iters={cfg_training['max_iters']}, "
          f"batch_size={cfg_training['batch_size']}, "
          f"lr={cfg_training['lr']}, "
          f"layers={cfg_model['n_layer']}, "
          f"heads={cfg_model['n_head']}, "
          f"embd={cfg_model['n_embd']}")

    # Training loop
    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    patience = cfg_training["patience"]
    no_improve = 0
    start_time = time.time()

    eval_interval = cfg_training.get("eval_interval", 1)

    for epoch in range(cfg_training["max_iters"]):
        # LR
        if cfg_training["decay_lr"]:
            lr_epoch = get_lr(epoch)
        else:
            lr_epoch = cfg_training["lr"]
        optimizer.param_groups[0]["lr"] = lr_epoch


        train_loss = train(model, train_dl, criterion, optimizer, device, R_smooth)
        val_loss = validate(model, val_dl, criterion, device, R_smooth) if (epoch % eval_interval) == 0 else np.nan


        # Track
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(lr_epoch),
            "best_val_loss_so_far": float(best_val_loss),
        }
        history.append(row)

        # Early stopping logic on actual val_loss
        if not np.isnan(val_loss):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve = 0

                checkpoint = {
                    "model": model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": epoch,
                    "train_time": time.time() - start_time,
                    "history": history,
                    "best_val_loss": best_val_loss,
                    "cfg": cfg,
                }
                torch.save(checkpoint, run_dir / f"{cfg_logging['checkpoint_stem']}_best.pt")
            else:
                no_improve += 1

        if (epoch % max(1, cfg_training["eval_interval"])) == 0:
            print(
                f"[epoch {epoch}] "
                f"train={train_loss:.4e} val={val_loss:.4e} "
                f"best={best_val_loss:.4e} (epoch={best_epoch}) "
                f"lr={lr_epoch:.3e} no_improve={no_improve}/{patience}"
            )

        if no_improve >= patience:
            print(f"[early-stop] epoch {epoch}, patience={patience}")
            break

    # Final checkpoint (loss trajectory etc)
    train_time = time.time() - start_time
    final_ckpt = {
        "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": epoch,
        "train_time": train_time,
        "history": history,
        "best_val_loss": best_val_loss,
        "cfg": cfg,
    }
    torch.save(final_ckpt, run_dir / f"{cfg_logging['checkpoint_stem']}_last.pt")

    # Save history as CSV
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(run_dir / "history.csv", index=False)

    # Save the config actually used for this run
    import yaml
    with (run_dir / "config_used.yaml").open("w") as f:
        yaml.safe_dump(cfg, f)

    print(f"[run] Done. best_val_loss={best_val_loss:.4e} at epoch {best_epoch} (time: {train_time})")
    return float(best_val_loss)
