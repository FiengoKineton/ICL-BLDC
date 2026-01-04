# run_experiment.py     | SET (leave it as it is)

import time, torch, torch.nn as nn, numpy as np, pandas as pd

from pathlib import Path
from functools import partial
from copy import deepcopy
from typing import Dict, Any, List
from torch.utils.data import DataLoader

from engine import train, validate
from engine_utils import build_device, build_model, configure_optimizer, warmup_cosine_lr, TimeWeightedMSELoss
from plot_testing import run_testing
from plot_training import run_training_plots
from resource_monitor import ResourceMonitor


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
    print_flag = bool(cfg.get("plot", {}).get("print", True))

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
    regression_mode = cfg_training.get("regression_mode", "time_last")

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

    # Device + model
    if regression_mode == "one_step":
        cfg_data["nu"] = cfg_data["nu"] - 1

    device, device_type = build_device(cfg_compute, print_flag)
    model, model_args = build_model(cfg_model, cfg_data, device, device_type, print_flag)
    optimizer = configure_optimizer(model, cfg_training, device_type, print_flag)

    # Criterion
    loss = cfg_training.get("loss", "MSE")
    if loss == "MSE":
        criterion = torch.nn.MSELoss()
    elif loss == "TimeWeightedMSE":
        criterion = TimeWeightedMSELoss(mode=cfg_training["loss_mode"])
    num_params = model.get_num_params()

    # LR schedule
    get_lr = partial(
        warmup_cosine_lr,
        lr=cfg_training["lr"],
        min_lr=cfg_training["min_lr"],
        warmup_iters=cfg_training["warmup_iters"],
        lr_decay_iters=cfg_training["lr_decay_iters"],
    )

    if print_flag: print(f"[run] Running experiment in {run_dir}")
    if print_flag: print(f"[run] seq_len={cfg_data['seq_len']}, "
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

    # -------- resource monitor (CPU/RAM/GPU/etc) --------
    mon = ResourceMonitor(
        sample_interval_s=5.0,                 # change as you like
        device_index=0,                        # GPU index
        enabled=True,                          # can be cfg-driven
    )
    mon.start()
    start_time = time.time()

    eval_interval = cfg_training.get("eval_interval", 1)

    for epoch in range(cfg_training["max_iters"]):
        # LR
        if cfg_training["decay_lr"]:
            lr_epoch = get_lr(epoch)
        else:
            lr_epoch = cfg_training["lr"]
        optimizer.param_groups[0]["lr"] = lr_epoch


        train_loss = train(model, train_dl, criterion, optimizer, device, R_smooth, regression_mode)
        val_loss = validate(model, val_dl, criterion, device, R_smooth, regression_mode) if (epoch % eval_interval) == 0 else np.nan


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
                best_model = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

                checkpoint = {
                    "model": best_model,
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": best_epoch,
                    "train_time": time.time() - start_time,
                    "history": history,
                    "best_val_loss": best_val_loss,
                    "cfg": cfg,
                    "num_params": num_params,
                    "device_type": device_type,
                }
                torch.save(checkpoint, run_dir / f"{cfg_logging['checkpoint_stem']}_best.pt")
            else:
                no_improve += 1

        if (epoch % max(1, eval_interval)) == 0:
            if print_flag: print(
                f"[epoch {epoch}] "
                f"train={train_loss:.4e} val={val_loss:.4e} "
                f"best={best_val_loss:.4e} (epoch={best_epoch}) "
                f"lr={lr_epoch:.3e} no_improve={no_improve}/{patience}"
            )

        if no_improve >= patience:
            if print_flag: print(f"[early-stop] epoch {epoch}, patience={patience}")
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
        "num_params": num_params,
        "device_type": device_type,
    }
    torch.save(final_ckpt, run_dir / f"{cfg_logging['checkpoint_stem']}_last.pt")

    # Save history as CSV
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(run_dir / "history.csv", index=False)

    # Stop monitor + save CSV
    df_res = mon.stop()
    res_csv = cfg.get("experiment", {}).get("resources", "resources.csv")
    df_res.to_csv(run_dir / res_csv, index=False)


    # Save the config actually used for this run
    import yaml
    with (run_dir / "config_used.yaml").open("w") as f:
        yaml.safe_dump(cfg, f)


    run_testing(
        run_dir=run_dir,
        split="test",
        epoch=best_epoch,      # you can leave this None since it's only at the end
        cfg=cfg,
        model_dir=best_model,
        device=device,
        data_set=test_ds,
    ) 

    run_training_plots(
        run_dir=run_dir,
        cfg=cfg,
        history=history,     # no checkpoint read
        train_time=train_time,
        best_val_loss=best_val_loss,
        num_params=num_params,
        device_type=device_type,
        ckpt_name=None,
        show=False,
    )

    print(f"\n\n[run] Done. best_val_loss={best_val_loss:.4e} at epoch {best_epoch} (time: {train_time})")
    return float(best_val_loss)
