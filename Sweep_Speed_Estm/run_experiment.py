# run_experiment.py     | SET (leave it as it is)

import os, time, torch, json, random
import torch.nn as nn, numpy as np, pandas as pd, matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
from copy import deepcopy
from typing import Dict, Any, List
from torch.utils.data import DataLoader
from datetime import timedelta

from engine import train, validate, test
from engine_utils import build_device, build_model, configure_optimizer, teacher_prob_schedule, warmup_cosine_lr, TimeWeightedMSELoss
from plot_testing import run_testing
from plot_training import run_training_plots
from resource_monitor import ResourceMonitor



def _evaluate(epoch, I: int = 50, N: int = 1000) -> bool:
    # Example: evaluate every 10 epochs
    return (epoch % I) == 0 or epoch + 1 == N

def _format_seconds(seconds: float) -> str:
    try:
        return str(timedelta(seconds=int(round(seconds))))
    except Exception:
        return f"{seconds:.3f}s"

def write_progress_pdf(
    history,
    epoch: int,
    y_true_1d,
    y_pred_1d,
    pdf_path,
    exp_name: str = "",
    log_loss: bool = True,
    ):
    """
    Overwrites a single PDF on disk each call.
    Top: loss curves + vertical dotted line at best val epoch so far.
    Bottom: test prediction vs true for one example (flattened).
    """
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.array([r["epoch"] for r in history], dtype=int)
    train  = np.array([r["train_loss"] for r in history], dtype=float)
    val    = np.array([r["val_loss"] for r in history], dtype=float)
    test   = np.array([r["test_loss"] for r in history], dtype=float)

    p_curve = np.array([r.get("theacher_prob", np.nan) for r in history], dtype=float)
    selection = np.array([r.get("actual_gt_ratio", np.nan) for r in history], dtype=float)

    best_idx = int(np.argmin(val))
    best_epoch = int(epochs[best_idx])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=False)

    # --- losses ---
    ax1.plot(epochs, train, label="train")
    ax1.plot(epochs, val,   label="val")
    if test.mean() < train.mean() * 10:
        ax1.plot(epochs, test,  label="test")
    ax1.axvline(best_epoch, linestyle=":", linewidth=2, label=f"best val @ {best_epoch}")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best")
    if log_loss:
        ax1.set_yscale("log")
    ax1.set_title(f"{exp_name} | epoch={epoch} | best_val_epoch={best_epoch}")

    # --- scheduled sampling ---
    if not np.all(np.isnan(p_curve)):
        ax2.plot(epochs, p_curve, color="tab:blue", linewidth=2, label="p (Target)")
        # Filter out Nones for selection scatter
        valid_mask = ~np.isnan(selection)
        if np.any(valid_mask):
            ax2.scatter(epochs[valid_mask], selection[valid_mask], 
                        color="tab:orange", s=15, alpha=0.6, label="Actual GT Ratio")
        
        ax2.set_ylabel("Prob / Ratio")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")
        ax2.set_title("Scheduled Sampling Progress")
    else:
        ax2.text(0.5, 0.5, "Scheduled Sampling not active", ha="center", va="center")

    # --- prediction ---
    if y_true_1d is not None and y_pred_1d is not None:
        yt = np.asarray(y_true_1d).reshape(-1)
        yp = np.asarray(y_pred_1d).reshape(-1)

        n = min(len(yt), len(yp))
        x = np.arange(n)

        yt = yt[:n]
        yp = yp[:n]

        err = yp - yt
        mse = float(np.mean(err**2))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(err))
        mape = np.mean(np.abs(err) / (np.abs(yt) + 1e-12)) * 100

        ax3.plot(x, yt, label="test y (true)")
        ax3.plot(x, yp, label="test y (pred)")
        ax3.plot(x, err, label=f"error (MSE={mse:.6g}), MAE={mae:.6g}, MAPE={mape:.2f}%)", linestyle="--")

        ax3.set_xlabel("Index (flattened)")
        ax3.set_ylabel("Value")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="best")
        ax3.set_title("Test prediction (one representative batch/sequence)")

    else:
        ax3.text(0.5, 0.5, "No example available", ha="center", va="center")
        ax3.axis("off")

    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf")  # overwrite same file
    plt.close(fig)

def save_run_summary(
    run_dir: Path,
    *,
    exp_name: str,
    cfg: Dict[str, Any],
    best_val_loss: float,
    best_epoch: int,
    train_time: float,
    test_loss: float,
    final_epoch: int,
    num_params: int,
    device_type: str,
    history_len: int,
    resources_csv_name: str,
    checkpoint_stem: str,
    ) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_training = cfg.get("training", {})
    cfg_model = cfg.get("model", {})
    cfg_exp = cfg.get("experiment", {})
    cfg_data = cfg.get("data", {})

    summary = {
        "exp_name": exp_name,
        "seed": cfg_exp.get("seed", None),

        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "test_loss": float(test_loss),

        "train_time_seconds": float(train_time),
        "train_time_pretty": _format_seconds(train_time),

        "final_epoch": int(final_epoch),
        "epochs_ran": int(history_len),

        "device_type": device_type,
        "num_params": int(num_params),

        # Key training knobs (so you can compare runs quickly)
        "batch_size": cfg_training.get("batch_size", None),
        "lr": cfg_training.get("lr", None),
        "max_iters": cfg_training.get("max_iters", None),
        "patience": cfg_training.get("patience", None),
        "eval_interval": cfg_training.get("eval_interval", None),
        "loss": cfg_training.get("loss", None),
        "loss_mode": cfg_training.get("loss_mode", None),
        "smoothness": cfg_training.get("smoothness", None),
        "regression_mode": cfg_training.get("regression_mode", None),

        # Model shape
        "n_layer": cfg_model.get("n_layer", None),
        "n_head": cfg_model.get("n_head", None),
        "n_embd": cfg_model.get("n_embd", None),

        # Data shape hints
        "seq_len": cfg_data.get("seq_len", None),
        "nu": cfg_data.get("nu", None),

        # Artifacts
        "artifacts": {
            "history_csv": "history.csv",
            "resources_csv": resources_csv_name,
            "config_used": "config_used.yaml",
            "checkpoint_best": f"{checkpoint_stem}_best.pt",
            "checkpoint_last": f"{checkpoint_stem}_last.pt",
        },
    }

    # JSON (machine-friendly)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # TXT (human-friendly)
    lines = [
        f"exp_name: {summary['exp_name']}",
        f"seed: {summary['seed']}",
        "",
        f"best_val_loss: {summary['best_val_loss']:.6e}",
        f"best_epoch: {summary['best_epoch']}",
        f"test_loss: {summary['test_loss']:.6e}",
        "",
        f"train_time: {summary['train_time_pretty']} ({summary['train_time_seconds']:.3f}s)",
        f"final_epoch: {summary['final_epoch']}",
        f"epochs_ran: {summary['epochs_ran']}",
        "",
        f"device_type: {summary['device_type']}",
        f"num_params: {summary['num_params']}",
        "",
        "artifacts:",
        f"  - {summary['artifacts']['history_csv']}",
        f"  - {summary['artifacts']['resources_csv']}",
        f"  - {summary['artifacts']['config_used']}",
        f"  - {summary['artifacts']['checkpoint_best']}",
        f"  - {summary['artifacts']['checkpoint_last']}",
    ]
    with (run_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def save_testing_results(
    dir,
    t_ex,
    u_ex,
    y_ex,
    y_pred_ex,
    N: int,
    T: int,
    k: int = 5,
    filename: str = "test_results.npz",
    seed: int | None = None,
    ):
    out_dir = Path(dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Ensure numpy arrays (handles torch tensors too) ---
    def to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    t_ex = to_numpy(t_ex).squeeze()
    y_ex = to_numpy(y_ex).squeeze()
    y_pred_ex = to_numpy(y_pred_ex).squeeze()
    u_ex = to_numpy(u_ex)

    NT = N * T
    if t_ex.shape[0] != NT:
        raise ValueError(f"t_ex length {t_ex.shape[0]} != N*T ({NT})")
    if y_ex.shape[0] != NT:
        raise ValueError(f"y_ex length {y_ex.shape[0]} != N*T ({NT})")
    if y_pred_ex.shape[0] != NT:
        raise ValueError(f"y_pred_ex length {y_pred_ex.shape[0]} != N*T ({NT})")

    # --- Save as NPZ ---
    npz_path = out_dir / filename
    np.savez(
        npz_path,
        N=np.int64(N),
        T=np.int64(T),
        t=t_ex,
        u=u_ex,
        y_true=y_ex,
        y_pred=y_pred_ex,
    )

    # --- Plot k random trajectories (each in its own figure, 2 subplots) ---
    if seed is not None:
        random.seed(seed)

    k = min(k, N)
    chosen = random.sample(range(N), k) if N >= k else [random.randint(0, N - 1) for _ in range(k)]

    for j, i in enumerate(chosen):
        sl = slice(i * T, (i + 1) * T)
        t_1d = t_ex[sl]
        y_true_1d = y_ex[sl]
        y_pred_1d = y_pred_ex[sl]

        err_1d = y_pred_1d - y_true_1d
        mse = float(np.mean(err_1d ** 2))
        mae = np.mean(np.abs(err_1d))
        mape = np.mean(np.abs(err_1d) / (np.abs(y_true_1d) + 1e-12)) * 100
        mape_str = f"={mape:.2f}%" if mape < 300 else f">300%"

        fig, axes = plt.subplots(2, 1, sharex=True)
        fig.suptitle(f"Test sample {i}  ({j+1}/{k})")

        # Top: dynamics
        axes[0].plot(t_1d, y_true_1d, label="y_true")
        axes[0].plot(t_1d, y_pred_1d, label="y_pred")
        axes[0].set_ylabel("y")
        axes[0].legend()
        axes[0].grid(True)

        # Bottom: error + MSE in legend
        axes[1].plot(t_1d, err_1d, label=f"error (MSE={mse:.6g}, MAE={mae:.6g}, MAPE{mape_str}")
        axes[1].axhline(0.0, linewidth=1)
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("y_pred - y_true")
        axes[1].legend()
        axes[1].grid(True)

        fig.tight_layout()
        fig.savefig(out_dir / f"pred_error_sample_{i}.pdf", dpi=150)

    # plt.show()  # uncomment if you want figures to pop up when running as script
    return str(npz_path)


# Core function: trains one model, saves checkpoints + history.

def run_single_experiment(
    cfg: Dict[str, Any],
    train_ds,
    val_ds,
    test_ds,
    run_dir: Path,
    exp_name: str,
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
    cfg_plot = deepcopy(cfg.get("plot", {}))
    ss_cfg = cfg.get("scheduled_sampling", {})

    # Slightly stupid thing: attach compile flag to model
    cfg_model["compile"] = cfg_compute.get("compile", False)
    print_flag = bool(cfg.get("plot", {}).get("print", True))

    # Seed
    seed = cfg_exp.get("seed", 42)
    dt = cfg.get("simulation", {}).get("dt", 0.01)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Scheduled sampling params
    ss_enabled = bool(ss_cfg.get("sched_enabled", False))
    ss_mode = ss_cfg.get("sched_mode", "stochastic")
    p0 = float(ss_cfg.get("p0", 1.0))
    decay_epochs = int(ss_cfg.get("decay_epochs", 0))
    p_min = float(ss_cfg.get("p_min", 0.0))

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

    eval_interval = cfg_training.get("eval_interval", 50)
    tex_filename = os.path.join(run_dir, "run_ongoing.tex")
    pdf_file = os.path.join(run_dir, "run_ongoing_progress.pdf")
    N_epochs = cfg_training["max_iters"]

    for epoch in range(N_epochs):
        # LR
        if cfg_training["decay_lr"]:
            lr_epoch = get_lr(epoch)
        else:
            lr_epoch = cfg_training["lr"]
        optimizer.param_groups[0]["lr"] = lr_epoch

        if ss_enabled:
            teacher_prob = teacher_prob_schedule(
                epoch=epoch,
                p0=p0,
                decay_epochs=decay_epochs,
                p_min=p_min,
            )
        else:
            teacher_prob = 0.0

        train_loss, selection = train(model, train_dl, criterion, optimizer, device, R_smooth, regression_mode, teacher_prob, ss_mode)
        val_loss = validate(model, val_dl, criterion, device, R_smooth, regression_mode, teacher_prob, ss_mode)
        test_loss, (t_ex, u_ex, y_ex, y_pred_ex), (N, T) = test(model, test_ds, device, cfg_data["seq_len"], dt) if _evaluate(epoch, eval_interval, N_epochs) else (float("nan"), (None, None, None, None), (None, None))

        # Track
        if selection is not None:
            if isinstance(selection, torch.Tensor):
                actual_ratio = selection.mean().item()  # Convert multi-element tensor to mean float
            elif isinstance(selection, (list, np.ndarray)):
                actual_ratio = np.mean(selection)       # Convert list/array to mean float
            else:
                actual_ratio = float(selection)         # It's already a single number
        else:
            actual_ratio = None

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "test_loss": test_loss,
            "lr": float(lr_epoch),
            "best_val_loss_so_far": float(best_val_loss),
            "theacher_prob": float(teacher_prob),
            "actual_gt_ratio": actual_ratio,
        }
        history.append(row)

        if _evaluate(epoch, eval_interval, N_epochs):
            with open(tex_filename, "w") as f:
                f.write(r"\begin{tabular}{|l|l|}" + "\n")
                f.write(r"\hline" + "\n")
                f.write(r"\textbf{Metric} & \textbf{Value} \\\\" + "\n")
                f.write(f"Exp Name & {exp_name} \\\\ \n")
                f.write(f"Directory & {run_dir} \\\\ \n")
                f.write(f"Epoch & {epoch} \\\\ \n")
                f.write(f"Train Loss & {train_loss:.4e} \\\\ \n")
                f.write(f"Val Loss & {val_loss:.4e} \\\\ \n")
                f.write(f"Test Loss & {test_loss:.4e} \\\\ \n")
                f.write(f"Best Val Loss & {best_val_loss:.4e} \\\\ \n")
                f.write(f"Learning Rate & {lr_epoch:.3e} \\\\ \n")
                f.write(f"Patience Count & {no_improve}/{patience} \\\\ \n")
                f.write(f"Device & {device_type} \\\\ \n")
                f.write(f"Elapsed Time & {_format_seconds(time.time() - start_time)} \\\\ \n")
                f.write(f"Scheduled Sampling p & {teacher_prob:.3f}, selection = {selection if selection is not None else 0.0} \\\\ \n")
                f.write(r"\hline" + "\n")
                f.write(r"\end{tabular}")

            i = random.randint(0, N - 1)
            write_progress_pdf(
                history=history,
                epoch=epoch,
                y_true_1d=y_ex[i*T:(i+1)*T],
                y_pred_1d=y_pred_ex[i*T:(i+1)*T],
                pdf_path= pdf_file,
                exp_name=exp_name,
                log_loss=True,
            )

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

        if print_flag: print(
            f"[epoch {epoch}] "
            f"train={train_loss:.4e} val={val_loss:.4e} test={test_loss:.4e} "
            f"best={best_val_loss:.4e} (epoch={best_epoch}) "
            f"lr={lr_epoch:.3e} no_improve={no_improve}/{patience}"
        )

        if no_improve >= patience:
            if print_flag: print(f"[early-stop] epoch {epoch}, patience={patience}")
            break

    #if os.path.exists(tex_filename):    os.remove(tex_filename)
    #if os.path.exists(pdf_file):        os.remove(pdf_file)

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
    
    try: 
        save_testing_results(
            dir=run_dir/"plots"/"testing"/"test_fn",
            t_ex=t_ex,
            u_ex=u_ex,
            y_ex=y_ex,
            y_pred_ex=y_pred_ex,
            N=N,
            T=T,
            k=cfg_plot.get("n_test_exps", 5),
            seed=seed,
        )
    except Exception as e:
        print(f"[warning] Could not save test results: {e}")


    test_loss = run_testing(
        run_dir=run_dir,
        split="test",
        epoch=best_epoch,      # you can leave this None since it's only at the end
        cfg=cfg,
        model_dir=best_model,
        device=device,
        data_set=test_ds,
        n_exps=cfg_plot.get("n_test_exps", 5),
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

    # ---- Save run summary (JSON + TXT) ----
    save_run_summary(
        run_dir=run_dir,
        exp_name=exp_name,
        cfg=cfg,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        train_time=train_time,
        test_loss=test_loss,
        final_epoch=epoch,
        num_params=num_params,
        device_type=device_type,
        history_len=len(history),
        resources_csv_name=res_csv,
        checkpoint_stem=cfg_logging["checkpoint_stem"],
    )


    print(f"\n\n[run] Done. best_val_loss={best_val_loss:.4e} at epoch {best_epoch} (time: {train_time})")
    return float(best_val_loss), float(train_time), float(test_loss), int(num_params)

