# plot_training.py      | SET (leave it as it is)

"""
Config-driven training history plotting.

This script:
  - Reads 'config_used.yaml' from the run directory.
  - Loads train/val loss either from:
        - checkpoint["history"], or
        - old keys ("LOSS", "LOSS_VAL"), or
        - history.csv
  - Uses plotting options from cfg["plot"], e.g.:
        plot:
          output_subdir: "plots"
          show: false
          logy: true
          ma: 1
          dpi: 160
          export_csv: true
          fmt: "pdf"

It can be used in two ways:

1) From CLI (as before):
    python plot_training.py --dir runs/my_run --ckpt test_best.pt

2) From training code:
    from plot_training import run_training_plots

    run_training_plots(
        run_dir=run_dir,
        cfg=cfg,           # config used for the run
        history=history,   # list[dict] built during training
        ckpt_name=None,    # optional, ignored if history is provided
        show=False,
    )
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any

import math, argparse, sys, yaml, torch
import numpy as np, pandas as pd, matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def load_cfg_used(run_dir: Path) -> Dict[str, Any]:
    """
    Load config_used.yaml from a run directory, if present.
    Returns an empty dict if the file does not exist.
    """
    cfg_path = run_dir / "config_used.yaml"
    if not cfg_path.exists():
        print(f"[warn] {cfg_path} not found. Proceeding with default plot settings.")
        return None
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_yaml_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Simple centered moving average with window w (odd preferred)."""
    if w <= 1 or w > len(x):
        return x
    kernel = np.ones(w, dtype=float) / w
    y = np.convolve(x, kernel, mode="same")
    half = w // 2
    for i in range(half):
        y[i] = x[: i + half + 1].mean()
        y[-(i + 1)] = x[-(i + half + 1):].mean()
    return y


def load_cfg_from_ckpt(path: Path): 
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("cfg", {})


def load_losses_from_ckpt(path: Path) -> Dict[str, np.ndarray]:
    """
    Load train/val losses from a checkpoint saved by run_experiment.py.

    Priority:
      1) Use ckpt["history"] if present.
      2) Fallback to old-style keys ("LOSS", "LOSS_VAL") if they exist.
      3) Fallback to history.csv in the same run directory.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    train = val = None

    # 1) Preferred: extract from history list
    hist = ckpt.get("history", None)
    train_time = ckpt.get("train_time", None)
    num_params = ckpt.get("num_params", None)
    best_val_loss = ckpt.get("best_val_loss", None)
    device_type = ckpt.get("device_type", None)

    if isinstance(hist, list) and len(hist) > 0:
        train = np.asarray(
            [row.get("train_loss", np.nan) for row in hist],
            dtype=float,
        )
        val = np.asarray(
            [row.get("val_loss", np.nan) for row in hist],
            dtype=float,
        )

    # 2) Backward compatibility: old keys
    if train is None or val is None or len(train) == 0 or len(val) == 0:
        if "LOSS" in ckpt and "LOSS_VAL" in ckpt:
            train = np.asarray(ckpt["LOSS"], dtype=float)
            val = np.asarray(ckpt["LOSS_VAL"], dtype=float)

    # 3) Fallback: history.csv in same directory as checkpoint
    if train is None or val is None or len(train) == 0 or len(val) == 0:
        csv_path = path.parent / "history.csv"
        if not csv_path.exists():
            raise KeyError(
                f"{path} has no 'history' and no 'LOSS'/'LOSS_VAL', "
                f"and {csv_path} does not exist."
            )
        df = pd.read_csv(csv_path)
        if "train_loss" not in df.columns or "val_loss" not in df.columns:
            raise KeyError(
                f"{csv_path} does not contain 'train_loss' and 'val_loss' columns."
            )
        train = df["train_loss"].to_numpy(dtype=float)
        val = df["val_loss"].to_numpy(dtype=float)

    # Align lengths if mismatched
    n = min(len(train), len(val))
    train, val = train[:n], val[:n]

    # Best val / epoch
    best_val_loss = float(ckpt.get("best_val_loss", np.nan))
    if not math.isfinite(best_val_loss) and n > 0:
        best_idx = int(np.nanargmin(val))
        best_val_loss = float(val[best_idx])
    best_epoch = int(np.nanargmin(val)) if n > 0 else -1

    iter_num = int(ckpt.get("iter_num", n))

    out: Dict[str, Any] = {
        "train": train,
        "val": val,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "iter_num": iter_num,
    }

    # Optional: reconstruct LR curve from history if present
    if isinstance(hist, list) and len(hist) > 0:
        if "lr" in hist[0]:
            lr_arr = np.asarray([row.get("lr", np.nan) for row in hist], dtype=float)
            out["LR"] = lr_arr[:n]

    return out, train_time, best_val_loss, num_params, device_type


def build_losses_from_history(history: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Build the same losses dict as load_losses_from_ckpt, but from in-memory history.

    history: list of dicts, each with keys "train_loss", "val_loss", optionally "lr".
    """
    if not isinstance(history, list) or len(history) == 0:
        raise ValueError("history must be a non-empty list of dicts.")

    train = np.asarray([row.get("train_loss", np.nan) for row in history], dtype=float)
    val = np.asarray([row.get("val_loss", np.nan) for row in history], dtype=float)
    n = min(len(train), len(val))
    train, val = train[:n], val[:n]

    best_epoch = int(np.nanargmin(val)) if n > 0 else -1
    best_val_loss = float(val[best_epoch]) if best_epoch >= 0 else float("nan")

    out: Dict[str, Any] = {
        "train": train,
        "val": val,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "iter_num": n,
    }

    # optional LR
    if "lr" in history[0]:
        lr_arr = np.asarray([row.get("lr", np.nan) for row in history], dtype=float)
        out["LR"] = lr_arr[:n]

    return out


def load_resources_csv(run_dir: Path, res_csv: str) -> pd.DataFrame | None:
    """
    Load resources.csv if present in run_dir.
    Returns None if not found or unreadable.
    """
    path = run_dir / res_csv
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Failed to read {path}: {e}")
        return None

    # prefer hours if available
    if "t_hours" not in df.columns and "t_seconds" in df.columns:
        df["t_hours"] = df["t_seconds"] / 3600.0

    return df


def simulate_early_stopping(
    val: np.ndarray,
    patience: int,
    tol_rel: float = 0.01,
    eval_interval: int = 1,
    min_epochs: int = 0,
) -> Dict[str, Any]:
    """
    Improvement condition:
        val[t] <= best_so_far * (1 + tol_rel)

    Stops when:
        epoch >= min_epochs and no_improve >= patience
    """
    n = len(val)
    best = np.inf
    best_epoch = -1
    baseline_epoch = -1
    no_improve = 0
    stop_epoch = n - 1
    stopped_due_to_patience = False

    # patience "can" only start after we have a finite best
    patience_active_from_epoch = -1

    for epoch in range(n):
        # mimic eval schedule
        if (epoch % max(1, eval_interval)) != 0:
            continue

        v = float(val[epoch])
        if not math.isfinite(v):
            continue

        if best_epoch < 0:
            best = v
            best_epoch = epoch
            baseline_epoch = epoch
            no_improve = 0
            # earliest epoch where patience counting is meaningful:
            patience_active_from_epoch = max(epoch + eval_interval, int(min_epochs))
            continue

        within = v <= best * (1.0 + tol_rel)
        if within:
            if v < best:
                best = v
                best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch >= int(min_epochs) and no_improve >= int(patience):
            stop_epoch = epoch
            stopped_due_to_patience = True
            break

    return {
        "patience": int(patience),
        "tol_rel": float(tol_rel),
        "eval_interval": int(eval_interval),
        "min_epochs": int(min_epochs),
        "baseline_epoch": int(baseline_epoch),
        "patience_active_from_epoch": int(patience_active_from_epoch),
        "stop_epoch": int(stop_epoch),
        "stopped_due_to_patience": bool(stopped_due_to_patience),
        "best_epoch": int(best_epoch),
        "best_val": float(best) if math.isfinite(best) else np.nan,
    }


def recommend_patience_from_val(
    val: np.ndarray,
    tol_rel: float = 0.01,
    eval_interval: int = 1,
    min_epochs: int = 0,
    patience_grid: Tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50),
    # selection rule:
    max_degrade_rel: float = 0.002,   # allow up to +0.2% worse best_val than full run
) -> Dict[str, Any]:
    """
    Choose the smallest patience that:
      - stops earliest
      - but does NOT worsen best_val by more than max_degrade_rel relative to best over full run.

    Returns:
      {
        "recommended_patience": ...,
        "table": pd.DataFrame([...])
      }
    """
    # full run baseline (patience huge)
    full_best = float(np.nanmin(val)) if np.any(np.isfinite(val)) else np.nan

    rows = []
    sims = []
    for p in patience_grid:
        sim = simulate_early_stopping(
            val=val,
            patience=p,
            tol_rel=tol_rel,
            eval_interval=eval_interval,
            min_epochs=min_epochs,
        )
        sims.append(sim)

        best_p = sim["best_val"]
        # relative degrade vs global best (full run)
        degrade_rel = np.nan
        if np.isfinite(full_best) and full_best > 0 and np.isfinite(best_p):
            degrade_rel = (best_p - full_best) / full_best

        rows.append({
            "patience": p,
            "baseline_epoch": sim["baseline_epoch"],
            "patience_active_from_epoch": sim["patience_active_from_epoch"],
            "stop_epoch": sim["stop_epoch"],
            "best_epoch": sim["best_epoch"],
            "best_val": best_p,
            "best_val_degrade_rel": degrade_rel,
            "epochs_saved_vs_full": (len(val) - 1) - sim["stop_epoch"],
        })

    tab = pd.DataFrame(rows).sort_values(["stop_epoch", "patience"], ascending=[True, True])

    # feasibility: not too much worse than full-run best
    feasible = tab[
        tab["best_val_degrade_rel"].isna() | (tab["best_val_degrade_rel"] <= max_degrade_rel)
    ]

    if len(feasible) == 0:
        # if everything degrades too much, pick the least bad (smallest degrade, then earliest stop)
        tab2 = tab.sort_values(["best_val_degrade_rel", "stop_epoch", "patience"], ascending=[True, True, True])
        rec = int(tab2.iloc[0]["patience"])
        patience_active_from_epoch = int(tab2.iloc[0]["patience_active_from_epoch"])
    else:
        # choose earliest stop among feasible (tie-break by smaller patience)
        rec = int(feasible.iloc[0]["patience"])
        patience_active_from_epoch = int(feasible.iloc[0]["patience_active_from_epoch"])

    return {
        "recommended_patience": rec,
        "patience_active_from_epoch": patience_active_from_epoch,
        "full_run_best_val": full_best,
        "table": tab,
    }


# ---------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------


def plot_losses(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    outdir: Path,
    show: bool,
    logy: bool,
    ma: int,
    dpi: int,
    fmt: str,
    print_flag: bool,
    cfg: Dict[str, Any],
    prefix: str = "",
) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: train/val curves (possibly multiple runs) ---
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    for label, data in series:
        n = len(data["train"])
        epochs = np.arange(1, n + 1)
        train = data["train"]
        val = data["val"]
        if ma and ma > 1:
            train = moving_average(train, ma)
            val = moving_average(val, ma)

        ax1.plot(epochs, train, linestyle="-", linewidth=1.5, label=f"{label} – train")
        ax1.plot(epochs, val, linestyle="--", linewidth=1.5, label=f"{label} – val")

        # Mark best val epoch
        if 0 <= data["best_epoch"] < n and math.isfinite(data["best_val_loss"]):
            be = data["best_epoch"] + 1
            ax1.axvline(be, alpha=0.15)
            ax1.scatter([be], [data["val"][data["best_epoch"]]], s=20)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    if logy:
        ax1.set_yscale("log")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", frameon=True)
    fig1.tight_layout()
    f1 = outdir / f"{prefix}_loss_curves.{fmt}"
    fig1.savefig(f1, dpi=dpi, format=fmt)

    # --- Figure 2: box plots (train vs val per run) ---
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    all_boxes = []
    labels = []
    for label, data in series:
        all_boxes += [data["train"], data["val"]]
        labels += [f"{label} – train", f"{label} – val"]
    ax2.boxplot(all_boxes, tick_labels=labels, showmeans=True, meanline=True)
    ax2.set_ylabel("Loss")
    if logy:
        ax2.set_yscale("log")
    ax2.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    f2 = outdir / f"{prefix}_loss_boxplots.{fmt}"
    fig2.savefig(f2, dpi=dpi, format=fmt)


    # ---- Suggest patience based on val trajectory ----
    val_arr = np.asarray(data["val"], dtype=float)
    eval_interval = int(cfg.get("training", {}).get("eval_interval", 1))
    tol_rel = float(cfg.get("training", {}).get("early_stop_tol_rel", 0.01))  # add to yaml if you want
    min_epochs = int(cfg.get("training", {}).get("min_epochs", 0))

    rec = recommend_patience_from_val(
        val=val_arr,
        tol_rel=tol_rel,
        eval_interval=eval_interval,
        min_epochs=min_epochs,
        patience_grid=(20, 30, 50, 100, 150, 300, 500, 750, 1000, 1200, 2000, 5000),
        max_degrade_rel=0.002,
    )

    if print_flag:
        print("\n[patience-suggest]")
        print(f"  tol_rel={tol_rel}  eval_interval={eval_interval}  min_epochs={min_epochs}")
        print(f"  full-run best val: {rec['full_run_best_val']:.6g}")
        print(f"  recommended patience: {rec['recommended_patience']}")
        print(f"  (patience_active_from_epoch epoch: {rec['patience_active_from_epoch']})")
        # optional: dump table
        # print(rec["table"].to_string(index=False))


    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
    return f1, f2


def export_csv(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    outdir: Path,
    prefix: str = "",
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    max_len = max(len(d["train"]) for _, d in series) if series else 0
    epochs = np.arange(1, max_len + 1)
    cols = ["epoch"]
    arrs = [epochs.reshape(-1, 1)]
    for label, data in series:
        n = len(data["train"])
        t_pad = np.full((max_len,), np.nan)
        v_pad = np.full((max_len,), np.nan)
        t_pad[:n] = data["train"]
        v_pad[:n] = data["val"]
        arrs.append(t_pad.reshape(-1, 1))
        cols.append(f"{label}_train")
        arrs.append(v_pad.reshape(-1, 1))
        cols.append(f"{label}_val")

    table = np.hstack(arrs) if arrs else np.empty((0, 0))
    out_csv = outdir / f"{prefix}_loss_history.csv"

    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for i in range(table.shape[0]):
            row = []
            for j in range(table.shape[1]):
                val = table[i, j]
                row.append("" if np.isnan(val) else f"{val:.10g}")
            f.write(",".join(row) + "\n")
    return out_csv


def plot_generalization_gap(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    outdir: Path,
    logy: bool,
    dpi: int,
    fmt: str,
    prefix: str = "",
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for label, data in series:
        n = len(data["train"])
        epochs = np.arange(1, n + 1)
        gap = data["val"] - data["train"]
        ax.plot(epochs, gap, linewidth=1.5, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Generalization gap (val - train)")
    if logy:
        # gap can be negative; log scale would be invalid, so skip
        pass
    ax.axhline(0.0, linestyle=":", linewidth=1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    f = outdir / f"{prefix}_gap_val_minus_train.{fmt}"
    fig.savefig(f, dpi=dpi, format=fmt)
    plt.close(fig)
    return f


def plot_running_best_val(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    outdir: Path,
    logy: bool,
    dpi: int,
    fmt: str,
    prefix: str = "",
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for label, data in series:
        val = data["val"]
        runmin = np.minimum.accumulate(val)
        epochs = np.arange(1, len(val) + 1)
        ax.plot(epochs, runmin, linewidth=1.5, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Running best validation loss")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    f = outdir / f"{prefix}_running_best_val.{fmt}"
    fig.savefig(f, dpi=dpi, format=fmt)
    plt.close(fig)
    return f


def plot_train_vs_val_scatter(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    outdir: Path,
    logxy: bool,
    dpi: int,
    fmt: str,
    prefix: str = "",
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    all_train = np.concatenate([d["train"] for _, d in series])
    all_val = np.concatenate([d["val"] for _, d in series])
    lo = float(np.nanmin([all_train.min(), all_val.min()]))
    hi = float(np.nanmax([all_train.max(), all_val.max()]))

    ax.plot([lo, hi], [lo, hi], linestyle=":", linewidth=1.0)  # y=x
    for label, data in series:
        ax.scatter(data["train"], data["val"], s=10, alpha=0.6, label=label)
    ax.set_xlabel("Train loss per epoch")
    ax.set_ylabel("Validation loss per epoch")
    if logxy:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    f = outdir / f"{prefix}_scatter_train_vs_val.{fmt}"
    fig.savefig(f, dpi=dpi, format=fmt)
    plt.close(fig)
    return f


def plot_val_improvement(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    outdir: Path,
    percent: bool,
    dpi: int,
    fmt: str,
    prefix: str = "",
) -> Path:
    """Plot per-epoch improvement: Δval (or %) where Δ = val[t] - val[t-1] (negative is good)."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for label, data in series:
        v = data["val"]
        if len(v) < 2:
            continue
        dv = np.diff(v)
        y = (dv / v[:-1] * 100.0) if percent else dv
        epochs = np.arange(2, len(v) + 1)
        ax.plot(epochs, y, linewidth=1.2, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ validation loss" + (" (%)" if percent else ""))
    ax.axhline(0.0, linestyle=":", linewidth=1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    suffix = "_val_improvement_pct" if percent else "_val_improvement"
    f = outdir / f"{prefix}{suffix}.{fmt}"
    fig.savefig(f, dpi=dpi, format=fmt)
    plt.close(fig)
    return f


def maybe_plot_optional_curves(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    key: str,
    ylabel: str,
    fname: str,
    outdir: Path,
    logy: bool,
    dpi: int,
    fmt: str,
    prefix: str = "",
) -> Path | None:
    """If checkpoints include an array under `key`, plot it vs epoch."""
    have_any = any(key in d for _, d in series)
    if not have_any:
        return None

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for label, data in series:
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=float)
        epochs = np.arange(1, len(arr) + 1)
        ax.plot(epochs, arr, linewidth=1.2, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / f"{prefix}_{fname}.{fmt}"
    fig.savefig(f, dpi=dpi, format=fmt)
    plt.close(fig)
    return f


def plot_resource_timeseries(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    outdir: Path,
    ylabel: str,
    title: str,
    fname: str,
    dpi: int,
    fmt: str,
):
    outdir.mkdir(parents=True, exist_ok=True)

    if x_col not in df.columns or y_col not in df.columns:
        return None

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    if np.all(~np.isfinite(y)):
        return None

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.plot(x, y, linewidth=1.2)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = outdir / f"{fname}.{fmt}"
    fig.savefig(path, dpi=dpi, format=fmt)
    plt.close(fig)
    return path


def plot_resources_over_time(
    df: pd.DataFrame,
    outdir: Path,
    dpi: int,
    fmt: str,
    prefix: str = "",
) -> List[Path]:
    """
    Plot key CPU / RAM / GPU metrics vs time.
    Returns list of saved plot paths.
    """
    saved: List[Path] = []
    x = "t_hours"

    specs = [
        # ---- GPU ----
        ("gpu_util_percent", "GPU Utilization (%)", "GPU Utilization", "gpu_util"),
        ("gpu_mem_used_mb", "GPU Memory Used (MB)", "GPU Memory Usage", "gpu_mem_used"),
        ("gpu_mem_used_percent", "GPU Memory Used (%)", "GPU Memory %", "gpu_mem_pct"),
        ("gpu_temp_c", "GPU Temperature (°C)", "GPU Temperature", "gpu_temp"),
        ("gpu_power_w", "GPU Power (W)", "GPU Power Usage", "gpu_power"),
        ("gpu_sm_clock_mhz", "GPU SM Clock (MHz)", "GPU SM Clock", "gpu_sm_clock"),
        ("gpu_mem_clock_mhz", "GPU Memory Clock (MHz)", "GPU Memory Clock", "gpu_mem_clock"),

        # ---- CPU / process ----
        ("cpu_percent", "CPU Utilization (%)", "System CPU Utilization", "cpu_util"),
        ("proc_cpu_percent", "Process CPU (%)", "Process CPU Utilization", "proc_cpu"),
        ("proc_threads", "Threads", "Process Threads", "proc_threads"),

        # ---- RAM ----
        ("ram_percent", "RAM Utilization (%)", "System RAM Usage", "ram_pct"),
        ("proc_rss_mb", "Process RSS (MB)", "Process Memory RSS", "proc_rss"),

        # ---- IO ----
        ("disk_read_bytes", "Disk Read (bytes)", "Disk Read", "disk_read"),
        ("disk_write_bytes", "Disk Write (bytes)", "Disk Write", "disk_write"),
        ("net_recv_bytes", "Network RX (bytes)", "Network Receive", "net_rx"),
        ("net_sent_bytes", "Network TX (bytes)", "Network Transmit", "net_tx"),
    ]

    for col, ylabel, title, stem in specs:
        p = plot_resource_timeseries(
            df=df,
            x_col=x,
            y_col=col,
            outdir=outdir,
            ylabel=ylabel,
            title=title,
            fname=f"{prefix}_resource_{stem}",
            dpi=dpi,
            fmt=fmt,
        )
        if p is not None:
            saved.append(p)

    return saved


# ---------------------------------------------------------------------
# Modular entry point: can be called from training or CLI
# ---------------------------------------------------------------------


def run_training_plots(
    run_dir: Path | str,
    cfg: Dict[str, Any] | None = None,
    ckpt_name: str | None = "test_best.pt",
    history: List[Dict[str, Any]] | None = None,
    train_time: float = None,
    best_val_loss: float = None,
    num_params: int = None,
    device_type: str = None,
    label: str | None = None,
    show: bool | None = None,
    prnt: bool = True,
):
    """
    Main plotting entry point.

    Modes:
      - If `history` is provided (from training code):
            * losses are built from in-memory history (list of dicts)
            * ckpt_name is ignored

      - Else:
            * losses are loaded from `run_dir / ckpt_name` using load_losses_from_ckpt

    Parameters
    ----------
    run_dir : Path | str
        Directory of the run (contains config_used.yaml, history.csv, etc.).
    cfg : dict | None
        Config used for the run. If None, tries to load config_used.yaml.
    ckpt_name : str | None
        Checkpoint filename relative to run_dir (default: "test_best.pt").
    history : list[dict] | None
        Training history as built in run_experiment.py.
    label : str | None
        Optional label for plots. If None, uses checkpoint stem.
    show : bool | None
        If True, show plots. If None, uses cfg["plot"]["show"] or False.
    """
    run_dir = Path(run_dir).expanduser().resolve()

    # Config
    if cfg is None:
        cfg = load_cfg_used(run_dir)
        if cfg is None:
            ckpt_name = "test_best.pt"
            cfg = load_cfg_from_ckpt(run_dir / ckpt_name)
    plot_cfg = cfg.get("plot", {})

    # plotting options
    output_subdir = plot_cfg.get("output_subdir", "plots")
    logy = bool(plot_cfg.get("logy", False))
    ma = int(plot_cfg.get("ma", 1))
    dpi = int(plot_cfg.get("dpi", 160))
    export_csv_flag = bool(plot_cfg.get("export_csv", False))
    fmt = str(plot_cfg.get("fmt", "pdf"))

    if Path(output_subdir).is_absolute():
        outdir = Path(output_subdir)
    else:
        outdir = run_dir / output_subdir
    outdir = outdir / "train"

    # show override
    show_flag = bool(plot_cfg.get("show", False)) if show is None else bool(show)
    print_flag = bool(plot_cfg.get("print", False)) if prnt is None else bool(prnt)

    # Load losses
    if history is not None:
        data = build_losses_from_history(history)
        ckpt_path = run_dir / (ckpt_name or "from_history")
        label_final = label or ckpt_path.stem
        if print_flag: print(f"[plot] Using in-memory history, label={label_final}")
    else:
        ckpt_path = run_dir / ckpt_name
        if print_flag: print(f"[plot] Using run directory  : {run_dir}")
        if print_flag: print(f"[plot] Using checkpoint file: {ckpt_path}")
        try:
            data, train_time, best_val_loss, num_params, device_type = \
                  load_losses_from_ckpt(ckpt_path)
        except Exception as e:
            if print_flag: print(f"ERROR loading {ckpt_path}: {e}", file=sys.stderr)
            sys.exit(1)
        label_final = label or ckpt_path.stem

    series: List[Tuple[str, Dict[str, np.ndarray]]] = [(label_final, data)]

    # prefix: single run -> use stem
    prefix = f"{series[0][0]}" if len(series) == 1 else "_multi"

    # 1) Loss curves + boxplots
    f1, f2 = plot_losses(
        series,
        outdir,
        show=show_flag,
        logy=logy,
        ma=ma,
        dpi=dpi,
        fmt=fmt,
        prefix=prefix,
        print_flag=print_flag,
        cfg=cfg,
    )
    if print_flag: print(f"[OK] Saved figures:\n  - {f1}\n  - {f2}")

    # 2) Optional CSV export
    if export_csv_flag:
        csv_path = export_csv(series, outdir, prefix=prefix)
        if print_flag: print(f"[OK] Saved CSV: {csv_path}")

    # 3) Other diagnostics
    f_gap = plot_generalization_gap(series, outdir, logy=logy, dpi=dpi, fmt=fmt, prefix=prefix)
    if print_flag: print(f"  - {f_gap}")
    f_runbest = plot_running_best_val(series, outdir, logy=logy, dpi=dpi, fmt=fmt, prefix=prefix)
    if print_flag: print(f"  - {f_runbest}")
    f_scatter = plot_train_vs_val_scatter(series, outdir, logxy=logy, dpi=dpi, fmt=fmt, prefix=prefix)
    if print_flag: print(f"  - {f_scatter}")
    f_dv = plot_val_improvement(series, outdir, percent=False, dpi=dpi, fmt=fmt, prefix=prefix)
    if print_flag: print(f"  - {f_dv}")
    f_dv_pct = plot_val_improvement(series, outdir, percent=True, dpi=dpi, fmt=fmt, prefix=prefix)
    if print_flag: print(f"  - {f_dv_pct}")

    # 4) Optional curves if present in checkpoints/history
    f_lr = maybe_plot_optional_curves(
        series,
        key="LR",
        ylabel="Learning rate",
        fname="lr_curve",
        outdir=outdir,
        logy=logy,
        dpi=dpi,
        fmt=fmt,
        prefix=prefix,
    )
    if f_lr is not None:
        if print_flag: print(f"  - {f_lr}")

    f_gn = maybe_plot_optional_curves(
        series,
        key="GRAD_NORM",
        ylabel="Gradient L2 norm",
        fname="grad_norm",
        outdir=outdir,
        logy=logy,
        dpi=dpi,
        fmt=fmt,
        prefix=prefix,
    )
    if f_gn is not None:
        if print_flag: print(f"  - {f_gn}")

    f_pn = maybe_plot_optional_curves(
        series,
        key="PARAM_NORM",
        ylabel="Parameter L2 norm",
        fname="param_norm",
        outdir=outdir,
        logy=logy,
        dpi=dpi,
        fmt=fmt,
        prefix=prefix,
    )
    if f_pn is not None:
        if print_flag: print(f"  - {f_pn}")

    f_et = maybe_plot_optional_curves(
        series,
        key="EPOCH_TIME",
        ylabel="Epoch time (s)",
        fname="epoch_time",
        outdir=outdir,
        logy=False,
        dpi=dpi,
        fmt=fmt,
        prefix=prefix,
    )
    if f_et is not None:
        if print_flag: print(f"  - {f_et}")
    
    train_time = int(train_time)

    days, rem = divmod(train_time, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)


    # -------------------------------------------------
    # Resource usage plots (CPU / RAM / GPU)
    # -------------------------------------------------
    res_csv = cfg.get("experiment", {}).get("resources", "resources.csv")
    df_res = load_resources_csv(run_dir, res_csv)
    if df_res is not None:
        res_outdir = outdir / "resources"
        res_plots = plot_resources_over_time(
            df=df_res,
            outdir=res_outdir,
            dpi=dpi,
            fmt=fmt,
            prefix=prefix,
        )
        if print_flag and res_plots:
            print("[OK] Saved resource plots:")
            for p in res_plots:
                print(f"  - {p}")


    run_name = cfg["experiment"]["name"]
    print(f"\n\n"
          f"Run: {run_name}\n"
          f"Best validation loss: {best_val_loss}\n"
          f"Number of parameters: {num_params}\n"
          f"Device type: {device_type}\n"
          f"Training time: {days:2d}d {hours:2d}h {minutes:2d}m {seconds:2d}s")


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot training history from a run directory.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Run directory containing config_used.yaml and checkpoint",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="test_last.pt",
        help="Checkpoint filename inside the run directory (default: test_best.pt)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of just saving them.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print debug information.",
    )

    args = parser.parse_args()
    run_dir = Path(args.dir).expanduser().resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # CLI mode: let run_training_plots handle cfg + checkpoint
    run_training_plots(
        run_dir=run_dir,
        cfg=None,
        ckpt_name=args.ckpt,
        history=None,
        train_time=None,
        label=None,
        show=args.show,
        prnt=args.print,
    )


if __name__ == "__main__":
    # >> python.exe .\plot_training.py --dir runs/{name_run}
    # >> python.exe .\plot_training.py --dir runs/sweeps/{name_sweep}
    main()
