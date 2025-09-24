# plot_training.py
"""
Read one or more PyTorch checkpoints (.pt) that contain training history and
produce:
  1) Loss curves (train & val on the same axes)
  2) Box plots for train/val loss distributions
  3) (Optional) CSV export of the loss histories

Expected checkpoint structure (as in your training loop):
{
  'LOSS': [train_loss_0, train_loss_1, ...],
  'LOSS_VAL': [val_loss_0, val_loss_1, ...],
  'best_val_loss': float,
  'iter_num': int,
  'cfg': argparse.Namespace or dict,
  ... (model, optimizer, etc.)
}

Usage examples
--------------
# Single checkpoint, save PNGs in ./plots and open an interactive window
python plot_training.py out/my_model.pt --show

# Multiple checkpoints compared on one figure
python plot_training.py out/best.pt out/last.pt -o plots_compare

# With moving-average smoothing and log-scale on Y
python plot_training.py out/best.pt --ma 5 --logy

# Export CSV next to the figures
python plot_training.py out/best.pt --export-csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import math
import sys

try:
    import torch
except Exception as e:
    print("ERROR: torch is required to load .pt checkpoints.\n"
          "Install with: pip install torch", file=sys.stderr)
    raise

import numpy as np
import matplotlib.pyplot as plt


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Simple centered moving average with window w (odd preferred)."""
    if w <= 1 or w > len(x):
        return x
    # Use convolution; 'same' pads equally on both sides
    kernel = np.ones(w, dtype=float) / w
    y = np.convolve(x, kernel, mode="same")
    # Edge correction (avoid underweight at boundaries)
    # Recompute for edges with smaller effective window
    half = w // 2
    for i in range(half):
        y[i] = x[: i + half + 1].mean()
        y[-(i + 1)] = x[-(i + half + 1):].mean()
    return y


def load_losses(path: Path) -> Dict[str, np.ndarray]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "LOSS" not in ckpt or "LOSS_VAL" not in ckpt:
        raise KeyError(f"{path} does not contain 'LOSS' and 'LOSS_VAL' lists.")
    train = np.asarray(ckpt["LOSS"], dtype=float)
    val = np.asarray(ckpt["LOSS_VAL"], dtype=float)
    # Align lengths if mismatched
    n = min(len(train), len(val))
    train, val = train[:n], val[:n]
    best_val_loss = float(ckpt.get("best_val_loss", np.nan))
    best_epoch = int(np.nanargmin(val)) if len(val) else -1
    iter_num = int(ckpt.get("iter_num", n))
    out = {
        "train": train,
        "val": val,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "iter_num": iter_num,
    }

    # Optional histories if you log them during training (see note below)
    for k in ("LR", "GRAD_NORM", "PARAM_NORM", "EPOCH_TIME"):
        if k in ckpt and isinstance(ckpt[k], (list, tuple)):
            arr = np.asarray(ckpt[k], dtype=float)
            out[k] = arr[:n]  # align to epochs length
    return out


def plot_losses(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    outdir: Path,
    show: bool,
    logy: bool,
    ma: int,
    dpi: int,
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
            ax1.scatter([be], [data["val"][data['best_epoch']]], s=20)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    if logy:
        ax1.set_yscale("log")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", frameon=True)
    fig1.tight_layout()
    f1 = outdir / f"{prefix}_loss_curves.pdf"
    fig1.savefig(f1, dpi=dpi)

    # --- Figure 2: box plots (train vs val per run) ---
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    all_boxes = []
    labels = []
    for label, data in series:
        all_boxes += [data["train"], data["val"]]
        labels += [f"{label} – train", f"{label} – val"]
    bp = ax2.boxplot(all_boxes, labels=labels, showmeans=True, meanline=True)
    ax2.set_ylabel("Loss")
    if logy:
        ax2.set_yscale("log")
    ax2.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    f2 = outdir / f"{prefix}_loss_boxplots.pdf"
    fig2.savefig(f2, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
    return f1, f2


def export_csv(series: List[Tuple[str, Dict[str, np.ndarray]]], outdir: Path, prefix: str = "") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    # Write a simple wide CSV: columns per run/phase
    # Header: epoch, <run>-train, <run>-val, ...
    # Epochs aligned to min length per run already.
    max_len = max(len(d["train"]) for _, d in series) if series else 0
    epochs = np.arange(1, max_len + 1)
    cols = ["epoch"]
    arrs = [epochs.reshape(-1, 1)]
    for label, data in series:
        n = len(data["train"])
        # pad with NaN to max_len for consistent width
        t_pad = np.full((max_len,), np.nan); t_pad[:n] = data["train"]
        v_pad = np.full((max_len,), np.nan); v_pad[:n] = data["val"]
        arrs.append(t_pad.reshape(-1, 1)); cols.append(f"{label}_train")
        arrs.append(v_pad.reshape(-1, 1)); cols.append(f"{label}_val")
    table = np.hstack(arrs) if arrs else np.empty((0, 0))

    # Manual CSV (avoid pandas dependency)
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
    outdir: Path, logy: bool, dpi: int, prefix: str = ""
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
        # gap can be negative; avoid log scale if sign-flipping is possible
        pass
    ax.axhline(0.0, linestyle=":", linewidth=1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    f = outdir / f"{prefix}_gap_val_minus_train.pdf"
    fig.savefig(f, dpi=dpi); plt.close(fig)
    return f

def plot_running_best_val(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    outdir: Path, logy: bool, dpi: int, prefix: str = ""
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
    f = outdir / f"{prefix}_running_best_val.pdf"
    fig.savefig(f, dpi=dpi); plt.close(fig)
    return f

def plot_train_vs_val_scatter(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    outdir: Path, logxy: bool, dpi: int, prefix: str = ""
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    # Determine a global range for y=x
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
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    f = outdir / f"{prefix}_scatter_train_vs_val.pdf"
    fig.savefig(f, dpi=dpi); plt.close(fig)
    return f

def plot_val_improvement(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    outdir: Path, percent: bool, dpi: int, prefix: str = ""
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
    f = outdir / f"{prefix}_val_improvement{'_pct' if percent else ''}.pdf"
    fig.savefig(f, dpi=dpi); plt.close(fig)
    return f

def maybe_plot_optional_curves(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    key: str, ylabel: str, fname: str,
    outdir: Path, logy: bool, dpi: int, prefix: str = ""
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
    f = outdir / f"{prefix}_{fname}.pdf"
    fig.savefig(f, dpi=dpi); plt.close(fig)
    return f


def main():
    p = argparse.ArgumentParser(description="Plot train/val loss curves and box plots from .pt checkpoints")
    p.add_argument("checkpoints", nargs="+", type=Path, help="Path(s) to .pt file(s)")
    p.add_argument("-o", "--outdir", type=Path, default=Path("speed_estimator\plots"), help="Output directory for figures/CSV")
    p.add_argument("--show", action="store_true", help="Show interactive windows")
    p.add_argument("--logy", action="store_true", help="Log scale on Y axis")
    p.add_argument("--ma", type=int, default=1, help="Moving-average window (epochs); 1 disables smoothing")
    p.add_argument("--dpi", type=int, default=160, help="Figure resolution")
    p.add_argument("--export-csv", action="store_true", help="Also export a CSV with loss histories")
    
    p.add_argument("--fmt", default="pdf", choices=["pdf","png","svg"], help="Output image format per-figure (default: pdf)")
    # ext = args.fmt, f1 = outdir / f"loss_curves{suffix}.{ext}", fig1.savefig(f1, format=ext)

    args = p.parse_args()

    series: List[Tuple[str, Dict[str, np.ndarray]]] = []
    print(args.checkpoints)
    for ckpt_path in args.checkpoints:
        if not ckpt_path.exists():
            print(f"WARNING: {ckpt_path} not found; skipping.", file=sys.stderr)
            continue
        try:
            data = load_losses(ckpt_path)
            label = ckpt_path.stem
            series.append((label, data))
        except Exception as e:
            print(f"ERROR loading {ckpt_path}: {e}", file=sys.stderr)

    if not series:
        print("No valid checkpoints loaded. Nothing to plot.", file=sys.stderr)
        sys.exit(2)

    # If exactly one run, prefix with its stem; else use '_multi'
    prefix = f"{series[0][0]}" if len(series) == 1 else "_multi"

    f1, f2 = plot_losses(series, args.outdir, args.show, args.logy, args.ma, args.dpi, prefix=prefix)
    print(f"[OK] Saved figures:\n  - {f1}\n  - {f2}")

    if args.export_csv:
        csv_path = export_csv(series, args.outdir, prefix=prefix)
        print(f"[OK] Saved CSV: {csv_path}")

    f_gap = plot_generalization_gap(series, args.outdir, args.logy, args.dpi, prefix=prefix)
    print(f"  - {f_gap}")
    f_runbest = plot_running_best_val(series, args.outdir, args.logy, args.dpi, prefix=prefix)
    print(f"  - {f_runbest}")
    f_scatter = plot_train_vs_val_scatter(series, args.outdir, logxy=args.logy, dpi=args.dpi, prefix=prefix)
    print(f"  - {f_scatter}")
    f_dv = plot_val_improvement(series, args.outdir, percent=False, dpi=args.dpi, prefix=prefix)
    print(f"  - {f_dv}")
    f_dv_pct = plot_val_improvement(series, args.outdir, percent=True, dpi=args.dpi, prefix=prefix)
    print(f"  - {f_dv_pct}")

    # Optional curves if present in checkpoints
    f_lr  = maybe_plot_optional_curves(series, "LR", "Learning rate", "lr_curve",
                                       args.outdir, args.logy, args.dpi, prefix=prefix)
    f_gn  = maybe_plot_optional_curves(series, "GRAD_NORM", "Gradient L2 norm", "grad_norm",
                                       args.outdir, args.logy, args.dpi, prefix=prefix)
    f_pn  = maybe_plot_optional_curves(series, "PARAM_NORM", "Parameter L2 norm", "param_norm",
                                       args.outdir, args.logy, args.dpi, prefix=prefix)
    f_et  = maybe_plot_optional_curves(series, "EPOCH_TIME", "Epoch time (s)", "epoch_time",
                                       args.outdir, False, args.dpi, prefix=prefix)


if __name__ == "__main__":
    # >> python.exe .\speed_estimator\plot_training.py .\out\test.pt --show --logy
    main()
