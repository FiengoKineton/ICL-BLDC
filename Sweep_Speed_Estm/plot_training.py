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
    best_val_loss = ckpt.get("best_val_loss", None)

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

    return out, train_time, best_val_loss


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
    label: str | None = None,
    show: bool | None = None,
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
    if show is None:
        show_flag = bool(plot_cfg.get("show", False))
        print_flag = bool(plot_cfg.get("print", False))
    else:
        show_flag = bool(show)
        print_flag = show_flag

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
            data, train_time, best_val_loss = load_losses_from_ckpt(ckpt_path)
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


    run_name = cfg["experiment"]["name"]
    print(f"\n\n"
          f"Run: {run_name}\n"
          f"Best validation loss: {best_val_loss}\n"
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
    )


if __name__ == "__main__":
    # >> python.exe .\plot_training.py --dir runs/{name_run}
    # >> python.exe .\plot_training.py --dir sweeps/{name_sweep}
    main()
