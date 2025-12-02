# plot_testing.py
"""
Evaluation / testing plots for zero-step Transformer runs.

Usage example:
    python plot_testing.py --dir runs/test_run
    python plot_testing.py --dir sweeps/my_sweep_run --split val --n-exps 5

Contract:
  - --dir must contain:
        config_used.yaml
        <checkpoint_stem>_best.pt   (saved by run_experiment.py)
  - config_used.yaml must contain the full cfg dict used at training:
        {experiment, data, model, training, compute, logging, plot, ...}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml, sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_datasets
from run_experiment import build_device, build_model
from dataset import reverse_normalization


# ---------------------------------------------------------------------
# Config & model loading
# ---------------------------------------------------------------------


def load_cfg_used(run_dir: Path) -> Dict[str, Any]:
    """
    Load config_used.yaml from a run directory.
    """
    cfg_path = run_dir / "config_used.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found. You must pass a run dir with config_used.yaml."
        )
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def build_model_from_cfg(cfg: Dict[str, Any], device, device_type: str) -> nn.Module:
    """
    Rebuild the GPT model exactly as in run_experiment.py, using config_used.yaml.
    """
    cfg_data = cfg["data"]
    cfg_model = dict(cfg["model"])
    cfg_compute = cfg["compute"]

    # Attach compile flag as in run_experiment
    cfg_model["compile"] = cfg_compute.get("compile", False)

    model, _ = build_model(cfg_model, cfg_data, device, device_type)
    return model


def load_checkpoint_model(
    run_dir: Path,
    cfg: Dict[str, Any],
    device,
    device_type: str,
    ckpt_name: str | None = None,
) -> nn.Module:
    """
    Build model from cfg and load weights from checkpoint.

    By default, uses "<checkpoint_stem>_best.pt" where checkpoint_stem
    is in cfg["logging"]["checkpoint_stem"] (default "test").
    """
    cfg_logging = cfg.get("logging", {})
    stem = cfg_logging.get("checkpoint_stem", "test")
    if ckpt_name is None:
        ckpt_name = f"{stem}_best.pt"

    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")

    print(f"[test] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model_from_cfg(cfg, device, device_type)
    state_dict = ckpt.get("model", ckpt)

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    return model


def prepare_datasets(cfg: Dict[str, Any]):
    """
    Reload datasets exactly as in train_zerostep.py via data_utils.load_datasets.
    """
    cfg_data = cfg["data"]
    train_ds, val_ds = load_datasets(cfg_data)
    return train_ds, val_ds


# ---------------------------------------------------------------------
# Closed-loop ICL-style testing (sliding window, respects block_size)
# ---------------------------------------------------------------------


def run_model_on_experiment(
    model: nn.Module,
    ds,
    idx: int,
    device,
):
    """
    Exact single-experiment port of the old notebook logic.

    Returns:
        t, y_true_den, y_hat_den, u_den  (all numpy)
    """
    # 1) Get full normalized experiment
    u_full, y_full = ds.get_full_experiment(idx)   # (T,5), (T,1)
    u_full = u_full.to(device)
    y_full = y_full.to(device)

    T = y_full.shape[0]
    H = getattr(ds, "seq_len", None)
    if H is None:
        raise AttributeError("Dataset has no attribute 'seq_len'.")
    H = min(H, T)

    # 2) Add fake batch dimension so shapes match the notebook
    df_len = 1
    u_full_all = u_full.unsqueeze(0)        # (1, T, 5)
    y_full_all = y_full.unsqueeze(0)        # (1, T, 1)
    y_pred_all = torch.zeros_like(y_full_all, device=device)  # (1, T, 1)

    # 3) Same last_omega logic as notebook
    last_omega = torch.zeros((df_len, H, 1), device=device)

    with torch.no_grad():
        for j in range(T):
            if j < H:
                # (1, j+1, 5)
                input_val = u_full_all[:, :j+1, :].clone()
                # fill last_omega for the available j+1 steps
                input_val[:, :j+1, 4] = last_omega[:, -j-1:, 0]
            else:
                # (1, H, 5)
                input_val = u_full_all[:, j-H+1:j+1, :].clone()
                input_val[:, :, 4] = last_omega[:, :, 0]

            pred = model(input_val)[:, -1, :]          # (1,1)
            y_pred_all[:, j, 0] = pred[:, 0]

            last_omega = torch.roll(last_omega, shifts=-1, dims=1)
            last_omega[:, -1, 0] = y_pred_all[:, j, 0]

    # 4) Denormalize exactly as in the notebook
    u_den, y_den, y_pred_den = reverse_normalization(
        u_full_all.cpu(), y_full_all.cpu(), y_pred_all.cpu()
    )

    t = np.arange(T)
    u_den = u_den[0].numpy()              # (T,5)
    y_true = y_den[0, :, 0].numpy()       # (T,)
    y_hat = y_pred_den[0, :, 0].numpy()   # (T,)

    return t, y_true, y_hat, u_den


# ---------------------------------------------------------------------
# Plotting (3-panel figure like the original notebook)
# ---------------------------------------------------------------------


def plot_experiment_timeseries(
    t: np.ndarray,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    u_den: np.ndarray,
    outdir: Path,
    idx: int,
    dpi: int,
    fmt: str,
    split: str,
) -> Path:
    """
    Plot one experiment's time series and save to disk.

    Layout:
      - Top    : ω_real, ω_est (Transformer)
      - Middle : ia, ib
      - Bottom : va, vb
    """
    outdir.mkdir(parents=True, exist_ok=True)

    ia = u_den[:, 0]
    ib = u_den[:, 1]
    va = u_den[:, 2]
    vb = u_den[:, 3]

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(10, 4), sharex=True
    )

    # Top: speed
    ax0.plot(t, y_true, label=r"$\omega_{\mathrm{real}}$", linewidth=1.0)
    ax0.plot(t, y_hat, label=r"$\omega_{\mathrm{est}}$", linewidth=1.0, alpha=0.9)
    ax0.set_ylabel(r"$\omega$ [rad/s]")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")

    # Middle: currents
    ax1.plot(t, ia, label=r"$I_\alpha$", linewidth=0.8)
    ax1.plot(t, ib, label=r"$I_\beta$", linewidth=0.8, alpha=0.9)
    ax1.set_ylabel("Current [A]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Bottom: voltages
    ax2.plot(t, va, label=r"$V_\alpha$", linewidth=0.8)
    ax2.plot(t, vb, label=r"$V_\beta$", linewidth=0.8, alpha=0.9)
    ax2.set_ylabel("Voltage [V]")
    ax2.set_xlabel("Sample")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    fig.tight_layout()

    fname = outdir / f"{split}_experiment_{idx:03d}.{fmt}"
    fig.savefig(fname, dpi=dpi, format=fmt)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot testing / evaluation trajectories for a run."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Run directory containing config_used.yaml and checkpoint (e.g. runs/test_run)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="Which dataset split to use for experiments (default: val).",
    )
    parser.add_argument(
        "--n-exps",
        type=int,
        default=10,
        help="Number of experiments to plot from the chosen split.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="test_best.pt",
        help="Optional checkpoint filename inside --dir (default: '<stem>_best.pt').",
    )

    args = parser.parse_args()
    run_dir = Path(args.dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist.")

    # Load config_used.yaml
    cfg = load_cfg_used(run_dir)

    # Plot options
    cfg_plot = cfg.get("plot", {})
    dpi = int(cfg_plot.get("dpi", 160))
    fmt = cfg_plot.get("fmt", "pdf")
    output_subdir = cfg_plot.get("output_subdir", "plots")
    outdir = run_dir / output_subdir / "testing"

    # Device + model
    cfg_compute = cfg["compute"]
    device, device_type = build_device(cfg_compute)
    model = load_checkpoint_model(
        run_dir, cfg, device, device_type, ckpt_name=args.ckpt
    )

    # Datasets
    train_ds, val_ds = prepare_datasets(cfg)
    ds = train_ds if args.split == "train" else val_ds

    # How many experiments are actually available
    n_available = len(ds.dfs)
    n_exps = min(args.n_exps, n_available)
    if n_exps <= 0:
        raise RuntimeError(
            f"No experiments available in {args.split} dataset (len(dfs)={n_available})."
        )

    print(
        f"[test] Using split={args.split}, plotting {n_exps} experiments (out of {n_available})."
    )

    # Loop over experiments
    for idx in range(n_exps):
        t, y_true, y_hat, u_den = run_model_on_experiment(model, ds, idx, device)

        mask = np.isfinite(y_true) & np.isfinite(y_hat)
        mse = float(np.mean((y_hat[mask] - y_true[mask]) ** 2)) if np.any(mask) else float(
            "nan"
        )
        mse = float(((y_true - y_hat)**2).mean().item())
        rmse = np.sqrt(mse)

        print(f"[test] exp {idx:03d}: MSE={mse:.4e}, RMSE={rmse}, T={len(t)}")

        fname = plot_experiment_timeseries(
            t,
            y_true,
            y_hat,
            u_den,
            outdir,
            idx,
            dpi,
            fmt,
            split=args.split,
        )
        print(f"[test]   -> saved {fname}")



if __name__ == "__main__":
    # >> python.exe .\plot_testing.py --dir runs/{name_run}
    # >> python.exe .\plot_testing.py --dir sweeps/{name_sweep}
    main()
