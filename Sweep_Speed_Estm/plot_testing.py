# plot_testing.py       | SET (leave it as it is)
   
"""
Evaluation / testing plots for zero-step Transformer runs.

Usage examples:
    # From CLI (uses saved config + checkpoint)
    python plot_testing.py --dir runs/test_run --split test --n-exps 5

    # From training code (after training completes)
    from plot_testing import run_testing
    run_testing(
        run_dir=run_dir,
        split="test",
        n_exps=5,
        cfg=cfg,
        model=model,
        device=device,
    )

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

import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_datasets
from engine_utils import build_device, build_model
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
        return None
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg #or {}


def load_cfg_from_ckpt(path: Path): 
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("cfg", {})


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
    model_dir = None,
) -> nn.Module:
    """
    Build model from cfg and load weights from checkpoint.

    By default, uses "<checkpoint_stem>_best.pt" where checkpoint_stem
    is in cfg["logging"]["checkpoint_stem"] (default "test").
    """
    model = build_model_from_cfg(cfg, device, device_type)

    if model_dir is None:
        cfg_logging = cfg.get("logging", {})
        stem = cfg_logging.get("checkpoint_stem", "test")
        if ckpt_name is None:
            ckpt_name = f"{stem}_best.pt"

        ckpt_path = run_dir / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")

        print(f"[test] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model", ckpt)
    else: 
        state_dict = model_dir

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def prepare_datasets(cfg: Dict[str, Any]):
    """
    Reload datasets exactly as in train_zerostep.py via data_utils.load_datasets.

    Supports both:
      - load_datasets(...) -> (train_ds, val_ds)
      - load_datasets(...) -> (train_ds, val_ds, test_ds)

    Returns:
        train_ds, val_ds, test_ds
    """
    cfg_data = cfg["data"]
    result = load_datasets(cfg_data)

    # Handle both 2-split and 3-split variants
    if isinstance(result, tuple) and len(result) == 2:
        train_ds, val_ds = result
        test_ds = val_ds  # fallback: reuse val as test if no separate test split
    elif isinstance(result, tuple) and len(result) == 3:
        train_ds, val_ds, test_ds = result
    else:
        raise RuntimeError(
            f"Unexpected result from load_datasets (expected 2 or 3 items, got {len(result)})"
        )

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------
# Closed-loop ICL-style testing (sliding window, respects seq_len)
# ---------------------------------------------------------------------


def run_model_on_experiment(
    model: nn.Module,
    ds,
    idx: int,
    device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    # 2) Add fake batch dimension
    df_len = 1
    u_full_all = u_full.unsqueeze(0)        # (1, T, 5)
    y_full_all = y_full.unsqueeze(0)        # (1, T, 1)
    y_pred_all = torch.zeros_like(y_full_all, device=device)  # (1, T, 1)

    # 3) Autoregressive omega logic
    last_omega = torch.zeros((df_len, H, 1), device=device)

    with torch.no_grad():
        for j in range(T):
            if j < H:
                # (1, j+1, 5)
                input_val = u_full_all[:, : j + 1, :].clone()
                # fill last_omega for the available j+1 steps
                input_val[:, : j + 1, 4] = last_omega[:, -j - 1 :, 0]
            else:
                # (1, H, 5)
                input_val = u_full_all[:, j - H + 1 : j + 1, :].clone()
                input_val[:, :, 4] = last_omega[:, :, 0]

            pred = model(input_val)[:, -1, :]          # (1,1)
            y_pred_all[:, j, 0] = pred[:, 0]

            last_omega = torch.roll(last_omega, shifts=-1, dims=1)
            last_omega[:, -1, 0] = y_pred_all[:, j, 0]

    # 4) Denormalize exactly as in training utilities
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

    Additionally:
      - Save a second figure with the speed error e_ω = ω_real - ω_est
        and show the MSE in the legend.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    ia = u_den[:, 0]
    ib = u_den[:, 1]
    va = u_den[:, 2]
    vb = u_den[:, 3]

    # ------------------------------------------------------------------
    # Main time–series plot
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Error plot (ω_real - ω_est) with MSE in the legend
    # ------------------------------------------------------------------
    err = y_true - y_hat
    mse = float(np.mean(err**2))

    fig_err, ax_err = plt.subplots(1, 1, figsize=(10, 3))

    ax_err.plot(
        t,
        err,
        linewidth=1.0,
        label=rf"$e_\omega = \omega_\mathrm{{real}} - \omega_\mathrm{{est}}$"
              rf"\ (MSE={mse:.3e})",
    )
    ax_err.set_ylabel(r"$e_\omega$ [rad/s]")
    ax_err.set_xlabel("Sample")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="upper right")

    fig_err.tight_layout()

    fname_err = outdir / f"{split}_experiment_{idx:03d}_error.{fmt}"
    fig_err.savefig(fname_err, dpi=dpi, format=fmt)
    plt.close(fig_err)

    # Keep original return to avoid breaking callers
    return fname


# ---------------------------------------------------------------------
# Modular testing entry point
# ---------------------------------------------------------------------


def run_testing(
    run_dir: Path | str,
    split: str = "val",
    n_exps: int = 5,
    epoch: int | None = None,
    cfg: Dict[str, Any] | None = None,
    model_dir = None,
    device=None,
    ckpt_name: str | None = None,
    data_set = None,
):
    """
    General testing+plotting entry point.

    Two usage modes:

      1) From CLI:
           - pass run_dir, split, n_exps, ckpt_name
           - cfg/model/device are None
           -> function loads cfg, builds device+model, loads checkpoint.

      2) From training:
           - pass run_dir, cfg, model, device, split, n_exps
           - ckpt_name is ignored (model is already loaded)
           -> function just plots using the in-memory model.

    Plots are saved under:
        <run_dir>/<plot.output_subdir>/testing[/epoch_xxxx]/<split>_experiment_XXX.<fmt>

    For your use case (only at the end of training), just call once with epoch=None.
    """
    # Coerce run_dir to Path
    run_dir = Path(run_dir)

    # 1) Config
    if cfg is None:
        cfg = load_cfg_used(run_dir)
        if cfg is None: 
            ckpt_name = "test_best.pt"
            cfg = load_cfg_from_ckpt(run_dir / ckpt_name)
    
    # 2) Device
    if device is None:
        cfg_compute = cfg["compute"]
        device, device_type = build_device(cfg_compute)
    else:
        device_type = "cuda" if getattr(device, "type", "") == "cuda" else "cpu"

    # 3) Model
    model = load_checkpoint_model(
        run_dir, cfg, device, device_type, ckpt_name=ckpt_name, model_dir=model_dir
    )


    # 4) Plot options
    cfg_plot = cfg.get("plot", {})
    dpi = int(cfg_plot.get("dpi", 160))
    fmt = cfg_plot.get("fmt", "pdf")
    output_subdir = cfg_plot.get("output_subdir", "plots")

    outdir = run_dir / output_subdir / "testing"
    # For your “only at end of training” use, epoch can stay None.
    if epoch is not None:
        outdir = outdir / f"BestRun_epoch{epoch:05d}"

    # 5) Datasets
    if data_set is None:
        train_ds, val_ds, test_ds = prepare_datasets(cfg)
        if split == "train":
            ds = train_ds
        elif split == "val":
            ds = val_ds
        elif split == "test":
            ds = test_ds
        else:
            raise ValueError(f"Unknown split {split!r}, must be 'train', 'val', or 'test'.")
    else: 
        ds = data_set
    
    n_available = len(ds.dfs)
    n_exps = min(n_exps, n_available)
    if n_exps <= 0:
        raise RuntimeError(
            f"No experiments available in {split} dataset (len(dfs)={n_available})."
        )

    print(
        f"[test] split={split}, epoch={epoch}, plotting {n_exps} experiments "
        f"(out of {n_available}). Saving to {outdir}"
    )

    # 6) Loop over experiments
    #model.eval()
    for idx in range(n_exps):
        t, y_true, y_hat, u_den = run_model_on_experiment(model, ds, idx, device)

        mse = float(np.mean((y_true - y_hat) ** 2))
        rmse = float(np.sqrt(mse))
        print(f"[test] exp {idx:03d}: MSE={mse:.4e}, RMSE={rmse:.4e}, T={len(t)}")

        fname = plot_experiment_timeseries(
            t,
            y_true,
            y_hat,
            u_den,
            outdir,
            idx,
            dpi,
            fmt,
            split=split,
        )
        print(f"[test]   -> saved {fname}")


# ---------------------------------------------------------------------
# CLI entry point
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
        choices=["train", "val", "test"],
        default="val",
        help="Which dataset split to use for experiments (default: val).",
    )
    parser.add_argument(
        "--n-exps",
        type=int,
        default=5,
        help="Number of experiments to plot from the chosen split.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional checkpoint filename inside --dir "
             "(default: '<checkpoint_stem>_best.pt').",
    )

    args = parser.parse_args()
    run_dir = Path(args.dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist.")

    # Use the modular function in "CLI mode":
    # cfg/model/device are None, so it'll load everything itself.
    run_testing(
        run_dir=run_dir,
        split=args.split,
        n_exps=args.n_exps,
        epoch=None,
        cfg=None,
        model_dir=None,
        device=None,
        ckpt_name=args.ckpt,
        data_set=None,
    )


if __name__ == "__main__":
    # Example CLI:
    # >> python.exe .\plot_testing.py --dir runs/{name_run}
    # >> python.exe .\plot_testing.py --dir sweeps/{name_sweep}
    main()
