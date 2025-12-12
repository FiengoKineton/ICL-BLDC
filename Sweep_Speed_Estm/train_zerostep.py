# main.py               | SET (leave it as it is)

import itertools
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

import yaml
import pandas as pd

from data_utils import load_datasets
from run_experiment import run_single_experiment


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def run_single(cfg: Dict[str, Any]):
    # Datasets loaded once
    train_ds, val_ds, test_ds = load_datasets(cfg["data"])

    exp_name = cfg["experiment"]["name"]
    root = Path(cfg["experiment"]["output_root"])
    run_dir = root / exp_name

    best = run_single_experiment(cfg, train_ds, val_ds, test_ds, run_dir)
    print(f"[main] Single run finished, best_val_loss={best:.4e}\n")


def run_sweep(cfg: Dict[str, Any]):
    """
    Cartesian sweep over cfg["sweep"].
    Same datasets reused for all runs.
    """
    base_cfg = deepcopy(cfg)
    sweep_def = base_cfg.get("sweep", {})
    if not sweep_def:
        raise ValueError("Sweep mode requested but no 'sweep' block in config.")

    keys, values = zip(*sweep_def.items())

    train_ds, val_ds, test_ds = load_datasets(base_cfg["data"])

    root = Path(base_cfg["experiment"]["output_root"]) / base_cfg["experiment"]["output_sweep"]
    root.mkdir(parents=True, exist_ok=True)

    results_rows = []

    for i, combo in enumerate(itertools.product(*values)):
        overrides = dict(zip(keys, combo))

        # Build name
        name = "_".join(f"{k}{v}" for k, v in overrides.items())
        cfg_run = deepcopy(base_cfg)
        cfg_run["experiment"]["name"] = name

        # Apply overrides to relevant blocks
        # here we assume all swept keys belong to either `model` or `training`
        for k, v in overrides.items():
            if k in cfg_run["model"]:
                cfg_run["model"][k] = v
            elif k in cfg_run["training"]:
                cfg_run["training"][k] = v
            else:
                # if you sweep something else, handle accordingly
                cfg_run["training"][k] = v

        run_dir = root / name
        print(f"[main] Running sweep combo {i}: {name}\n")
        best_val_loss = run_single_experiment(cfg_run, train_ds, val_ds, test_ds, run_dir)

        row = {"name": name, "best_val_loss": best_val_loss}
        row.update(overrides)
        results_rows.append(row)

    if results_rows:
        df = pd.DataFrame(results_rows)
        out_csv = root / "sweep_results.csv"
        df.to_csv(out_csv, index=False)
        print(f"[main] Saved sweep summary to {out_csv}")
    else:
        print("[main] No sweep runs executed.")


if __name__ == "__main__":
    config_path = "configs.yaml"
    cfg = load_config(config_path)

    mode = cfg["experiment"].get("mode", "single")
    if mode == "single":
        run_single(cfg)
    elif mode == "sweep":
        run_sweep(cfg)
    else:
        raise ValueError(f"Unknown experiment.mode='{mode}' (expected 'single' or 'sweep').")
