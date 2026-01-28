# main.py               | SET (leave it as it is)

import itertools
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

import yaml, pandas as pd

from data_utils import load_datasets
from run_experiment import run_single_experiment
from run_sweep import Sweeper
from plot_sweep_analisys import analyze_runs_csv


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

    best_val, train_time, test_loss = run_single_experiment(cfg, train_ds, val_ds, test_ds, run_dir, exp_name)
    print(f"[main] Single run finished, best_val_loss={best_val:.4e}, test_loss={test_loss:.4e}, train_time={train_time:.2f}s\n")



if __name__ == "__main__":
    config_path = "configs.yaml"
    cfg = load_config(config_path)

    mode = cfg["experiment"].get("mode", "single")
    if mode == "single":
        run_single(cfg)
    elif mode == "sweep":
        sweeper = Sweeper(cfg, load_datasets_fn=load_datasets, run_single_experiment_fn=run_single_experiment, analyze_runs_csv_fn=analyze_runs_csv)
        sweeper.run()
    else:
        raise ValueError(f"Unknown experiment.mode='{mode}' (expected 'single' or 'sweep').")
