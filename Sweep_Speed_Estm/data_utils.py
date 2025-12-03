# data_utils.py         | SET (leave it as it is)

import random 

from pathlib import Path
from typing import Dict, Any, List
from dataset import Dataset, load_dataframes_from_folder


def resolve_data_root(cfg_data: Dict[str, Any]) -> Path:
    """
    Mimics your current_path split("ICL-BLDC")[0] logic.
    """
    cwd = Path().resolve()
    marker = cfg_data.get("project_root_marker", "ICL-BLDC")
    parts = str(cwd).split(marker)
    if len(parts) < 2:
        # fallback: assume cwd IS project root
        project_root = cwd
    else:
        project_root = Path(parts[0]) / marker
    return project_root / cfg_data.get("data_subdir", "data")


def load_datasets(cfg_data: Dict[str, Any], print_flag: bool = False):
    """
    Reads all DataFrames once and splits into train / val / test.
    Split is done at the file level using a 7:2:1 ratio by default.

    Returns:
        train_ds, val_ds, test_ds
    """
    data_root = resolve_data_root(cfg_data)

    try: 
        folders: List[str]  = cfg_data["folders"]
    except Exception as e: 
        folders: List[str]  = cfg_data["train_folders"]
    seq_len: int            = cfg_data["seq_len"]

    # Ratios for train / val / test: 7, 2, 1  (i.e. 0.7, 0.2, 0.1)
    r_train = cfg_data.get("train_ratio", 0.7)
    r_val   = cfg_data.get("val_ratio",   0.2)
    r_test  = cfg_data.get("test_ratio",  0.1)

    total_ratio = r_train + r_val + r_test
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Train/val/test ratios must sum to 1.0, got {total_ratio:.3f}"
        )

    # Load all dataframes from the union of folders
    dfs_all: List[Any] = []
    folders = sorted(set(folders)) #  train_folders + val_folders))
    for folder in folders:
        path = data_root / folder
        new_dfs = load_dataframes_from_folder(str(path))
        dfs_all += new_dfs
        if print_flag: print(f"[data] Loaded {len(new_dfs)} DataFrames from {path}")

    if not dfs_all:
        raise RuntimeError("[data] No CSV files found in the specified folders")

    # Deterministic shuffle
    split_seed = cfg_data.get("split_seed", 42)
    rng = random.Random(split_seed)
    rng.shuffle(dfs_all)

    n_total = len(dfs_all)
    n_train = int(r_train * n_total)
    n_val   = int(r_val   * n_total)
    # Whatever is left goes to test
    n_test  = n_total - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test == 0:
        print(
            f"[data][WARN] Very few files ({n_total}). "
            f"Split gives train={n_train}, val={n_val}, test={n_test}."
        )

    dfs_train = dfs_all[:n_train]
    dfs_val   = dfs_all[n_train:n_train + n_val]
    dfs_test  = dfs_all[n_train + n_val:]

    if print_flag: print(f"[data] Split {n_total} DataFrames into "
          f"{len(dfs_train)} train, {len(dfs_val)} val, {len(dfs_test)} test.")

    train_ds = Dataset(dfs=dfs_train, seq_len=seq_len)
    val_ds   = Dataset(dfs=dfs_val,   seq_len=seq_len)
    test_ds  = Dataset(dfs=dfs_test,  seq_len=seq_len)

    return train_ds, val_ds, test_ds
