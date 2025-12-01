# data_utils.py
from pathlib import Path
from typing import Tuple, Dict, Any, List

from torch.utils.data import DataLoader
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


def load_datasets(cfg_data: Dict[str, Any]):
    """
    Reads all training and validation DataFrames once.
    Returns (train_ds, val_ds).
    """
    data_root = resolve_data_root(cfg_data)

    train_folders: List[str] = cfg_data["train_folders"]
    val_folders: List[str] = cfg_data["val_folders"]
    seq_len: int = cfg_data["seq_len"]

    # Train
    dfs_train = []
    for folder in train_folders:
        path = data_root / folder
        new_dfs = load_dataframes_from_folder(str(path))
        dfs_train += new_dfs
        print(f"[data] Loaded {len(new_dfs)} train DataFrames from {path}")

    train_ds = Dataset(dfs=dfs_train, seq_len=seq_len)

    # Val
    dfs_val = []
    for folder in val_folders:
        path = data_root / folder
        new_dfs = load_dataframes_from_folder(str(path))
        dfs_val += new_dfs
        print(f"[data] Loaded {len(new_dfs)} val DataFrames from {path}")

    val_ds = Dataset(dfs=dfs_val, seq_len=seq_len)

    return train_ds, val_ds
