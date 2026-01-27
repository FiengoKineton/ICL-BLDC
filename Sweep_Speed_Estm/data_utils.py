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
    Generalized loader for train / val / test datasets.

    Logic:
      - If train_folders, val_folders, test_folders are all the same
        -> behave like the old implementation: union, shuffle, global split
           according to (train_ratio, val_ratio, test_ratio).

      - If exactly two of them are equal and one is different:
          * The different one is used entirely for its split.
          * The two equal ones are merged into a pool and split between them
            using a 2-way ratio, with priority:
                train >> val >> test
            That is:
                - if train == val:    train keeps r_train, val gets (1 - r_train)
                - if train == test:   train keeps r_train, test gets (1 - r_train)
                - if val == test:     val   keeps r_val,   test gets (1 - r_val)

      - If all three are different:
          -> each split uses only its own folders, fully.

    Returns:
        train_ds, val_ds, test_ds
    """
    data_root = resolve_data_root(cfg_data)

    # --- Ratios (used in some cases) ---
    r_train = cfg_data.get("train_ratio", 0.7)
    r_val   = cfg_data.get("val_ratio",   0.2)
    r_test  = cfg_data.get("test_ratio",  0.1)
    total_ratio = r_train + r_val + r_test
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Train/val/test ratios must sum to 1.0, got {total_ratio:.3f}"
        )

    seq_len: int = cfg_data["seq_len"]

    # --- Read folder configuration ---
    train_folders = cfg_data.get("train_folders")
    val_folders   = cfg_data.get("val_folders")
    test_folders  = cfg_data.get("test_folders")

    # Backward compatibility: if only "folders" is given, use it for all three
    if train_folders is None and val_folders is None and test_folders is None:
        base_folders: List[str] = cfg_data["folders"]
        train_folders = base_folders
        val_folders   = base_folders
        test_folders  = base_folders
    else:
        # If some are missing, default them to train_folders (or others)
        if train_folders is None:
            # Prefer "folders" if present, else reuse val/test
            base_folders = cfg_data.get("folders", val_folders or test_folders)
            train_folders = base_folders
        if val_folders is None:
            val_folders = train_folders
        if test_folders is None:
            test_folders = train_folders

    # Normalize to lists
    train_folders = list(train_folders)
    val_folders   = list(val_folders)
    test_folders  = list(test_folders)

    set_train = set(train_folders)
    set_val   = set(val_folders)
    set_test  = set(test_folders)

    # --- Load all DataFrames once per folder and cache ---
    all_folder_names = sorted(set_train | set_val | set_test)
    folder_to_dfs: Dict[str, List[Any]] = {}
    for folder in all_folder_names:
        path = data_root / folder
        new_dfs = load_dataframes_from_folder(str(path))
        folder_to_dfs[folder] = new_dfs
        if print_flag:
            print(f"[data] Loaded {len(new_dfs)} DataFrames from {path}")

    def collect_dfs(folders: List[str]) -> List[Any]:
        out: List[Any] = []
        for f in sorted(set(folders)):
            out.extend(folder_to_dfs.get(f, []))
        return out

    # Global RNG for deterministic shuffles
    split_seed = cfg_data.get("split_seed", 42)
    rng = random.Random(split_seed)

    # --- Case 1: all the same -> old behavior ---
    if set_train == set_val == set_test:
        dfs_all = collect_dfs(train_folders)
        if not dfs_all:
            raise RuntimeError("[data] No CSV files found in the specified folders")

        rng.shuffle(dfs_all)
        n_total = len(dfs_all)
        n_train = int(r_train * n_total)
        n_val   = int(r_val   * n_total)
        n_test  = n_total - n_train - n_val

        if n_train == 0 or n_val == 0 or n_test == 0:
            print(
                f"[data][WARN] Very few files ({n_total}). "
                f"Split gives train={n_train}, val={n_val}, test={n_test}."
            )

        dfs_train = dfs_all[:n_train]
        dfs_val   = dfs_all[n_train:n_train + n_val]
        dfs_test  = dfs_all[n_train + n_val:]

    # --- Case 2: exactly two equal ---
    elif set_train == set_val != set_test:
        # train == val, test different
        dfs_test = collect_dfs(test_folders)
        dfs_pool = collect_dfs(train_folders)  # shared by train & val

        if not dfs_pool and not dfs_test:
            raise RuntimeError("[data] No CSV files found in any folders")

        rng.shuffle(dfs_pool)

        n_pool = len(dfs_pool)
        # train keeps r_train, val gets (1 - r_train)
        n_train = int(r_train * n_pool)
        n_val   = n_pool - n_train

        dfs_train = dfs_pool[:n_train]
        dfs_val   = dfs_pool[n_train:]
        # dfs_test already full

    elif set_train == set_test != set_val:
        # train == test, val different
        dfs_val  = collect_dfs(val_folders)
        dfs_pool = collect_dfs(train_folders)  # shared by train & test

        if not dfs_pool and not dfs_val:
            raise RuntimeError("[data] No CSV files found in any folders")

        rng.shuffle(dfs_pool)

        n_pool = len(dfs_pool)
        # train keeps r_train, test gets (1 - r_train)
        n_train = int(r_train * n_pool)
        n_test  = n_pool - n_train

        dfs_train = dfs_pool[:n_train]
        dfs_test  = dfs_pool[n_train:]
        # dfs_val already full

    elif set_val == set_test != set_train:
        # val == test, train different
        dfs_train = collect_dfs(train_folders)
        dfs_pool  = collect_dfs(val_folders)  # shared by val & test

        if not dfs_pool and not dfs_train:
            raise RuntimeError("[data] No CSV files found in any folders")

        rng.shuffle(dfs_pool)

        n_pool = len(dfs_pool)
        # val keeps r_val, test gets (1 - r_val)
        n_val  = int(r_val * n_pool)
        n_test = n_pool - n_val

        dfs_val  = dfs_pool[:n_train]
        dfs_test = dfs_pool[n_train:]

    # --- Case 3: all different ---
    else:
        dfs_train = collect_dfs(train_folders)
        dfs_val   = collect_dfs(val_folders)
        dfs_test  = collect_dfs(test_folders)

        if not (dfs_train or dfs_val or dfs_test):
            raise RuntimeError("[data] No CSV files found in any folders")

        # Shuffle each split independently for randomness
        rng.shuffle(dfs_train)
        rng.shuffle(dfs_val)
        rng.shuffle(dfs_test)

    # --- Final sanity / logging ---
    n_train = len(dfs_train)
    n_val   = len(dfs_val)
    n_test  = len(dfs_test)

    if print_flag:
        print(f"[data] Final split: "
              f"{n_train} train, {n_val} val, {n_test} test.")

    if n_train == 0 or n_val == 0 or n_test == 0:
        print(
            f"[data][WARN] Some splits are empty: "
            f"train={n_train}, val={n_val}, test={n_test}."
        )

    train_ds = Dataset(dfs=dfs_train, seq_len=seq_len)
    val_ds   = Dataset(dfs=dfs_val,   seq_len=seq_len)
    test_ds  = Dataset(dfs=dfs_test,  seq_len=seq_len)

    return train_ds, val_ds, test_ds

