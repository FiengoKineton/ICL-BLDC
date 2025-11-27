from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple
from .transforms import fixed_range_normalize

# All possible columns you *might* have
ALL = ["t", "iq", "id", "vq", "vd", "ia", "ib", "va", "vb", "theta_e", "omega", "r"]

# What we actually need for this dataset
REQUIRED = ["ia", "ib", "va", "vb", "omega", "r"]   # include 'r' to detect steps
FEATURES = ["ia", "ib", "va", "vb", "omega"]        # columns we normalize / feed
INPUT_CHANNELS = ["ia", "ib", "va", "vb", "last_omega"]

COLUMNS = [c for c in REQUIRED]


class BLDCSequenceDataset(Dataset):
    """
    Drop-in-ish replacement for the old BLDC Dataset:
      - Random experiment
      - Step vs constant window selection using 'r'
      - last_omega = shifted omega
      - Normalization: fixed ranges (same as old normalize_fixed_ranges)
      - Returns:
          x: (H, 5)  = [ia, ib, va, vb, last_omega]   (normalized)
          y: (H, 1)  = omega                          (normalized)
    """

    def __init__(self, cfg: Dict[str, Any], split: str):
        self.cfg = cfg
        self.split = split
        root = cfg["root"]             # e.g. "data/raw"
        patterns = cfg["split"][split] # e.g. ["simulated/50_percent_low_speed"]

        # 1) Collect all CSV files matching patterns
        files: List[str] = []
        for pat in patterns:
            base = os.path.join(root, pat)
            if os.path.isdir(base):
                # pattern is a folder -> all CSVs inside
                files.extend(glob.glob(os.path.join(base, "*.csv")))
            else:
                # pattern is a glob / filename relative to root
                files.extend(glob.glob(os.path.join(root, pat)))
                files.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))

        if not files:
            raise FileNotFoundError(f"No CSV files matched under {root} for patterns {patterns}")

        # 2) Load each CSV as a separate experiment (like the old dfs list)
        dfs: List[pd.DataFrame] = []
        for f in files:
            try:
                df = pd.read_csv(f, encoding="utf-8-sig")
                df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

                if not set(REQUIRED).issubset(df.columns):
                    print(f"Skipping {f}, missing columns {set(REQUIRED) - set(df.columns)}")
                    continue

                # Keep at least required columns; you can keep more if you want
                df = df[REQUIRED].copy()
                dfs.append(df)
            except Exception as e:
                print(f"Failed on {f} with error: {e!r}")
                continue

        if not dfs:
            raise RuntimeError("No valid CSV with required columns found.")

        self.dfs = dfs
        self.seq_len = int(cfg["seq_len"])
        self.inject_last = bool(cfg.get("inject_last_channel", True))

        # 3) Normalization ranges (same effect as normalize_fixed_ranges)
        norm = cfg.get("normalize", {})
        if norm.get("method", "fixed_range") == "fixed_range":
            r = norm.get("ranges", {})
            # only for FEATURES (ia, ib, va, vb, omega)
            self.ranges = {k: r.get(k, [0.0, 1.0]) for k in FEATURES}
        else:
            raise NotImplementedError("Only fixed_range supported in this version.")

        # 4) Virtual length like the old Dataset (__len__ = 512)
        self.virtual_len = int(cfg.get("virtual_len", 512))

        # 5) Probability of picking a "step" window vs "constant" window
        #    old code used prob_ratio = 0.5 (50/50)
        self.prob_step = float(cfg.get("prob_step", 0.5))

    def __len__(self) -> int:
        # Same idea as the old Dataset: length is arbitrary,
        # sampling is random inside __getitem__.
        return self.virtual_len

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fixed-range normalization inplace on a copy."""
        df = df.copy()
        for k in FEATURES:
            lo, hi = self.ranges[k]
            df[k] = fixed_range_normalize(df[k].to_numpy(), lo, hi)
        return df

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1) Randomly pick one experiment (as in old Dataset)
        df_idx = np.random.randint(len(self.dfs))
        df_raw = self.dfs[df_idx]

        # Work on a copy, we'll normalize it
        df = self._normalize_df(df_raw)

        # 2) Step vs constant window selection using 'r'
        #    diff(-H) ~ r_t - r_{t+H}, shape ~ len(df) - H
        diff_array = df["r"].diff(-self.seq_len).to_numpy()
        diff_array = diff_array[~np.isnan(diff_array)]

        # good_idx: indices where a window of length H can start
        if np.random.rand() >= self.prob_step:
            # Prefer constant windows (no step)
            good_idx = np.flatnonzero(diff_array == 0)
            if len(good_idx) == 0:
                # fallback: any step window
                good_idx = np.flatnonzero(diff_array != 0)
        else:
            # Prefer step windows
            good_idx = np.flatnonzero(diff_array != 0)
            if len(good_idx) == 0:
                # fallback: constant
                good_idx = np.flatnonzero(diff_array == 0)

        if len(good_idx) == 0:
            # degenerate case: just pick any valid start
            max_start = len(df) - self.seq_len
            if max_start <= 0:
                raise ValueError("Experiment too short for requested seq_len.")
            start_idx = np.random.randint(0, max_start)
        else:
            start_idx = int(np.random.choice(good_idx))

        # 3) Build last_omega as shifted omega (normalized)
        if self.inject_last:
            omega = df["omega"].to_numpy()
            last_omega = omega.copy()
            last_omega[1:] = last_omega[:-1]
            last_omega[0] = 0.0
            df["last_omega"] = last_omega
        else:
            df["last_omega"] = 0.0

        # 4) Extract window
        win = df.iloc[start_idx : start_idx + self.seq_len]

        # x: (H,5) -> [ia, ib, va, vb, last_omega]
        x_np = win[INPUT_CHANNELS].to_numpy(dtype=np.float32)

        # y: (H,1) -> omega
        y_np = win["omega"].to_numpy(dtype=np.float32).reshape(-1, 1)

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        return {"x": x, "y": y}


class __BLDCSequenceDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], split: str):
        self.cfg = cfg
        self.split = split
        root = cfg["root"]                 # e.g. "data/processed"
        patterns = cfg["split"][split]     # e.g. ["simulated/50_percent_low_speed"]

        files: List[str] = []
        for pat in patterns:
            # absolute-ish path for the pattern
            base = os.path.join(root, pat)

            if os.path.isdir(base):
                # pattern is a folder -> take all CSVs inside
                files.extend(glob.glob(os.path.join(base, "*.csv")))
                # if you want recursion:
                # for dirpath, _, _ in os.walk(base):
                #     files.extend(glob.glob(os.path.join(dirpath, "*.csv")))
            else:
                # pattern is a glob / filename relative to root
                files.extend(glob.glob(os.path.join(root, pat)))
                files.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))

        if not files:
            raise FileNotFoundError(f"No CSV files matched under {root} for patterns {patterns}")

        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f, encoding="utf-8-sig")
                # clean weird BOM / spaces
                df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

                if not set(COLUMNS).issubset(df.columns):
                    print(f"Skipping {f}, missing columns {set(COLUMNS) - set(df.columns)}")
                    continue

                dfs.append(df[COLUMNS])
            except Exception as e:
                print(f"Failed on {f} with error: {e!r}")
                continue

        if not dfs:
            raise RuntimeError("No valid CSV with required columns found.")

        self.df = pd.concat(dfs, axis=0, ignore_index=True)
        self.seq_len = int(cfg["seq_len"])
        self.inject_last = bool(cfg.get("inject_last_channel", True))

        norm = cfg.get("normalize", {})
        if norm.get("method", "fixed_range") == "fixed_range":
            r = norm.get("ranges", {})
            self.ranges = {k: r.get(k, [0.0, 1.0]) for k in COLUMNS}
        else:
            raise NotImplementedError("Only fixed_range supported in this minimal version.")

        self.n = len(self.df) - self.seq_len - 1
        if self.n <= 0:
            raise ValueError("Not enough rows for the requested seq_len.")


    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        xslice = self.df.iloc[idx: idx + self.seq_len].copy()
        yslice = self.df.iloc[idx: idx + self.seq_len].copy()
        # Normalize
        for k in COLUMNS:
            lo, hi = self.ranges[k]
            xslice[k] = fixed_range_normalize(xslice[k].to_numpy(), lo, hi)
            yslice[k] = fixed_range_normalize(yslice[k].to_numpy(), lo, hi)
        x = xslice[["ia","ib","va","vb"]].to_numpy(dtype=np.float32)  # (H, 4)
        y = yslice["omega"].to_numpy(dtype=np.float32)                # (H,)
        if self.inject_last:
            last_omega = np.zeros_like(y)
            #last_omega[0] = y[0]
            x = np.concatenate([x, last_omega[:, None]], axis=1)      # (H, 5)
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y)}


def collate_batch(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Stack samples into:
      X: (B, H, 5)
      Y: (B, H, 1)
    """
    X = torch.stack([s["x"] for s in samples], dim=0)
    Y = torch.stack([s["y"] for s in samples], dim=0)
    return {"x": X, "y": Y}


# CHECKED -- almost good!