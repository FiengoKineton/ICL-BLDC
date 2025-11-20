from __future__ import annotations
import os, glob, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple
from .transforms import fixed_range_normalize

ALL = ["t", "iq", "id", "vq", "vd", "ia", "ib", "va", "vb", "theta_e", "omega", "r"]
REQUIRED = ["ia", "ib", "va", "vb", "omega"]
COLUMNS = [c for c in REQUIRED]

class BLDCSequenceDataset(Dataset):
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

def collate_batch(samples: List[Dict[str, torch.Tensor]]):
    X = torch.stack([s["x"] for s in samples], dim=0)
    Y = torch.stack([s["y"] for s in samples], dim=0)
    return {"x": X, "y": Y}
