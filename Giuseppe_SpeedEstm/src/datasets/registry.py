from __future__ import annotations
from typing import Dict, Any
from .bldc_csv import BLDCSequenceDataset, collate_batch

_DATASETS = {
    "bldc_csv": BLDCSequenceDataset,
}

def build_dataset(name: str, cfg: Dict[str, Any], split: str):
    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset: {name}")
    return _DATASETS[name](cfg, split)

__all__ = ["build_dataset", "collate_batch"]


# CHECKED -- all good!