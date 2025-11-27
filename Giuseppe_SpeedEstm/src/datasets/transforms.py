from __future__ import annotations
import numpy as np

def fixed_range_normalize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (arr - lo) / (hi - lo + 1e-12)

def zscore_normalize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (arr - mean) / max(std, 1e-12)


# NOT USED!