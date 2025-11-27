from __future__ import annotations
import math, os

class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = math.inf
        self.count = 0
    def step(self, val_metric: float) -> bool:
        if val_metric < self.best - 1e-12:
            self.best = val_metric
            self.count = 0
            return False
        self.count += 1
        return self.count > self.patience


# NOT USED!