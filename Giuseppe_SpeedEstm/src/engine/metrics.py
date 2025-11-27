from __future__ import annotations
import torch

@torch.no_grad()
def mse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean((y_hat - y) ** 2).cpu().item())


# NOT USED!