from __future__ import annotations
import torch

def loss_mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_hat - y) ** 2)


# CHECKED -- all good!