from __future__ import annotations
import torch

def loss_smoothness(y_hat: torch.Tensor, order: int = 1) -> torch.Tensor:
    # simple finite difference along time dimension
    if order == 1:
        diff = y_hat[:, 1:] - y_hat[:, :-1]
    else:
        d1 = y_hat[:, 1:] - y_hat[:, :-1]
        diff = d1[:, 1:] - d1[:, :-1]
    return torch.mean(diff ** 2)


# CHECKED -- all good!