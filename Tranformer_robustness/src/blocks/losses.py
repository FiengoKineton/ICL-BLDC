# src/blocks/losses.py
import torch

def mse_loss(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    L = mean((yhat - y)^2) over all entries (B,T,1)
    """
    assert yhat.shape == y.shape
    return torch.mean((yhat - y) ** 2)