# src/blocks/ar.py
from __future__ import annotations
import torch
import torch.nn as nn

def ar_rollout(
    batch_u: torch.Tensor,
    model: nn.Module,
    *,
    feedback_channel: int = 4,
) -> torch.Tensor:
    """
    Pure AR rollout:
      - zero feedback channel in a copy
      - yhat_0 = 0
      - for t: inject yhat_{t-1} into U[:,t,feedback_channel]
              run model(prefix)
              take last output as yhat_t
      - return (B,T,1)
    """
    assert batch_u.ndim == 3, "batch_u must be (B,T,n_u)"
    B, T, _ = batch_u.shape

    u = batch_u.clone()
    u[:, :, feedback_channel] = 0.0

    last_yhat = torch.zeros(B, device=batch_u.device, dtype=batch_u.dtype)
    preds = []

    for t in range(T):
        u_step = u.clone()
        if t > 0:
            u_step[:, t, feedback_channel] = last_yhat

        prefix = u_step[:, :t+1, :]
        y_full = model(prefix)                 # (B,t+1,1)
        last_yhat = y_full[:, -1, 0]           # (B,)
        preds.append(last_yhat.unsqueeze(1))   # (B,1)

    return torch.cat(preds, dim=1).unsqueeze(-1)  # (B,T,1)