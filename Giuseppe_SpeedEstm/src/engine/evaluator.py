from __future__ import annotations
import torch

@torch.no_grad()
def rollout(model, batch: dict, device: torch.device) -> torch.Tensor:
    """
    Strict autoregressive rollout, same schedule as train_zerostep.validate:
      - omega-hat channel (last) is zeroed initially
      - ω̂_0 = 0
      - at step t, inject ω̂_{t-1} at time t, feed prefix 0..t
      - read last time step output
    Returns: y_hat of shape (B, H)
    """
    model.eval()
    x = batch["x"].to(device)   # (B, H, C)
    y = batch["y"].to(device)   # (B, H)  (we only use shape)
    B, H, C = x.shape

    # zero omega-hat input channel
    x0 = x.clone()
    x0[:, :, -1] = 0.0

    last = torch.zeros(B, device=device)  # ω̂_0
    y_hat_steps = []

    for t in range(H):
        x_step = x0.clone()
        x_step[:, t, -1] = last
        x_prefix = x_step[:, :t+1, :]

        out = model(x_prefix)

        if out.ndim == 3 and out.shape[-1] == 1:
            y_t = out[:, -1, 0]
        elif out.ndim == 2:
            y_t = out[:, -1]
        else:
            raise RuntimeError(f"Unexpected model output shape {out.shape}")

        y_hat_steps.append(y_t.unsqueeze(1))
        last = y_t

    y_hat = torch.cat(y_hat_steps, dim=1)  # (B, H)
    return y_hat


# NOT USED!