from __future__ import annotations
import torch

@torch.no_grad()
def rollout(model, batch: dict, device: torch.device) -> torch.Tensor:
    # expects batch["x"]: (B, H, C_in), batch["y"]: (B, H)
    model.eval()
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    B, H, C = x.shape
    last = y[:, :1]  # teacher init for t=0
    y_hat = []
    for t in range(H):
        x_t = x.clone()
        x_t[:, :, -1] = torch.cat([last.squeeze(-1), torch.zeros(B, H-1, device=x.device)], dim=1)[:, t]
        out = model(x_t)  # model should output (B, H)
        pred_t = out[:, t:t+1]
        y_hat.append(pred_t)
        last = pred_t
    y_hat = torch.cat(y_hat, dim=1)
    return y_hat
