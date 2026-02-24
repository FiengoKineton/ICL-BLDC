# src/analysis/layer_a.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch, math
import torch.nn as nn

from src.blocks.ar import ar_rollout

@dataclass
class LayerAConfig:
    feedback_channel: int = 4
    noise_std: float = 0.0
    eps: float = 1e-4

def _finite_diff_scalar_sensitivity(f, x: torch.Tensor, eps: float) -> float:
    """
    f: maps x -> scalar tensor
    returns approx |df/dx| using symmetric finite difference (x scalar).
    """
    x1 = x + eps
    x2 = x - eps
    y1 = f(x1)
    y2 = f(x2)
    return float(torch.abs((y1 - y2) / (2 * eps)).detach().cpu())

@torch.no_grad()
def robustness_to_input_noise(
    model: nn.Module,
    batch_u: torch.Tensor,
    *,
    cfg: LayerAConfig,
    trials: int = 20,
) -> Dict[str, float]:
    """
    Empirical robustness: add Gaussian noise to selected input channels and measure output deviation.
    """
    model.eval()
    base = ar_rollout(batch_u, model, feedback_channel=cfg.feedback_channel)

    deltas = []
    for _ in range(trials):
        noisy = batch_u + cfg.noise_std * torch.randn_like(batch_u)
        yhat = ar_rollout(noisy, model, feedback_channel=cfg.feedback_channel)
        deltas.append(torch.mean(torch.abs(yhat - base)).item())

    return {
        "mean_abs_output_change": sum(deltas) / max(1, len(deltas)),
        "max_abs_output_change": max(deltas) if deltas else 0.0,
    }

@torch.no_grad()
def closed_loop_gain_proxy_worstcase(
    model: nn.Module,
    batch_u: torch.Tensor,
    *,
    cfg: LayerAConfig,
    t_start: int = 1,
    t_end: int | None = None,
    b: int = 0,
) -> dict:
    """
    Computes g_max = max_t |∂ yhat_t / ∂ yhat_{t-1}| over t in [t_start, t_end).
    Returns both the max value and which timestep achieved it.
    """
    assert batch_u.ndim == 3
    T = batch_u.shape[1]
    if t_end is None:
        t_end = T

    gains = []
    for t in range(t_start, min(t_end, T)):
        g = closed_loop_gain_proxy(model, batch_u, cfg=cfg, t=t, b=b)
        gains.append((t, g))

    if not gains:
        return {"gain_worst": 0.0, "t_worst": None}

    t_worst, g_worst = max(gains, key=lambda x: x[1])
    return {"gain_worst": float(g_worst), "t_worst": int(t_worst)}

def closed_loop_gain_proxy(
    model: nn.Module,
    batch_u: torch.Tensor,
    *,
    cfg: LayerAConfig,
    t: int,
    b: int = 0,
) -> float:
    """
    Proxy for |∂ yhat_t / ∂ yhat_{t-1}| by perturbing the injected feedback at time t.
    We treat the injected scalar as the variable and keep everything else fixed.
    """
    assert batch_u.ndim == 3
    B, T, _ = batch_u.shape
    assert 0 <= t < T
    assert 0 <= b < B

    model.eval()

    # freeze a single sample b to keep it scalar-ish
    u = batch_u[b:b+1].clone()  # (1,T,n_u)
    u[:, :, cfg.feedback_channel] = 0.0

    # we need the "baseline" last prediction up to t-1
    with torch.no_grad():
        if t == 0:
            y_prev = torch.zeros((), device=u.device, dtype=u.dtype)
        else:
            # rollout to get yhat_{t-1}
            yhat = ar_rollout(u, model, feedback_channel=cfg.feedback_channel)  # (1,T,1)
            y_prev = yhat[:, t-1, 0].squeeze(0)  # scalar

    # define a function of injected value at time t (scalar)
    def f(injected_scalar: torch.Tensor) -> torch.Tensor:
        u_step = u.clone()
        if t > 0:
            u_step[:, t, cfg.feedback_channel] = injected_scalar
        prefix = u_step[:, :t+1, :]
        y_full = model(prefix)
        return y_full[:, -1, 0].squeeze()  # scalar

    return _finite_diff_scalar_sensitivity(f, y_prev, cfg.eps)


@torch.no_grad()
def closed_loop_amplification_worstcase(
    model: nn.Module,
    batch_u: torch.Tensor,
    *,
    cfg: LayerAConfig,
    t_start: int = 1,
    t_end: int | None = None,
    b: int = 0,
    clamp_min: float = 1e-6,
) -> dict:
    """
    Computes:
      g_t = |∂ yhat_t / ∂ yhat_{t-1}|  (finite-diff proxy)
      G_t = Π_{k=1..t} g_k
      G_max = max_t G_t

    Returns:
      - G_worst: worst-case cumulative amplification
      - t_worst: time where it occurs
      - g_list: list of (t, g_t)
      - G_list: list of (t, G_t)
    """
    assert batch_u.ndim == 3
    T = batch_u.shape[1]
    if t_end is None:
        t_end = T

    # 1) local gains g_t
    g_list: list[tuple[int, float]] = []
    for t in range(t_start, min(t_end, T)):
        g = closed_loop_gain_proxy(model, batch_u, cfg=cfg, t=t, b=b)
        g_list.append((t, float(g)))

    if not g_list:
        return {"G_worst": 0.0, "t_worst": None, "g_list": [], "G_list": []}

    # 2) cumulative products in log-space
    logG = 0.0
    G_list: list[tuple[int, float]] = []
    best_t, best_G = None, -float("inf")

    for t, g in g_list:
        g_safe = max(g, clamp_min)               # avoid log(0)
        logG += math.log(g_safe)
        G = float(math.exp(logG))
        G_list.append((t, G))
        if G > best_G:
            best_G, best_t = G, t

    return {"G_worst": best_G, "t_worst": best_t, "g_list": g_list, "G_list": G_list}