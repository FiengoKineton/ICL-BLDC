# src/analysis/layer_b.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional

import torch
import torch.nn as nn

from src.blocks.model import DecoderOnlyCausalTransformer, GPTConfig
from src.blocks.optim import build_optimizer, OptimConfig
from src.blocks.train import fit
from src.analysis.layer_a import LayerAConfig, robustness_to_input_noise, closed_loop_gain_proxy_worstcase

@dataclass
class LayerBConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 5
    feedback_channel: int = 4
    noise_std: float = 1e-3

def apply_input_mask(batch_u: torch.Tensor, keep: List[int]) -> torch.Tensor:
    """
    Keep only selected channels by zeroing the others (shape stays (B,T,n_u)).
    This makes comparisons fair without changing model input dim.
    """
    mask = torch.zeros(batch_u.shape[-1], device=batch_u.device, dtype=batch_u.dtype)
    mask[keep] = 1.0
    return batch_u * mask.view(1, 1, -1)

def train_and_score_inputset(
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]],
    *,
    model_cfg: GPTConfig,
    optim_cfg: OptimConfig,
    layerb_cfg: LayerBConfig,
    input_keep: List[int],
) -> Dict[str, float]:
    device = torch.device(layerb_cfg.device)

    model = DecoderOnlyCausalTransformer(model_cfg).to(device)
    opt = build_optimizer(model, optim_cfg)

    # wrap loaders: apply mask on-the-fly
    def masked(loader):
        for u, y in loader:
            u = apply_input_mask(u, input_keep)
            yield u, y

    hist = fit(
        model=model,
        optimizer=opt,
        train_loader=masked(train_loader),
        val_loader=(masked(val_loader) if val_loader is not None else None),
        device=device,
        epochs=layerb_cfg.epochs,
        feedback_channel=layerb_cfg.feedback_channel,
    )

    # Evaluate Layer A robustness on one validation batch (simple, extend as you like)
    u0, _y0 = next(iter(masked(val_loader if val_loader is not None else train_loader)))
    u0 = u0.to(device)

    a_cfg = LayerAConfig(
        feedback_channel=layerb_cfg.feedback_channel,
        noise_std=layerb_cfg.noise_std,
        eps=1e-4,
    )

    rob = robustness_to_input_noise(model, u0, cfg=a_cfg, trials=20)
    wc = closed_loop_gain_proxy_worstcase(model, u0, cfg=a_cfg, t_start=1, b=0)
    
    return {
        "final_train_loss": float(hist["train_loss"][-1]),
        "final_val_loss": float(hist["val_loss"][-1]) if hist["val_loss"] else float("nan"),
        "robust_mean_abs_output_change": rob["mean_abs_output_change"],
        "closed_loop_gain_proxy_worst": wc["gain_worst"],
        "t_worst": wc["t_worst"],
    }