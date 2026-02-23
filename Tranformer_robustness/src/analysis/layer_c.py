# src/analysis/layer_c.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, List, Optional

import torch

from src.blocks.model import DecoderOnlyCausalTransformer, GPTConfig
from src.blocks.optim import build_optimizer, OptimConfig
from src.blocks.train import fit

@dataclass
class LayerCConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 5

def sweep_optimizer_sensitivity(
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]],
    *,
    model_cfg: GPTConfig,
    base_optim: OptimConfig,
    sweep_lrs: List[float],
    sweep_wds: List[float],
    layerc_cfg: LayerCConfig,
) -> List[Dict[str, float]]:
    """
    Sensitivity analysis: vary LR and WD, record final train/val loss + grad norms.
    This is empirical (practical) rather than formal theorem guarantees.
    """
    device = torch.device(layerc_cfg.device)
    results = []

    for lr in sweep_lrs:
        for wd in sweep_wds:
            model = DecoderOnlyCausalTransformer(model_cfg).to(device)
            opt_cfg = OptimConfig(
                name=base_optim.name, lr=lr, weight_decay=wd,
                betas=base_optim.betas, eps=base_optim.eps,
                momentum=base_optim.momentum, nesterov=base_optim.nesterov
            )
            opt = build_optimizer(model, opt_cfg)

            hist = fit(
                model=model, optimizer=opt,
                train_loader=train_loader, val_loader=val_loader,
                device=device, epochs=layerc_cfg.epochs
            )

            results.append({
                "lr": lr,
                "weight_decay": wd,
                "final_train_loss": float(hist["train_loss"][-1]),
                "final_val_loss": float(hist["val_loss"][-1]) if hist["val_loss"] else float("nan"),
                "final_grad_norm": float(hist["train_gnorm"][-1]),
            })

    return results

def algorithmic_stability_proxy_leave_one_batch_out(
    train_loaders: List[Iterable[Tuple[torch.Tensor, torch.Tensor]]],
    val_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    model_cfg: GPTConfig,
    optim_cfg: OptimConfig,
    layerc_cfg: LayerCConfig,
) -> Dict[str, float]:
    """
    Proxy for algorithmic stability:
    Train multiple models with small dataset perturbations (e.g., remove one batch),
    then measure variation in validation loss.
    """
    device = torch.device(layerc_cfg.device)
    val_losses = []

    for tr_loader in train_loaders:
        model = DecoderOnlyCausalTransformer(model_cfg).to(device)
        opt = build_optimizer(model, optim_cfg)

        hist = fit(
            model=model, optimizer=opt,
            train_loader=tr_loader, val_loader=val_loader,
            device=device, epochs=layerc_cfg.epochs
        )

        if hist["val_loss"]:
            val_losses.append(hist["val_loss"][-1])

    if not val_losses:
        return {"val_loss_mean": float("nan"), "val_loss_std": float("nan")}

    mean = sum(val_losses) / len(val_losses)
    var = sum((x - mean) ** 2 for x in val_losses) / max(1, len(val_losses) - 1)
    return {"val_loss_mean": float(mean), "val_loss_std": float(var ** 0.5)}