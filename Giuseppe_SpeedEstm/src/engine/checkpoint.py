from __future__ import annotations
import os, json, torch
from typing import Dict, Any

def save_checkpoint(path: str, model, optimizer, epoch: int, history: list[dict], cfg: dict, best: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "history": history,
        "cfg": cfg,
    }, path)
    if best:
        with open(os.path.join(os.path.dirname(path), "metrics_val.json"), "w", encoding="utf-8") as f:
            json.dump(history[-1], f, indent=2)
