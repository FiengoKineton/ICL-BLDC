from __future__ import annotations
from typing import Dict, Any
# Expect user to place transformer_zerostep.py under src/models with class GPT and GPTConfig
try:
    from .transformer_zerostep import GPT, GPTConfig
except Exception as e:
    GPT = None
    GPTConfig = None

_MODELS = {
    "gpt_zerostep": ("GPT", "GPTConfig")
}

def build_model(name: str, cfg: Dict[str, Any]):
    if name not in _MODELS or GPT is None or GPTConfig is None:
        raise RuntimeError("Model 'gpt_zerostep' requires src/models/transformer_zerostep.py with GPT/GPTConfig.")
    mcfg = GPTConfig(
        n_layer=cfg.get("n_layer", 4),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("d_model", 128),
        block_size=cfg.get("block_size", 128),
        dropout=cfg.get("dropout", 0.1),
        bias=cfg.get("bias", False),
        n_u=cfg.get("n_u", 5),
        n_y=cfg.get("n_y", 1),
        n_x=cfg.get("n_x", 4),
    )
    return GPT(mcfg)


# CHECKED -- all good!