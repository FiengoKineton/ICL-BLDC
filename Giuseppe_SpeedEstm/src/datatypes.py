from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml
import copy

@dataclass
class DataCfg:
    name: str = "bldc_csv"
    root: str = "data/processed"
    split: Dict[str, List[str]] = field(default_factory=lambda: {"train": [], "val": []})
    seq_len: int = 128
    batch_size: int = 64
    num_workers: int = 8
    sampler: str = "default"
    normalize: Dict[str, Any] = field(default_factory=dict)
    inject_last_channel: bool = True

@dataclass
class ModelCfg:
    name: str = "gpt_zerostep"
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1
    n_u: int = 5
    n_y: int = 1
    n_x: int = 2
    block_size: int = 128
    bias: bool = False

@dataclass
class TrainCfg:
    epochs: int = 200
    eval_interval: int = 1
    eval_iters: int = 10
    patience: int = 20
    grad_clip: Optional[float] = 1.0
    mixed_precision: str = "off"  # "off"|"fp16"|"bf16"

@dataclass
class OptimCfg:
    name: str = "adamw"
    lr: float = 3e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.01
    scheduler: Dict[str, Any] = field(default_factory=lambda: {"name": "warmup_cosine", "warmup_steps": 1000, "max_steps": 60000, "min_lr_scale": 0.1})

@dataclass
class LossCfg:
    components: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {"mse": {"weight": 1.0}})

@dataclass
class RootCfg:
    seed: int = 1337
    device: str = "cuda"
    exp_name: str = "bldc_zerostep"
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    loss: LossCfg = field(default_factory=LossCfg)

def _deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = _deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def merge_cfg(*dicts: dict) -> dict:
    out = {}
    for d in dicts:
        _deep_update(out, copy.deepcopy(d))
    return out

def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        ref = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in ref or not isinstance(ref[p], dict):
                ref[p] = {}
            ref = ref[p]
        # try to parse numbers/booleans
        v = val
        if v.lower() in {"true","false"}:
            v = v.lower() == "true"
        else:
            try:
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except Exception:
                pass
        ref[parts[-1]] = v
    return cfg

def dict_to_cfg(d: dict) -> RootCfg:
    # simple manual mapping
    return RootCfg(
        seed=d.get("seed", 1337),
        device=d.get("device", "cuda"),
        exp_name=d.get("exp_name", "bldc_zerostep"),
        data=DataCfg(**d.get("data", {})),
        model=ModelCfg(**d.get("model", {})),
        train=TrainCfg(**d.get("train", {})),
        optim=OptimCfg(**d.get("optim", {})),
        loss=LossCfg(**d.get("loss", {})),
    )


# CHECKED -- all good!