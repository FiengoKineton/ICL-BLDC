# scripts/run_layer_b.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.blocks.model import GPTConfig
from src.blocks.optim import OptimConfig
from src.analysis.layer_b import LayerBConfig, train_and_score_inputset

def main():
    # dummy data (replace with your real loaders)
    cfg = GPTConfig()
    B, T, n_u = 64, cfg.block_size, cfg.n_u

    U = torch.randn(512, T, n_u)
    Y = torch.randn(512, T, 1)

    ds = TensorDataset(U, Y)
    train_loader = DataLoader(ds, batch_size=B, shuffle=True)
    val_loader = DataLoader(ds, batch_size=B, shuffle=False)

    model_cfg = cfg
    optim_cfg = OptimConfig(name="adamw", lr=1e-3, weight_decay=0.0)

    layerb_cfg = LayerBConfig(epochs=3, noise_std=1e-3)

    candidates = [
        [0,1,2,3,4],  # all
        [0,1,2,3],    # no feedback channel (still exists but zeroed)
        [2,3,4],      # voltages + feedback only
    ]

    for keep in candidates:
        stats = train_and_score_inputset(
            train_loader, val_loader,
            model_cfg=model_cfg, optim_cfg=optim_cfg,
            layerb_cfg=layerb_cfg,
            input_keep=keep
        )
        print(f"keep={keep} -> {stats}")

if __name__ == "__main__":
    main()
    # >> python -m scripts.run_layer_b