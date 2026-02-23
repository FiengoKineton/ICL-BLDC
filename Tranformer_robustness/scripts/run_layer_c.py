# scripts/run_layer_c.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.blocks.model import GPTConfig
from src.blocks.optim import OptimConfig
from src.analysis.layer_c import LayerCConfig, sweep_optimizer_sensitivity, algorithmic_stability_proxy_leave_one_batch_out

def main():
    cfg = GPTConfig()
    B, T, n_u = 64, cfg.block_size, cfg.n_u

    U = torch.randn(1024, T, n_u)
    Y = torch.randn(1024, T, 1)

    ds = TensorDataset(U, Y)
    base_train = DataLoader(ds, batch_size=B, shuffle=True)
    val_loader = DataLoader(ds, batch_size=B, shuffle=False)

    model_cfg = cfg
    base_optim = OptimConfig(name="adamw", lr=1e-3, weight_decay=0.0)
    layerc_cfg = LayerCConfig(epochs=3)

    # 1) Sensitivity sweep
    res = sweep_optimizer_sensitivity(
        train_loader=base_train,
        val_loader=val_loader,
        model_cfg=model_cfg,
        base_optim=base_optim,
        sweep_lrs=[3e-4, 1e-3, 3e-3],
        sweep_wds=[0.0, 1e-4, 1e-3],
        layerc_cfg=layerc_cfg
    )
    print("Sensitivity sweep results:")
    for r in res:
        print(r)

    # 2) Algorithmic stability proxy: “remove one batch”
    # Build multiple train loaders where we drop the first k batches (simple proxy)
    loaders = []
    full_batches = list(iter(DataLoader(ds, batch_size=B, shuffle=False)))
    for drop_idx in range(min(3, len(full_batches))):
        kept = [b for i,b in enumerate(full_batches) if i != drop_idx]
        # wrap kept batches as an iterable
        def iter_kept(kept_batches):
            for u,y in kept_batches:
                yield u, y
        loaders.append(iter_kept(kept))

    stab = algorithmic_stability_proxy_leave_one_batch_out(
        train_loaders=loaders,
        val_loader=val_loader,
        model_cfg=model_cfg,
        optim_cfg=base_optim,
        layerc_cfg=layerc_cfg
    )
    print("Algorithmic stability proxy:", stab)

if __name__ == "__main__":
    main()
    # >> python -m scripts.run_layer_c