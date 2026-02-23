# scripts/run_layer_a.py
import torch
from src.blocks.model import DecoderOnlyCausalTransformer, GPTConfig
from src.analysis.layer_a import LayerAConfig, robustness_to_input_noise, closed_loop_gain_proxy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = GPTConfig()
    model = DecoderOnlyCausalTransformer(cfg).to(device)
    model.eval()

    # dummy batch (replace with real)
    B, T, n_u = 8, cfg.block_size, cfg.n_u
    batch_u = torch.randn(B, T, n_u, device=device)

    a_cfg = LayerAConfig(feedback_channel=4, noise_std=1e-3, eps=1e-4)

    rob = robustness_to_input_noise(model, batch_u, cfg=a_cfg, trials=50)
    gain = closed_loop_gain_proxy(model, batch_u, cfg=a_cfg, t=1, b=0) if T > 1 else 0.0

    print("Layer A robustness:", rob)
    print("Closed-loop gain proxy (t=1):", gain)

if __name__ == "__main__":
    main()
    # >> python -m scripts.run_layer_a