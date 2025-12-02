import torch, torch.nn as nn
from transformer_zerostep import GPTConfig, GPT
from typing import Dict, Any


def build_device(cfg_compute: Dict[str, Any]):
    torch.set_num_threads(cfg_compute.get("threads", 16))

    no_cuda = cfg_compute.get("no_cuda", False)
    cuda_device = cfg_compute.get("cuda_device", "cuda:0")

    use_cuda = (not no_cuda) and torch.cuda.is_available()
    device_name = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = "cuda" if "cuda" in device_name else "cpu"

    if device_type == "cuda":
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
        print(f"[device] CUDA available: {torch.cuda.is_available()}")
        print(f"[device] Current device: {torch.cuda.current_device()}")

    print(f"[device] Using device: {device_name}")
    return device, device_type


def build_model(cfg_model: Dict[str, Any],
                cfg_data: Dict[str, Any],
                device,
                device_type: str):
    model_args = dict(
        n_layer=cfg_model["n_layer"],
        n_head=cfg_model["n_head"],
        n_embd=cfg_model["n_embd"],
        n_x=cfg_data["nx"],
        n_y=cfg_data["ny"],
        n_u=cfg_data["nu"],
        block_size=cfg_data["seq_len"],
        bias=cfg_model.get("bias", False),
        dropout=cfg_model.get("dropout", 0.0),
    )

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    if torch.cuda.device_count() > 1 and device_type == "cuda":
        print("[model] Using DataParallel on all GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    if cfg_model.get("compile", False):
        model = torch.compile(model)

    return model, model_args


def configure_optimizer(model, cfg_training: Dict[str, Any], device_type: str):
    if isinstance(model, nn.DataParallel):
        optim_target = model.module
    else:
        optim_target = model

    optimizer = optim_target.configure_optimizers(
        cfg_training.get("weight_decay", 0.0),
        cfg_training["lr"],
        (0.9, 0.95),
        device_type
    )
    return optimizer

