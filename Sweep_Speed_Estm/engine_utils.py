# engine_utils.py

import math, torch, torch.nn as nn
from transformer_zerostep import GPTConfig, GPT
from typing import Dict, Any



def build_device(cfg_compute: Dict[str, Any], print_flag: bool = True):
    torch.set_num_threads(cfg_compute.get("threads", 16))

    no_cuda = cfg_compute.get("no_cuda", False)
    cuda_device = cfg_compute.get("cuda_device", "cuda:0")

    use_cuda = (not no_cuda) and torch.cuda.is_available()
    device_name = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = "cuda" if "cuda" in device_name else "cpu"

    if device_type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        #torch.set_float32_matmul_precision("high")

        if print_flag: print(f"[device] CUDA available: {torch.cuda.is_available()}")
        if print_flag: print(f"[device] Current device: {torch.cuda.current_device()}")
    else: 
        raise Warning("Running on CPU. This will be slow!")

    if print_flag: print(f"[device] Using device: {device_name}")
    return device, device_type


def build_model(cfg_model: Dict[str, Any],
                cfg_data: Dict[str, Any],
                device,
                device_type: str, 
                print_flag: bool = True, ):
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
        activation_function=cfg_model.get("activation_function", "gelu"),
    )

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, print_flag)

    if torch.cuda.device_count() > 1 and device_type == "cuda":
        if print_flag: print("[model] Using DataParallel on all GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    if cfg_model.get("compile", False):
        model = torch.compile(model)

    return model, model_args


def configure_optimizer(model, cfg_training: Dict[str, Any], device_type: str, print_flag: bool):
    if isinstance(model, nn.DataParallel):
        optim_target = model.module
    else:
        optim_target = model

    optimizer = optim_target.configure_optimizers(
        cfg_training.get("weight_decay", 0.0),
        cfg_training["lr"],
        (0.9, 0.95),
        device_type, 
        print_flag=print_flag,
    )
    return optimizer


def warmup_cosine_lr(iter, lr, min_lr, warmup_iters, lr_decay_iters):
    # LR schedule:
    # 1) Warmup: lr * (iter / warmup_iters), for iter < warmup_iters
    # 2) Cosine decay from lr → min_lr over [warmup_iters, lr_decay_iters]
    #    lr(iter) = min_lr + 0.5*(lr - min_lr)*(1 + cos(pi * progress))
    # 3) Clamp to min_lr after lr_decay_iters

    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return lr * iter / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)

