# engine_utils.py       | SET (leave it as it is)

import math, torch, torch.nn as nn
from transformer_zerostep import GPTConfig, GPT
from typing import Dict, Any, Optional



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

    # NOTE:
    # A GPT-style Transformer is used here to model continuous-time signals by
    # learning long-range temporal dependencies via attention, without imposing
    # an explicit state-space or differential equation structure. This is useful
    # for capturing nonlocal and multiscale temporal correlations in the data,
    # but does not guarantee physical consistency or stability outside the
    # observed time horizon.


class TimeWeightedMSELoss(nn.Module):
    """
    Time-weighted MSE for sequences: y_pred, y_true in (B, T, D).
    Weights increase with time so later steps matter more.
    """
    def __init__(self, mode: str = "linear", alpha: float = 2.0, eps: float = 1e-12):
        super().__init__()
        if mode not in ("linear", "exp"):
            raise ValueError("mode must be 'linear' or 'exp'")
        self.mode = mode
        self.alpha = alpha
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: {y_pred.shape=} vs {y_true.shape=}")
        if y_pred.dim() != 3:
            raise ValueError(f"Expected (B,T,D) tensors, got {y_pred.dim()} dims")

        B, T, D = y_pred.shape

        # per-time MSE: (B,T)
        mse_t = ((y_pred - y_true) ** 2).mean(dim=-1)

        # weights: (T,)
        if self.mode == "linear":
            w = torch.arange(1, T + 1, device=y_pred.device, dtype=y_pred.dtype) / T
        else:  # "exp"
            if T == 1:
                return mse_t.mean()
            t = torch.linspace(0, 1, T, device=y_pred.device, dtype=y_pred.dtype)
            w = torch.exp(self.alpha * t)

        # normalize weights to sum to 1 (stable across different T)
        w = w / (w.sum() + self.eps)          # (T,)
        w = w.view(1, T)                      # (1,T) broadcasts over batch

        # weighted mean over time, then mean over batch
        loss = (mse_t * w).sum(dim=1).mean()
        return loss



def teacher_prob_schedule(epoch: int, p0: float, decay_epochs: int, p_min: float = 0.0) -> float:
    """Linear decay from p0 to p_min over decay_epochs, then stays at p_min."""
    if decay_epochs is None or decay_epochs <= 0:
        return float(p_min)
    frac = min(1.0, max(0.0, epoch / float(decay_epochs)))
    p = p0 * (1.0 - frac)
    return float(max(p_min, p))

def regression(
        batch_u,
        model,
        device,
        regression_mode: str = "time_last",
        batch_y: torch.Tensor | None = None,
        teacher_prob: float = 0.0,
        ss_mode: str = "stochastic",   # "stochastic" or "soft"
):
    # Create a copy of batch_u and zero the feedback channel
    batch_u_copy = batch_u.clone()
    batch_u_copy[:, :, 4] = 0

    if regression_mode != "one_step":

        # Initialize ω̂_0 = 0
        last_predictions = torch.zeros(
            batch_u_copy.shape[0],
            device=device,
            requires_grad=True
        )

        batch_y_pred_list = []

        for t in range(batch_u_copy.shape[1]):

            batch_u_step = batch_u_copy.clone()

            if t > 0 and teacher_prob > 0.0 and batch_y is not None:
                y_prev = batch_y[:, t-1, 0]

                if ss_mode == "stochastic":
                    m = (torch.rand_like(last_predictions) < teacher_prob).float()
                    inject = m * y_prev + (1.0 - m) * last_predictions
                elif ss_mode == "soft":
                    inject = teacher_prob * y_prev + (1.0 - teacher_prob) * last_predictions
                else:
                    raise ValueError(f"Unknown ss_mode: {ss_mode}")
            else:
                inject = last_predictions

            batch_u_step[:, t, 4] = inject
            batch_u_tmp = batch_u_step[:, :t+1, :]

            prediction_full = model(batch_u_tmp)
            last_predictions = prediction_full[:, -1, :].view(-1)

            batch_y_pred_list.append(last_predictions.unsqueeze(1))

        if regression_mode == "time_last":
            batch_y_pred = torch.cat(batch_y_pred_list, dim=1).unsqueeze(-1)
        elif regression_mode == "time_full":
            batch_y_pred = prediction_full

    else:
        batch_y_pred = model(batch_u_copy[:, :, :-1])

    return batch_y_pred

def smooth_dynamics_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    R_smooth: float,
    thresh: float = 0.01,
    criterion: Optional[torch.nn.Module] = None,
    ):
    """
    Computes the dynamics-matching regularization term.

    Args:
        y_true: (B, H, 1)
        y_pred: (B, H, 1)
        R_smooth: regularization weight (scalar)
        thresh: threshold for weighting the derivative penalty

    Returns:
        scalar tensor (already scaled by R_smooth^2)
    """
    if R_smooth is None or R_smooth == 0.0:
        return 0.0

    dy_true = y_true.diff(dim=1)      # (B, H-1, 1)
    dy_pred = y_pred.diff(dim=1)      # (B, H-1, 1)

    mag = dy_true.abs()
    w_dyn = (mag / thresh).clamp(0.0, 1.0)

    target_dy = w_dyn * dy_true #+ (1.0 - w_dyn) * 0.0
    try:    
        loss_dyn = criterion(target_dy, dy_pred)
    except Exception as e:
        loss_dyn = ((dy_pred - target_dy) ** 2).mean()

    return (R_smooth ** 2) * loss_dyn

def _regression(
        batch_u, model, device, 
        regression_mode: str = "time_last"
        ):
    # Create a copy of batch_u to work with, and set the velocity column (index 4) to zero
    batch_u_copy = batch_u.clone()
    batch_u_copy[:,:,4] = 0 

    if regression_mode != "one_step":
        # Store predictions
        # Initialize ω̂_0 = 0 for all sequences in the batch
        # Requires grad=True so gradients can flow through the unrolled recursion.
        last_predictions = torch.zeros(batch_u_copy.shape[0], device=device, requires_grad=True) # batch_u_copy.shape[0] is the batch size
        batch_y_pred_list = []  # list to accumulate outputs

        # Simulate step by step
        for t in range(batch_u_copy.shape[1]):
            # Inject previous estimate into the 5th channel at current step t
            # u_step[:, :t+1, :] provides a strictly-causal prefix to the Transformer
            # ŷ_t is obtained by taking the last time position of the model output
            batch_u_step = batch_u_copy.clone()  # Clone to avoid modification issues
            batch_u_step[:, t, 4] = last_predictions  # Inject last predictions
            batch_u_tmp = batch_u_step[:, :t+1, :]  # Take relevant time slice

            # Forward pass
            prediction_full = model(batch_u_tmp)
            last_predictions = prediction_full[:, -1, :].view(-1)  # Ensure shape matches

            batch_y_pred_list.append(last_predictions.unsqueeze(1))  # Store prediction



        if regression_mode == "time_last":
            # Concatenate all predictions along time dimension
            batch_y_pred = torch.cat(batch_y_pred_list, dim=1).unsqueeze(-1)  # Ensure shape matches batch_y
            # batch_y_pred: concatenated per-step predictions ŷ_{1:H} with shape (B,H,1)
            # Criterion compares full sequences (teacher-free schedule because we feed our own ŷ).
        elif regression_mode == "time_full":
            batch_y_pred = prediction_full
        
    else: 
        # One-step prediction mode: directly predict from full input sequence
        batch_y_pred = model(batch_u_copy[:,:,:-1])
    
    return batch_y_pred


