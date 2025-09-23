from pathlib import Path
import time
import torch
import numpy as np
import math
import gc
from functools import partial
from torch.utils.data import DataLoader
from transformer_zerostep import GPTConfig, GPT, warmup_cosine_lr
import argparse
import warnings
import wandb
import torch.nn as nn
import pandas as pd
import copy
import os

### quick param selection
"""
# --- EXPERIMENT TAGS / I/O ---
# checkpoint_name_to_save controls the filename prefix and the W&B run name.
# mode:
#   "scratch"   → new model from cfg
#   "resume"    → load state dict + optimizer + losses and continue training
#   "pretrained"→ load weights only, keep new optimizer/lr schedule.
#
# sequence_length (H) is the causal context length used by the Transformer.
# It MUST be ≤ GPTConfig.block_size and equals the window length extracted by Dataset.
#
# NOTE (dataset choice):
# - dataset.py  → uniform random windows (no stratification).
# - dataset_alt.py → "alternative batch extractor": draws windows with constraints
#   (e.g., include at least one sample above 2000 rpm). This is *important* to keep
#   a balanced curriculum across speed bands and avoid low-rpm bias."""


checkpoint_name_to_save = "noise_h10"
checkpoint_name_to_open = "noise_h10"
mode = "scratch"  # resume / scratch / pretrained

# model parameters
sequence_length = 10 #h
layers_number = 8 #8
heads_number = 4 #4
embd_number = 16 #16

# training parameters
batch_size_ = 128
max_iteration_number = 40_000
learning_rate_value = 1e-5


# standard batch extractor selects a random window of length h, from a random experiment, with a uniform probability. 
# the alternative one enforces the extraction of windows that possess specific characteristics with a certain probability,
# e.g. 50% chance of extracting a sample window in which the speed is >2000RPM at least once
alternative_batch_extractor = True

# whether or not to log training data on wandb
"""
# Set wandb_record=False to avoid logging when prototyping. When True, each run is
# tagged by checkpoint_name_to_save; metrics you want to see there: loss, val_loss,
# current lr, and optionally banded NRMSE (0–300/300–800/800–2500 rpm).
#
# data_path is built assuming the repo is named "in-context-bldc".
# If you move folders around, prefer Path(...).resolve() to avoid fragile splits."""

wandb_record = False

current_path = os.getcwd().split("ICL-BLDC")[0]
data_path = os.path.join(current_path,"ICL-BLDC", "data")


# multiple folders can be selected
folder_training = ["simulated/50_percent_control_with_noise/training", "simulated/50_percent_control_current_disturbance_with_noise/training", "simulated/50_percent_control_perturbed_with_noise/training"]
# folder_training = ["simulated/50_percent_control/training", "simulated/50_percent_control_current_disturbance/training", "simulated/50_percent_control_perturbed/training"]
# folder_training = ["simulated/50_percent_control/training", "simulated/50_percent_control_perturbed/training"]
folder_path_training = [os.path.join(data_path, folder) for folder in folder_training]

folder_vaildation = ["simulated/50_percent_control_with_noise/validation", "simulated/50_percent_control_current_disturbance_with_noise/validation", "simulated/50_percent_control_perturbed_with_noise/validation"]
# folder_vaildation = ["simulated/50_percent_control/validation", "simulated/50_percent_control_perturbed/validation"]
folder_path_val = [os.path.join(data_path, folder) for folder in folder_vaildation]

if alternative_batch_extractor:
    """
    # Both Datasets must return:
    #   batch_u: float32 tensor [B, H, n_u]  (normalized inputs)
    #   batch_y: float32 tensor [B, H, 1]    (normalized targets, ω)
    # IMPORTANT: If you rely on "teacher forcing" of ω_{t-1}, the Dataset should fill
    # the 'last_omega' channel with *ground-truth* ω_{t-1}. If you want robustness
    # to deployment (autoregressive), see the scheduled sampling snippet below.
    """
    from dataset_alt import Dataset, load_dataframes_from_folder
else:
    from dataset import Dataset, load_dataframes_from_folder




# Disable all user warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Re-enable user warnings
# warnings.filterwarnings("default")



# def train(model, dataloader, criterion, optimizer, device):
#     '''
#     Trains the model over the given data batches. Along the windows of length h, the model estimates recursively the output omega_hat_t, with t = 1...h.
#     At each iteration the model receives as input the previous estimated outputs, which is initialized at 0 e.g. omega_hat_3 = f(..., [0, omega_hat_1, omega_hat_2]). Performs back-propagation to update the model weights. Returns the training loss, as the mse between the recursively obtained output estimations, and the real outputs inside the window.
#     '''
#     torch.autograd.set_detect_anomaly(True)
#     model.train()
#     running_loss = 0.0
    
#     for batch in dataloader:
#         batch_u, batch_y = batch
#         batch_u, batch_y = batch_u.to(device), batch_y.to(device)

#         optimizer.zero_grad()  # Clear previous gradients

#         # Create a copy of batch_u to work with, and set the velocity column (index 4) to zero
#         batch_u_copy = batch_u.clone()
#         batch_u_copy[:,:,4] = 0  

#         # Store predictions
#         last_predictions = torch.zeros(batch_u_copy.shape[0], device=device, requires_grad=True) # batch_u_copy.shape[0] is the batch size
#         batch_y_pred_list = []  # list to accumulate outputs

#         # Simulate step by step
#         for t in range(batch_u_copy.shape[1]):
#             batch_u_step = batch_u_copy.clone()  # Clone to avoid modification issues
#             batch_u_step[:, t, 4] = last_predictions  # Inject last predictions
#             batch_u_tmp = batch_u_step[:, :t+1, :]  # Take relevant time slice

#             # Forward pass
#             last_predictions = model(batch_u_tmp)[:, -1, :].view(-1)  # Ensure shape matches

#             batch_y_pred_list.append(last_predictions.unsqueeze(1))  # Store prediction

#         # Concatenate all predictions along time dimension
#         batch_y_pred = torch.cat(batch_y_pred_list, dim=1).unsqueeze(-1)  # Ensure shape matches batch_y

#         # Compute loss
#         loss = criterion(batch_y, batch_y_pred)

#         # Backpropagation
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#         # Debugging: Check if all parameters have gradients
#         for name, param in model.named_parameters():
#             if param.grad is None:
#                 print(f"Warning: No gradient computed for {name}")

#     return running_loss / len(dataloader)



# def validate(model, dataloader, criterion, device):
#     '''
#     Evaluates the model over the given data batches. Along the windows of length h, the model estimates recursively the output omega_hat_t, with t = 1...h.
#     At each iteration the model receives as input the previous estimated outputs, which is initialized at 0 e.g. omega_hat_3 = f(..., [0, omega_hat_1, omega_hat_2]). Returns the validation loss, as the mse between the recursively obtained output estimations, and the real outputs inside the window.
#     '''
#     model.eval()
#     running_loss = 0.0
#     with torch.no_grad():
#         for batch in dataloader:
#             batch_u, batch_y = batch
#             batch_u, batch_y = batch_u.to(device), batch_y.to(device)

#             batch_y_pred = torch.zeros_like(batch_y)
        
#             # create a copy of batch_u to work with, then overwrite the real velocity (symbolic, may not be needed for the code)
#             batch_u_copy = batch_u.clone().detach()
#             batch_u_copy[:,:,4] = 0

#             # simulate step by step
#             last_predictions = torch.zeros(batch_u_copy.shape[0], device=device) # batch_u_copy.shape[0] is the batch size

#             for t in range(batch_u_copy.shape[1]):
#                 batch_u_step = batch_u_copy.clone()
#                 batch_u_step[:,t,4] = last_predictions
#                 batch_u_tmp = batch_u_step[:,:t+1,:]
#                 #update last predictions
#                 last_predictions = model(batch_u_tmp)[:,-1,:].view(-1)
#                 batch_y_pred[:,t,0] = last_predictions

#             loss = criterion(batch_y, batch_y_pred)

#             running_loss += loss.item()

#     return running_loss / len(dataloader)


def train(model, dataloader, criterion, optimizer, device):
    """
    One-step/parallel training:
      - Expects batch_u=[B,H,n_u], batch_y=[B,H,1].
      - The model outputs [B,H,1] in parallel (no inner loop), i.e. we DO NOT
        overwrite the 'last_omega' channel with the model's own prediction during
        training (pure teacher forcing if the Dataset fills last_omega with truth).
      - Pros: faster, stable gradients. Cons: train/test mismatch if at inference
        you feed back ω̂_{t-1}. Mitigation: scheduled sampling (see below).
    """
    torch.autograd.set_detect_anomaly(True)
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        batch_u, batch_y = batch
        batch_u, batch_y = batch_u.to(device), batch_y.to(device)

        optimizer.zero_grad()

        """# Forward pass: model returns ω̂ over the whole window in one shot
        # (causal attention ensures x[:,t] never sees future tokens)."""
        batch_y_pred = model(batch_u)

        """# MSE between normalized ω and ω̂. If you later add uncertainty (μ, logσ²),
        # replace this with a Gaussian NLL."""
        loss = criterion(batch_y[:, :, 0], batch_y_pred[:, :, 0])

        loss.backward()
        ### torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ← safe clip
        optimizer.step()

        running_loss += loss.item()

        # for name, param in model.gpt_model.named_parameters():
        #     if name == "proportional_coefficient":
        #         print(f"Parameter Value: {param}")
        for name, param in model.named_parameters():
            if param.grad is None:
                # Debug: check missing gradients (usually means the parameter is detached or unused).
                print(f"No gradient computed for {name}")

    return running_loss / len(dataloader)


def train_autoreg(model, dataloader, criterion, optimizer, device, p_sched=0.3):
    """
    Scheduled-sampling autoregressive train:
    at each t, with prob p_sched feed model's ω̂_{t-1}, else feed ground-truth ω_{t-1}.
    This shrinks train→test mismatch and stabilizes branch selection beyond aliasing.
    """
    model.train(); total=0.0
    for batch_u, batch_y in dataloader:
        batch_u = batch_u.to(device); batch_y = batch_y.to(device)
        optimizer.zero_grad()
        # start from a copy with zeroed feedback
        bu = batch_u.clone(); bu[:,:,4] = 0.0   # 5th channel = last_omega
        preds = []
        last = torch.zeros(bu.size(0), device=device)
        for t in range(bu.size(1)):
            feed = bu.clone()
            feed[:, t, 4] = last
            out = model(feed)[:, t, 0]  # predict ω̂_t
            preds.append(out)
            # scheduled sampling
            gt = batch_y[:, t, 0]
            use_model = (torch.rand_like(gt) < p_sched).float()
            last = use_model * out.detach() + (1 - use_model) * gt
        yhat = torch.stack(preds, dim=1)  # [B,H]
        loss = criterion(batch_y[:,:,0], yhat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total/len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validation mirrors training mode: if you trained autoregressively, validate the same way.
    Otherwise do the fast parallel pass to monitor MSE. Keep shapes: [B,H,1].
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch_u, batch_y = batch
            batch_u, batch_y = batch_u.to(device), batch_y.to(device)

            batch_y_pred = model(batch_u)

            loss = criterion(batch_y[:, :, 0], batch_y_pred[:, :, 0])
            running_loss += loss.item()

    """
    # --- OPTIONAL: band-wise metrics (requires denormalized rpm or a mask from Dataset) ---
    # try:
    #     # Suppose dataloader yields (batch_u, batch_y, rpm_denorm)
    #     # and you collected yhat for the whole val set into lists yhat_list, rpm_list.
    #     # Here only a schematic example:
    #     pass
    # except Exception:
    #     pass"""
    return running_loss / len(dataloader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meta system identification with transformers')
    """
    # nx/nu/ny are *not* state-space orders; here they are dimensionalities:
    #   n_u = number of input channels (e.g., ia, ib, va, vb, last_omega, ...),
    #   n_y = outputs (1: ω),
    #   n_x is unused in the GPT but kept for compatibility with other models.
    # seq-len = window H used as the Transformer context (also set as block_size)."""

    # Overall
    parser.add_argument('--model-dir', type=str, default="out", metavar='S',
                        help='Saved model folder')
    parser.add_argument('--out-file', type=str, default=checkpoint_name_to_save, metavar='S',
                        help='Saved model name')
    parser.add_argument('--in-file', type=str, default=checkpoint_name_to_open, metavar='S',
                        help='Loaded model name (when resuming)')
    parser.add_argument('--init-from', type=str, default=mode, metavar='S',
                        help='Init from (scratch|resume|pretrained)')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='Seed for random number generation')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='disables CUDA training')

    # Dataset
    parser.add_argument('--nx', type=int, default=4, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--nu', type=int, default=6, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--ny', type=int, default=1, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--seq-len', type=int, default=sequence_length, metavar='N',
                        help='sequence length (default: 600)')
    parser.add_argument('--mag_range', type=tuple, default=(0.5, 0.97), metavar='N',
                        help='sequence length (default: 600)')
    parser.add_argument('--phase_range', type=tuple, default=(0.0, math.pi/2), metavar='N',
                        help='sequence length (default: 600)')
    parser.add_argument('--fixed-system', action='store_true', default=False,
                        help='If True, keep the same model all the times')

    # Model
    parser.add_argument('--n-layer', type=int, default=layers_number, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--n-head', type=int, default=heads_number, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--n-embd', type=int, default=embd_number, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--dropout', type=float, default=0, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='bias in model')

    # Training
    parser.add_argument('--batch-size', type=int, default=batch_size_, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--max-iters', type=int, default= max_iteration_number, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--warmup-iters', type=int, default=5_000, metavar='N',
                        help='number of iterations (default: 1000)')
    parser.add_argument('--lr', type=float, default=learning_rate_value, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='D',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--eval-interval', type=int, default=10, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--eval-iters', type=int, default=10, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--fixed-lr', action='store_true', default=False,
                        help='disables CUDA training')

    # Compute
    parser.add_argument('--threads', type=int, default=16,
                        help='number of CPU threads (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda-device', type=str, default="cuda:0", metavar='S',
                        help='cuda device (default: "cuda")')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='disables CUDA training')

    cfg = parser.parse_args()

    # Other settings
    cfg.beta1 = 0.9
    cfg.beta2 = 0.95

    print(cfg.seq_len)

    # Derived settings
    n_skip = 0
    cfg.block_size = cfg.seq_len
    cfg.lr_decay_iters = cfg.max_iters
    cfg.min_lr = cfg.lr/10.0  #
    cfg.decay_lr = not cfg.fixed_lr
    cfg.eval_batch_size = cfg.batch_size

    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed) # not needed? All randomness now handled with generators

    # Create out dir
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(exist_ok=True)

    # Configure compute
    cuda_device = "cuda:0"

    torch.set_num_threads(cfg.threads)
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device_name = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = 'cuda' if 'cuda' in device_name else 'cpu' # for later use in torch.autocast
    torch.set_float32_matmul_precision("high")

    """# Device selection. Avoid calling CUDA APIs when running on CPU.
    if device_type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")  # tf32 for speed on Ampere+
    else:
        print("Running on CPU. Consider enabling CUDA for speed.")"""
    torch.cuda.set_device(device)
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())

    """
    # ---------------------------------------------------------------------------
    # DATA EXPECTATIONS (folders → CSVs)
    # Each folder must contain CSV files with columns:
    #   t, theta, omega, r, ia, ib, iq_ref, va, vb
    # The Dataset will:
    #   - normalize physical ranges (i, v, ω),
    #   - build sliding windows of length H=cfg.seq_len,
    #   - stack inputs as [ia, ib, va, vb, last_omega, ...] → n_u channels,
    #   - set targets to ω (1 channel).
    # NOTE: If you rely on a feedback channel 'last_omega', the Dataset should
    # fill it with ground-truth ω_{t-1} (teacher forcing). If you later switch to
    # scheduled sampling, you’ll overwrite this channel inside train().
    # ---------------------------------------------------------------------------"""
    # Load all your DataFrames (replace with your data loading code)
    # folder_path = '../data/CL_experiments/train/inertia13_ki-0.0061-kp-11.8427'
    dfs = []
    for path_iter in folder_path_training:
        new_dfs = load_dataframes_from_folder(path_iter)
        dfs= dfs + new_dfs
        print(f"Loaded {len(new_dfs)} DataFrames from {path_iter}.")

    train_ds = Dataset(dfs=dfs, seq_len=cfg.seq_len)

    """
    # --- OPTIONAL: stratified sampling by speed bands ---
    # from torch.utils.data import WeightedRandomSampler
    # weights = train_ds.window_weights_by_speed([0,300,800,2500])  # implement in Dataset
    # sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    # train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, pin_memory=True, sampler=sampler)"""
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, pin_memory=True, shuffle=True)

    dfs_val = []
    for path_iter in folder_path_val:
        new_dfs = load_dataframes_from_folder(path_iter)
        dfs_val = dfs_val + new_dfs
        print(f"Loaded {len(new_dfs)} DataFrames from {path_iter}.")

    val_ds = Dataset(dfs=dfs_val, seq_len=cfg.seq_len)
    val_dl = DataLoader(val_ds, batch_size=cfg.eval_batch_size, pin_memory=True, shuffle=True)


    print("saving model in: ", checkpoint_name_to_save)
    if mode != "scratch":
        print("starting from model: ", checkpoint_name_to_open, " (", mode, ")")
    print("sequence length: ", sequence_length)
    print("max iterations: ", max_iteration_number)
    print("batch size: ", batch_size_)
    print("learning rate: ", learning_rate_value)
    print("layers: ", layers_number)
    print("heads: ", heads_number)
    print("embd: ", embd_number)
    
    if alternative_batch_extractor:
        print("using alternative batch extractor")
    
    """# Print the full config so checkpoints are reproducible. Consider removing the input()
    # when running unattended (or guard it under a flag)."""
    input("everything ok?")

    
    if wandb_record:
        os.environ["WANDB_ENTITY"]  = "g7-fiengo"
        os.environ["WANDB_PROJECT"] = "in-context-bldc"
        os.environ.pop("WANDB_BASE_URL", None)  # assicura cloud standard

        run = wandb.init(
            entity="g7-fiengo",
            project="in-context-bldc",
            name=checkpoint_name_to_save,
            reinit=True,
            mode="online",
        )
        print("W&B -> entity:", run.entity, "| project:", run.project)

    # Model
    model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, n_x=cfg.nx, n_y=cfg.ny, n_u=cfg.nu, block_size=cfg.block_size,
                      bias=cfg.bias, dropout=cfg.dropout)  # start with model_args from command line

    if cfg.init_from == "scratch":
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif cfg.init_from == "resume" or cfg.init_from == "pretrained":
        ckpt_path = model_dir / f"{cfg.in_file}.pt"
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        print("Using all the GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    if cfg.compile:
        model = torch.compile(model)  # requires PyTorch 2.0

    # Optimizer
    # Check if model is wrapped by DataParallel
    if isinstance(model, torch.nn.DataParallel):
        optimizer = model.module.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)
    else:
        optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)

    if cfg.init_from == "resume":
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Criterion
    criterion = torch.nn.MSELoss()

    # Training and validation loop
    LOSS_ITR = []
    LOSS_VAL = []
    best_val_loss = float('inf')

    if cfg.init_from == ("scratch") or cfg.init_from == "pretrained":
        # Training and validation loop
        LOSS_ITR = []
        LOSS_VAL = []
        iter_num = 0
        best_val_loss = np.inf
        train_time = 0.0
    elif cfg.init_from == "resume":
        # Training and validation loop
        LOSS_ITR = checkpoint['LOSS']
        LOSS_VAL = checkpoint['LOSS_VAL']
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint['best_val_loss']
        train_time = checkpoint['train_time']

    get_lr = partial(warmup_cosine_lr, lr=cfg.lr, min_lr=cfg.min_lr,
                     warmup_iters=cfg.warmup_iters, lr_decay_iters=cfg.lr_decay_iters)
    time_start = time.time()

    best_epoch = iter_num -1
    for epoch in range(iter_num+1, cfg.max_iters):
        patience = 5000  # iterations without improvement (tune this)
        no_improve = 0

        #########################################
        # aggiungi qua una stringa da mette nel file di check per savere il best model ogni 10k tipo iterazioni (e.g. checkpoint_stocazzo_***10k***.pt)

        frequency = 10 #k

        current_block = (epoch // (frequency * 1000) + 1) * frequency

        """# Save "best so far" as <name>_10k.pt, <name>_20k.pt, ... for long runs,
        # so you can backtrack to earlier minima if you overfit later."""
        name_suffix = '_' + str(current_block) + 'k'

        checkpoint_name_to_save_file = checkpoint_name_to_save + name_suffix




        ## I COMMENTED THIS PART BECAUSE THERE WAS A PROBLEM WITH LR : IT WAS STUCK TO 0
        if cfg.decay_lr:
            lr_iter = get_lr(epoch)
        else:
            lr_iter = cfg.lr
        optimizer.param_groups[0]['lr'] = lr_iter

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.lr_decay_iters)
        train_loss = train(model, train_dl, criterion, optimizer, device)
        val_loss = validate(model, val_dl, criterion, device)
        #scheduler.step()
        # print("...")

        LOSS_ITR.append(train_loss)
        LOSS_VAL.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': epoch,
                'train_time': time.time() - time_start + train_time,
                'LOSS': LOSS_ITR,
                'LOSS_VAL': LOSS_VAL,
                'best_val_loss': best_val_loss,
                'cfg': cfg,
            }

            torch.save(checkpoint, model_dir / f"{checkpoint_name_to_save_file}.pt")
            no_improve = 0
        else:
            no_improve += 1
        # --- EARLY STOP (optional) ---
        # if no_improve >= patience:
        #     print(f"Early stopping at iter {epoch} (patience={patience})")
        #     break

        
        print("model: ", checkpoint_name_to_save)
        print(f"Epoch [{epoch}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}, best val loss was: {best_val_loss:.4f}")
        if wandb_record:
            wandb.log({"epoch": epoch, "loss": train_loss, "val_loss": val_loss, "best_epoch": best_epoch})

    print("Training complete. Best model saved as: ", checkpoint_name_to_save)


    # when the last iteration is reached, a checkpoint is saved, just to be able to see how the training and validation losses progressed after finding a minimum validation loss point
    checkpoint = {
                'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': cfg.max_iters,
                'train_time': time.time() - time_start + train_time,
                'LOSS': LOSS_ITR,
                'LOSS_VAL': LOSS_VAL,
                'best_val_loss': best_val_loss,
                'cfg': cfg,
    }

    """# Save the "last" checkpoint (even if not best) to inspect loss curves.
    # For production use, load the best_k.pt; for research, sometimes the final model has
    # interesting generalization on specific bands even if val MSE is worse globally."""
    torch.save(checkpoint, model_dir / f"{cfg.out_file}_loss_check.pt")