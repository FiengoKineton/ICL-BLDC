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
import os, sys
import itertools


# -----------------------------------------------------------------------------
# Training script for a causal (decoder-only) Transformer used as an ICL estimator.
# The model consumes a window of continuous tokens u_{1:T} (e.g. [ia, ib, va, vb, ω̂_{k-1}])
# and predicts the full sequence ŷ_{1:T} (e.g. speed). During training/inference we do
# closed-loop recursion: at time t we feed the *previous prediction* ω̂_{t-1} into channel 4.
#
# Math (autoregressive recursion):
#   ŷ_t = f_ϕ(u_{1:t}, ŷ_{1:t-1})  with  ŷ_0 ≜ 0
# Loss: MSE( y_{1:T}, ŷ_{1:T} )
#
# References: ICL state estimators (Busetto et al. IFAC'24), BLDC speed ICL (Colombo et al. 2025).
#
# Notes:
# - Data windows come normalized from dataset.py.
# - Channel index 4 (last_omega) is overwritten step-by-step with predictions.
# - This is *not* teacher forcing: the loop uses its own predictions → evaluates robustness.
# -----------------------------------------------------------------------------


### quick param selection
# Quick knobs for experimentation. Consider mirroring these into argparse defaults
# to have a single source of truth (prevents mismatches between CLI and here).


checkpoint_name_to_save = "test"
checkpoint_name_to_open = "test"
mode = "scratch"  # resume / scratch / pretrained


# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. #
# model parameters
sequence_length = 10 #h
layers_number = 8 #8
heads_number = 4 #4
embd_number = 16 #16

# training parameters
batch_size_ = 128
max_iteration_number = 10_000
learning_rate_value = 1e-5



# Winner: n_layer4_n_head4_n_embd32_lr0.0001_patience20_max_iters500_batch_size128
SWEEP_MODE = True
sweep = {
    "n_layer":    [4, 8],                # was "layers"
    "n_head":     [2, 4],                # was "heads" 
    "n_embd":     [16, 32],              # was "embd" | n_embd/n_head: int
    "lr":         [1e-4, 5e-5, 1e-5],
    "patience":   [20],
    "max_iters":  [500],
    "batch_size": [64, 128],
    # optionally also:
    # "seq_len":   [10, 20],
}


# standard batch extractor selects a random window of length h, from a random experiment, with a uniform probability. 
# the alternative one enforces the extraction of windows that possess specific characteristics with a certain probability,
# e.g. 50% chance of extracting a sample window in which the speed is >2000RPM at least once
alternative_batch_extractor = False

# whether or not to log training data on wandb
wandb_record = False

current_path = os.getcwd().split("ICL-BLDC")[0]
data_path = os.path.join(current_path,"ICL-BLDC", "data")


# multiple folders can be selected
folder_training = ["simulated/50_percent_low_speed"]
folder_path_training = [os.path.join(data_path, folder) for folder in folder_training]

folder_vaildation = ["simulated/50_percent_low_speed"]
folder_path_val = [os.path.join(data_path, folder) for folder in folder_vaildation]

if alternative_batch_extractor:
# Dataset contract:
#  - __getitem__ returns (batch_u, batch_y) where
#      batch_u: (B,H,5) with channel 4 reserved for ω̂_{t-1} injection at train/val time
#      batch_y: (B,H,1) ground-truth speed ω_t
#  - load_dataframes_from_folder aggregates per-experiment CSV/Parquet files
    from dataset_alt import Dataset, load_dataframes_from_folder
else:
    from dataset import Dataset, load_dataframes_from_folder
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. #



# Disable all user warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Re-enable user warnings
# warnings.filterwarnings("default")

if wandb_record:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="in-context bldc estimator",
        name=checkpoint_name_to_save
    )
    # We log only scalars (train/val loss). Avoid logging full batches to keep runs light.


# ========================================================================================================= #
# =========================================== UTILS ======================================================= #
# ========================================================================================================= #

def train(model, dataloader, criterion, optimizer, device):
    '''
    Trains the model over the given data batches. Along the windows of length h, the model estimates recursively the output omega_hat_t, with t = 1...h.
    At each iteration the model receives as input the previous estimated outputs, which is initialized at 0 e.g. omega_hat_3 = f(..., [0, omega_hat_1, omega_hat_2]). Performs back-propagation to update the model weights. Returns the training loss, as the mse between the recursively obtained output estimations, and the real outputs inside the window.
    One epoch of autoregressive training over windows of length H.
    At step t we form input_{1:t} by injecting the last prediction:
        ω̂_0 := 0
        for t = 1..H:
            u_step = u.copy(); u_step[:, t, 4] = ω̂_{t-1}
            ŷ_t = model(u_step[:, :t+1, :])[:, -1, 0]
    We collect ŷ_{1:H}, compute MSE(y_{1:H}, ŷ_{1:H}), backprop, and update.
    Shapes:
      batch_u: (B,H,5)  batch_y: (B,H,1)  batch_y_pred: (B,H,1)
    '''
    torch.autograd.set_detect_anomaly(True)
    model.train()
    running_loss = 0.0
    
    for batch in dataloader:
        batch_u, batch_y = batch
        batch_u, batch_y = batch_u.to(device), batch_y.to(device)

        optimizer.zero_grad()  # Clear previous gradients

        # Create a copy of batch_u to work with, and set the velocity column (index 4) to zero
        batch_u_copy = batch_u.clone()
        batch_u_copy[:,:,4] = 0  

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
            last_predictions = model(batch_u_tmp)[:, -1, :].view(-1)  # Ensure shape matches

            batch_y_pred_list.append(last_predictions.unsqueeze(1))  # Store prediction

        # Concatenate all predictions along time dimension
        batch_y_pred = torch.cat(batch_y_pred_list, dim=1).unsqueeze(-1)  # Ensure shape matches batch_y
        # batch_y_pred: concatenated per-step predictions ŷ_{1:H} with shape (B,H,1)
        # Criterion compares full sequences (teacher-free schedule because we feed our own ŷ).

        # Compute loss
        loss = criterion(batch_y, batch_y_pred)

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Debugging: Check if all parameters have gradients
        # If some parameters have no grads, common culprits:
        #  - using .detach() inadvertently on inputs
        #  - not using last token output ([:, -1, :]) in the forward recurrence
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"Warning: No gradient computed for {name}")

    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    '''
    Evaluates the model over the given data batches. Along the windows of length h, the model estimates recursively the output omega_hat_t, with t = 1...h.
    At each iteration the model receives as input the previous estimated outputs, which is initialized at 0 e.g. omega_hat_3 = f(..., [0, omega_hat_1, omega_hat_2]). Returns the validation loss, as the mse between the recursively obtained output estimations, and the real outputs inside the window.
    Autoregressive rollout without gradient:
      ω̂_0 := 0, then for t=1..H inject ω̂_{t-1}, read ŷ_t = model(... )[:, -1, 0].
    Returns mean MSE over validation batches.
    Note: identical schedule to training (no teacher forcing), so val is faithful to inference.
    '''

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch_u, batch_y = batch
            batch_u, batch_y = batch_u.to(device), batch_y.to(device)

            batch_y_pred = torch.zeros_like(batch_y)
        
            # create a copy of batch_u to work with, then overwrite the real velocity (symbolic, may not be needed for the code)
            batch_u_copy = batch_u.clone().detach()
            batch_u_copy[:,:,4] = 0

            # simulate step by step
            last_predictions = torch.zeros(batch_u_copy.shape[0], device=device) # batch_u_copy.shape[0] is the batch size

            for t in range(batch_u_copy.shape[1]):
                batch_u_step = batch_u_copy.clone()
                batch_u_step[:,t,4] = last_predictions
                batch_u_tmp = batch_u_step[:,:t+1,:]
                #update last predictions
                last_predictions = model(batch_u_tmp)[:,-1,:].view(-1)
                batch_y_pred[:,t,0] = last_predictions

            loss = criterion(batch_y, batch_y_pred)

            running_loss += loss.item()

    return running_loss / len(dataloader)


# ========================================================================================================= #
# =========================================== RUNS ======================================================== #
# ========================================================================================================= #

def run_single_experiment(cfg, sweep_name=None, out_dir="runs"):  
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
    if out_dir is None:
        out_dir = cfg.model_dir

    if sweep_name is None:
        sweep_name = cfg.out_file     # reuse the file name for folder naming

    # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. #
    model_dir = Path(out_dir) / sweep_name
    model_dir.mkdir(parents=True, exist_ok=True)


    # Configure compute
    cuda_device = "cuda:0"

    torch.set_num_threads(cfg.threads)
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device_name = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = 'cuda' if 'cuda' in device_name else 'cpu' # for later use in torch.autocast
    print("device_type: ", device_type, "\n\n\n")
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(device)
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())

    # Load all your DataFrames (replace with your data loading code)
    # folder_path = '../data/CL_experiments/train/inertia13_ki-0.0061-kp-11.8427'
    dfs = []
    for path_iter in folder_path_training:
        new_dfs = load_dataframes_from_folder(path_iter)
        dfs= dfs + new_dfs
        print(f"Loaded {len(new_dfs)} DataFrames from {path_iter}.")

    train_ds = Dataset(dfs=dfs, seq_len=cfg.seq_len)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, pin_memory=True, shuffle=True)

    dfs_val = []
    for path_iter in folder_path_val:
        dfs_val = dfs_val + load_dataframes_from_folder(path_iter)
        print(f"Loaded {len(dfs_val)} DataFrames from {path_iter}.")

    val_ds = Dataset(dfs=dfs_val, seq_len=cfg.seq_len)
    val_dl = DataLoader(val_ds, batch_size=cfg.eval_batch_size, pin_memory=True, shuffle=True)
    # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. #


    # ===========================================================================================================================================================================================================
    # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. #
    print("saving model in: ", cfg.out_file)
    if cfg.init_from != "scratch":
        print("starting from model: ", cfg.in_file, " (", cfg.init_from, ")")

    print(f"sequence length: {cfg.seq_len}")
    print(f"max iterations:  {cfg.max_iters}")
    print(f"batch size:      {cfg.batch_size}")
    print(f"learning rate:   {cfg.lr}")
    print(f"layers:          {cfg.n_layer}")
    print(f"heads:           {cfg.n_head}")
    print(f"embd:            {cfg.n_embd}")
    print(f"patience:        {cfg.patience}")

    
    if alternative_batch_extractor:
        print("using alternative batch extractor")
    
    if not SWEEP_MODE: input("everything ok?")

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
    # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. #


    # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. #
    # Optimizer
    # Check if model is wrapped by DataParallel
    if isinstance(model, torch.nn.DataParallel):
        optimizer = model.module.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)
    else:
        optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)

    if cfg.init_from == "resume":
        optimizer.load_state_dict(checkpoint['optimizer'])
    # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. #

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
    patience = cfg.patience 
    no_improve = 0
    tol = 0.001

    for epoch in range(iter_num+1, cfg.max_iters):
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

            torch.save(checkpoint, model_dir / f"{cfg.out_file}.pt")
            
            no_improve = 0 #if val_loss < best_val_loss - tol else no_improve + 1
        else:
            no_improve += 1
        # --- EARLY STOP (optional) ---
        if no_improve >= patience:
            print(f"Early stopping at iter {epoch} (patience={patience})")
            break

        digits = len(str(cfg.max_iters))
        print("-----\n",
            "model: ", checkpoint_name_to_save, "\tno_improve: ", no_improve, "/", patience)
        print(f"Epoch [{epoch:>{digits}}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}, best_val_loss: {best_val_loss:.4f}")
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

    torch.save(checkpoint, model_dir / f"{cfg.out_file}_loss_check.pt")
    return best_val_loss

def sweep_hyperparams(base_cfg, sweep, out_root="sweep_runs"):
    keys, values = zip(*sweep.items())
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []  # collect all runs here
    i = 0

    for combo in itertools.product(*values):
        overrides = dict(zip(keys, combo))

        # build experiment name: e.g. layers8_heads4_lr1e-4
        name = "_".join(f"{k}{v}" for k, v in overrides.items())

        cfg = copy.deepcopy(base_cfg)
        for k, v in overrides.items():
            setattr(cfg, k, v)

        # optional: encode name into output file
        cfg.out_file = f"sweep_{name}"

        print(f"\n\n=== Running {name} ===")
        print("Resolved cfg for this run:")
        print(f"  n_layer   = {cfg.n_layer}")
        print(f"  n_head    = {cfg.n_head}")
        print(f"  n_embd    = {cfg.n_embd}")
        print(f"  batch_size= {cfg.batch_size}")
        print(f"  max_iters = {cfg.max_iters}")
        print(f"  lr        = {cfg.lr}")
        print(f"  patience  = {cfg.patience}")
        if i==0: input("Everything ok?")
        i += 1

        score = run_single_experiment(cfg, sweep_name=name, out_dir=str(out_root))
        print(f"Val score = {score}")

        # store this run
        row = {
            "name": name,
            "score": score,
        }
        # flatten hyperparams into the row
        row.update(overrides)
        results.append(row)

    # ---- save all results at the end ----
    if results:
        df = pd.DataFrame(results)
        results_path = out_root / "sweep_results.csv"
        df.to_csv(results_path, index=False)
        print(f"\nSaved sweep results to {results_path}")
    else:
        print("\nNo sweep runs executed, nothing to save.")


# ========================================================================================================= #
# =========================================== MAIN ======================================================== #
# ========================================================================================================= #

def make_parser(
    *,
    checkpoint_name_to_save,
    checkpoint_name_to_open,
    mode,
    sequence_length,
    layers_number,
    heads_number,
    embd_number,
    batch_size_,
    max_iteration_number,
    learning_rate_value,
):
    """
    Build an argparse.ArgumentParser for the training script.
    Mirrors the options in the 'NEW FILE' snippet.
    """
    parser = argparse.ArgumentParser(description='Meta system identification with transformers')

    # --- Overall ---
    parser.add_argument('--model-dir', type=str, default="out", metavar='S',
                        help='Saved model folder')
    parser.add_argument('--out-file', type=str, default=checkpoint_name_to_save, metavar='S',
                        help='Saved model name (filename stem)')
    parser.add_argument('--in-file', type=str, default=checkpoint_name_to_open, metavar='S',
                        help='Loaded model name (when resuming)')
    parser.add_argument('--init-from', type=str, default=mode, metavar='S',
                        help='Init from (scratch|resume|pretrained)')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='Seed for random number generation')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='Log training to Weights & Biases')

    # --- Dataset ---
    parser.add_argument('--nx', type=int, default=4, metavar='N',
                        help='state dimension proxy (not SS order)')
    parser.add_argument('--nu', type=int, default=5, metavar='N',
                        help='number of input channels')
    parser.add_argument('--ny', type=int, default=1, metavar='N',
                        help='number of output channels')
    parser.add_argument('--seq-len', type=int, default=sequence_length, metavar='N',
                        help='sequence length H (must be ≤ block_size)')
    parser.add_argument('--mag_range', type=tuple, default=(0.5, 0.97), metavar='TUP',
                        help='magnitude range (tuple)')
    parser.add_argument('--phase_range', type=tuple, default=(0.0, math.pi/2), metavar='TUP',
                        help='phase range (tuple)')
    parser.add_argument('--fixed-system', action='store_true', default=False,
                        help='If True, keep the same plant/system across epochs')

    # --- Model ---
    parser.add_argument('--n-layer', type=int, default=layers_number, metavar='N',
                        help='number of Transformer blocks (depth)')
    parser.add_argument('--n-head', type=int, default=heads_number, metavar='N',
                        help='number of attention heads')
    parser.add_argument('--n-embd', type=int, default=embd_number, metavar='N',
                        help='model width (embedding dim)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='P',
                        help='dropout rate (0–0.2 typical)')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='use bias in Linear/LayerNorm')

    # --- Training ---
    parser.add_argument('--batch-size', type=int, default=batch_size_, metavar='N',
                        help='batch size')
    parser.add_argument('--max-iters', type=int, default=max_iteration_number, metavar='N',
                        help='number of training epochs/iterations')
    parser.add_argument('--warmup-iters', type=int, default=5_000, metavar='N',
                        help='LR warmup steps/iters')
    parser.add_argument('--lr', type=float, default=learning_rate_value, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='WD',
                        help='weight decay (AdamW)')
    parser.add_argument('--eval-interval', type=int, default=10, metavar='N',
                        help='evaluate every N iters')
    parser.add_argument('--eval-iters', type=int, default=10, metavar='N',
                        help='batches per evaluation')
    parser.add_argument('--fixed-lr', action='store_true', default=False,
                        help='disable LR scheduling (use fixed lr)')
    parser.add_argument('--patience', type=int, default=200,
                        help='early stopping patience')

    # --- Compute ---
    parser.add_argument('--threads', type=int, default=16,
                        help='number of CPU dataloader/BLAS threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='force CPU (disable CUDA)')
    parser.add_argument('--cuda-device', type=str, default="cuda:0", metavar='S',
                        help='CUDA device, e.g. "cuda:0"')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='enable torch.compile for speed')

    return parser

if __name__ == '__main__':
    # These symbols must exist in your script/environment:
    # checkpoint_name_to_save, checkpoint_name_to_open, mode,
    # sequence_length, layers_number, heads_number, embd_number,
    # batch_size_, max_iteration_number, learning_rate_value
    parser = make_parser(
        checkpoint_name_to_save=checkpoint_name_to_save,
        checkpoint_name_to_open=checkpoint_name_to_open,
        mode=mode,
        sequence_length=sequence_length,
        layers_number=layers_number,
        heads_number=heads_number,
        embd_number=embd_number,
        batch_size_=batch_size_,
        max_iteration_number=max_iteration_number,
        learning_rate_value=learning_rate_value,
    )
    cfg = parser.parse_args()
    # ... training code using cfg ...

    if SWEEP_MODE:
        sweep_hyperparams(cfg, sweep)
    else:
        # Run ONE experiment normally
        best = run_single_experiment(cfg)
        print("Best loss:", best)


    # GOTCHA: Keep seq_len (H) ≤ model.block_size; else positional embedding lookup will fail.
    # GOTCHA: batch_u_copy[:, :, 4] is assumed to be the ω̂ channel; keep dataset channel order consistent.
    # GOTCHA: last_predictions must keep requires_grad=True in training; do not detach it.
    # GOTCHA: If validation loss >> train loss, check normalization consistency and that val uses the same autoregressive schedule.
    # GOTCHA: Warmup too long can freeze LR near 0; ensure warmup_iters ≪ max_iters.

