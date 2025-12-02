import torch
import matplotlib.pyplot as plt
from pathlib import Path

def train(model, dataloader, criterion, optimizer, device, R_smooth: float = None):
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

        if R_smooth is not None: 
            # ---- Dynamics-matching term ----
            # Finite differences over time, along the sequence dimension
            dy_true = batch_y.diff(dim=1)        # (B, H-1, 1),   y_t - y_{t-1}
            dy_pred = batch_y_pred.diff(dim=1)   # (B, H-1, 1),   ŷ_t - ŷ_{t-1}

            # Weight for the derivative penalty
            mag = dy_true.abs()
            thresh = 0.05   # to tune
            w_dyn = (mag / thresh).clamp(0.0, 1.0)

            target_dy = w_dyn * dy_true #+ (1.0 - w_dyn) * 0.0
            loss_dyn = ((dy_pred - target_dy) ** 2).mean()

            # Total loss
            loss += R_smooth**2 * loss_dyn


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

def validate(model, dataloader, criterion, device, R_smooth: float = None):
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

            if R_smooth is not None:
                # ---- Dynamics-matching term (same as in train) ----
                dy_true = batch_y.diff(dim=1)        # (B, H-1, 1)
                dy_pred = batch_y_pred.diff(dim=1)   # (B, H-1, 1)

                mag = dy_true.abs()
                thresh = 0.05   # use the SAME thresh as in train
                w_dyn = (mag / thresh).clamp(0.0, 1.0)

                target_dy = w_dyn * dy_true  # + (1.0 - w_dyn) * 0.0
                loss_dyn = ((dy_pred - target_dy) ** 2).mean()

                loss += R_smooth**2 * loss_dyn

            running_loss += loss.item()

    return running_loss / len(dataloader)

def evaluate_and_plot_test(
    model,
    test_dl,
    device,
    run_dir,
    epoch: int,
    R_smooth: float | None = None,
    max_trajs: int = 4,
):
    """
    Run one batch from test_dl autoregressively, compute loss,
    and save a plot of true vs predicted trajectories.

    This is for diagnostics/visualization, not for early stopping.
    """
    model.eval()
    run_dir = Path(run_dir)

    with torch.no_grad():
        try:
            batch_u, batch_y = next(iter(test_dl))
        except StopIteration:
            print("[test-plot] Empty test_dl, skipping.")
            return

        batch_u = batch_u.to(device)
        batch_y = batch_y.to(device)

        B, H, _ = batch_y.shape
        batch_y_pred = torch.zeros_like(batch_y)

        # autoregressive loop (same as train/validate)
        last_predictions = torch.zeros(B, device=device)
        batch_u_copy = batch_u.clone()

        for t in range(batch_u_copy.shape[1]):
            batch_u_step = batch_u_copy.clone()
            batch_u_step[:, t, 4] = last_predictions
            batch_u_tmp = batch_u_step[:, :t+1, :]

            last_predictions = model(batch_u_tmp)[:, -1, :].view(-1)
            batch_y_pred[:, t, 0] = last_predictions

        # base value loss (optional, for logging)
        criterion = torch.nn.MSELoss()
        loss = criterion(batch_y, batch_y_pred)

        if R_smooth is not None:
            dy_true = batch_y.diff(dim=1)        # (B, H-1, 1)
            dy_pred = batch_y_pred.diff(dim=1)   # (B, H-1, 1)

            mag = dy_true.abs()
            thresh = 0.05   # must match training
            w_dyn = (mag / thresh).clamp(0.0, 1.0)
            w_smooth = 1.0 - w_dyn

            target_dy = w_dyn * dy_true  # + w_smooth * 0.0

            loss_dyn = ((dy_pred - target_dy) ** 2).mean()
            loss = loss + R_smooth**2 * loss_dyn

        test_loss = float(loss.item())
        print(f"[test-plot] epoch={epoch} test_loss={test_loss:.4e}")

        # ---- plotting ----
        y_true = batch_y.detach().cpu().numpy()      # (B, H, 1)
        y_pred = batch_y_pred.detach().cpu().numpy() # (B, H, 1)

        T = y_true.shape[1]
        t_axis = range(T)

        n_plot = min(B, max_trajs)

        plt.figure(figsize=(8, 6))
        for i in range(n_plot):
            plt.subplot(n_plot, 1, i + 1)
            plt.plot(t_axis, y_true[i, :, 0], label="true", linewidth=1.2)
            plt.plot(t_axis, y_pred[i, :, 0], label="pred", linewidth=1.2, linestyle="--")
            if i == 0:
                plt.legend(loc="upper right", fontsize=8)
            plt.ylabel(f"traj {i}", fontsize=8)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
        plt.xlabel("time step", fontsize=9)
        plt.suptitle(f"Test trajectories @ epoch {epoch} (loss={test_loss:.3e})", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plots_dir = run_dir / "test_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / f"test_epoch_{epoch:04d}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"[test-plot] Saved test plot to: {out_path}")
