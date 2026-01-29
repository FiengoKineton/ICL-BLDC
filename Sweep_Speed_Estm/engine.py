# engine.py             | TESTING

import torch, sys, numpy as np
from typing import Optional
from dataset import reverse_normalization
from engine_utils import _regression, regression, smooth_dynamics_loss



def train(              # to check hyperparameters change
        model, 
        dataloader, 
        criterion, 
        optimizer, 
        device, 
        R_smooth: 
        float = None, 
        regression_mode: str = "time_last",
        teacher_prob: float = 0.0,
        ss_mode: str = "stochastic",
        ):
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
        batch_y_pred = regression(batch_u, model, device, regression_mode, batch_y, teacher_prob, ss_mode)
        loss = criterion(batch_y, batch_y_pred) + smooth_dynamics_loss(batch_y, batch_y_pred, R_smooth, criterion=criterion)

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

def validate(           # to check the best model
        model, 
        dataloader, 
        criterion, 
        device, 
        R_smooth: float = None, 
        regression_mode: str = "time_last",
        teacher_prob: float = 0.0,
        ss_mode: str = "stochastic",
        ):
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

            batch_y_pred = regression(batch_u, model, device, regression_mode, batch_y, teacher_prob, ss_mode)
            """batch_y_pred = torch.zeros_like(batch_y)
        
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
                batch_y_pred[:,t,0] = last_predictions"""

            loss = criterion(batch_y, batch_y_pred) + smooth_dynamics_loss(batch_y, batch_y_pred, R_smooth, criterion=criterion)
            running_loss += loss.item()

    return running_loss / len(dataloader)


def test(           # to check over/under-fitting
    model,
    dataset,
    device,
    seq_len: int = 10,
    dt: Optional[float] = None,
):
    """
    Generic evaluation loop: runs autoregressive regression under no_grad,
    returns mean loss over batches. Use it for BOTH validation and test.
    """
    ds_len = len(dataset)
    loss = np.zeros(ds_len)
    u_full_all, y_full_all = [], []


    for i in range(ds_len):
        try:
            u_full, y_full = dataset.get_full_experiment(i)  # (T, n_u), (T, 1)
            T = u_full.shape[0]
            u_full_all.append(u_full.to(device))
            y_full_all.append(y_full.to(device))
        except Exception as e:
            ds_len = i
            break

    u_full_all = torch.stack(u_full_all, dim=0)  # (N, T, n_u)
    y_full_all = torch.stack(y_full_all, dim=0)  # (N, T, 1)
    y_pred_all = torch.zeros_like(y_full_all)

    last_omega = torch.zeros((ds_len, seq_len, 1))

    with torch.no_grad():
        for j in range(y_full_all.shape[1]):
            if j < seq_len:
                input_val = u_full_all[:, :j+1, :]                  # first j samples of the experiment
                input_val[:, :j+1, 4] = last_omega[:,-j-1:,0]       # last j samples of the past estimations window

                pred = model(input_val)[:,-1,:]                     # we consider only the last value of the model estimation
            else:
                input_val = u_full_all[:,j-seq_len+1:j+1, :]
                input_val[:,:, 4] = last_omega[:,:,0]

                pred = model(input_val)[:,-1,:]                     # we consider only the last value of the model estimation
            
            
            y_pred_all[:,j,0] = pred[:,0]                           # the list of estimations is updated
            last_omega = torch.roll(last_omega, -1, 1)              # the window of last estimation is slid of 1 sample
            last_omega[:,-1,0] = y_pred_all[:,j,0]                  # last estimation is added to the window
            
        u_full_all, y_full_all, y_pred_all = reverse_normalization(u_full_all, y_full_all, y_pred_all)

    for i in range(ds_len):
        y_tmp = y_full_all[i,:,:].cpu().numpy()
        y_pred_tmp = y_pred_all[i,:,:].cpu().numpy()
        loss[i] = np.sqrt(((y_tmp-y_pred_tmp)**2).mean())

    #test_loss = loss.mean()
    test_loss = loss.mean()

    # Shapes
    N, T, n_u = u_full_all.shape

    base_dt = 1.0 if dt is None else dt
    t_flat = (torch.arange(N * T, device=device) * base_dt).cpu().numpy()

    # Remove trailing dim=1 from y if present, then flatten
    y_true_flat = y_full_all.view(N, T, -1).squeeze(-1).reshape(-1).cpu().numpy()  # (N*T,)
    y_pred_flat = y_pred_all.view(N, T, -1).squeeze(-1).reshape(-1).cpu().numpy()  # (N*T,)

    # Inputs: (N, T, n_u) -> (N*T, n_u)
    u_flat = u_full_all.reshape(N * T, n_u).cpu().numpy()  # (N*T, n_u)

    traj = (t_flat, u_flat, y_true_flat, y_pred_flat)

    return test_loss, traj, (N, T)


def _test(               # to check predicted trajectories
        model,
        dataloader,
        device: torch.device,
        dt: Optional[float] = None,
        regression_mode: str = "time_last",
        ): # -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run test (evaluation) with the same recursive structure used in training.

    Returns
    -------
    t : (T,) tensor
        Time vector. If dt is provided, t = [0, dt, 2*dt, ..., (T-1)*dt].
        Otherwise t = [0, 1, ..., T-1].
    u_true : (N, T, n_u) tensor
        True input sequences (concatenated over all batches).
    y_true : (N, T, 1) tensor
        True output sequences.
    y_pred : (N, T, 1) tensor
        Predicted output sequences from the model, using recursive injection.
    """
    model.eval()

    all_u_true = []
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            batch_u, batch_y = batch
            batch_u = batch_u.to(device)
            batch_y = batch_y.to(device)

            """# Copy and zero velocity channel (index 4), same as in training
            batch_u_copy = batch_u.clone()
            batch_u_copy[:, :, 4] = 0.0

            B, T, _ = batch_u_copy.shape

            # Initialize ω̂_0 = 0 for all sequences in the batch
            last_predictions = torch.zeros(B, device=device)
            batch_y_pred_list = []

            # Simulate step by step (teacher-free)
            for t_step in range(T):
                # Inject previous estimate into channel 4 at time t_step
                batch_u_step = batch_u_copy.clone()
                batch_u_step[:, t_step, 4] = last_predictions

                # Prefix up to current time
                batch_u_tmp = batch_u_step[:, :t_step + 1, :]

                # Forward pass, take last time step
                prediction_full = model(batch_u_tmp)
                last_predictions = prediction_full[:, -1, :].view(-1)
                batch_y_pred_list.append(last_predictions.unsqueeze(1))

            # Concatenate predictions along time dimension and add output dim
            batch_y_pred = torch.cat(batch_y_pred_list, dim=1).unsqueeze(-1)  # (B, T, 1)"""

            batch_y_pred = _regression(batch_u, model, device, regression_mode)

            all_u_true.append(batch_u)
            all_y_true.append(batch_y)
            all_y_pred.append(batch_y_pred)

    # Concatenate over batches
    u_true = torch.cat(all_u_true, dim=0)   # (N, T, n_u)
    y_true = torch.cat(all_y_true, dim=0)   # (N, T, 1)
    y_pred = torch.cat(all_y_pred, dim=0)   # (N, T, 1)


    u_true, y_true, y_pred = reverse_normalization(
        u_true, y_true, y_pred
    )


    # Shapes
    N, T, n_u = u_true.shape

    base_dt = 1.0 if dt is None else dt
    t_flat = (torch.arange(N * T, device=u_true.device) * base_dt).cpu().numpy()

    # Remove trailing dim=1 from y if present, then flatten
    y_true_flat = y_true.view(N, T, -1).squeeze(-1).reshape(-1).cpu().numpy()  # (N*T,)
    y_pred_flat = y_pred.view(N, T, -1).squeeze(-1).reshape(-1).cpu().numpy()  # (N*T,)

    # Inputs: (N, T, n_u) -> (N*T, n_u)
    u_flat = u_true.reshape(N * T, n_u).cpu().numpy()  # (N*T, n_u)

    return t_flat, y_true_flat, y_pred_flat, u_flat

