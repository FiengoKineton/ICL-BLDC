import numpy as np
import pandas as pd
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import copy

# -----------------------------------------------------------------------------
# Dataset for ICL-style BLDC speed estimation.
# Contract:
#   __getitem__ -> (batch_u, batch_y)
#     batch_u: (H, 5) = [ia, ib, va, vb, last_omega]  (normalized)
#     batch_y: (H, 1) = omega                        (normalized)
# Windowing:
#   - Random experiment, random start index.
#   - 50% probability to sample a "step" window (r changes inside H) vs "constant" window.
# Autoregression helper:
#   - 'last_omega' is a placeholder copy of ω_{t-1}; training/validation overwrite it with model predictions.
# Normalization: fixed ranges -> map raw signals to ~[0,1] magnitudes.
# -----------------------------------------------------------------------------


class Dataset(Dataset):
    """Sequence dataset for autoregressive estimation.
    Args:
        dfs: list[pd.DataFrame] each with columns at least
             ['t','theta','omega','r','ia','ib','iq_ref','va','vb'] (or remapped)
        seq_len: window length H
    Returns (__getitem__):
        batch_u: torch.float32 (H,5)  -> [ia, ib, va, vb, last_omega]
        batch_y: torch.float32 (H,1)  -> omega
    Notes:
        - Values are normalized by normalize_fixed_ranges().
        - 'last_omega' is shifted ω (ω_{t-1}); at training we overwrite it with predictions ω̂_{t-1}.
        - Channel order is fixed; training code assumes index 4 is the ω̂ channel.
    """

    def __init__(self, dfs, seq_len):
        self.dfs = dfs
        self.seq_len = seq_len

    def __len__(self):
        # Upper bound on samples per epoch (virtual length).
        # Keeps each epoch lightweight regardless of how many CSVs are loaded.
        # maximum set of samples considered at each iteration
        return 512

    def __getitem__(self, idx):
        # Randomly select a DataFrame
        df_idx = np.random.choice(len(self.dfs))
        df = self.dfs[df_idx]

        # evaluate whether the first and last element of the window of length H of the reference speed are different (e.g. if a step is present inside the window)
        diff_array = df['r'].diff(-self.seq_len).to_numpy()
        diff_array = diff_array[~np.isnan(diff_array)]

        prob_ratio = 0.5 # ratio between constant samples and step samples
        if np.random.rand() >= prob_ratio: # filter indices that correspond to a window without a step in it
            good_idx = np.flatnonzero(diff_array == 0)
            if len(good_idx) == 0: # if no indices satisfy the request, look for windows with no step 
                good_idx = np.flatnonzero(diff_array != 0)
        else:
            good_idx = np.flatnonzero(diff_array != 0) # filter indices that correspond to a window with a step in it
            if len(good_idx) == 0: # if no indices satisfy the request, look for windows with steps
                good_idx = np.flatnonzero(diff_array == 0)

        start_idx = np.random.choice(good_idx)  # select a random starting index among the one filtered above

        # generate a column in the dataset for the "past" values of omega. These are not actually used in training, but are overwritten with the past estimations
        tmp = copy.deepcopy(df['omega'].to_numpy())
        tmp[1:-1] = tmp[0:-2]
        tmp[0] = 0
        df['last_omega'] = tmp


        # Get the sequence for batch_u and batch_y
        # batch_y: (H,1) target speeds (normalized)
        # batch_u: (H,5) inputs  [ia, ib, va, vb, last_omega] (normalized)
        # NOTE: channel 4 MUST remain reserved to ω̂ during training/validation.
        batch_y = torch.tensor(df['omega'].iloc[start_idx:start_idx + self.seq_len].values, dtype=torch.float32)
        batch_u = torch.tensor(df[['ia', 'ib', 'va', 'vb', 'last_omega']].iloc[start_idx:start_idx + self.seq_len].values,
                               dtype=torch.float32)

        # Add a batch dimension
        batch_y = batch_y.view(-1,1)  # Shape (1, seq_len, 1)

        return batch_u, batch_y

    def get_full_experiment(self, idx):
        """Return entire normalized experiment as tensors.
        Shapes:
          batch_u: (T,5), batch_y: (T,1)
        Includes 'last_omega' = shifted ω (ω_{t-1}); at inference you can ignore/overwrite it.

        Outputs the entirety of the experiment at index idx as a torch tensor (normalized if the data files were passed to the Dataset object correctly)
        """
        df = self.dfs[idx]
        tmp = copy.deepcopy(df['omega'].to_numpy())
        tmp[1:-1] = tmp[0:-2]
        tmp[0] = 0
        df['last_omega'] = tmp
        batch_y = torch.tensor(df['omega'].to_numpy(), dtype=torch.float32)
        batch_u = torch.tensor(df[['ia', 'ib', 'va', 'vb', 'last_omega']].to_numpy(), dtype=torch.float32)
        # Add a batch dimension
        batch_y = batch_y.view(-1,1)  # Shape (1, seq_len, 1)

        return batch_u, batch_y
    
    def get_experiment_observer(self, idx):
        """
        returns pll observer estimated speed, non-normalized
        """
        df = self.dfs[idx]
        obs_y = df['omega_obs'].to_numpy()
        
        return obs_y
    
    def get_experiment_ekf(self, idx):
        """
        returns ekf estimated speed, non-normalized
        """
        df = self.dfs[idx]
        obs_y = df['omega_ekf'].to_numpy()
        
        return obs_y




# Normalization function
def normalize_fixed_ranges(df):
    '''
    Transforms the relevant column of the dataframe so that their valuse is in the range [0,1], or at least in its order of magnitude
    Normalize raw signals to fixed ranges (~[0,1]).
    Mapping:
      ia, ib:  [-5, +5]   -> (x+5)/10
      va, vb:  [-24,+24]  -> (x+24)/48
      omega:   [0, 2500]  -> x/2500
    WARNING:
      - Assumes currents and voltages stay within these physical bounds.
      - If your data exceed these ranges, values may go <0 or >1; either clip or adapt the constants.
      - Keep this mapping in sync with reverse_normalization().
    '''
    df['ia'] = (df['ia'] + 5) / 10  # Normalize iq from -5 to 5 -> [0, 1]
    df['ib'] = (df['ib'] + 5) / 10  # Normalize id from -5 to 5 -> [0, 1]
    df['va'] = (df['va'] + 24) / 48  # Normalize vq from -24 to 24 -> [0, 1]
    df['vb'] = (df['vb'] + 24) / 48  # Normalize vd from -24 to 24 -> [0, 1]
    df['omega'] = df['omega'] / 2500  # Normalize omega from 0 to 2500 -> [0, 1]
    return df


def reverse_normalization(batch_u, batch_y, batch_y_pred):
    '''
    Transforms the batch values into their orignal values, inverting the transfotrmation of "normalized_fixed_ranges()"
    Invert normalize_fixed_ranges() for visualization/evaluation on raw units.
    Args:
      batch_u: (B,H,5), batch_y: (B,H,1), batch_y_pred: (B,H,1)
    Returns:
      (batch_u_denorm, batch_y_denorm, batch_y_pred_denorm)
    Notes:
      - Expects the same channel order: [ia, ib, va, vb, last_omega].
      - Keep constants consistent with normalize_fixed_ranges().
      - Make sure tensors are cloned if you need original normalized values later.
    '''
    # Define the normalization constants
    min_currents = -5
    max_currents = 5
    min_voltages = -24
    max_voltages = 24
    min_speed = 0
    max_speed = 2500

    # Reverse normalization for currents (iq, id)
    # Assuming batch_u contains currents in the first two columns
    batch_u[:, :, 0] = batch_u[:, :, 0] * (max_currents - min_currents) + min_currents  # ia
    batch_u[:, :, 1] = batch_u[:, :, 1] * (max_currents - min_currents) + min_currents  # ib

    # Reverse normalization for voltages (vq, vd)
    batch_u[:, :, 2] = batch_u[:, :, 2] * (max_voltages - min_voltages) + min_voltages  # va
    batch_u[:, :, 3] = batch_u[:, :, 3] * (max_voltages - min_voltages) + min_voltages  # vb

    # Reverse normalization for speed (omega)
    batch_u[:, :, 4] = batch_u[:, :, 4] * (max_speed - min_speed) + min_speed  # last_omega
    batch_y = batch_y * (max_speed - min_speed) + min_speed
    batch_y_pred = batch_y_pred * (max_speed - min_speed) + min_speed

    return batch_u, batch_y, batch_y_pred


def load_dataframes_from_folder(folder_path):
    '''
    Genertes a list of dataframes corresponding to all csv files in the given folder "folder_path".
    
    Load and normalize all CSVs in folder_path.
    Steps:
      - Read CSV
      - Map columns to ['t','theta','omega','r','ia','ib','iq_ref','va','vb'] if possible
      - Apply normalize_fixed_ranges()
    Returns:
      list[pd.DataFrame] normalized
    GOTCHA:
      - If CSV columns differ, extend the mapping or handle exceptions explicitly.
      - Ensure units match the assumed ranges before normalization.
    '''
    # Create a list to hold all DataFrames
    dataframes = []
    # Use glob to find all CSV files in the specified folder
    for file in glob.glob(os.path.join(folder_path, '*.csv')):
        df = pd.read_csv(file)
        try:
            df.columns = ['t', 'theta', 'omega', 'r', 'ia', 'ib', 'iq_ref', 'va', 'vb']
        except:
            pass
        df = normalize_fixed_ranges(df)
        dataframes.append(df)

    return dataframes

# Example usage
if __name__ == "__main__":
    
    current_path = os.getcwd().split("in-context-bldc")[0]
    data_path = os.path.join(current_path,"in-context-bldc", "data")

    # folder = "CL_experiments_double_sensor_low_speed_ekf_and_meta/final/inertia13_ki-0.0029-kp-3.0000"
    folder = "CL_experiments_double_sensor_high_speed_ekf_and_meta/final/inertia13_ki-0.0061-kp-11.8427"
    folder_path = os.path.join(data_path, folder)

    dfs = load_dataframes_from_folder(folder_path)
    # Log the number of DataFrames loaded
    print(f"Loaded {len(dfs)} DataFrames from {folder_path}.")

    seq_len = 10

    # Create an instance of the dataset
    dataset = Dataset(dfs=dfs, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Example of accessing an item
    batch_u, batch_y = next(iter(dataloader))
    print(f'batch_u: {batch_u.shape}, batch_y: {batch_y.shape}')

    # Convert batch tensors to numpy for plotting
    batch_u_np = batch_u.squeeze(0).numpy()  # Shape (seq_len, n_u)
    batch_y_np = batch_y.squeeze(0).numpy()  # Shape (seq_len, 1)

    # Plotting
    fig = plt.figure(figsize=(12, 6))

    # Plot batch_y (omega)
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(batch_y_np[:,:,0].T, label='Batch y (omega)', color='blue')
    ax1.set_title('Batch y (omega)')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Value')
    
    # Plot each component of batch_u
    ax2 = fig.add_subplot(2,1,2, sharex = ax1)
    ax2.plot(batch_u_np[:, :, 0].T, label='Batch u (ia)', color='orange')
    ax2.plot(batch_u_np[:, :, 1].T, label='Batch u (ib)', color='green')
    ax2.plot(batch_u_np[:, :, 2].T, label='Batch u (va)', color='red')
    ax2.plot(batch_u_np[:, :, 3].T, label='Batch u (vb)', color='purple')
    ax2.plot(batch_u_np[:, :, 4].T, label='Batch u (last_omega)', color='grey')
    ax2.set_title('Batch u (ia, ib, va, vb, last_omega)')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Value')

    plt.tight_layout()


    # plot some window examples
    batch_u, batch_y, _ = reverse_normalization(batch_u, batch_y, batch_y)

    for i in range(2):
        fig = plt.figure()
        ax0 = fig.add_subplot(4,1,1)
        ax0.plot(batch_y[i,:,:],label = "$\omega$")
        ax0.legend()
        # ax0.set_ylim(-50,3050)
        ax1 = fig.add_subplot(4,1,2, sharex = ax0)
        ax1.plot(batch_u[i,:,0],label = "$I_a$")
        ax1.plot(batch_u[i,:,1],label = "$I_b$")
        ax1.legend()
        ax2 = fig.add_subplot(4,1,3, sharex = ax0)
        ax2.plot(batch_u[i,:,2],label = "$V_a$")
        ax2.plot(batch_u[i,:,3],label = "$V_b$")
        ax2.legend()
        ax3 = fig.add_subplot(4,1,4, sharex = ax0, sharey = ax0)
        ax3.plot(batch_u[i,:,4],label = "$\omega_{k-1}$")
        ax3.legend()
        # ax3.set_ylim(-50,3050)



    plt.show()


# GOTCHA: Channel order is hard-coded: [ia, ib, va, vb, last_omega]; training assumes index 4 is ω̂.
# GOTCHA: If your currents/voltages exceed the assumed ranges, update normalize_fixed_ranges() (and reverse).
# GOTCHA: __len__ returns 512 (virtual length). If you need epoch-size proportional to data, change it.
# GOTCHA: Sampling balance (prob_ratio) affects curriculum: 0.5 enforces 50/50 step vs constant windows.
# GOTCHA: Copy-on-write: modifying df in-place (df['last_omega'] = ...) will persist across calls if dfs are reused.
#         If you want immutability, work on a df copy inside __getitem__.
