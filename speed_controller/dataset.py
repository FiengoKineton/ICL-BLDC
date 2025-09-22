import numpy as np
import pandas as pd
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import copy

class Dataset(Dataset):
    def __init__(self, dfs, seq_len):
        self.dfs = dfs
        self.seq_len = seq_len

    def __len__(self):
        # maximum set of samples considered at each iteration
        return 512

    def __getitem__(self, idx):
        # Randomly select a DataFrame
        df_idx = np.random.choice(len(self.dfs))
        # df_idx = 582
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

        # metadata = df.keys()[-1].split(',')
        # print(df_idx)
        # print(metadata)
        # T_ass = float(metadata[0].split(":")[1]) / 3
        # S_pct = float(metadata[1].split(":")[1]) / 40
        # Kp = float(metadata[2].split(":")[1])
        # Ki = float(metadata[3].split(":")[1])

        # if "T_ass" not in df.keys():  
        #     # print("adding T_ass")    
        #     location = len(df.keys())-1
        #     df.insert(loc=location, column='T_ass', value=np.ones_like(df["omega"].to_numpy())*T_ass)
        # if "S_pct" not in df.keys():            
        #     # print("adding S_pct")        
        #     location = len(df.keys())-1  
        #     df.insert(loc=location, column='S_pct', value=np.ones_like(df["omega"].to_numpy())*S_pct)

        if "next_iq_ref" not in df.keys(): 
            tmp = copy.deepcopy(df['iq_ref'].to_numpy())
            tmp[0:-2] = tmp[1:-1]
            location = len(df.keys())-1 
            df.insert(loc=location, column='next_iq_ref', value=tmp)
        
        if "next_omega" not in df.keys(): 
            tmp = copy.deepcopy(df['omega'].to_numpy())
            tmp[0:-2] = tmp[1:-1]
            location = len(df.keys())-1 
            df.insert(loc=location, column='next_omega', value=tmp)
        
        if "next_omega_ref" not in df.keys(): 
            tmp = copy.deepcopy(df['r'].to_numpy())
            tmp[0:-2] = tmp[1:-1]
            location = len(df.keys())-1 
            df.insert(loc=location, column='next_omega_ref', value=tmp)


        
        # df["T_ass"] = np.ones_like(df["omega"].to_numpy())*T_ass
        # df["S_pct"] = np.ones_like(df["omega"].to_numpy())*S_pct



        # Get the sequence for batch_u and batch_y
        batch_y = torch.tensor(df['next_iq_ref'].iloc[start_idx:start_idx + self.seq_len].values, dtype=torch.float32)
        batch_u = torch.tensor(df[['id', 'iq', 'vd', 'vq', 'next_omega', 'next_omega_ref']].iloc[start_idx:start_idx + self.seq_len].values,
                               dtype=torch.float32)

        # Add a batch dimension
        batch_y = batch_y.view(-1,1)  # Shape (1, seq_len, 1)

        return batch_u, batch_y

    def get_full_experiment(self, idx):
        '''
        Outputs the entirety of the experiment at index idx as a torch tensor (normalized if the data files were passed to the Dataset object correctly)
        '''
        df = self.dfs[idx]
        # metadata = df.keys()[-1].split(',')
        # # print(metadata)
        # T_ass = float(metadata[0].split(":")[1]) / 3
        # S_pct = float(metadata[1].split(":")[1]) / 40
        # if "T_ass" not in df.keys():  
        #     # print("adding T_ass")    
        #     location = len(df.keys())-1
        #     df.insert(loc=location, column='T_ass', value=np.ones_like(df["omega"].to_numpy())*T_ass)
        # if "S_pct" not in df.keys():            
        #     # print("adding S_pct")        
        #     location = len(df.keys())-1  
        #     df.insert(loc=location, column='S_pct', value=np.ones_like(df["omega"].to_numpy())*S_pct)
        
        if "next_iq_ref" not in df.keys(): 
            tmp = copy.deepcopy(df['iq_ref'].to_numpy())
            tmp[0:-2] = tmp[1:-1]
            location = len(df.keys())-1 
            df.insert(loc=location, column='next_iq_ref', value=tmp)
        
        if "next_omega" not in df.keys(): 
            tmp = copy.deepcopy(df['omega'].to_numpy())
            tmp[0:-2] = tmp[1:-1]
            location = len(df.keys())-1 
            df.insert(loc=location, column='next_omega', value=tmp)
        
        if "next_omega_ref" not in df.keys(): 
            tmp = copy.deepcopy(df['r'].to_numpy())
            tmp[0:-2] = tmp[1:-1]
            location = len(df.keys())-1 
            df.insert(loc=location, column='next_omega_ref', value=tmp)
        batch_y = torch.tensor(df['next_iq_ref'].to_numpy(), dtype=torch.float32)
        batch_u = torch.tensor(df[['id', 'iq', 'vd', 'vq', 'next_omega', 'next_omega_ref']].to_numpy(), dtype=torch.float32)
        # Add a batch dimension
        batch_y = batch_y.view(-1,1)  # Shape (1, seq_len, 1)

        return batch_u, batch_y
    
    # def get_experiment_observer(self, idx):
    #     """
    #     returns pll observer estimated speed, non-normalized
    #     """
    #     df = self.dfs[idx]
    #     obs_y = df['omega_obs'].to_numpy()
        
    #     return obs_y
    
    # def get_experiment_ekf(self, idx):
    #     """
    #     returns ekf estimated speed, non-normalized
    #     """
    #     df = self.dfs[idx]
    #     obs_y = df['omega_ekf'].to_numpy()
        
    #     return obs_y




# Normalization function
def normalize_fixed_ranges(df):
    '''
    Transforms the relevant column of the dataframe so that their valuse is in the range [0,1], or at least in its order of magnitude
    '''
    df['id'] = (df['id'] + 5) / 10  # Normalize iq from -5 to 5 -> [0, 1]
    df['iq'] = (df['iq'] + 5) / 10  # Normalize id from -5 to 5 -> [0, 1]
    df['iq_ref'] = (df['iq_ref'] + 5) / 10  # Normalize iq_ref from -5 to 5 -> [0, 1]
    df['vd'] = (df['vd'] + 24) / 48  # Normalize vq from -24 to 24 -> [0, 1]
    df['vq'] = (df['vq'] + 24) / 48  # Normalize vd from -24 to 24 -> [0, 1]
    df['omega'] = df['omega'] / 2500  # Normalize omega from 0 to 2500 -> [0, 1]
    df['r'] = df['r'] / 2500  # Normalize speed reference from 0 to 2500 -> [0, 1]
    return df


def reverse_normalization(batch_u, batch_y, batch_y_pred):
    '''
    Transforms the batch values into their orignal values, inverting the transformation of "normalized_fixed_ranges()"
    '''
    # Define the normalization constants
    min_currents = -5
    max_currents = 5
    min_voltages = -24
    max_voltages = 24
    min_speed = 0
    max_speed = 2500
    # min_T_ass = 0
    # max_T_ass = 3
    # min_S_pct = 0
    # max_S_pct = 40

    # ['id', 'iq', 'vd', 'vq', 'omega', 'r', 'T_ass', 'S_pct']
    # Reverse normalization for currents (iq, id)
    # Assuming batch_u contains currents in the first two columns
    batch_u[:, :, 0] = batch_u[:, :, 0] * (max_currents - min_currents) + min_currents  # id
    batch_u[:, :, 1] = batch_u[:, :, 1] * (max_currents - min_currents) + min_currents  # iq
    batch_y = batch_y * (max_currents - min_currents) + min_currents                    # iq_ref
    batch_y_pred = batch_y_pred * (max_currents - min_currents) + min_currents          # iq_ref_est

    # Reverse normalization for voltages (vq, vd)
    batch_u[:, :, 2] = batch_u[:, :, 2] * (max_voltages - min_voltages) + min_voltages  # vd
    batch_u[:, :, 3] = batch_u[:, :, 3] * (max_voltages - min_voltages) + min_voltages  # vq

    # Reverse normalization for speed (omega)
    batch_u[:, :, 4] = batch_u[:, :, 4] * (max_speed - min_speed) + min_speed           # omega
    batch_u[:, :, 5] = batch_u[:, :, 5] * (max_speed - min_speed) + min_speed           # omega_ref

    # # Reverse normalization for control param (T_ass, S_pct)
    # batch_u[:, :, 6] = batch_u[:, :, 6] * (max_T_ass - min_T_ass) + min_T_ass           # T_ass
    # batch_u[:, :, 7] = batch_u[:, :, 7] * (max_S_pct - min_S_pct) + min_S_pct           # S_pct


    return batch_u, batch_y, batch_y_pred


def load_dataframes_from_folder(folder_path):
    '''
    Generates a list of dataframes corresponding to all csv files in the given folder "folder_path".
    '''
    # Create a list to hold all DataFrames
    dataframes = []
    # Use glob to find all CSV files in the specified folder
    for file in glob.glob(os.path.join(folder_path, '*.csv')):
        df = pd.read_csv(file)
        metadata = df.keys()[-1].split(',')
        # print(metadata)
        T_ass = float(metadata[0].split(":")[1])
        S_pct = float(metadata[1].split(":")[1])
        if T_ass > 1.5 or S_pct > 20:
            continue
        df = normalize_fixed_ranges(df)
        dataframes.append(df)

    return dataframes

# Example usage
if __name__ == "__main__":
    
    current_path = os.getcwd().split("in-context-bldc")[0]
    data_path = os.path.join(current_path,"in-context-bldc", "data")

    folder = "simulated/50_percent_control/training"
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
    ax1.plot(batch_y_np[:,:,0].T, label='Batch y (i_q_ref)', color='blue')
    ax1.set_title('Batch y (i_q_ref)')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Value')
    
    # Plot each component of batch_u
    ax2 = fig.add_subplot(2,1,2, sharex = ax1)
    ax2.plot(batch_u_np[:, :, 0].T, label='Batch u (id)', color='orange')
    ax2.plot(batch_u_np[:, :, 1].T, label='Batch u (iq)', color='green')
    ax2.plot(batch_u_np[:, :, 2].T, label='Batch u (vd)', color='red')
    ax2.plot(batch_u_np[:, :, 3].T, label='Batch u (vq)', color='purple')
    ax2.plot(batch_u_np[:, :, 4].T, label='Batch u (omega)', color='grey')
    ax2.plot(batch_u_np[:, :, 5].T, label='Batch u (omega_ref)')
    # ax2.plot(batch_u_np[:, :, 6].T, label='Batch u (T_ass)')
    # ax2.plot(batch_u_np[:, :, 7].T, label='Batch u (S_pct)')
    ax2.set_title('Batch u (id,iq,vd,vq,omega,omega_ref)')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Value')

    plt.tight_layout()


    # plot some window examples
    batch_u, batch_y, _ = reverse_normalization(batch_u, batch_y, batch_y)

    for i in range(10):
        fig = plt.figure()
        ax0 = fig.add_subplot(4,1,1)
        ax0.plot(batch_y[i,:,:],label = "$iq_ref$")
        ax0.legend()
        # ax0.set_ylim(-50,3050)
        ax1 = fig.add_subplot(4,1,2, sharex = ax0, sharey = ax0)
        ax1.plot(batch_u[i,:,0],label = "$I_d$")
        ax1.plot(batch_u[i,:,1],label = "$I_q$")
        ax1.legend()
        ax2 = fig.add_subplot(4,1,3, sharex = ax0)
        ax2.plot(batch_u[i,:,2],label = "$V_d$")
        ax2.plot(batch_u[i,:,3],label = "$V_q$")
        ax2.legend()
        ax3 = fig.add_subplot(4,1,4, sharex = ax0)
        ax3.plot(batch_u[i,:,4],label = "$omega$")
        ax3.plot(batch_u[i,:,5],label = "$omega_ref$")
        ax3.legend()
        # ax4 = fig.add_subplot(5,2,9)
        # ax4.plot(batch_u[i,:,6],label = "$T_{ass}$")
        # ax4.legend()
        # ax5 = fig.add_subplot(5,2,10)
        # ax5.plot(batch_u[i,:,7],label = "$S_{pct}$")
        # ax5.legend()
        # ax3.set_ylim(-50,3050)



    plt.show()
