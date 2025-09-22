import numpy as np
import pandas as pd
import glob
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import copy
import seaborn as sns


current_path = os.getcwd().split("in-context-bldc")[0]
data_path = os.path.join(current_path,"in-context-bldc", "data")

# folder = "CL_experiments_double_sensor_low_speed_ekf_and_meta/final/inertia13_ki-0.0029-kp-3.0000"
# folder = "simulated/50_percent_control/training"
data_folders = ["simulated/50_percent_control/training"]
# data_folders = ["simulated/50_percent_control/training","simulated/50_percent_control_perturbed/training","simulated/50_percent_control_current_disturbance/training"]
# data_folders = ["simulated/50_percent_control/validation","simulated/50_percent_control_perturbed/validation","simulated/50_percent_control_current_disturbance/validation"]
# data_folders = ["simulated/50_percent_control/training","simulated/50_percent_control_perturbed/training","simulated/50_percent_control_current_disturbance/training", "simulated/50_percent_control/validation","simulated/50_percent_control_perturbed/validation","simulated/50_percent_control_current_disturbance/validation"]
folder_path_list = [os.path.join(data_path, folder) for folder in data_folders]
# folder_path = os.path.join(data_path, folder)
file_list = []
for folder_path in folder_path_list:
    file_list = file_list + glob.glob(os.path.join(folder_path, '*.csv'))


# file_list = glob.glob(os.path.join(folder_path, '*.csv'))

metadata_matrix = np.zeros((len(file_list),4))
i = 0
for file in file_list:
    df = pd.read_csv(file)
    metadata = df.keys()[-1].split(',')
    # print(metadata)
    T_ass = float(metadata[0].split(":")[1])
    S_pct = float(metadata[1].split(":")[1])
    Kp = float(metadata[2].split(":")[1])
    Ki = float(metadata[3].split(":")[1])
    # print(T_ass, S_pct, Kp, Ki)
    metadata_matrix[i,:] = [T_ass, S_pct, Kp, Ki]
    i+=1


valid = np.count_nonzero(np.logical_and(metadata_matrix[:,0] <= 1.5, metadata_matrix[:,1] <= 20))
print(valid)
full = np.shape(metadata_matrix)[0]
print(full)
print(valid/full)





# fig = plt.figure()
# plt.scatter(metadata_matrix[:,0],metadata_matrix[:,1], s= 10)
# plt.xlabel("$T_{set}$")
# plt.ylabel("$OS_\%$")


# fig = plt.figure()
# plt.scatter(metadata_matrix[:,2],metadata_matrix[:,3])
# plt.xlabel("$K_p$")
# plt.ylabel("$K_i$")
# ax = plt.gca()
# ax.set_xscale('log')
# ax.set_yscale('log')


# fig = plt.figure()
# sns.histplot(data=metadata_matrix[:,0], stat='density')
# ax = plt.gca()
# ax.set(xlabel="$T_{set}$")
# ax.grid()
# ax.set_title("Settling time distribution")

# fig = plt.figure()
# sns.histplot(data=metadata_matrix[:,1], stat='density')
# ax = plt.gca()
# ax.set(xlabel="$OS_\%$")
# ax.grid()
# ax.set_title("Overshoot distribution")


# data_tmp = pd.DataFrame()
# data_tmp['T_set'] = metadata_matrix[:,0]
# data_tmp['S_pct'] = metadata_matrix[:,1]



# # fig = plt.figure()
# # sns.kdeplot(data=data_tmp, x='T_set', y = 'S_pct')
# # ax = plt.gca()
# # ax.set(xlabel="$T_{set}$", ylabel="$OS_\%$")
# # ax.grid()
# # ax.set_title("cross distribution")


# fig = plt.figure()
# plt.scatter(metadata_matrix[:,0],metadata_matrix[:,1], s= 10)
# plt.scatter([0.05, 1.7],[0, -2], c= 'r', s = 20)
# plt.xlabel("$T_{set}$")
# plt.ylabel("$OS_\%$")





# fig = plt.figure()
# plt.scatter(metadata_matrix[:,0],metadata_matrix[:,1], s= 10)
# plt.xlabel("$T_{set}$")
# ax = plt.gca()
# ax.set_xscale('log')
# plt.ylabel("$OS_\%$")

# fig = plt.figure()
# plt.scatter(metadata_matrix[:,0],metadata_matrix[:,1], s= 10)
# plt.scatter([0.05, 1.7],[0, -2], c= 'r', s = 20)
# plt.xlabel("$T_{set}$")
# ax = plt.gca()
# ax.set_xscale('log')
# plt.ylabel("$OS_\%$")


# fig = plt.figure()
# sns.histplot(data=metadata_matrix[:,1], stat='density', log_scale=True)
# ax = plt.gca()
# ax.set(xlabel="$OS_\%$")
# ax.grid()
# ax.set_title("Overshoot distribution")



# fig = plt.figure(figsize=(6,3))
# sns.histplot(data=metadata_matrix[:,0], stat='density')
# ax = plt.gca()
# ylim = ax.get_ylim()
# plt.vlines([0.05, 1.7],ymin=0, ymax=1, colors='r', linestyles='--')
# ax.set_ylim(ylim)
# ax.set(xlabel="$T_{set}$")
# ax.grid()
# ax.set_title("Settling time distribution")
# plt.tight_layout()




# fig = plt.figure(figsize=(6,3))
# sns.histplot(data=metadata_matrix[:,0], stat='density', log_scale=True)
# ax = plt.gca()
# ylim = ax.get_ylim()
# plt.vlines([0.05, 1.7],ymin=0, ymax=2, colors='r', linestyles='--')
# ax.set_ylim(ylim)
# ax.set(xlabel="$T_{set}$")
# ax.grid()
# ax.set_title("Settling time distribution")
# plt.tight_layout()



# plt.show()






