import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')) # Get the directory TWO level up from the script
sys.path.insert(0, parent_dir)  # Add the parent directory to the system path 
import GLM_Tools.PowerSystemModel as psm
import pickle

start_idx = (31+30+27)*24 # hours before July 28th

# load the pkl file
substation_name = "Burton_Hill_ESE"
pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
with open(pkl_file, 'rb') as file:
    pkl = pickle.load(file)

# Determine which nodes have loads/gens
load_nodes = []
gen_nodes = []
for (ii,Node) in enumerate(pkl.Nodes):
    # print(ii)
    if len(Node.loads) > 0:
        load_nodes.append(ii)
    if len(Node.gens) > 0:
        gen_nodes.append(ii)

# Extract Load and Gen data
load_all = np.zeros((len(load_nodes), 24))
gen_all = np.zeros((len(gen_nodes), 24))
for (ii,load) in enumerate(pkl.Loads):
    load_all[ii,:] = load.Sload[start_idx:start_idx+24,1].real # units MW I think
for (ii,gen) in enumerate(pkl.Generators):
    gen_all[1-ii,:] = gen.Sgen[start_idx:start_idx+24,1].real       # flipping the order so it matches the load nodes

# # plot power
# colors = ['blue','orange','green','red']
# fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True)
# time_hours = np.arange(24)
# # Plot load and gen
# for ii in range(load_all.shape[0]):
#     ax1.plot(time_hours, load_all[ii,:]*1000, color=colors[ii], label=f'Load @ Node {load_nodes[ii]}') # convert to kW
# for ii in range(gen_all.shape[0]):
#     ax1.plot(time_hours, gen_all[ii,:]*1000, '--', color=colors[ii],  label=f'Solar @ Node {gen_nodes[ii]}') # convert to kW
# ax1.set_ylabel('Load (kW)')
# ax1.grid(True)
# ax1.legend()
# plt.xlabel('Time (hours)')
# plt.show()

# interpolate load data to minute scale
t_uniform = np.arange(0, 24*60*60, 60)  # every minute
t_hrs = np.arange(0,24,1)
load_i = np.zeros((t_uniform.shape[0], load_all.shape[0]))
for ii in range(load_all.shape[0]):
    load_i[:,ii] = np.interp(t_uniform, t_hrs * 3600, load_all[ii,:] * 1000)  # in kW

gen_i = np.zeros((t_uniform.shape[0], gen_all.shape[0]))
for ii in range(gen_all.shape[0]):
    gen_i[:,ii] = np.interp(t_uniform, t_hrs * 3600, gen_all[ii,:] * 1000)  # in kW

load_gen_i = np.hstack((load_i, gen_i))
np.savetxt("djr-all/ESE-Project/Loaddata-from-WH/load_gen_minute.csv",
            load_gen_i, delimiter=",", 
            header="Load1,Load2,Load3,Load4,Gen1,Gen2", 
            comments='')

# colors = ['blue','orange','green','red']
# fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True)
# time_hours = np.arange(24)
# # Plot load and gen
# for ii in range(load_all.shape[0]):
#     ax1.plot(t_uniform/3600, load_i[:,ii], color=colors[ii], label=f'Load @ Node {load_nodes[ii]}') # convert to kW
# for ii in range(load_all.shape[0]):
#     ax1.plot(time_hours, load_all[ii,:]*1000, '--', color=colors[ii],  label=f'Solar @ Node {load_nodes[ii]}') # convert to kW
# ax1.set_ylabel('Load (kW)')
# ax1.grid(True)
# ax1.legend()
# plt.xlabel('Time (hours)')
# plt.show()