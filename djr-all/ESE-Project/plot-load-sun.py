## Make a plot of voltage, load, and solar over time for July 28th 2024
# July 28th picked because of https://weatherspark.com/h/d/24985/2024/7/28/Historical-Weather-on-Sunday-July-28-2024-in-Burlington-Vermont-United-States#Figures-CloudCover
# says it was mostly not cloudy that day
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')) # Get the directory TWO level up from the script
sys.path.insert(0, parent_dir)  # Add the parent directory to the system path 
import GLM_Tools.PowerSystemModel as psm
import pickle

start_idx = (31+30+27)*24 # hours before July 28th

show_old_plots = False

# mapping different indices
map_to_loads = [3, 0, 2, 1]
map_to_volts = [0, 3, 1, 2]

# load the pkl file
substation_name = "Burton_Hill_ESE"
pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
with open(pkl_file, 'rb') as file:
    pkl = pickle.load(file)

# Determine which nodes have loads/gens
load_nodes = []
gen_nodes = []
for (ii,Node) in enumerate(pkl.Nodes):
    if len(Node.loads) > 0:
        load_nodes.append(ii)
    if len(Node.gens) > 0:
        gen_nodes.append(ii)
print(gen_nodes)
print(load_nodes)

# Extract Load and Gen data
load_all = np.zeros((len(load_nodes), 24))
gen_all = np.zeros((len(gen_nodes), 24))
for (ii,load) in enumerate(pkl.Loads):
    load_all[ii,:] = load.Sload[start_idx:start_idx+24,1].real # units MW I think
for (ii,gen) in enumerate(pkl.Generators):
    gen_all[1-ii,:] = gen.Sgen[start_idx:start_idx+24,1].real       # flipping the order so it matches the load nodes

# Simulate to get voltage
# done in Julia script: voltages-over-time.jl
df = pd.read_csv("djr-all/ESE-Project/V_allAMI_BH_ESE.csv", header=None, skiprows=1)
V_all = df.values.astype(float)
V_day_nodes = V_all[start_idx:start_idx+24, load_nodes].round(6)

if show_old_plots:
    # Plot: two subplots, one for load/gen at all 4 load nodes, one for voltage at all nodes
    colors = ['blue','orange','green','red']
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,6), sharex=True)
    time_hours = np.arange(24)
    # Plot load and gen

    for ii in range(load_all.shape[0]):
        idx = map_to_loads[ii]
        ax1.plot(time_hours, load_all[idx,:]*1000, color=colors[ii], label=f'Node {ii+1} Load') # convert to kW
    # for ii in range(gen_all.shape[0]):
    #     ax1.plot(time_hours, gen_all[ii,:]*1000, '--', color=colors[ii],  label=f'Solar @ Node {ii+1}') # convert to kW
    ax1.plot(time_hours, gen_all[1,:]*1000, '--', color=colors[0],  label=f'Node {1} Solar') # convert to kW
    ax1.plot(time_hours, gen_all[0,:]*1000, '--', color=colors[2],  label=f'Node {3} Solar') # convert to kW
    ax1.axvspan(6, 19, color='green', alpha=0.08)
    ax1.axvspan(19, 21, color='red', alpha=0.08)
    ax1.set_ylabel('Load (kW)')
    ax1.grid(True)
    ax1.legend()
    # plot voltage
    for ii in range(V_day_nodes.shape[1]):
        idx = map_to_volts[ii]
        ax2.plot(time_hours, V_day_nodes[:,idx], color=colors[ii], label=f'Node {ii+1}')
    ax2.axvspan(6, 19, color='green', alpha=0.08)
    ax2.axvspan(19, 21, color='red', alpha=0.08)
    ax2.set_ylabel('Voltage (p.u.)')
    ax2.grid(True)
    plt.xlabel('Time (hours)')
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # Make a duck-curve plot
    total_load = np.sum(load_all, axis=0) * 1000  # kW
    total_gen = np.sum(gen_all, axis=0) * 1000    # kW
    net_load = total_load - total_gen            # kW
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(time_hours, total_load, label='Total Load', color='red')
    ax.plot(time_hours, total_gen, label='Total Solar Generation', color='green')
    ax.plot(time_hours, net_load, linewidth=3, label='Net Load', color='blue')
    ax.axvspan(6, 19, color='green', alpha=0.08)
    ax.axvspan(19, 21, color='red', alpha=0.08)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Power (kW)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()



## Re-make the plots above using minutely data ##############################################################
# Import load data
def make_plots(load_file_name, volt_file_name):
    df = pd.read_csv(load_file_name, header=None, skiprows=1)
    load_all_min = df.values.astype(float)
    load_all_min = load_all_min.round(6)

    # Simulate to get voltage
    # done in Julia script: voltages-over-time.jl
    df = pd.read_csv(volt_file_name, header=None, skiprows=1)
    V_all = df.values.astype(float)
    V_all = V_all[:, load_nodes].round(6)

    # Plot: two subplots, one for load/gen at all 4 load nodes, one for voltage at all nodes
    colors = ['blue','orange','green','red']
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,6), sharex=True)
    time_hours = np.arange(24)
    time_minutes = np.arange(0, 24*60)/60
    # Plot load and gen
    for ii in range(load_all_min.shape[1]):
        idx = map_to_loads[ii]
        ax1.plot(time_minutes, load_all_min[:,idx]*1000, color=colors[ii], label=f'Node {ii+1} Load') # convert to kW
    ax1.plot(time_hours, gen_all[0,:]*1000, '--', color=colors[0],  label=f'Node {1} Solar') # convert to kW
    ax1.plot(time_hours, gen_all[1,:]*1000, '--', color=colors[2],  label=f'Node {3} Solar') # convert to kW
    ax1.axvspan(6, 19, color='green', alpha=0.08)
    ax1.axvspan(19, 21, color='red', alpha=0.08)
    ax1.set_ylabel('Load (kW)')
    ax1.grid(True)
    ax1.legend()
    # plot voltage
    for ii in range(V_all.shape[1]):
        idx = map_to_volts[ii]
        ax2.plot(time_minutes, V_all[:,idx], color=colors[ii], label=f'Node {ii+1}')
    ax2.axvspan(6, 19, color='green', alpha=0.08)
    ax2.axvspan(19, 21, color='red', alpha=0.08)
    ax2.set_ylabel('Voltage (p.u.)')
    ax2.grid(True)
    plt.xlabel('Time (hours)')
    ax2.legend()
    plt.tight_layout()
    plt.show()


# print("BASE CASE")
# load_name = "djr-all/ESE-Project/load_WH_modified_ESE_base.csv"
# volt_name = "djr-all/ESE-Project/V_all_ESE_base.csv"
# make_plots(load_name,volt_name)

# print("HIGH HOLD CASE")
# load_name = "djr-all/ESE-Project/load_WH_modified_ESE_hh.csv"
# volt_name = "djr-all/ESE-Project/V_all_ESE_hh.csv"
# make_plots(load_name,volt_name)

# print("HIGH LOW HOLD CASE")
# load_name = "djr-all/ESE-Project/load_WH_modified_ESE_hlh.csv"
# volt_name = "djr-all/ESE-Project/V_all_ESE_hlh.csv"
# make_plots(load_name,volt_name)

# print("SUPERCHARGE CASE")
# load_name = "djr-all/ESE-Project/load_WH_modified_ESE_sc.csv"
# volt_name = "djr-all/ESE-Project/V_all_ESE_sc.csv"
# make_plots(load_name,volt_name)

## Plot all voltages from all 4 trials

# Files
files = {
    "base": "djr-all/ESE-Project/V_all_ESE_base.csv",
    "hh":   "djr-all/ESE-Project/V_all_ESE_hh.csv",
    "hlh":  "djr-all/ESE-Project/V_all_ESE_hlh.csv",
    "sc":   "djr-all/ESE-Project/V_all_ESE_sc.csv",
}

# Build time vector (minutely, 24 hours)
time_minutes = np.arange(0, 24*60) / 60

trials = list(files.keys())
# trial_names = ['base','high hold','high low hold','supercharge']
style = ['-', '--', '-.', ':']

plt.figure(figsize=(6,3))

for trial_index, trial in enumerate(trials):
    # Load voltage file
    df = pd.read_csv(files[trial], header=None, skiprows=1)
    V_all = df.values.astype(float)
    V_all = V_all[:, load_nodes].round(6)
    plt.plot(time_minutes, V_all[:, 0], style[trial_index], alpha=0.7,label=f"1: {trial}")
    plt.plot(time_minutes, V_all[:, 1], style[trial_index], alpha=0.7,label=f"3: {trial}")

# Format plot
plt.axvspan(6, 19, color='green', alpha=0.08)
plt.axvspan(19, 21, color='red', alpha=0.08)
plt.xlabel("Time (hours)")
plt.ylabel("Voltage (p.u.)")
plt.grid(True)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()


## Make duck curve plot with all 4 trials:
# import minutely gen data
df = pd.read_csv("djr-all/ESE-Project/Load-Data/load_gen_minute.csv", header=0)
data = df.values.astype(float)
gen = data[:,4:]


# Files
files = {
    "base": "djr-all/ESE-Project/load_WH_modified_ESE_base.csv",
    "hh":   "djr-all/ESE-Project/load_WH_modified_ESE_hh.csv",
    "hlh":  "djr-all/ESE-Project/load_WH_modified_ESE_hlh.csv",
    "sc":   "djr-all/ESE-Project/load_WH_modified_ESE_sc.csv",
}


# Colors for each node, repeated for each trial
node_colors = ['blue', 'orange', 'green', 'red']
trials = list(files.keys())
# trial_names = ['base','high hold','high low hold','supercharge']
style = ['-', '--', '-.', ':']

plt.figure(figsize=(6,3))

for trial_index, trial in enumerate(trials):
    # Load voltage file
    df = pd.read_csv(files[trial], header=None, skiprows=1)
    load = df.values.astype(float) *1000  # to kW
    net = load.sum(axis=1) - gen.sum(axis=1)
    if trial_index == 0:
        lw = 1
    else:
        lw = 2
    plt.plot(time_minutes, net, style[trial_index],linewidth=lw, alpha=0.7,label=f"{trial}")

# Format plot
plt.axvspan(6, 19, color='green', alpha=0.08)
plt.axvspan(19, 21, color='red', alpha=0.08)
plt.xlabel("Time (hours)")
plt.ylabel("Net Load (kW)")
plt.grid(True)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
# plt.xlim([17,23])
# plt.ylim([-8,22])
plt.show()


## Compute some metrics

v_files = {
    "base": "djr-all/ESE-Project/V_all_ESE_base.csv",
    "hh":   "djr-all/ESE-Project/V_all_ESE_hh.csv",
    "hlh":  "djr-all/ESE-Project/V_all_ESE_hlh.csv",
    "sc":   "djr-all/ESE-Project/V_all_ESE_sc.csv",
}
l_files = {
    "base": "djr-all/ESE-Project/load_WH_modified_ESE_base.csv",
    "hh":   "djr-all/ESE-Project/load_WH_modified_ESE_hh.csv",
    "hlh":  "djr-all/ESE-Project/load_WH_modified_ESE_hlh.csv",
    "sc":   "djr-all/ESE-Project/load_WH_modified_ESE_sc.csv",
}


# average voltage deviation at all nodes and at load nodes
for trial_index, trial in enumerate(trials):
    # Load voltage file
    df = pd.read_csv(v_files[trial], header=None, skiprows=1)
    V_all = df.values.astype(float)
    V_ld = V_all[:, load_nodes].round(6)
    avg_volt_all = np.mean(abs(V_all - np.ones(V_all.shape)))
    avg_volt_ld = np.mean(abs(V_ld - np.ones(V_ld.shape)))
    print(f"Trial: {trial}, Avg Volt all nodes: {avg_volt_all:.7f}, Avg Volt load nodes: {avg_volt_ld:.7f}")


# load statistics computed in plot_poweruse.py


