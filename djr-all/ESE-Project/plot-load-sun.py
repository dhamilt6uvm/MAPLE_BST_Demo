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

# load the pkl file
substation_name = "Burton_Hill_ESE"
pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
with open(pkl_file, 'rb') as file:
    pkl = pickle.load(file)

# Determine which nodes have loads/gens
load_nodes = []
gen_nodes = []
for (ii,Node) in enumerate(pkl.Nodes):
    print(ii)
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


# Plot: two subplots, one for load/gen at all 4 load nodes, one for voltage at all nodes
colors = ['blue','orange','green','red']
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True)
time_hours = np.arange(24)
# Plot load and gen
for ii in range(load_all.shape[0]):
    ax1.plot(time_hours, load_all[ii,:]*1000, color=colors[ii], label=f'Load @ Node {load_nodes[ii]}') # convert to kW
for ii in range(gen_all.shape[0]):
    ax1.plot(time_hours, gen_all[ii,:]*1000, '--', color=colors[ii],  label=f'Solar @ Node {gen_nodes[ii]}') # convert to kW
ax1.set_ylabel('Load (kW)')
ax1.grid(True)
ax1.legend()
# plot voltage
for ii in range(V_day_nodes.shape[1]):
    ax2.plot(time_hours, V_day_nodes[:,ii], color=colors[ii], label=f'Node {load_nodes[ii]}')
ax2.set_ylabel('Voltage (p.u.)')
ax2.grid(True)
plt.xlabel('Time (hours)')
ax2.legend()
plt.show()

# Make a duck-curve plot
total_load = np.sum(load_all, axis=0) * 1000  # kW
total_gen = np.sum(gen_all, axis=0) * 1000    # kW
net_load = total_load - total_gen            # kW
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(time_hours, total_load, label='Total Load', color='red')
ax.plot(time_hours, total_gen, label='Total Solar Generation', color='green')
ax.plot(time_hours, net_load, linewidth=3, label='Net Load', color='blue')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Power (kW)')
ax.grid(True)
ax.legend()
plt.show()