import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def process(input_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv, header=None, skiprows=1)
    data = df.values.astype(float)
    
    # Filter to smooth data
    window_size = 500
    kernel = np.ones(window_size) / window_size
    u_f = np.convolve(data[:,1], kernel, mode='same')
    
    return data, u_f

u_base, u_f_base = process("djr-all/ESE-Project/Loaddata-from-WH/u_base.csv")
u_hh, u_f_hh = process("djr-all/ESE-Project/Loaddata-from-WH/u_hh.csv")
u_hlh, u_f_hlh = process("djr-all/ESE-Project/Loaddata-from-WH/u_hlh.csv")
u_sc, u_f_sc = process("djr-all/ESE-Project/Loaddata-from-WH/u_sc.csv")


# # Plot comparison - filtered vs unfiltered
# plt.figure(figsize=(7, 3.5))
# plt.plot(u_base[:,0]/3600, u_base[:,1]/1000,linewidth=1, label='Base')
# plt.plot(u_base[:,0]/3600, u_f_base/1000,linewidth=1, label='filtered')
# plt.xlabel('Time (hours)',fontsize=14)
# plt.ylabel('Power (kW)',fontsize=14)
# plt.legend(fontsize=14)
# plt.grid()
# plt.tight_layout()
# plt.show()

# # Plot comparison - filtered vs unfiltered
# plt.figure(figsize=(7, 3.5))
# plt.plot(u_base[:,0]/3600, (u_f_base-u_f_hh)/1000,linewidth=1, label='hh')
# plt.plot(u_base[:,0]/3600, (u_f_base-u_f_hlh)/1000,linewidth=1, label='hlh')
# plt.plot(u_base[:,0]/3600, (u_f_base-u_f_sc)/1000,linewidth=1, label='Sc')
# plt.xlabel('Time (hours)',fontsize=14)
# plt.ylabel('Power (kW)',fontsize=14)
# plt.legend(fontsize=14)
# plt.grid()
# plt.tight_layout()
# plt.show()

# make signals uniform on minute scale
t_uniform = np.arange(u_base[0,0], u_base[-1,0], 60)  # every minute
u_fi_base = np.interp(t_uniform, u_base[:,0], u_f_base)
u_fi_hh = np.interp(t_uniform, u_hh[:,0], u_f_hh)
u_fi_hlh = np.interp(t_uniform, u_hlh[:,0], u_f_hlh)
u_fi_sc = np.interp(t_uniform, u_sc[:,0], u_f_sc)
# pack into one outout to save
u_all = np.vstack((t_uniform, u_fi_base, u_fi_hh, u_fi_hlh, u_fi_sc)).T
# save to csv
np.savetxt("djr-all/ESE-Project/Loaddata-from-WH/u_all_minute.csv",
            u_all, delimiter=",", 
            header="Time_seconds,Base,HH,HLH,SC", 
            comments='') 

# # plot and compare all 4 filtered and shifted signals
# plt.figure(figsize=(7, 3.5))
# plt.plot(t_uniform/3600, u_fi_base/1000,linewidth=1, label='Base')
# plt.plot(t_uniform/3600, u_fi_hh/1000,linewidth=1, label='hh')
# plt.plot(t_uniform/3600, u_fi_hlh/1000,linewidth=1, label='hlh')
# plt.plot(t_uniform/3600, u_fi_sc/1000,linewidth=1, label='Sc')
# plt.xlabel('Time (hours)',fontsize=14)
# plt.ylabel('Power (kW)',fontsize=14)
# plt.legend(fontsize=14)
# plt.grid()
# plt.tight_layout()
# plt.show()

# # plot and compare an un-interpolated with interpolated
# plt.figure(figsize=(7, 3.5))
# plt.plot(u_base[:,0]/3600, u_f_hlh/1000,linewidth=1, label='filtered')
# plt.plot(t_uniform/3600, u_fi_hlh/1000,linewidth=1, label='hlh')
# plt.xlabel('Time (hours)',fontsize=14)
# plt.ylabel('Power (kW)',fontsize=14)
# plt.legend(fontsize=14)
# plt.grid()
# plt.tight_layout()
# plt.show()