###########################################################################################
## Script to visualize load and generation data from AMI data
###########################################################################################

# use this to determine what time of day to cut off into two different VVC schemes

## Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Data files to use
# change this in case of different data
fpgens = '/Users/danrussell/LOCAL/BST_repo/MAPLE_BST_Demo/Feeder_Data/Burton_Hill_small00/AMI_Data/Burton_Hill_small00_True_Gen_AMI_Data.csv'
fploads = '/Users/danrussell/LOCAL/BST_repo/MAPLE_BST_Demo/Feeder_Data/Burton_Hill_small00/AMI_Data/Burton_Hill_small00_True_Load_AMI_Data.csv'


## Import data
# define function to pull values
def extract_data(filepath):
    # extract data from CSV file and return data vectors
    data = pd.read_csv(filepath)
    index = data.iloc[:, 0].values
    # Extract datetime from second column
    timestamps = pd.to_datetime(data.iloc[:, 1])
    # Extract day of year and time
    day_of_year = timestamps.dt.dayofyear.values
    time_of_day = timestamps.dt.time
    hour_of_day = timestamps.dt.hour.values
    # Extract remaining data as matrix
    values = data.iloc[:, 2:].values
    return index, day_of_year, hour_of_day, values
# actually extract
genidx, genday, genhour, genvals = extract_data(fpgens)
loadidx, loadday, loadhour, loadvals = extract_data(fploads)

## Plot data
# scatter subplot of load and generation data
fig, axs = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
# Generation/Load data scatter
for i in range(genvals.shape[1]):
    axs[0].scatter(genhour, genvals[:, i], label=f'Gen {i+1}', s=10, alpha=0.01, color='blue')
    axs[1].scatter(loadhour, loadvals[:, i], label=f'Load {i+1}', s=10, alpha=0.01, color='red')
# Compute and plot average generation/load per hour
def compute_average(data, hour):
    df = pd.DataFrame(data)
    df['hour'] = hour               # return df.groupby('hour').mean()
    return df.groupby('hour').apply(lambda g: g.drop(columns='hour').replace(0, np.nan).mean())
gen_avg = compute_average(genvals, genhour)
load_avg = compute_average(loadvals, loadhour)
axs[0].plot(gen_avg.index, gen_avg.mean(axis=1), color='black', linewidth=2, label='Avg Gen')
axs[1].plot(load_avg.index, load_avg.mean(axis=1), color='black', linewidth=2, label='Avg Load')
# rest of plot
axs[0].set_xlabel('Hour')
axs[1].set_xlabel('Hour')
axs[0].set_ylabel('Generation')
axs[1].set_ylabel('Load')
axs[1].set_ylim(0, 2)
plt.tight_layout()
plt.show()

##### Based on plots!
# hours 0-10 have lower generation and also lower load (night time) (8pm to 6am)
# hours 11-23 have higher load and generation (day time) (7am to 7pm)