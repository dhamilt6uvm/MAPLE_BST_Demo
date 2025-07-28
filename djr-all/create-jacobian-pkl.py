## Script to create a pkl file that has loading conditions that will be used for finding the jacobian of power flow
###########################################################################################


# Things to do:
# finish this code obvi
# get fresh data from the server
# get updated solve_3ph_pf.jl code

## Import packages etc. ##################################################################
import pandas as pd
import numpy as np
import pickle
import re

import sys
import os 
os.system('clear')

substation_name = "Burton_Hill_small02" # change this to the substation you want to use

## Import pkl file & find meter IDs ########################################################
# load the file in
pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
with open(pkl_file, 'rb') as file:
    pkl_model = pickle.load(file)               # this references GLM_Tools somehow, so it won't run unless GLM_Tools is in the same folder
                                                # not sure where the reference is so I can't fix it... 
# # pull generator names - extract IDs from names
# gen_names = [gen.name for gen in pkl_model.Generator_Dict.values()]
# gen_ids = [int(re.search(r'gene_(\d+)_negLdGen', s).group(1)) for s in gen_names]
# # repeat for loads
# load_names = [load.name for load in pkl_model.Load_Dict.values()]
# ###########################
# # for name in load_names:
# #     if "OSMOSE" in name:
# #         print(name)
# # # print(pkl_model.Load_Dict['_3085424_cons'])
# # node = pkl_model.Load_Dict['_3085424_cons']
# # print("print(node)")
# # print(node)
# # print("dir(node)")
# # print(dir(node))        # List available attributes/methods
# # print("vars(node)")
# # print(vars(node))       # Dump the internal fields (may be large)
# ##########################
# load_ids = [
#     int(re.search(r'\d+', name).group())
#     for name in load_names
#     if not name.startswith("OSMOSE") and re.search(r'\d+', name)
# ]           # ignore OSMOSE loads, because they are all in the null meters list - in future, actually check the null meters list instead of just ignoring them

# # load meter number dictionaries to map the IDs to meter numbers
# gen_dict_file = f"Feeder_Data/{substation_name}/gen_meter_number_data.csv"
# load_dict_file = f"Feeder_Data/{substation_name}/meter_number_data.csv"
# gen_dict = pd.read_csv(gen_dict_file)
# load_dict = pd.read_csv(load_dict_file)

# # map it up
# filtered_gen_ids = gen_dict[gen_dict['Object ID'].isin(gen_ids)]                # filter rows with IDs
# filtered_load_ids = load_dict[load_dict['Service Number'].isin(load_ids)]       # filter rows with IDs
# gen_meters = filtered_gen_ids['Meter Number'].tolist()                          # get the meter numbers
# load_meters = filtered_load_ids['Meter Number'].tolist()                        # get the meter numbers



# Extract load names and IDs
gen_names = [gen.name for gen in pkl_model.Generator_Dict.values()]
gen_id_to_name = gen_id_to_name = {
    int(re.search(r'gene_(\d+)_negLdGen', s).group(1)): s
    for s in gen_names
    if re.search(r'gene_(\d+)_negLdGen', s)
}
load_names = [load.name for load in pkl_model.Load_Dict.values()]
load_id_to_name = {
    int(re.search(r'\d+', name).group()): name
    for name in load_names
    if not name.startswith("OSMOSE") and re.search(r'\d+', name)
}

# Load the meter number data
gen_dict_file = f"Feeder_Data/{substation_name}/gen_meter_number_data.csv"
load_dict_file = f"Feeder_Data/{substation_name}/meter_number_data.csv"
gen_dict = pd.read_csv(gen_dict_file)
load_dict = pd.read_csv(load_dict_file)

# Filter and build mapping from load_id to meter number
gen_ids = list(gen_id_to_name.keys())
load_ids = list(load_id_to_name.keys())
filtered_gen_ids = gen_dict[gen_dict['Object ID'].isin(gen_ids)]                # filter rows with IDs
filtered_load_ids = load_dict[load_dict['Service Number'].isin(load_ids)]
gen_id_to_meter = dict(zip(filtered_gen_ids['Object ID'], filtered_gen_ids['Meter Number']))
load_id_to_meter = dict(zip(filtered_load_ids['Service Number'], filtered_load_ids['Meter Number']))

# Build final mappings
gen_meter_to_name = {
    load_id_to_meter[load_id]: load_id_to_name[load_id]
    for load_id in load_id_to_meter
    if load_id in load_id_to_name
}
gen_name_to_meter = {
    load_id_to_name[load_id]: load_id_to_meter[load_id]
    for load_id in load_id_to_meter
    if load_id in load_id_to_name
}
load_meter_to_name = {
    load_id_to_meter[load_id]: load_id_to_name[load_id]
    for load_id in load_id_to_meter
    if load_id in load_id_to_name
}
load_name_to_meter = {
    load_id_to_name[load_id]: load_id_to_meter[load_id]
    for load_id in load_id_to_meter
    if load_id in load_id_to_name
}

# get lists of meters
gen_meters = filtered_gen_ids['Meter Number'].tolist()                          # get the meter numbers
load_meters = filtered_load_ids['Meter Number'].tolist()                        # get the meter numbers


## Process the .csv data into averages ###################################################
fpgens = f"Feeder_Data/{substation_name}/AMI_Data/{substation_name}_True_Gen_AMI_Data.csv"
fploads = f"Feeder_Data/{substation_name}/AMI_Data/{substation_name}_True_Load_AMI_Data.csv"

## Import data 
def extract_data(filepath, cols_to_keep):
    # Read CSV file
    data = pd.read_csv(filepath)
    # Filter columns: keep 'start_date_time' and the specified meter numbers
    columns_to_use = ['start_date_time'] + [str(col) for col in cols_to_keep if str(col) in data.columns]
    missing_cols = [str(col) for col in cols_to_keep if str(col) not in data.columns]
    if missing_cols:
        # raise ValueError(f"Missing expected columns in CSV: {missing_cols}")
        print(f"Missing expected columns in CSV: {missing_cols}")
        print(filepath)
    # pare down the data
    data = data[columns_to_use]
    # Extract index (assuming row numbers in first column before read_csv)
    index = data.index.values
    # Extract datetime
    timestamps = pd.to_datetime(data['start_date_time'])
    day_of_year = timestamps.dt.dayofyear.values
    hour_of_day = timestamps.dt.hour.values
    # Extract the selected meter data
    values = data.iloc[:, 1:].values  # exclude index, use the filtered meters
    return index, day_of_year, hour_of_day, values

# actually extract
genidx, genday, genhour, genvals = extract_data(fpgens, gen_meters)
loadidx, loadday, loadhour, loadvals = extract_data(fploads, load_meters)
nloads = loadvals.shape[1]  # number of loads
ngens = genvals.shape[1]    # number of generators

# print(genvals.shape)
# print(loadvals.shape)

## Divide into hour-sets and get average
day_hrs = np.arange(11,24)                      # adjust if needed, based on VisualizeLoadsGens.py
night_hrs = np.arange(0,11)
# function for getting day and night averages
def get_daynight_avgs(data,day_hrs,night_hrs):
    nsamp = data.shape[0]                       # get number of time stamps
    day_data = []                                   # initialize empty lists for day and night data     
    night_data = []
    for ii in range(nsamp):                         # loop through all samples
        samp = data[ii,:]                        # get the sample
        if genhour[ii] in day_hrs:
            day_data.append(samp)
        elif genhour[ii] in night_hrs:
            night_data.append(samp)
        else:
            print(f"Error: hour {genhour[ii]} not in day or night hours")
    # convert to numpy arrays
    day_data = np.array(day_data)
    night_data = np.array(night_data)
    ## Calculate averages
    day_avg = np.mean(day_data, axis=0)            # average over the day data
    night_avg = np.mean(night_data, axis=0)        # average over the night
    return day_avg, night_avg
# get day and night averages
gen_day_avg, gen_night_avg = get_daynight_avgs(genvals, day_hrs, night_hrs)
load_day_avg, load_night_avg = get_daynight_avgs(loadvals, day_hrs, night_hrs)


## Perturb the csv data to make arrays for each loading condition ##################
eps = 10 # perturbation value
# function for perturbing
def perturb_data(data, eps, n):
    plus = []               # init lists
    minus = []
    for ii in range(n):     # loop over all values
        perturb = np.zeros([n]) # perturb just one of them
        perturb[ii] = eps
        plus.append(data + perturb) # plus perturbation
        minus.append(data - perturb)    # minus pert...
    plus = np.array(plus)           # convery to np arrays
    minus = np.array(minus)
    return plus, minus

# fit into big arrays
def combine_mats(data_avg, n, eps, n_other):
    # n is the number of things that are in data
    # n other is the number of unmodified data rows to insert
    data_p, data_m = perturb_data(data_avg, eps, n)
    data_unmod = np.tile(data_avg, (n_other,1))
    if n > n_other:     # for loads, modified comes first (assuming more loads than gens)
        data_p_with = np.vstack((data_p, data_unmod))       # with unmodified rows
        data_m_with = np.vstack((data_m, data_unmod))
    else:               # for gens, unmodified comes first (assuming more loads than gens)  
        data_p_with = np.vstack((data_unmod, data_p))
        data_m_with = np.vstack((data_unmod, data_m))
    return np.vstack((data_p_with, data_m_with))

# create the big matrices
load_day = combine_mats(load_day_avg, nloads, eps, ngens)       # samples x load-nodes where samples is [ + eps, 0, -eps, 0] (0's for gens)
gen_day = combine_mats(gen_day_avg, ngens, eps, nloads)         # samples x gen-nodes where samples is [ 0, + eps, 0, -eps] (0's for loads)
load_night = combine_mats(load_night_avg, nloads, eps, ngens)   # samples x load-nodes where samples is [ + eps, 0, -eps, 0] (0's for gens)
gen_night = combine_mats(gen_night_avg, ngens, eps, nloads)     # samples x gen-nodes where samples is [ 0, + eps, 0, -eps] (0's for loads)



# check if the loads are in the null meters list
fp = f"Feeder_Data/{substation_name}/null_meters.csv"
null_loads_df = pd.read_csv(fp)
null_load_ids = set(null_loads_df['Service Number'])  # use set for faster lookup

## Overwrite the values in pkl file ################################################
# print the current load values
# for load in pkl_model.Loads:
#     if not hasattr(load, 'Sload'):
#         print(f"Load {load.name}") 
#         # print(f"current value: {load.Sload}")

for load in pkl_model.Loads:
    if hasattr(load, 'Sload'):
        print(1)
        # overwrite the Sload value with appropriate column from load_day or load_night
        # ideally use the list mapping thing that we made up above to smoothly get the meter number from the name
        # figure out how to map meter number to column of load_day/night

# for load in pkl_model.Loads:
#     if not hasattr(load, 'Sload'):
#         print(load.name)



# node = pkl_model.Load_Dict['_13277_cons']
# print("print(node)")
# print(node)
# print("dir(node)")
# print(dir(node))        # List available attributes/methods
# print("vars(node)")
# print(vars(node))       # Dump the internal fields (may be large)

# need to find a way to manage the loads that are in null meters list
# need to find a way to map columns of my Sload data to the node names


## save the new pkl file ###########################################################



######
# Various things I printed to see stuff: 
# print(list(vars(pkl_model).keys()))
# type(pkl_model.Nodes)
# type(pkl_model.Node_Dict)
# print(pkl_model.Nodes[:3])         # likely a list of objects
# print(list(pkl_model.Node_Dict))   # prints the keys (e.g., node names)
# print(pkl_model.Generators[:3])         # likely a list of objects
# node = pkl_model.Generator_Dict['gene_1276_negLdGen']
# print("print(node)")
# print(node)
# print("dir(node)")
# print(dir(node))        # List available attributes/methods
# print("vars(node)")
# print(vars(node))       # Dump the internal fields (may be large)
# index_list = [gen.name for gen in pkl_model.Generator_Dict.values()]
# print(index_list)
# print(vars(pkl_model))
# print(type(pkl_model))