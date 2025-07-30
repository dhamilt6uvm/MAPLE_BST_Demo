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

# Create map from load names to meter numbers and make list of meter numbers
load_names = [load.name for load in pkl_model.Load_Dict.values()]                   # list of all load names
load_serv_num = [int(re.search(r'\d+', name).group()) for name in load_names]       # list of service numbers from names
load_dict_file = f"Feeder_Data/{substation_name}/meter_number_data.csv"             # import serv num > meter num map
load_dict = pd.read_csv(load_dict_file)             
serv_to_meter = {                                                                   # map from service number to meter number
    int(serv): meter                                                                # convert to integer because reads it in as a string
    for serv, meter in zip(load_dict.iloc[:, 0], load_dict.iloc[:, 1])
}
load_name_to_meter = {                                                              # map from load name to meter number                        
    name: serv_to_meter[serv_num]
    for name, serv_num in zip(load_names, load_serv_num)
    if serv_num in serv_to_meter
}
load_meters = list(load_name_to_meter.values())                                     # list of all load meter numbers

# Create map from gen names to meter numbers and make list of meter numbers
gen_names = [gen.name for gen in pkl_model.Generator_Dict.values()]                 # list of all gen names
gen_objID = [int(re.search(r'\d+', name).group()) for name in gen_names]            # list of object IDs from names
gen_dict_file = f"Feeder_Data/{substation_name}/gen_meter_number_data.csv"          # import object id > meter num map
gen_dict = pd.read_csv(gen_dict_file)
objID_to_meter = {                                                                  # map from object ID to meter number
    int(objID): meter                                                               # convert to integer because reads it in as a string
    for objID, meter in zip(gen_dict.iloc[:, 0], gen_dict.iloc[:, 2])
}
gen_name_to_meter = {                                                               # map from gen name to meter number                        
    name: objID_to_meter[objID]
    for name, objID in zip(gen_names, gen_objID)
    if objID in objID_to_meter
}
gen_meter_to_name = {                                                               # map from meter number to gen name                     
    meter: name
    for name, meter in gen_name_to_meter.items()
}
gen_meters = list(gen_name_to_meter.values())                                       # list of all load meter numbers


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
    # Extract datetime
    timestamps = pd.to_datetime(data['start_date_time'])
    hour_of_day = timestamps.dt.hour.values
    # Extract the selected meter data
    values = data.iloc[:, 1:].values  # exclude index, use the filtered meters
    # map meter numbers to column indices
    meter_to_col = {col: i for i, col in enumerate(data.columns[1:])}
    return hour_of_day, values, meter_to_col

# actually extract
genhour, genvals, gen_meter_col_dict = extract_data(fpgens, gen_meters)
loadhour, loadvals, load_meter_col_dict = extract_data(fploads, load_meters)

# compute how many loads and gens
nloads = loadvals.shape[1]  # number of loads
ngens = genvals.shape[1]    # number of generators

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


## Overwrite the values in pkl file ################################################
# determine what column Sload should be in 
phase = 'B'
if phase == 'A':
    col = 0
elif phase == 'B':
    col = 1
elif phase == 'C':  
    col = 2
## Create and save day pickle file
# load day
for load in pkl_model.Loads:
    if hasattr(load, 'Sload'):                                  # make sure it has Sload attribute
        if load.name in load_name_to_meter:                     # if the load is in the meter map 
            load_meter = load_name_to_meter[load.name]          # get the meter number from the map
            if str(load_meter) in load_meter_col_dict:               # check if the meter is in the map - there's one node that isn't
                load_col_idx = load_meter_col_dict[str(load_meter)]     # get the column index from the map
                load_day_col = load_day[:, load_col_idx]                # get the column from the load_day matrix
            else:
                print(f"Warning: Meter {load_meter} not found in load_meter_col_dict for load {load.name}")
                continue
            load.Sload = np.zeros([load_day.shape[0],3], dtype=complex)  # initialize Sload with zeros
            load.Sload[:,col] = load_day_col/pkl_model.Sbase_1ph         # overwrite the Sload value with jacobian stuff (and convert to PU)
        else:
            print(f"Warning: No meter found for load {load.name}")
# gen day
for gen in pkl_model.Generators:
    if hasattr(gen, 'Sgen'):                                   # make sure it has Sload attribute
        if gen.name in gen_name_to_meter:                       # if the gen is in the meter map
            gen_meter = gen_name_to_meter[gen.name]             # get the meter number
            if str(gen_meter) in gen_meter_col_dict:               # check if the meter is in the map - there's one node that isn't
                gen_col_idx = gen_meter_col_dict[str(gen_meter)]     # get the column index from the map
                gen_day_col = gen_day[:, gen_col_idx]                # get the column from the load_day matrix
            else:
                print(f"Warning: Meter {gen_meter} not found in gen_meter_col_dict for gen {gen.name}")
                continue
            gen.Sgen = np.zeros([gen_day.shape[0],3], dtype=complex)  # initialize Sload with zeros
            gen.Sgen[:,col] = gen_day_col/pkl_model.Sbase_1ph         # overwrite the Sload value with jacobian stuff (and convert to PU)
        else:
            print(f"Warning: No meter found for load {gen.name}")
            
# COMPLEX VALUES!! DAKOTA PROBABLY ALREADY HANDLED THIS BUT NEED TO FIGURE OUT



# check dimensions of things match
if False:  # just to make sure it doesn't run
    print(f"num loads: {nloads}, num gens: {ngens}")
    print(f"load_day shape: {load_day.shape}, gen_day shape: {gen_day.shape}")
    print(f"load_night shape: {load_night.shape}, gen_night shape: {gen_night.shape}")
    print(f"Num Names in loads: {len(load_names)}, Num Names in gens: {len(gen_names)}")
    print(f"Num Meters in loads: {len(load_meters)}, Num Meters in gens: {len(gen_meters)}")


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

#### Old code: 
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

# # check if the loads are in the null meters list
# fp = f"Feeder_Data/{substation_name}/null_meters.csv"
# null_loads_df = pd.read_csv(fp)
# null_load_ids = set(null_loads_df['Service Number'])  # use set for faster lookup

# meter = load_meters[7]  # just pick the first one for now
# col_idx = load_meter_col_dict[str(meter)]
# print(meter)
# print(col_idx)
# print(load_day[:9,col_idx])

# sys.exit("testing")

# # modify gens to remove the meters where loads also exist - where a load and gen share a meter, the gen and load will be summed together and called 
# # a load, and the gen should be ignored
# common_meters = set(gen_meters) & set(load_meters)  # find common elements
# for meter in common_meters:
#     if meter in gen_meters:
#         gen_meters.remove(meter)  # remove the meter from the gens list
#     else:
#         print(f"couldn't find {meter} in gen_meters list, but it was in common_meters list")
#     name = gen_meter_to_name[meter]  # get the name of the gen
#     if name in gen_name_to_meter:
#         del gen_name_to_meter[name]
#     if name in gen_names:
#         gen_names.remove(name)
#     else:
#         print(f"couldn't find {name} in gen_names list, but it was in gen_name_to_meter map")

# for meter in common_meters:
#     load_col = load_meter_col_dict[str(meter)]
#     gen_col = gen_meter_col_dict[str(meter)]
#     # sum the gen and load values together
#     loadvals[:, load_col] -= genvals[:, gen_col]  # subtract the gen values from the load values
# # remove the gen values from the genvals array
# common_meters_cols = [gen_meter_col_dict[str(i)] for i in common_meters if str(i) in gen_meter_col_dict]  # get the column indices of the common meters
# genvals = np.delete(genvals, common_meters_cols, axis=1)  # remove the common meters from the genvals array