## Script to create a pkl file that has loading conditions that will be used for finding the jacobian of power flow
###########################################################################################

## Import packages etc. ##################################################################
import pandas as pd
import numpy as np
import pickle
import re

substation_name = "Burton_Hill_small02" # change this to the substation you want to use
phase = 'B'  # change this to the phase you want to use - A, B, or C (need a single phase for now)
check_out = False  # set to True to check the output of the jacobian pkl files via printing statements at the end

## Import pkl file  ########################################################################
# load the file in
pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
with open(pkl_file, 'rb') as file:
    pkl_model = pickle.load(file)               # this references GLM_Tools somehow, so it won't run unless GLM_Tools is in the same folder
                                                # not sure where the reference is so I can't fix it... 


## Extract Gen and Load Data Averages ######################################################
# init storage
loads_avg_day = []          # average of load values for each load day time
loads_avg_night = []        # average of load values for each load night time
gens_avg_day = []           # average of gen values for each gen day time
gens_avg_night = []         # average of gen values for each gen night time
# find first load and pull shape - get size of data
for load in pkl_model.Loads:
    if hasattr(load, 'Sload'):  # make sure it has Sload attribute
        num_samples = load.Sload.shape[0]  # get number of samples (should be 96)
        break
# determine time indices to pull for day/night
idx_day = []                        # initialize empty lists for day and night indices
idx_night = []
day_hrs = np.arange(11,24)                      # actual hours to use - adjust if needed, based on VisualizeLoadsGens.py
night_hrs = np.arange(0,11)
idx_samples = np.arange(num_samples)            # create an array of sample indices
hr_of_day = idx_samples % 24                    # determine hour of day for each sample
for ii in range(num_samples):  # loop through all samples
    if hr_of_day[ii] in day_hrs:
        idx_day.append(ii)
    elif hr_of_day[ii] in night_hrs:
        idx_night.append(ii)    
# loads
for load in pkl_model.Loads:
    if hasattr(load, 'Sload'):                  # make sure it has Sload attribute
        num_hrs = load.Sload.shape[0]           # get number of hours (samples) in Sload
        if num_hrs != num_samples:              # make sure it has same number of samples as the first load
            raise ValueError(f"Load {load.name} has {num_hrs} hours in Sload, but expected {num_samples}.")
        loads_avg_day.append(np.mean(load.Sload[idx_day,:], axis=0))       # average over the day data
        loads_avg_night.append(np.mean(load.Sload[idx_night,:], axis=0))   # average over the night data
# gens
for gen in pkl_model.Generators:
    if hasattr(gen, 'Sgen'):                    # make sure it has Sgen attribute
        num_hrs = gen.Sgen.shape[0]             # get number of hours (samples) in Sgen
        if num_hrs != num_samples:              # make sure it has same number of samples as the first load
            raise ValueError(f"Gen {gen.name} has {num_hrs} hours in Sgen, but expected {num_samples}.")
        gens_avg_day.append(np.mean(gen.Sgen[idx_day,:], axis=0))           # average over the day data
        gens_avg_night.append(np.mean(gen.Sgen[idx_night,:], axis=0))       # average over the night data

# find number of loads and gens
nloads = len(loads_avg_day)  # number of loads
ngens = len(gens_avg_day)    # number of generators
# conver to np arrays
loads_avg_day = np.array(loads_avg_day)  # convert to numpy array
loads_avg_night = np.array(loads_avg_night)  # convert to numpy array
gens_avg_day = np.array(gens_avg_day)  # convert to numpy array
gens_avg_night = np.array(gens_avg_night)  # convert to numpy array


## Perturb the data to create jacobian loading #############################################
# determine what column to use based on phase
if phase == 'A':
    ph_col = 0
elif phase == 'B':
    ph_col = 1
elif phase == 'C':  
    ph_col = 2

# function for perturbing
def perturb_data(data, eps):
    # input: data - nx1 array of data to perturb
    # output: nxn arrays of (data + eps) and (data - eps) with eps on diagonal
    n = data.shape[0]       # get number of data values
    plus = []               # init lists
    minus = []
    for ii in range(n):     # loop over all values
        perturb = np.zeros([n], dtype=complex) # perturb just one of them
        perturb[ii] = eps
        plus.append(data + perturb) # plus perturbation
        minus.append(data - perturb)    # minus pert...
    plus = np.array(plus)           # convert to np arrays
    minus = np.array(minus)
    return plus, minus
# function for combining perturbed data
def combine_mats(data, eps, n_other, pert_first):
    # input: data - nx1 array of data to perturb
    #        eps - perturbation value
    #        n_other - number of rows in gen if data is loads, or loads if data is gens
    #        pert_first - if True, perturbed data comes first, if False, unperturbed data comes first
    # output: 8n x n array of perturbed and unperturbed data (flip perturbed and unperturbed if pert_first is False)
    #        [ data + R eps;   n_data rows 
    #          data - R eps;   n_data rows    
    #          data + C eps;   n_data rows 
    #          data - C eps;   n_data rows
    #          data unpert;    n_other rows
    #          data unpert;    n_other rows
    #          data unpert;    n_other rows
    #          data unpert;    n_other rows]
    plus_R, minus_R = perturb_data(data, eps)  # perturb the data with real epsilon
    plus_C, minus_C = perturb_data(data, 1j*eps)  # perturb the data with complex epsilon
    unperturbed = np.tile(data, (4*n_other,1))  # unperturbed data
    if pert_first:  # if perturbed data comes first
        out = np.vstack((plus_R, minus_R, plus_C, minus_C, unperturbed))  # combine the perturbed and unperturbed data
    else:  # if unperturbed data comes first
        out = np.vstack((unperturbed, plus_R, minus_R, plus_C, minus_C))
    return out

# create the perturbed data matrices - reference data as columns for each node
eps = 1e-6        # perturbation amount
load_day = combine_mats(loads_avg_day[:,ph_col], eps, ngens, pert_first=True)   # (4*nloads + 4*ngens) x nloads array
load_night = combine_mats(loads_avg_night[:,ph_col], eps, ngens, pert_first=True)  # (4*nloads + 4*ngens) x nloads array
gen_day = combine_mats(gens_avg_day[:,ph_col], eps, nloads, pert_first=False)     # (4*ngens + 4*nloads) x ngens array
gen_night = combine_mats(gens_avg_night[:,ph_col], eps, nloads, pert_first=False)  # (4*ngens + 4*nloads) x ngens array


## Mod and Save the new pkl file ###########################################################
idx = [0, 0, 0, 0]          # indices for load and gen columns

def save_jacobian_pkl(pkl_model, new_file_name, load_data, gen_data, ph_col):
    idx = [0, 0]  # reset indices for load and gen columns
    # loop over loads
    for load in pkl_model.Loads:
        if hasattr(load, 'Sload'):
            load.Sload = np.zeros([load_data.shape[0],3], dtype=complex) # initialize Sload with zeros
            load.Sload[:,ph_col] = load_data[:,idx[0]]                      # overwrite the Sload value with jacobian stuff
            idx[0] += 1
    # loop over gens
    for gen in pkl_model.Generators:
        if hasattr(gen, 'Sgen'):
            gen.Sgen = np.zeros([gen_data.shape[0],3], dtype=complex) # initialize Sload with zeros
            gen.Sgen[:,ph_col] = gen_data[:,idx[1]]                      # overwrite the Sload value with jacobian stuff
            idx[1] += 1
    # save the modified model
    with open(new_file_name, 'wb') as file:
        pickle.dump(pkl_model, file) 
    print(f"Saved Jacobian pkl file to {new_file_name}")

# Day time model
pkl_file_day = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model_DAY.pkl"
save_jacobian_pkl(pkl_model, pkl_file_day, load_day, gen_day, ph_col)
# Night time model
pkl_file_night = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model_NIGHT.pkl"
save_jacobian_pkl(pkl_model, pkl_file_night, load_night, gen_night, ph_col)


## Make sure that it worked ##############################################################
if check_out:
    n_checks = 2
    # load the files back in and check the data
    with open(pkl_file_day, 'rb') as file:
        pkl_model_day = pickle.load(file)
    with open(pkl_file_night, 'rb') as file:
        pkl_model_night = pickle.load(file) 

    print("Checking the Jacobian pkl files...")
    print("Day time model:")
    ii = 0
    for load in pkl_model_day.Loads:
        if hasattr(load, 'Sload'):
            print(f"Load {load.name}, #{ii}, Sload: {load.Sload[:n_checks,ph_col]}")
            ii += 1
            if ii > n_checks: break
    ii = 0
    for gen in pkl_model_day.Generators:
        if hasattr(gen, 'Sgen'):
            print(f"Gen {gen.name}, #{ii}, Sgen: {gen.Sgen[4*nloads:4*nloads+n_checks,ph_col]}")
            ii += 1
            if ii > n_checks: break
    print("Night time model:")
    ii = 0
    for load in pkl_model_night.Loads:  
        if hasattr(load, 'Sload'):
            print(f"Load {load.name}, #{ii}, Sload: {load.Sload[:n_checks,ph_col]}")
            ii += 1
            if ii > n_checks: break
    ii = 0
    for gen in pkl_model_night.Generators:
        if hasattr(gen, 'Sgen'):
            print(f"Gen {gen.name}, #{ii}, Sgen: {gen.Sgen[4*nloads:4*nloads+n_checks,ph_col]}")
            ii += 1
            if ii > n_checks: break
