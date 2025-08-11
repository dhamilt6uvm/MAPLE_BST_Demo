## Script to create a pkl file that has loading conditions that will be used for finding the jacobian of power flow
###########################################################################################

# to create new system: 
# n = 1072 (nodes)
# j = 209 (nodes with load, gen, or both)
# need jxj dV/dP and jxj dV/dQ
# isolate the 209 nodes with loads
# find the average NET loads at those important nodes
# make day and night pkl's with Sload 209 (x4 +- and P/Q) long and Sgen unchanged (provided there are no nodes with just gen)
# use this compute jacobians script to get the jacobians out

## Import packages etc. ##################################################################
import pandas as pd
import numpy as np
import pickle
import sys
import os
# os.system('clear')
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Get the directory one level up from the script
sys.path.insert(0, parent_dir)  # Add the parent directory to the system path   
import GLM_Tools.PowerSystemModel as psm # Now import the package


substation_name = "Burton_Hill_small02" # change this to the substation you want to use
phase = 'B'                 # change this to the phase you want to use - A, B, or C (need a single phase for now)
check_out = False           # set to True to check the output of the jacobian pkl files via printing statements at the end

## Import pkl file  ########################################################################
# load the file in
pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
with open(pkl_file, 'rb') as file:
    pkl_model = pickle.load(file)               # this references GLM_Tools somehow, so it won't run unless GLM_Tools is in the same folder
                                                # not sure where the reference is so I can't fix it... 


## Extract Gen and Load Data Averages ######################################################
# init storage - even just gen nodes called loads
loads_avg_day = []          # average of load values for each load day time
loads_avg_night = []        # average of load values for each load night time
nodes = []                  # unique nodes with a load or a gen
idx_day = []                # initialize empty lists for day and night time-indices
idx_night = []
# find first load and pull shape - get size of data
for load in pkl_model.Loads:
    if hasattr(load, 'Sload'):  # make sure it has Sload attribute
        num_samples = load.Sload.shape[0]  # get number of samples
        break
# Extract nodes with load/gen/both (check for all gens also having load?)
for load in pkl_model.Loads:
    if hasattr(load, 'Sload'):
        ind = load.parent_node_ind          # pull out parent node
        if ind not in nodes:                # make sure node isn't already accounted for
            nodes.append(ind)               # add it to the list
for gen in pkl_model.Generators:            # check generator nodes too
    if hasattr(gen, 'Sgen'):
        ind = gen.parent_node_ind
        if ind not in nodes:
            nodes.append(ind)
nloads = len(nodes)                         # get number of nodes with loads/gens
# determine time indices to pull for day/night
day_hrs = np.arange(11,23)                      # actual hours to use - adjust if needed, based on VisualizeLoadsGens.py
night_hrs = np.arange(0,11)
idx_samples = np.arange(num_samples)            # create an array of sample indices
hr_of_day = idx_samples % 24                    # determine hour of day for each sample
for ii in range(num_samples):  # loop through all samples
    if hr_of_day[ii] in day_hrs:
        idx_day.append(ii)
    elif hr_of_day[ii] in night_hrs:
        idx_night.append(ii)    
# get average load values (combining common nodes)
node_idx_map = {}                           # maps parent_node_ind to index in loads_avg_day
for load in pkl_model.Loads:
    if hasattr(load, 'Sload'):
        ind = load.parent_node_ind          # pull out index
        avg_day = np.mean(load.Sload[idx_day,:], axis=0)                    # compute average of Sload for day
        avg_night = np.mean(load.Sload[idx_night,:], axis=0)                # compute average of Sload for night
        if ind in node_idx_map:                                             # check if node already found
            loads_avg_day[node_idx_map[ind]] += avg_day                     # add average value to existing average
            loads_avg_night[node_idx_map[ind]] += avg_night                 # " "
        else:                                                               # new node
            loads_avg_day.append(avg_day)                                   # put average value in list of avgs
            loads_avg_night.append(avg_night)                               # " "
            node_idx_map[ind] = len(loads_avg_day) - 1                      # add mapping
# subtract gens at each to get net load 
for gen in pkl_model.Generators:
    if hasattr(gen, 'Sgen'):
        ind = gen.parent_node_ind
        avg_day = np.mean(gen.Sgen[idx_day,:], axis=0)
        avg_night = np.mean(gen.Sgen[idx_night,:], axis=0)
        if ind in node_idx_map:                                             # check if node already found
            loads_avg_day[node_idx_map[ind]] -= avg_day                     # subtract average value from existing average
            loads_avg_night[node_idx_map[ind]] -= avg_night                 # " "
        else:                                                               # new node
            loads_avg_day.append(-1*avg_day)                                # put average value in list of avgs
            loads_avg_night.append(-1*avg_night)                            # " "
            node_idx_map[ind] = len(loads_avg_day) - 1                      # add mapping
# convert to np arrays and flip sign (gen positive, load negative)
loads_avg_day = np.array(loads_avg_day)                     # convert to numpy array
loads_avg_night = np.array(loads_avg_night)                 # convert to numpy array


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
    # input: data - nx1 complex array of data to perturb
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
def combine_mats(data, eps):
    # input: data - nx1 complex array of data to perturb
    #        eps - perturbation value
    #        n_other - number of rows in gen if data is loads, or loads if data is gens
    #        pert_first - if True, perturbed data comes first, if False, unperturbed data comes first
    # output: 4n x n array of perturbed and unperturbed data (flip perturbed and unperturbed if pert_first is False)
    #        [ data + R eps;   n_data rows 
    #          data - R eps;   n_data rows    
    #          data + C eps;   n_data rows 
    #          data - C eps;   n_data rows ]
    plus_R, minus_R = perturb_data(data, eps)               # perturb the data with real epsilon
    plus_C, minus_C = perturb_data(data, 1j*eps)            # perturb the data with complex epsilon
    out = np.vstack((plus_R, minus_R, plus_C, minus_C))     # combine the perturbed and unperturbed data
    return out

# create the perturbed data matrices - reference data as columns for each node
eps = 1e-6        # perturbation amount
load_day = combine_mats(loads_avg_day[:,ph_col], eps)           # 4*nloads x nloads array
load_night = combine_mats(loads_avg_night[:,ph_col], eps)       # 4*nloads x nloads array


## Mod and Save the new pkl file ###########################################################
def save_jacobian_pkl(pkl_model, new_file_name, load_data, ph_col):
    ct = 0                      # counter to move through nodes
    node_idx_map = {}    
    # loop over loads
    for load in pkl_model.Loads:
        if hasattr(load, 'Sload'):
            ind = load.parent_node_ind
            load.Sload = np.zeros([load_data.shape[0],3], dtype=complex)    # initialize Sload with zeros
            if ind not in node_idx_map:
                load.Sload[:,ph_col] = load_data[:,ct]                      # overwrite the Sload value with jacobian stuff
                node_idx_map[ind] = ct                                      # store the mapping to not repeat
                ct += 1                                                     # index the count
    # loop over gens
    for gen in pkl_model.Generators:
        if hasattr(gen, 'Sgen'):
            ind = gen.parent_node_ind
            gen.Sgen = np.zeros([load_data.shape[0],3], dtype=complex)      # initialize Sgen with zeros
            if ind not in node_idx_map:
                gen.Sgen[:,ph_col] = -load_data[:,ct]                       # overwrite the Sgen value with jacobian stuff (minus because gen vals should be positive when gen-ing)
                node_idx_map[ind] = ct                                      # store the mapping to not repeat
                ct += 1                                                     # index the count
    # save the modified model
    with open(new_file_name, 'wb') as file:
        pickle.dump(pkl_model, file) 
    print(f"Saved Jacobian pkl file to {new_file_name}")

# Day time model
pkl_file_day = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model_DAY1.pkl"
save_jacobian_pkl(pkl_model, pkl_file_day, load_day, ph_col)
# Night time model
pkl_file_night = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model_NIGHT1.pkl"
save_jacobian_pkl(pkl_model, pkl_file_night, load_night, ph_col)


## Make sure that it worked ##############################################################
if check_out:
    n_checks = 3
    # load the files back in and check the data
    with open(pkl_file_day, 'rb') as file:
        pkl_model_day = pickle.load(file)
    with open(pkl_file_night, 'rb') as file:
        pkl_model_night = pickle.load(file) 

    print("Checking the Jacobian pkl files...")
    print("Day time model:")
    print("===Expect real component + eps on the diagonal, shape is nnodes*4 x 3")
    ii = 0
    for load in pkl_model_day.Loads:
        if hasattr(load, 'Sload'):
            if ii == 0:
                print(load.Sload.shape) 
            print(f"Load {load.name}, #{ii}, Sload:") 
            print(load.Sload[:n_checks,ph_col])
            ii += 1
            if ii >= n_checks: break
    print("===Expect real component - eps on the diagonal")
    ii = 0
    for load in pkl_model_day.Loads:
        if hasattr(load, 'Sload'):
            print(f"Load {load.name}, #{ii}, Sload:")
            print(load.Sload[nloads:nloads+n_checks,ph_col])
            ii += 1
            if ii >= n_checks: break
    print("===Expect complex component + eps on the diagonal")
    ii = 0
    for load in pkl_model_day.Loads:
        if hasattr(load, 'Sload'):
            print(f"Load {load.name}, #{ii}, Sload:") 
            print(load.Sload[2*nloads:2*nloads+n_checks,ph_col])
            ii += 1
            if ii >= n_checks: break
    print("===Expect complex component - eps on the diagonal")
    ii = 0
    for load in pkl_model_day.Loads:
        if hasattr(load, 'Sload'):
            print(f"Load {load.name}, #{ii}, Sload:") 
            print(load.Sload[3*nloads:3*nloads+n_checks,ph_col])
            ii += 1
            if ii >= n_checks: break
    print("===Expect 0's unless there are nodes with only gens, size should be 4*nnodes x 3")
    ii = 0
    for gen in pkl_model_day.Generators:
        if hasattr(gen, 'Sgen'):
            if ii == 0:
                print(gen.Sgen.shape)
            print(f"Gen {gen.name}, #{ii}, Sgen:")
            print(gen.Sgen[:n_checks,ph_col])
            ii += 1
            if ii > n_checks: break
    # add more checks as needed