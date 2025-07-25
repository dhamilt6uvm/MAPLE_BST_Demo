## Script to create a pkl file that has loading conditions that will be used for finding the jacobian of power flow
###########################################################################################


# Things to do:
# finish this code obvi
# get fresh data from the server
# get updated solve_3ph_pf.jl code

## Import packages #######################################################################
import pandas as pd
import numpy as np


## Process the .csv data into averages ###################################################
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
    data_p_with = np.vstack((data_p, data_unmod))       # with unmodified rows
    data_m_with = np.vstack((data_m, data_unmod))
    return np.vstack((data_p_with, data_m_with))

# create the big matrices
load_day = combine_mats(load_day_avg, nloads, eps, ngens)       # samples x load-nodes where samples is [ + eps, 0, -eps, 0] (0's for gens)
gen_day = combine_mats(gen_day_avg, ngens, eps, nloads)         # samples x gen-nodes where samples is [ 0, + eps, 0, -eps] (0's for loads)
load_night = combine_mats(load_night_avg, nloads, eps, ngens)   # samples x load-nodes where samples is [ + eps, 0, -eps, 0] (0's for gens)
gen_night = combine_mats(gen_night_avg, ngens, eps, nloads)     # samples x gen-nodes where samples is [ 0, + eps, 0, -eps] (0's for loads)


## Load the pkl file ###############################################################

## Overwrite the values in pkl file ################################################

## save the new pkl file ###########################################################