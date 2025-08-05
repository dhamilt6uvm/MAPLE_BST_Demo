## Compare performance of linearized power flow to actual BST solutions
##############################################################################
## Import packages etc. 
using PyCall
using SparseArrays
using Plots
gr()
using ColorTypes
using Colors
using JuMP
using Ipopt
# import HSL_jll
using LinearAlgebra
using CSV
using DataFrames
using Random
using Serialization
using LinearAlgebra
hasattr = pyimport("builtins").hasattr
@pyimport matplotlib.pyplot as plt

# Import Python modules
pickle = pyimport("pickle")
pyopen = pyimport("builtins").open
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")


substation_name = "Burton_Hill_small02"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model_DAY.pkl"
pkl_file = pyopen(fname, "rb")
psm_day = pickle.load(pkl_file)
pkl_file.close()

# get network info
n_nodes = length(psm.Nodes)
n_branches = length(psm.Branches)
n_loads = length(psm.Loads)
n_gens = length(psm.Generators)

# print attributes of a node
node = psm.Generators[5]
# len_elems = length(py"dir"(node))
# for ii in 1:len_elems
#     println(py"dir"(node)[ii])
# end

for gen in psm.Generators
    println(gen.parent_node_ind)
end



# # Substation Voltage
# V0_mag = 1
# V0_ref = V0_mag*[1,exp(-im*2*pi/3),exp(im*2*pi/3)]

# ## extract loads so that I can play with them
# # get indices to pull out correct loads/gens
# n_hours = length(psm.Loads[1].Sload)        # total number of n_hours
# n_days = div(n_hours, 24)                   # integer days 
# start_hour = 0                              # start of time (hour of the day)
# end_hour = 10                               # end of time (hour of the day)

# data_night = []                             # init data



# for ii in 1:n_loads
#     # get the load data for each load

#     for day in 0:n_days
#         start_idx = day * 24 + start_hour
#         end_idx = start_idx + end_hour
#         push!(daily_chunks, psm.Loads[ii].Sload[start_idx:end_idx])
#     end

#     # flatten the daily chunks into a single vector
#     data_night = vcat(data_night, daily_chunks...)
# end
# for day in 0:n_days
#     start_idx = day * 24 + start_hour
#     end_idx = start_idx + end_hour
#     push!(data_night, psm.Loads[ii].Sload[start_idx:end_idx])
# end

# result = vcat(daily_chunks...)  # flatten into a single vector


# avg_loads = zeros(n_loads, 3)



## Look at parsing tools .py
# find one that loads, modifies, saves .pkl file

# t_ind is specific time-stamp of ami data to use

# preprocess of ami loads
# load .pkl as psmf
# for load in psm.loads
# load.sload = 2x3 mat of day and night averages
# dump (save) the new pkl file

