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

# include("BST_func.jl")         # BST function: value.(Vph) = solve_pf(psm::PyObject, V0_ref::Vector{ComplexF64}, t_ind::Int64, linear_solver::String)
include(joinpath(@__DIR__, "..", "BST_func.jl"))

function save_data(A, name)
    df = DataFrame(A, :auto)
    CSV.write("$name.csv", df)
end


# Import Python modules
pickle = pyimport("pickle")
pyopen = pyimport("builtins").open
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")


## Set up for BST function use: 
V0_mag = 1                          # substation voltage
V0_ref = V0_mag*[1,exp(-im*2*pi/3),exp(im*2*pi/3)]
linear_solver = "mumps"
ph_col = 2          # phase B


## Load the .pkl file for all data ########################################################
substation_name = "Burton_Hill_ESE"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model.pkl"
pkl_file = pyopen(fname, "rb")
psm = pickle.load(pkl_file)
pkl_file.close()
# mapping loads to nodes:
# psm.Loads[1] = span_69719_n = psm.Nodes[22]
# psm.Loads[2] = span_69717_n = psm.Nodes[20]
# psm.Loads[3] = span_69718_n = psm.Nodes[17] = psm.Generators[1]
# psm.Loads[4] = span_69720_n = psm.Nodes[5]  = psm.Generators[2]


## Overwrite load data using loaded data
case_no = 1         # 1- base, 2-hh, 3-hlh, 4-sc
# load the power data at minute scale: cols: load @ node 1, 2, 3, 4, gen @ node 1, 2
base_load = CSV.read("djr-all/ESE-Project/Load-Data/load_gen_minute.csv", DataFrame)
base_load = base_load./1000      # convert to MW
# load the WH power data
wh_load = CSV.read("djr-all/ESE-Project/Load-Data/u_all_minute.csv", DataFrame)
wh_load_all = Matrix(wh_load[:,2:end])./1e6      # skip time column & convert to MW
# compute the modified power data (final = base - diff of base and case) (or maybe plus?)
wh_diff = wh_load_all[:,case_no] - wh_load_all[:,1]   # case_no=0 is base
load3_new = base_load.Load3 + wh_diff
load4_new = base_load.Load4 + wh_diff
# update all the power data in the psm to minute scale
for ii = 1:4
    new_load = zeros(ComplexF64,(1440,3))
    new_load[:,2] = base_load[:,ii] + base_load[:,ii].*0.2im
    psm.Loads[ii].Sload = new_load
end
# update specific loads to modified values
psm.Loads[3].Sload[:,2] = load3_new + load3_new.*0.2im
psm.Loads[4].Sload[:,2] = load4_new + load4_new.*0.2im



## Solve power-flow with BST ##############################################################
nnodes = length(psm.Nodes)
ntest = size(psm.Loads[1].Sload,1)
V_nodes = zeros(Float64, ntest, nnodes)
for ii in 1:ntest
    Vtmp = solve_pf(psm, V0_ref, ii, linear_solver)
    V_nodes[ii,:] = abs.(Vtmp[ph_col,:])
    if ii%100 == 0
        println("Finished $(ii) solutions")
    end
end


## plot it in python


## Save the voltage data ###################################################################
save_data(V_nodes, "V_all_ESE_base")            # voltage at all nodes at all AMI loading conditions
load_new = hcat(load3_new, load4_new)
save_data(load_new, "load_WH_modified_ESE")     # modified WH load data
## NEXT NEXT
# put the other two load columns in here (but in order!!!)
# develop the python to plot the loads 
# make sure everything checks out with load and voltage
# try NOT case 1, repeat plots, shit may go haywire, in which case may need maaaas filtering on load