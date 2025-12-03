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


## Starting from scratch
## Load the .pkl file for all data ########################################################
substation_name = "Burton_Hill_ESE"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model.pkl"
pkl_file = pyopen(fname, "rb")
psm = pickle.load(pkl_file)
pkl_file.close()
# # mapping loads to nodes:
# # plot:2 = psm.Loads[1] = span_69719_n = psm.Nodes[22]
# # plot:4 = psm.Loads[2] = span_69717_n = psm.Nodes[20]
# # plot:3 = psm.Loads[3] = span_69718_n = psm.Nodes[17] = psm.Generators[1]
# # plot:1 = psm.Loads[4] = span_69720_n = psm.Nodes[5]  = psm.Generators[2]


## Overwrite load data using imported data
base_load_kw = CSV.read("djr-all/ESE-Project/Load-Data/load_gen_minute.csv", DataFrame)    # n x 6 (Load1, Load2, Load3, Load4, Gen1, Gen2)
base_load = base_load_kw./1e3      # convert to MW
# load the WH power data
wh_load_w = CSV.read("djr-all/ESE-Project/Load-Data/u_all_minute.csv", DataFrame)   # n x 5 (Time_seconds, Base, HH, HLH, SC)
wh_load = wh_load_w[:,2:end]./1e6      # skip time column & convert to MW
# compute the modified power data (final = base - diff of base and case) (or maybe plus?)
case_no = 4         # 1- base, 2-hh, 3-hlh, 4-sc
wh_diff = wh_load[:,case_no] - wh_load[:,1]   # case_no=1 is base
# update non-solar nodes power data in the psm to minute scale
load_zeros = zeros(ComplexF64,(1440,1))
base_load1 = base_load[:,1] + base_load[:,1].*0.2im
base_load2 = base_load[:,2] + base_load[:,2].*0.2im
# base_load3 = base_load[:,3] + base_load[:,3].*0.2im
# base_load4 = base_load[:,4] + base_load[:,4].*0.2im
psm.Loads[1].Sload = hcat(load_zeros, base_load1, load_zeros)
psm.Loads[2].Sload = hcat(load_zeros, base_load2, load_zeros)
# psm.Loads[3].Sload = hcat(load_zeros, base_load3, load_zeros)
# psm.Loads[4].Sload = hcat(load_zeros, base_load4, load_zeros)
# update specific loads to modified values
new_load3 = (base_load[:,3] + wh_diff) + (base_load[:,3] + wh_diff).*0.2im
new_load4 = (base_load[:,4] + wh_diff) + (base_load[:,4] + wh_diff).*0.2im
psm.Loads[3].Sload = hcat(load_zeros, new_load3, load_zeros)
psm.Loads[4].Sload = hcat(load_zeros, new_load4, load_zeros)
# update generation data to minute scale
base_gen1 = base_load[:,5] + base_load[:,5].*0.0im
base_gen2 = base_load[:,6] + base_load[:,6].*0.0im
psm.Generators[1].Sgen = hcat(load_zeros, base_gen1, load_zeros)
psm.Generators[2].Sgen = hcat(load_zeros, base_gen2, load_zeros)


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
save_data(V_nodes, "V_all_ESE_sc")            # voltage at all nodes at all AMI loading conditions
load_new = hcat(base_load[:,1], base_load[:,2], real(new_load3), real(new_load4))
save_data(load_new, "load_WH_modified_ESE_sc")     # modified WH load data