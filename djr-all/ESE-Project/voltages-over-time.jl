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


## Solve power-flow with BST ##############################################################plot_
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


## Save the voltage data ###################################################################
save_data(V_nodes, "V_allAMI_BH_ESE")            # voltage at all nodes at all AMI loading conditions



## Questions for Katy
# does it make sense to add the power the way that I am doing it?
# 

#### 
# what am I trying to do: 
# modfiy the load data and resimulate

# get the values into a table maybe

# need to make interpolated load profile data to simulate voltage with altered loads. May want to filter the load. 