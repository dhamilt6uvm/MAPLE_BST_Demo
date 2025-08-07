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

include("BST_func.jl")         # BST function: value.(Vph) = solve_pf(psm::PyObject, V0_ref::Vector{ComplexF64}, t_ind::Int64, linear_solver::String)

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
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model.pkl"
pkl_file = pyopen(fname, "rb")
psm = pickle.load(pkl_file)
pkl_file.close()


## Solve power-flow with BST ##############################################################
nnodes = length(psm.Nodes)
ntest = size(psm.Loads[1].Sload,1)
V_nodes = zeros(Float64, ntest, nnodes)
Vdiffnorm = zeros(Float64, ntest)
V_goal = ones(nnodes)
for ii in 1:ntest
    Vtmp = solve_pf(psm, V0_ref, ii, linear_solver)
    V_nodes[ii,:] = abs.(Vtmp[ph_col,:])
    Vdiffnorm[ii] = norm(V_goal - V_nodes[ii,:])
    if ii%100 == 0
        println("Finished $(ii) solutions")
    end
end


## Assess norms etc. ######################################################################
idx_sort = sortperm(Vdiffnorm, rev=true)        # indices of Vdiffnorm values sorted high to low
println("10 samples with most deviation:")
println(idx_sort[1:10])


## Save the found data ###################################################################
save_data(V_nodes, "V_allAMI_BHsmall02")            # voltage at all nodes at all AMI loading conditions
save_data(Vdiffnorm, "Vdiffnorm_BHsmall02")         # norm of 1 - voltage at all ... ^   ^ 
save_data(idx_sort, "Vdiff_idx_BHsmall02")          # [t_ind of highest voltage norm ... ... t_ind of lowest]