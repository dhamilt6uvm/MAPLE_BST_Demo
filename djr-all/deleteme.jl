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
hasattr = pyimport("builtins").hasattr
include("BST_func_ssP.jl")         # BST function: value.(Vph) = solve_pf(psm::PyObject, V0_ref::Vector{ComplexF64}, t_ind::Int64, linear_solver::String)
# Import Python modules
pickle = pyimport("pickle")
pyopen = pyimport("builtins").open
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")


t_ind = 8562 # 1 indexed
ph_col = 2


## Load the feeder ######################################################
substation_name = "Burton_Hill_small02"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model.pkl"
pkl_file = pyopen(fname, "rb")
psm = pickle.load(pkl_file)
pkl_file.close()


V0_mag = 1                          # substation voltage
V0_ref = V0_mag*[1,exp(-im*2*pi/3),exp(im*2*pi/3)]
linear_solver = "mumps"


V_tmp, ssP = solve_pf_ssP(psm, V0_ref, t_ind, linear_solver)
V_final = abs.(V_tmp[ph_col,:])

ssP_MW = ssP * psm.Sbase_1ph / 1e6

CSV.write("vector.csv", DataFrame(value = V_final))