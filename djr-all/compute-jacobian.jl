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
include("BST_func.jl")         # BST function: value.(Vph) = solve_pf(psm::PyObject, V0_ref::Vector{ComplexF64}, t_ind::Int64, linear_solver::String)
# Import Python modules
pickle = pyimport("pickle")
pyopen = pyimport("builtins").open
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")

# measure time taken
start_time = time()

save_new_jacobs = true         # change this to save results
############################################################################################

# phase?
ph_col = 2      # phase = B

# Load the .pkl files for day and night time
substation_name = "Burton_Hill_small02"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model_DAY1.pkl"
pkl_file = pyopen(fname, "rb")
psm_day = pickle.load(pkl_file)
pkl_file.close()
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model_NIGHT1.pkl"
pkl_file = pyopen(fname, "rb")
psm_night = pickle.load(pkl_file)
pkl_file.close()

# determine num of nodes and extract unique ones
function get_loadgen_nodes_LO(psm)
    # returns list of unique nodes that have an Sload or Sgen attribute - 1-indexed! 
    # returns length of that list
    # note: (L)OAD (O)RDER node numbers are in the order they appear in when looping "for load in psm.Loads", then again with gens
    nodes = Int[]               # unique nodes with a load or a gen
    # Extract nodes with load/gen/both
    for load in psm.Loads
        if hasattr(load, :Sload)
            ind = load.parent_node_ind
            if !(ind in nodes)
                push!(nodes, ind)
            end
        end
    end
    for gen in psm.Generators
        if hasattr(gen, :Sgen)
            ind = gen.parent_node_ind
            if !(ind in nodes)
                push!(nodes, ind)
            end
        end
    end
    n_loads = length(nodes)
    return nodes.+1, n_loads
end
nodes, n_loads = get_loadgen_nodes_LO(psm_day)

# Substation Voltage
V0_mag = 1
V0_ref = V0_mag*[1,exp(-im*2*pi/3),exp(im*2*pi/3)]

# find how many cases to solve
t_start = 1
t_end = size(psm_day.Loads[1].Sload,1)  # assuming all loads have the same number of time steps

# solve power flow for day and night
n_times = t_end-t_start+1
linear_solver = "mumps"
Vm_out_day = Array{Float64}(undef, n_loads, n_times)
Vm_out_night = Array{Float64}(undef, n_loads, n_times)
for t_ind in t_start:t_end
    Vtmp = solve_pf(psm_day, V0_ref, t_ind, linear_solver)      # day time avg
    Vm_out_day[:,t_ind] = abs.(Vtmp[ph_col,nodes])
    Vtmp = solve_pf(psm_night, V0_ref, t_ind, linear_solver)    # night time avg
    Vm_out_night[:,t_ind] = abs.(Vtmp[ph_col,nodes])
    if t_ind%50 == 0
        println("Finished $(t_ind) ops")
    end
end
println("Finished ops")

# figure out epsilon
epsilon = abs(psm_day.Loads[1].Sload[1,2] - psm_day.Loads[1].Sload[2,2])

# break up voltages into appropriate blocks to eventually get dVdP and dVdQ for both day and night
function separate_jacobians(Vm, nloads, epsilon)
    # compute indices of each matrix
    idx = [(i*nloads + 1):(i+1)*nloads for i in 0:3]
    # compute centered numerical derivative 
    dVdP = ( Vm[:,idx[1]] - Vm[:,idx[2]] ) / (2*epsilon)
    dVdQ = ( Vm[:,idx[3]] - Vm[:,idx[4]] ) / (2*epsilon)
    return dVdP, dVdQ
end

# use the function to get jacobians
dVdP_day, dVdQ_day = separate_jacobians(Vm_out_day, n_loads, epsilon)
dVdP_night, dVdQ_night = separate_jacobians(Vm_out_night, n_loads, epsilon)

# measure time taken to run:
end_time = time()
elap = round(end_time - start_time, digits=2)
println("Elapsed time: $(elap) seconds")        # expect around 430 seconds for ~2000 cases

# to save variables after completion, run this in REPL terminal:
"""
using Serialization
serialize("BH_small02_Jacobians00.jls", (dVdP_day, dVdQ_day, dVdP_night, dVdQ_night))
"""
# to load in next script
"""
using Serialization
dVdP_day, dVdQ_day, dVdP_night, dVdQ_night = deserialize("BH_small02_Jacobians00.jls")
"""

# Convert Jacobians to DataFrame and save them
function save_data(A, name)
    df = DataFrame(A, :auto)
    CSV.write("$name.csv", df)
end
if save_new_jacobs
    save_data(dVdP_day, "dVdP_day01_BHsmall02")
    save_data(dVdQ_day, "dVdQ_day01_BHsmall02")
    save_data(dVdP_night, "dVdP_night01_BHsmall02")
    save_data(dVdQ_night, "dVdQ_night01_BHsmall02")
end