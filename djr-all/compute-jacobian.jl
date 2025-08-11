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

save_new_jacobs = false         # change this to save results
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

# determine num of nodes
nodes = Int[]               # unique nodes with a load or a gen
# Extract nodes with load/gen/both
for load in psm_day.Loads
    if hasattr(load, :Sload)
        ind = load.parent_node_ind
        if !(ind in nodes)
            push!(nodes, ind)
        end
    end
end
for gen in psm_day.Generators
    if hasattr(gen, :Sgen)
        ind = gen.parent_node_ind
        if !(ind in nodes)
            push!(nodes, ind)
        end
    end
end
nloads = length(nodes)

# Substation Voltage
V0_mag = 1
V0_ref = V0_mag*[1,exp(-im*2*pi/3),exp(im*2*pi/3)]

# find how many cases to solve
t_start = 1
t_end = size(psm_day.Loads[1].Sload,1)  # assuming all loads have the same number of time steps
# nloads = 210
# ngens = 30

return

# solve power flow for day and night
n_times = t_end-t_start+1
linear_solver = "mumps"
Vph_out_day = Array{Float64}(undef, n_nodes, n_times)
for t_ind in t_start:t_end
    Vtmp = solve_pf(psm_day, V0_ref, t_ind, linear_solver)
    Vph_out_day[:,t_ind] = abs.(Vtmp[ph_col,:])
    if t_ind%50 == 0
        println("Finished $(t_ind) ops in Day")
    end
end
println("Finished day-time ops")
Vph_out_night = Array{Float64}(undef, n_nodes, n_times)
for t_ind in t_start:t_end
    Vtmp = solve_pf(psm_night, V0_ref, t_ind, linear_solver)
    Vph_out_night[:,t_ind] = abs.(Vtmp[ph_col,:])
    if t_ind%50 == 0
        println("Finished $(t_ind) ops in Night")
    end
end
println("Finished night-time ops")

# figure out epsilon
epsilon = abs(psm_day.Loads[1].Sload[1,2] - psm_day.Loads[1].Sload[2,2])

# break up voltages into appropriate blocks to eventually get dVdP and dVdQ for both day and night (probably make functions)
function separate_jacobians(Vph, nloads, ngens, epsilon)
    # compute indices of each matrix
    idx_l = [(i*nloads + 1):(i+1)*nloads for i in 0:3] 
    idx_g = [(4*nloads + i*ngens + 1):(4*nloads + (i+1)*ngens) for i in 0:3]
    # compute the centered numerical derivative
    dV_loads_dP = ( Vph[:,idx_l[1]] - Vph[:,idx_l[2]] ) / (2*epsilon)
    dV_loads_dQ = ( Vph[:,idx_l[3]] - Vph[:,idx_l[4]] ) / (2*epsilon)
    dV_gens_dP = ( Vph[:,idx_g[1]] - Vph[:,idx_g[2]] ) / (2*epsilon)
    dV_gens_dQ = ( Vph[:,idx_g[3]] - Vph[:,idx_g[4]] ) / (2*epsilon)
    # concat into dP and dQ
    dVdP = hcat(dV_loads_dP, dV_gens_dP)
    dVdQ = hcat(dV_loads_dQ, dV_gens_dQ)

    return dVdP, dVdQ
end

# use the function to get jacobians
dVdP_day, dVdQ_day = separate_jacobians(Vph_out_day, nloads, ngens, epsilon)
dVdP_night, dVdQ_night = separate_jacobians(Vph_out_night, nloads, ngens, epsilon)

# measure time taken to run:
end_time = time()
elap = round(end_time - start_time, digits=2)
println("Elapsed time: $(elap) seconds")        # expect around 430 seconds

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
    save_data(dVdP_day, "dVdP_day_BHsmall02")
    save_data(dVdQ_day, "dVdQ_day_BHsmall02")
    save_data(dVdP_night, "dVdP_night_BHsmall02")
    save_data(dVdQ_night, "dVdQ_night_BHsmall02")
end