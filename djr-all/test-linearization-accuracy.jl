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
include("BST_func.jl")         # BST function: value.(Vph) = solve_pf(psm::PyObject, V0_ref::Vector{ComplexF64}, t_ind::Int64, linear_solver::String)
hasattr = pyimport("builtins").hasattr
@pyimport matplotlib.pyplot as plt

# Import Python modules
pickle = pyimport("pickle")
pyopen = pyimport("builtins").open
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")

## Set up for function use: 
V0_mag = 1                          # substation voltage
V0_ref = V0_mag*[1,exp(-im*2*pi/3),exp(im*2*pi/3)]
linear_solver = "mumps"


## Load the .pkl files for day and night time #############################################
substation_name = "Burton_Hill_small02"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model_DAY1.pkl"
pkl_file = pyopen(fname, "rb")
psm_day = pickle.load(pkl_file)
pkl_file.close()
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model_NIGHT1.pkl"
pkl_file = pyopen(fname, "rb")
psm_night = pickle.load(pkl_file)
pkl_file.close()

ph_col = 2          # phase B


## Shift PSMs for day and night so that t_ind=1 is the average ############################
# determine epsilon (which was used to calculate jacobian numerically)
epsilon = abs(psm_day.Loads[1].Sload[1,ph_col] - psm_day.Loads[1].Sload[2,ph_col])
# subtract epsilon from first load in both
psm_day.Loads[1].Sload[1,ph_col] -= epsilon
psm_night.Loads[1].Sload[1,ph_col] -= epsilon


## Find the unique nodes with loads or gens or both #####################################
function get_loadgen_nodes(psm)
    # returns list of unique nodes that have an Sload or Sgen attribute - 1-indexed!
    # returns length of that list
    # note: node numbers are in numerical order, not necessarily in the order used by "for load in psm.Loads"
    nodes = Int[]
    for node in psm.Nodes                                       # loop through all nodes 
        idx = node.index + 1                                    # set node value (+1 b/c julia)
        found = false                                           # havent found an Sload yet
        if (length(node.loads) > 0 || length(node.gens) > 0) && idx ∉ nodes    # check if there are loads and if node already found 
            for load_ind in node.loads                          # loop through all attached loads
                load = psm.Loads[load_ind+1]                    # pull out appropriate load
                if hasattr(load, "Sload")                       # check if that load has an Sload
                    push!(nodes, idx)                           # add node index to the list
                    found = true
                    break
                end
            end
            if found                                            # did find, don't bother checking gens
                continue
            end        
            for gen_ind in node.gens                            # loop through all attached gens
                gen = psm.Generators[gen_ind+1]                 # pull out appropriate gen
                if hasattr(gen, "Sgen")                         # check if that gen has an Sgen
                    push!(nodes, idx)                           # add node index to the list
                    found = true
                    break
                end
            end
        end
    end
    return nodes.+1, length(nodes)
end

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


## Determine initial conditions for day and night loading from PKL files ##################
function get_netload_onetime(psm, nodes, t_ind)
    n_loads = length(nodes)
    Sload_tind = zeros(ComplexF64, n_loads)
    for (ii,node) in enumerate(psm.Nodes[nodes])            # loop over all supplied nodes (in supplied order) - need plus one
        tmp = 0
        for load_ind in node.loads                          # loop over all loads at that node
            load = psm.Loads[load_ind+1]
            if hasattr(load, "Sload")                       # check that load has an Sload attribute
                tmp += load.Sload[t_ind,ph_col]             # add up loads to temporary
            end                                             # load is positive because load goes into BST positive
        end
        for gen_ind in node.gens                            # loop over all gens at node
            gen = psm.Generators[gen_ind+1]
            if hasattr(gen, "Sgen")
                tmp -= gen.Sgen[t_ind,ph_col]               # subtract gen values from temp
            end                                             # taking a load positive convention.. NEED TO LOOK AT THIS IF THERE ARE JUST GEN NODES
        end
        Sload_tind[ii] = tmp
    end
    return Sload_tind
end
# find initial conditions of injections (day/night averages) from psm's - will be used in linearized calculation
Sload0_day = get_netload_onetime(psm_day, nodes, 1)
Sload0_night = get_netload_onetime(psm_night, nodes, 1)

# find initial voltages at these averaged conditions 
Vph0_day = solve_pf(psm_day, V0_ref, 1, linear_solver)
Vph0_night = solve_pf(psm_night, V0_ref, 1, linear_solver)


## Load the .pkl file for all data ########################################################
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model.pkl"
pkl_file = pyopen(fname, "rb")
psm = pickle.load(pkl_file)
pkl_file.close()


## Choose random subset of loading conditions #############################################
nsamp = size(psm.Loads[1].Sload,1)      # determine number of loading conditions
shuff = shuffle(1:nsamp)                # shuffle them
ntest = 5                             # pick the first ntest of them to use
test_idx = shuff[1:ntest]


## Determine if condition is day or night #################################################
day_hours = 12:24
night_hours = 1:11
is_day = Bool[]
for idx in test_idx
    hour = mod1(idx,24)
    if hour in day_hours
        push!(is_day, true)
    elseif hour in night_hours
        push!(is_day, false)
    else
        println("Error, hour not in either set")
    end
end


## Solve power-flow with BST ##############################################################
V_bst = zeros(Float64, ntest, n_loads)
for (ii, t_ind) in enumerate(test_idx)
    Vtmp = solve_pf(psm, V0_ref, t_ind, linear_solver)
    V_bst[ii,:] = abs.(Vtmp[ph_col,nodes])
end


## Solve power-flow with linearization ####################################################
# import jacobians
dVdP_day, dVdQ_day, dVdP_night, dVdQ_night = deserialize("djr-all/BH_small02_Jacobians01.jls")
# init voltage storage
V_lin = zeros(Float64, ntest, n_loads)

# loop and compute
for (ii, t_ind) in enumerate(test_idx)
    # decide day or night values
    if is_day[ii]
        Sload0 = Sload0_day
        V0 = abs.(Vph0_day[ph_col,nodes])
        dVdP = dVdP_day
        dVdQ = dVdQ_day
    else
        Sload0 = Sload0_night
        V0 = abs.(Vph0_night[ph_col,nodes])
        dVdP = dVdP_night
        dVdQ = dVdQ_night
    end
    # pull load values for specific loading condition
    Sload_t = get_netload_onetime(psm, nodes, t_ind)
    # Separate S into P/Q 
    dSload = Sload_t - Sload0
    dP = real(dSload)
    dQ = imag(dSload)
    # Compute linearized voltage
    V_lin[ii,:] = V0 + dVdP * dP + dVdQ * dQ
end


## Compare solutions ######################################################################
# compute norms
dV_inorm = zeros(Float64, ntest)
dV_2norm = zeros(Float64, ntest)
Vdiff2 = V_lin - V_bst
for ii in axes(V_lin,1)
    Vdiff = V_lin[ii,:] - V_bst[ii,:]
    dV_inorm[ii] = norm(Vdiff, Inf)
    dV_2norm[ii] = 1/sqrt(n_loads)*norm(Vdiff, 2)      # normalized for vector size (number of nodes)
end
mx = 1.01 * maximum(vcat(dV_2norm, dV_inorm))
# plot norms
fig, axs = plt.subplots(2, 1, figsize=(5, 4))
axs[1].scatter(test_idx, dV_2norm, s=10)#, c=colors)
axs[2].scatter(test_idx, dV_inorm, s=10)
axs[1].set_ylim([0,mx])
axs[2].set_ylim([0,mx])
axs[1].set_xlabel("Hour of Year")
axs[2].set_xlabel("Hour of Year")
axs[1].set_ylabel("2-norm")
axs[2].set_ylabel("Inf Norm")
plt.tight_layout()
plt.show()


## Save variables to csv ########################################################################
# Convert to DataFrame and save them
function save_data(A, name)
    if isa(A,Vector)
        df = DataFrame(A = A)
    else
        df = DataFrame(A, :auto)
    end
    CSV.write("$name.csv", df)
end
function comp_to_arr(v)             # convert complex vector to array of real and imaginary components
    return hcat(real(v), imag(v))
end
save_vars = true
if save_vars
    # op-point voltages
    save_data(abs.(Vph0_day[ph_col,nodes]), "Vph0_day01_BHsmall02")       # save existing stuff
    save_data(abs.(Vph0_night[ph_col,nodes]), "Vph0_night01_BHsmall02")
    # op-point day/night of P and Q
    save_data(comp_to_arr(Sload0_day), "S0_day01_BHsmall02")
    save_data(comp_to_arr(Sload0_night), "S0_night01_BHsmall02")
    # cases to test: x worst voltage deviations
    tbl = CSV.File("Vdiff_idx_BHsmall02.csv")           # extract Sload from worst voltage cases using the saved csv
    worst_V_idx = collect(tbl.A)
    cases_to_get = 20
    is_day_worst = Bool[]
    Sload_worst = zeros(ComplexF64, cases_to_get, n_loads)
    for (ii,t_ind) in enumerate(worst_V_idx[1:cases_to_get])
        Sload_worst[ii,:] = get_netload_onetime(psm, nodes, t_ind)
        hour = mod1(t_ind,24)
        if hour in day_hours
            push!(is_day_worst, true)
        elseif hour in night_hours
            push!(is_day_worst, false)
        end
    end
    save_data(is_day_worst, "WorstCases_isday_BHsmall02")
    save_data(comp_to_arr(Sload_worst), "S_01_BHsmall02")
    # get list of indices that have generators
    gen_idx_in_209 = []
    for (ii,node) in enumerate(psm.Nodes[nodes])
        if length(node.gens) > 0
            for gen_ind in node.gens
                gen = psm.Generators[gen_ind+1]
                if hasattr(gen, "Sgen")
                    push!(gen_idx_in_209, ii)
                end
            end
        end
    end
    save_data(gen_idx_in_209, "genidx_in209_BHsmall02")
end