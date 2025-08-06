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

# include("BST_funcs.jl")         # BST function: value.(Vph) = solve_pf(psm::PyObject, V0_ref::Vector{ComplexF64}, t_ind::Int64, linear_solver::String)

# Import Python modules
pickle = pyimport("pickle")
pyopen = pyimport("builtins").open
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")

## BST Function: ##########################################################################
function solve_pf(psm::PyObject, V0_ref::Vector{ComplexF64}, t_ind::Int64, linear_solver::String)

    n_nodes = length(psm.Nodes)
    n_branches = length(psm.Branches)

    # Model Setup
    model = Model(Ipopt.Optimizer)
    if linear_solver in ["ma27","ma57","ma77","ma86","ma97"]
        set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        set_attribute(model, "linear_solver", linear_solver)
    elseif linear_solver == "mumps"
        set_attribute(model, "linear_solver", linear_solver)
    else
        throw(ArgumentError("linear_solver $linear_solver not supported."))
    end
    set_optimizer_attribute(model, "print_level", 0)

    # Variable Definitions
    @variable(model, Vph_real[ph=1:3,1:n_nodes], start=real(V0_ref[ph]*exp(-im*pi/6)))
    @variable(model, Vph_imag[ph=1:3,1:n_nodes], start=imag(V0_ref[ph]*exp(-im*pi/6)))
    @variable(model, Iph_real[1:3,1:n_branches], start=0)
    @variable(model, Iph_imag[1:3,1:n_branches], start=0)

    set_start_value.(Vph_real[:,1], real(V0_ref))
    set_start_value.(Vph_imag[:,1], imag(V0_ref))

    # Complex Variable Expressions
    @expression(model, Vph, Vph_real.+im*Vph_imag)
    @expression(model, Iph, Iph_real.+im*Iph_imag)

    # Substation Voltage Constraint
    @constraint(model, Vph[:,1] .== V0_ref)

    # Power Flow Constraints
    pb_lhs = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
    pb_rhs = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
    for (br_ind,Branch) in enumerate(psm.Branches)
        # skip open branches
        if Branch.type == "switch" 
            if Branch.status == "OPEN"
                Iph[:,br_ind] .== 0.0
                continue
            end
        end
        from_node_ind = Branch.from_node_ind+1
        to_node_ind = Branch.to_node_ind+1
        # "Ohm's law"
        @constraint(model, Vph[:,from_node_ind] .== Branch.A_br*Vph[:,to_node_ind] + Branch.B_br*Iph[:,br_ind])
        # Add branch flows to power balance expressions
        pb_lhs[:,to_node_ind] += diag(Vph[:,to_node_ind]*Iph[:,br_ind]')
        pb_rhs[:,from_node_ind] += diag(Branch.A_br*Vph[:,to_node_ind]*Iph[:,br_ind]'*(Branch.D_br')+Branch.B_br*Iph[:,br_ind]*Iph[:,br_ind]'*(Branch.D_br'))
    end


    # Power Injection Constraints
    s_load = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
    for (ld_ind, Load) in enumerate(psm.Loads)
        if haskey(Load,"Sload")
            s_load[:,Load.parent_node_ind+1] += Load.Sload[t_ind,:]
        end
    end
    s_gen = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
    for (gen_ind, Gen) in enumerate(psm.Generators)
        if haskey(Gen,"Sgen")
            s_gen[:,Gen.parent_node_ind+1] += Gen.Sgen[t_ind,:]
        end
    end
    for (sht_ind, Shunt) in enumerate(psm.Shunts)
        if Shunt.type == "capacitor"
            status = zeros(Int, 3, 1)
            if Shunt.switchA == "CLOSED"
                status[1] = 1
            end
            if Shunt.switchB == "CLOSED"
                status[2] = 1
            end
            if Shunt.switchC == "CLOSED"
                status[3] = 1
            end
            parent_node_ind = Shunt.parent_node_ind+1
            s_load[:,parent_node_ind] += status.*diag(Vph[:,parent_node_ind]*Vph[:,parent_node_ind]'*conj(Shunt.Ycap))
        end
    end
    @constraint(model, pb_rhs[:,2:end] - pb_lhs[:,2:end] .== s_gen[:,2:end]-s_load[:,2:end])

    optimize!(model)

    # print status
    status = termination_status(model)
    if status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        print(".")
    else
        println("Solver did not find an optimal solution: $status")
    end

    return value.(Vph)#, value.(pb_rhs[:,1]-pb_lhs[:,1])
end
## Set up for function use: 
V0_mag = 1                          # substation voltage
V0_ref = V0_mag*[1,exp(-im*2*pi/3),exp(im*2*pi/3)]
linear_solver = "mumps"


## Load the .pkl files for day and night time #############################################
substation_name = "Burton_Hill_small02"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model_DAY.pkl"
pkl_file = pyopen(fname, "rb")
psm_day = pickle.load(pkl_file)
pkl_file.close()
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model_NIGHT.pkl"
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

## Determine initial conditions for day and night loading from PKL files ##################
# extract Sload and Sgen
function gather_loads(psm, t_ind, ph_col)          # gather loads out of psm
    Sload = []
    for load in psm.Loads
        if hasattr(load, "Sload")
            push!(Sload, load.Sload[t_ind,ph_col])
        end
    end
    return Sload
end
function gather_gens(psm, t_ind, ph_col)          # gather gens out of psm
    Sgen = []
    for gen in psm.Generators
        if hasattr(gen, "Sgen")
            push!(Sgen, gen.Sgen[t_ind,ph_col])
        end
    end
    return Sgen
end

# find initial conditions of injections (day/night averages) from psm's
Sload0_day = gather_loads(psm_day, 1, ph_col)
Sgen0_day = gather_gens(psm_day, 1, ph_col)
Sload0_night = gather_loads(psm_night, 1, ph_col)
Sgen0_night = gather_gens(psm_night, 1, ph_col)

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
ntest = 200                             # pick the first ntest of them to use
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
nnodes = length(psm.Nodes)
V_bst = zeros(Float64, ntest, nnodes)
for (ii, t_ind) in enumerate(test_idx)
    Vtmp = solve_pf(psm, V0_ref, t_ind, linear_solver)
    V_bst[ii,:] = abs.(Vtmp[ph_col,:])
end


## Solve power-flow with linearization ####################################################
# import jacobians
dVdP_day, dVdQ_day, dVdP_night, dVdQ_night = deserialize("djr-all/BH_small02_Jacobians00.jls")
# init voltage storage
V_lin = zeros(Float64, ntest, nnodes)
# loop and compute
for (ii, t_ind) in enumerate(test_idx)
    # decide day or night values
    if is_day[ii]
        Sload0 = Sload0_day
        Sgen0 = Sgen0_day
        V0 = abs.(Vph0_day[ph_col,:])
        dVdP = dVdP_day
        dVdQ = dVdQ_day
    else
        Sload0 = Sload0_night
        Sgen0 = Sgen0_night
        V0 = abs.(Vph0_night[ph_col,:])
        dVdP = dVdP_night
        dVdQ = dVdQ_night
    end
    # pull load and gen values for specific loading condition
    Sload_t = gather_loads(psm, t_ind, ph_col)
    Sgen_t = gather_gens(psm, t_ind, ph_col)
    # Separate S into P/Q 
    dSload = Sload_t - Sload0
    dSgen = Sgen_t - Sgen0
    dP = vcat(real(dSload), real(dSgen))
    dQ = vcat(imag(dSload), imag(dSgen))
    # Compute linearized voltage
    V_lin[ii,:] = V0 + dVdP * dP + dVdQ * dQ
end


## Compare solutions ######################################################################
# compute norms
dV_inorm = zeros(Float64, ntest)
dV_2norm = zeros(Float64, ntest)
for ii in axes(V_lin,1)
    Vdiff = V_lin[ii,:] - V_bst[ii,:]
    dV_inorm[ii] = norm(Vdiff, Inf)
    dV_2norm[ii] = 1/ntest*norm(Vdiff, 2)      # normalized for vector size (number of nodes)
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


## Save variables to csv
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
    save_data(abs.(Vph0_day[ph_col,:]), "Vph0_day_BHsmall02")       # save existing stuff
    save_data(abs.(Vph0_night[ph_col,:]), "Vph0_night_BHsmall02")
    save_data(comp_to_arr(Sload0_day), "Sload0_day_BHsmall02")
    save_data(comp_to_arr(Sgen0_day), "Sgen0_day_BHsmall02")
    save_data(comp_to_arr(Sload0_night), "Sload0_night_BHsmall02")
    save_data(comp_to_arr(Sgen0_night), "Sgen0_night_BHsmall02")
    Sload_day = gather_loads(psm, 15, ph_col)                       # pull out an operating condition from PSM to save
    Sgen_day = gather_gens(psm, 15, ph_col)
    Sload_night = gather_loads(psm, 5, ph_col)
    Sgen_night = gather_gens(psm, 5, ph_col)
    save_data(comp_to_arr(Sload_day), "Sload_day_BHsmall02")        # save those operating conditions
    save_data(comp_to_arr(Sgen_day), "Sgen_day_BHsmall02")
    save_data(comp_to_arr(Sload_night), "Sload_night_BHsmall02")
    save_data(comp_to_arr(Sgen_night), "Sgen_night_BHsmall02")
    # determine what nodes have Generators and loads
    gen_idx = []
    for gen in psm.Generators
        if hasattr(gen, "Sgen")
            push!(gen_idx, gen.parent_node_ind)
        end
    end
    load_idx = []
    for load in psm.Loads
        if hasattr(load, "Sload")
            push!(load_idx, load.parent_node_ind)
        end
    end
    save_data(gen_idx,"gen_index_BHsmall02")
    save_data(load_idx,"load_index_BHsmall02")
    # determine net load at nodes - for averages
    function get_gen_netload(psm, nodes, ph_col, t_ind)
    netload = zeros(Float64, nodes, 2)
    for gen in psm.Generators               # loop through day gens
        if hasattr(gen, "Sgen")                 
            idx = gen.parent_node_ind + 1          # pull out parent node and make 1-indexed
            netload[idx,:] += [real(gen.Sgen[t_ind, ph_col]), imag(gen.Sgen[t_ind, ph_col])]      # sub in the real and imaginary components
        end
    end
    return netload
    end
    function get_netload(psm, nodes, ph_col, t_ind)
    netload = get_gen_netload(psm, nodes, ph_col, t_ind)
    for load in psm.Loads               # loop through day gens
        if hasattr(load, "Sload")                 
            idx = load.parent_node_ind  + 1        # pull out parent node and make 1-indexed
            netload[idx,:] -= [real(load.Sload[t_ind, ph_col]), imag(load.Sload[t_ind, ph_col])]      # sub in the real and imaginary components
        end
    end
    return netload
    end
    netload_day = get_netload(psm_day, 1072, ph_col, 1)
    netload_night = get_netload(psm_night, 1072, ph_col, 1)
    save_data(netload_day,"netload_day_BHsmall02")
    save_data(netload_night,"netload_night_BHsmall02")
    # determine net load at nodes - for specific conditions
    netload_day_op = get_netload(psm, 1072, ph_col, 15)
    netload_night_op = get_netload(psm, 1072, ph_col, 5)
    save_data(netload_day,"netload_day_op_BHsmall02")
    save_data(netload_night,"netload_night_op_BHsmall02")
end


