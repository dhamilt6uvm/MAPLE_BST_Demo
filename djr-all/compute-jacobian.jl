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

##### Function for running power flow ########
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

    return value.(Vph)#, value.(pb_rhs[:,1]-pb_lhs[:,1])

end

############################################################################################

# Import Python modules
pickle = pyimport("pickle")
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")

# phase?
ph_col = 2      # phase = B

# Load the .pkl files for day and night time
substation_name = "Burton_Hill_small02"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model_DAY.pkl"
pkl_file = pyopen(fname, "rb")
psm_day = pickle.load(pkl_file)
pkl_file.close()
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model_NIGHT.pkl"
pkl_file = pyopen(fname, "rb")
psm_night = pickle.load(pkl_file)
pkl_file.close()

# determine num of nodes and branches
n_nodes = length(psm_day.Nodes)
n_branches = length(psm_day.Branches)

# Substation Voltage
V0_mag = 1
V0_ref = V0_mag*[1,exp(-im*2*pi/3),exp(im*2*pi/3)]

# find how many cases to solve
t_start = 1
t_end = size(psm_day.Loads[1].Sload,1)  # assuming all loads have the same number of time steps

nloads = 4#210              # delete all this and fix      
ngens = 2#30
t_end = nloads*2 + ngens*2

# solve power flow for day and night
n_times = t_end-t_start+1
linear_solver = "mumps"
Vph_out_day = Array{Float64}(undef, n_nodes, n_times)
for t_ind in t_start:t_end
    Vtmp = solve_pf(psm_day, V0_ref, t_ind, linear_solver)
    Vph_out_day[:,t_ind] = abs.(Vtmp[ph_col,:])
end
Vph_out_night = Array{Float64}(undef, n_nodes, n_times)
for t_ind in t_start:t_end
    Vtmp = solve_pf(psm_night, V0_ref, t_ind, linear_solver)
    Vph_out_night[:,t_ind] = abs.(Vtmp[ph_col,:])
end

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