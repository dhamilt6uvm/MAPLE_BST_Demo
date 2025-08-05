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
using Dates

function create_model(psm::PyObject, V0_ref::Vector{ComplexF64}, linear_solver::String)

    # get network info
    n_nodes = length(psm.Nodes)
    n_branches = length(psm.Branches)

    # find number of delta-connected loads (need to create additional variables for delta currents)
    global n_loads_D = 0
    for Load in psm.Loads
        if occursin("D", Load.phases)
            global n_loads_D
            n_loads_D = n_loads_D + 1
        end
    end

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
    @variable(model, Iload_D_real[1:3,1:n_loads_D], start=0)
    @variable(model, Iload_D_imag[1:3,1:n_loads_D], start=0)

    set_start_value.(Vph_real[:,1], real(V0_ref))
    set_start_value.(Vph_imag[:,1], imag(V0_ref))

    # Parameter Definitions
    @variable(model, s_load_Y_real[1:3,1:n_nodes] in Parameter(0.0))
    @variable(model, s_load_Y_imag[1:3,1:n_nodes] in Parameter(0.0))
    @variable(model, s_load_D_real[1:3,1:n_loads_D] in Parameter(0.0))
    @variable(model, s_load_D_imag[1:3,1:n_loads_D] in Parameter(0.0))
    @variable(model, s_gen_real[1:3,1:n_nodes] in Parameter(0.0))
    @variable(model, s_gen_imag[1:3,1:n_nodes] in Parameter(0.0))

    # Complex Variable Expressions
    @expression(model, Vph, Vph_real.+im*Vph_imag)
    @expression(model, Iph, Iph_real.+im*Iph_imag)
    @expression(model, Iload_D, Iload_D_real.+im*Iload_D_imag)

    @expression(model, s_load_Y_param, s_load_Y_real.+im*s_load_Y_imag)
    @expression(model, s_load_D_param, s_load_D_real.+im*s_load_D_imag)
    @expression(model, s_gen_param, s_gen_real.+im*s_gen_imag)

    # Substation Voltage Constraint
    @constraint(model, Vph[:,1] .== V0_ref)

    # Power Flow Constraints
    pb_in = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
    pb_out = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
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
        pb_in[:,to_node_ind] += diag(Vph[:,to_node_ind]*Iph[:,br_ind]')
        pb_out[:,from_node_ind] += diag(Branch.A_br*Vph[:,to_node_ind]*Iph[:,br_ind]'*(Branch.D_br')+Branch.B_br*Iph[:,br_ind]*Iph[:,br_ind]'*(Branch.D_br'))
    end


    # Load and power injection constraints  
    s_load_Y = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
    s_load_Y += s_load_Y_param
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
            s_load_Y[:,parent_node_ind] += status.*diag(Vph[:,parent_node_ind]*Vph[:,parent_node_ind]'*conj(Shunt.Ycap))
        end
    end

    t_ind = 1
    s_load_D = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
    gamma_D = [[1.0 -1.0 0.0];[0.0 1.0 -1.0];[-1.0 0.0 1.0]]
    global delta_node_ind = 1
    for (ld_ind, Load) in enumerate(psm.Loads)
        if haskey(Load,"Sload")
            parent_node_ind = Load.parent_node_ind+1
            if occursin("D", Load.phases)
                # @constraint(model, diag(gamma_D*Vph[:,parent_node_ind]*Iload_D[:,delta_node_ind]') .== Load.Sload[t_ind,:])
                @constraint(model, diag(gamma_D*Vph[:,parent_node_ind]*Iload_D[:,delta_node_ind]') .== s_load_D_param[:,delta_node_ind])
                s_load_D[:,parent_node_ind] += diag(Vph[:,parent_node_ind]*Iload_D[:,delta_node_ind]'*gamma_D)
                global delta_node_ind
                delta_node_ind = delta_node_ind + 1
            end
        end
    end

    s_gen = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
    s_gen += s_gen_param

    # Power balance constraint
    @constraint(model, pb_out[:,2:end] - pb_in[:,2:end] .== s_gen[:,2:end]-s_load_Y[:,2:end]-s_load_D[:,2:end])


    # store some expressions for later
    model.ext[:pb_in] = pb_in
    model.ext[:pb_out] = pb_out

    return model

end

function solve_pf(model::Model, psm::PyObject, t_ind::Int64)

    # get network info
    n_nodes = length(psm.Nodes)
    n_branches = length(psm.Branches)

    # find number of delta-connected loads (need to create additional variables for delta currents)
    global n_loads_D = 0
    for Load in psm.Loads
        if occursin("D", Load.phases)
            global n_loads_D
            n_loads_D = n_loads_D + 1
        end
    end

    # zero out parameters - could have values from a previous pf solve
    set_parameter_value.(model[:s_load_Y_real], zeros(3,n_nodes))
    set_parameter_value.(model[:s_load_Y_imag], zeros(3,n_nodes))
    set_parameter_value.(model[:s_load_D_real], zeros(3,n_loads_D))
    set_parameter_value.(model[:s_load_D_imag], zeros(3,n_loads_D))
    set_parameter_value.(model[:s_gen_real], zeros(3,n_nodes))
    set_parameter_value.(model[:s_gen_imag], zeros(3,n_nodes))

    # update load parameters
    global delta_node_ind = 1
    for (ld_ind, Load) in enumerate(psm.Loads)
        if haskey(Load,"Sload")
            parent_node_ind = Load.parent_node_ind+1
            if occursin("N", Load.phases)
                temp_s_load_Y_real = parameter_value.(model[:s_load_Y_real][:,parent_node_ind])
                temp_s_load_Y_imag = parameter_value.(model[:s_load_Y_imag][:,parent_node_ind])
                set_parameter_value.(model[:s_load_Y_real][:,parent_node_ind], temp_s_load_Y_real+real(Load.Sload[t_ind,:]))
                set_parameter_value.(model[:s_load_Y_imag][:,parent_node_ind], temp_s_load_Y_imag+imag(Load.Sload[t_ind,:]))
            elseif occursin("D", Load.phases)
                temp_s_load_D_real = parameter_value.(model[:s_load_D_real][:,delta_node_ind])
                temp_s_load_D_imag = parameter_value.(model[:s_load_D_imag][:,delta_node_ind])
                set_parameter_value.(model[:s_load_D_real][:,delta_node_ind], temp_s_load_D_real+real(Load.Sload[t_ind,:]))
                set_parameter_value.(model[:s_load_D_imag][:,delta_node_ind], temp_s_load_D_imag+imag(Load.Sload[t_ind,:]))
                global delta_node_ind
                delta_node_ind = delta_node_ind + 1
            else
                throw(ArgumentError("Can't determine if Load $(Load.name) is wye or delta connected."))
            end
        end
    end

    # update generation parameters
    for (gen_ind, Gen) in enumerate(psm.Generators)
        if haskey(Gen,"Sgen")
            parent_node_ind = Gen.parent_node_ind+1
            temp_s_gen_real = parameter_value.(model[:s_gen_real][:,parent_node_ind])
            temp_s_gen_imag = parameter_value.(model[:s_gen_imag][:,parent_node_ind])
            set_parameter_value.(model[:s_gen_real][:,parent_node_ind], temp_s_gen_real+real(Gen.Sgen[t_ind,:]))
            set_parameter_value.(model[:s_gen_imag][:,parent_node_ind], temp_s_gen_imag+imag(Gen.Sgen[t_ind,:]))
        end
    end

    optimize!(model)
    return value.(model[:Vph]), value.(model.ext[:pb_out][:,1]-model.ext[:pb_in][:,1])
end

############################################################################################

# Import Python modules
pickle = pyimport("pickle")
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")

# Load the .pkl file 
substation_name = "Burton_Hill"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model.pkl"
pkl_file = open(fname, "r")
psm = pickle.load(pkl_file)
close(pkl_file)

n_nodes = length(psm.Nodes)
n_branches = length(psm.Branches)

# Substation Voltage
V0_mag = 1
V0_ref = V0_mag*[1,exp(-im*2*pi/3),exp(im*2*pi/3)]

# Create model
linear_solver = "mumps"
pf_model = create_model(psm, V0_ref, linear_solver)

# solve power flow
t_start = 1
t_end = 168
# t_end = 24
n_times = t_end-t_start+1
Vph_out = Array{ComplexF64}(undef, 3, n_nodes, n_times)
Ssub_out = Array{ComplexF64}(undef, 3, n_times)
for t_ind in t_start:t_end
    Vph_out[:,:,t_ind], Ssub_out[:,t_ind] = solve_pf(pf_model, psm, t_ind)
end

##
p1 = plot(t_start:t_end,transpose(real.(Ssub_out).*psm.Sbase_1ph./1e6),label=["Phase A" "Phase B" "Phase C"])
xlabel!("Time (h)")
ylabel!("Substation Active Power (MW)")
display(p1)


################# Compare Results to GLD ####################

# substation powers
gld_sub_power_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/substation_power.csv",skipto=10,header=9) |> DataFrame

Psub_gld = gld_sub_power_df[t_start:t_end,"power_in.real"]

p2 = plot(t_start:t_end,transpose(sum(real.(Ssub_out).*psm.Sbase_1ph./1e6,dims=1)),label="BST")
plot!(t_start:t_end,Psub_gld./1e6,label="GridLAB-D")
xlabel!("Time (h)")
ylabel!("Substation Active Power (MW)")
display(p2)


# voltages
gld_node_mags_A_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_mags_A.csv",skipto=10,header=9) |> DataFrame
gld_node_mags_B_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_mags_B.csv",skipto=10,header=9) |> DataFrame
gld_node_mags_C_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_mags_C.csv",skipto=10,header=9) |> DataFrame

gld_node_angs_A_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_angs_A.csv",skipto=10,header=9) |> DataFrame
gld_node_angs_B_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_angs_B.csv",skipto=10,header=9) |> DataFrame
gld_node_angs_C_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_angs_C.csv",skipto=10,header=9) |> DataFrame

Vph_gld = zeros(ComplexF64, 3, n_nodes, n_times)
for t_ind in t_start:t_end
    for (nd_ind, Node) in enumerate(psm.Nodes)
        Vmag_a = gld_node_mags_A_df[t_ind,Node.name]
        Vang_a = gld_node_angs_A_df[t_ind,Node.name]
        Vmag_b = gld_node_mags_B_df[t_ind,Node.name]
        Vang_b = gld_node_angs_B_df[t_ind,Node.name]
        Vmag_c = gld_node_mags_C_df[t_ind,Node.name]
        Vang_c = gld_node_angs_C_df[t_ind,Node.name]
        Vph_gld[1,nd_ind,t_ind] = Vmag_a*exp(1im*Vang_a)/Node.Vbase
        Vph_gld[2,nd_ind,t_ind] = Vmag_b*exp(1im*Vang_b)/Node.Vbase
        Vph_gld[3,nd_ind,t_ind] = Vmag_c*exp(1im*Vang_c)/Node.Vbase
    end
end

# correct for missing phases
Vph_opt = value.(Vph_out)
for t_ind in t_start:t_end
    for (nd_ind,Node) in enumerate(psm.Nodes)
        if ~occursin("A",Node.phases)
            Vph_opt[1,nd_ind,t_ind] = 0.0
        end
        if ~occursin("B",Node.phases)
            Vph_opt[2,nd_ind,t_ind] = 0.0
        end
        if ~occursin("C",Node.phases)
            Vph_opt[3,nd_ind,t_ind] = 0.0
        end
    end
end

mag_err = abs.(Vph_opt)-abs.(Vph_gld)
ang_err = angle.(Vph_opt)-angle.(Vph_gld)

max_mag_err = maximum(abs.(mag_err))
max_ang_err = maximum(abs.(ang_err))

println("Maximum error in voltage magnitudes: ", max_mag_err)
println("Maximum error in voltage angles: ", max_ang_err)

