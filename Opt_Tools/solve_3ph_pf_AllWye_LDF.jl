using PyCall
using SparseArrays
using Plots
gr()
using ColorTypes
using Colors
using JuMP
using Ipopt
import HSL_jll
using LinearAlgebra
using CSV
using DataFrames

# Import Python modules
pickle = pyimport("pickle")
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")

# Load the .pkl file 
substation_name = "Burton_Hill_AllWye"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model.pkl"
pkl_file = open(fname, "r")
psm = pickle.load(pkl_file)
close(pkl_file)

# get network info
n_nodes = length(psm.Nodes)
n_branches = length(psm.Branches)

# Substation Voltage
V0_mag = 1
V0_ref = V0_mag*[1.0,exp(-im*2*pi/3),exp(im*2*pi/3)]

# Model Setup
model = Model(Ipopt.Optimizer)
model = Model(Ipopt.Optimizer)
linear_solver = "ma57"
if linear_solver in ["ma27","ma57","ma77","ma86","ma97"]
    set_attribute(model, "hsllib", HSL_jll.libhsl_path)
    set_attribute(model, "linear_solver", linear_solver)
elseif linear_solver == "mumps"
    set_attribute(model, "linear_solver", linear_solver)
else
    throw(ArgumentError("linear_solver $linear_solver not supported."))
end

# Variable Definitions
@variable(model, v2_j[ph=1:3,1:n_nodes], start=1.0)
@variable(model, p_ij[1:3,1:n_branches], start=0.0)
@variable(model, q_ij[1:3,1:n_branches], start=0.0)

# set_start_value.(Vph_real[:,1], real(V0_ref))
# set_start_value.(Vph_imag[:,1], imag(V0_ref))

# Complex Variable Expressions
@expression(model, s_ij, p_ij.+im*q_ij)

# Substation Voltage Constraint
@constraint(model, v2_j[:,1] .== abs.(V0_ref).^2)

# Power Flow Constraints
alph = exp(-im*2*pi/3)
beta_vect = [1.0,alph,conj(alph)]
Gamma = beta_vect*beta_vect'
pb_in = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
pb_out = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
for (br_ind,Branch) in enumerate(psm.Branches)
    # skip open branches
    if Branch.type == "switch" 
        if Branch.status == "OPEN"
            s_ij[:,br_ind] .== 0.0
            continue
        end
    end
    from_node_ind = Branch.from_node_ind+1
    to_node_ind = Branch.to_node_ind+1
    # "Ohm's law"
    @constraint(model, v2_j[:,from_node_ind] .== v2_j[:,to_node_ind] + 2*real.(diag(Gamma*diagm(s_ij[:,br_ind])*Branch.Z_pu_3ph')))
    # Add branch flows to power balance expressions
    pb_in[:,to_node_ind] += s_ij[:,br_ind]
    pb_out[:,from_node_ind] += s_ij[:,br_ind]
end


# Power Injection Constraints
t_ind = 10
s_load = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
for (ld_ind, Load) in enumerate(psm.Loads)
    if haskey(Load,"Sload")
        parent_node_ind = Load.parent_node_ind+1
        s_load[:,parent_node_ind] += Load.Sload[t_ind,:]
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
        s_load[:,parent_node_ind] += status.*conj(Shunt.Ycap)*v2_j[:,parent_node_ind] # assumes Shunt.Ycap is diagonal
    end
end
@constraint(model, pb_out[:,2:end] - pb_in[:,2:end] .== s_gen[:,2:end]-s_load[:,2:end])

# print(model)
optimize!(model)

## Compare results to GLD
gld_node_mags_A_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_mags_A.csv",skipto=10,header=9) |> DataFrame
gld_node_mags_B_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_mags_B.csv",skipto=10,header=9) |> DataFrame
gld_node_mags_C_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_mags_C.csv",skipto=10,header=9) |> DataFrame

gld_node_angs_A_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_angs_A.csv",skipto=10,header=9) |> DataFrame
gld_node_angs_B_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_angs_B.csv",skipto=10,header=9) |> DataFrame
gld_node_angs_C_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_angs_C.csv",skipto=10,header=9) |> DataFrame

Vph_gld = zeros(ComplexF64, 3, n_nodes)
for (nd_ind, Node) in enumerate(psm.Nodes)
    Vmag_a = gld_node_mags_A_df[t_ind,Node.name]
    Vang_a = gld_node_angs_A_df[t_ind,Node.name]
    Vmag_b = gld_node_mags_B_df[t_ind,Node.name]
    Vang_b = gld_node_angs_B_df[t_ind,Node.name]
    Vmag_c = gld_node_mags_C_df[t_ind,Node.name]
    Vang_c = gld_node_angs_C_df[t_ind,Node.name]
    Vph_gld[1,nd_ind] = Vmag_a*exp(1im*Vang_a)/Node.Vbase
    Vph_gld[2,nd_ind] = Vmag_b*exp(1im*Vang_b)/Node.Vbase
    Vph_gld[3,nd_ind] = Vmag_c*exp(1im*Vang_c)/Node.Vbase
end

# correct for missing phases
v2_j_opt = value.(v2_j)
for (nd_ind,Node) in enumerate(psm.Nodes)
    if ~occursin("A",Node.phases)
        v2_j_opt[1,nd_ind] = 0.0
    end
    if ~occursin("B",Node.phases)
        v2_j_opt[2,nd_ind] = 0.0
    end
    if ~occursin("C",Node.phases)
        v2_j_opt[3,nd_ind] = 0.0
    end
end

mag_err = sqrt.(v2_j_opt)-abs.(Vph_gld)
p2 = plot(1:n_nodes,transpose(mag_err),label=["Phase A" "Phase B" "Phase C"])
xlabel!("Node Index")
ylabel!("Absolute Voltage Magnitude Error (pu)")
display(p2)

# ang_err = angle.(Vph_opt)-angle.(Vph_gld)
# p3 = plot(1:n_nodes,transpose(ang_err),label=["Phase A" "Phase B" "Phase C"])
# xlabel!("Node Index")
# ylabel!("Absolute Voltage Angle Error (rad)")
# display(p3)

p4 = plot(1:n_nodes,transpose(sqrt.(v2_j_opt)),label=["Phase A" "Phase B" "Phase C"])
plot!(1:n_nodes,transpose(abs.(Vph_gld)),label=["Phase A" "Phase B" "Phase C"])
xlabel!("Node Index")
ylabel!("Voltage Magnitude (pu)")
display(p4)