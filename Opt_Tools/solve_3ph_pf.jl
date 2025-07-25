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
using Printf

# Import Python modules
pickle = pyimport("pickle")
pyopen = pyimport("builtins").open
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")

# Load the .pkl file 
substation_name = "Burton_Hill_DanR"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model.pkl"
# try
pkl_file = pyopen(fname, "rb")
psm = pickle.load(pkl_file)
pkl_file.close()
# catch e
#     showerror(stdout, e)
# end

# get network info
n_nodes = length(psm.Nodes)
n_branches = length(psm.Branches)

# find number of delta-connected loads (need to create additional variables for delta currents)
n_loads_D = 0
for Load in psm.Loads
    if occursin("D", Load.phases)
        global n_loads_D
        n_loads_D = n_loads_D + 1
    end
end

# Substation Voltage
V0_mag = 1
V0_ref = V0_mag*[1.0,exp(-im*2*pi/3),exp(im*2*pi/3)]

# Model Setup
model = Model(Ipopt.Optimizer)
model = Model(Ipopt.Optimizer)
linear_solver = "mumps"
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

# Complex Variable Expressions
@expression(model, Vph, Vph_real.+im*Vph_imag)
@expression(model, Iph, Iph_real.+im*Iph_imag)
@expression(model, Iload_D, Iload_D_real.+im*Iload_D_imag)

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
t_ind = 5
s_load_Y = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
s_load_D = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
gamma_D = [[1.0 -1.0 0.0];[0.0 1.0 -1.0];[-1.0 0.0 1.0]]
delta_node_ind = 1
for (ld_ind, Load) in enumerate(psm.Loads)
    if haskey(Load,"Sload")
        parent_node_ind = Load.parent_node_ind+1
        if occursin("N", Load.phases)
            s_load_Y[:,parent_node_ind] += Load.Sload[t_ind,:]
        elseif occursin("D", Load.phases)
            @constraint(model, diag(gamma_D*Vph[:,parent_node_ind]*Iload_D[:,delta_node_ind]') .== Load.Sload[t_ind,:])
            # @constraint(model, diag(gamma_D*Vph[:,parent_node_ind]*Iload_D[:,delta_node_ind]') .== zeros(3,1))
            s_load_D[:,parent_node_ind] += diag(Vph[:,parent_node_ind]*Iload_D[:,delta_node_ind]'*gamma_D)
            global delta_node_ind
            delta_node_ind = delta_node_ind + 1
        else
            throw(ArgumentError("Can't determine if Load $(Load.name) is wye or delta connected."))
        end
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
        s_load_Y[:,parent_node_ind] += status.*diag(Vph[:,parent_node_ind]*Vph[:,parent_node_ind]'*conj(Shunt.Ycap))
    end
end
@constraint(model, pb_rhs[:,2:end] - pb_lhs[:,2:end] .== s_gen[:,2:end]-s_load_Y[:,2:end]-s_load_D[:,2:end])

# Objective
# @objective(model, Min, sum(real(s_inj[:,1])))

# print(model)
optimize!(model)

# ## Compare results to GLD
# gld_node_mags_A_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_mags_A.csv",skipto=10,header=9) |> DataFrame
# gld_node_mags_B_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_mags_B.csv",skipto=10,header=9) |> DataFrame
# gld_node_mags_C_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_mags_C.csv",skipto=10,header=9) |> DataFrame

# gld_node_angs_A_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_angs_A.csv",skipto=10,header=9) |> DataFrame
# gld_node_angs_B_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_angs_B.csv",skipto=10,header=9) |> DataFrame
# gld_node_angs_C_df = CSV.File("Feeder_Data/$(substation_name)/Output_Data/node_voltage_angs_C.csv",skipto=10,header=9) |> DataFrame

# Vph_gld = zeros(ComplexF64, 3, n_nodes)
# for (nd_ind, Node) in enumerate(psm.Nodes)
#     Vmag_a = gld_node_mags_A_df[t_ind,Node.name]
#     Vang_a = gld_node_angs_A_df[t_ind,Node.name]
#     Vmag_b = gld_node_mags_B_df[t_ind,Node.name]
#     Vang_b = gld_node_angs_B_df[t_ind,Node.name]
#     Vmag_c = gld_node_mags_C_df[t_ind,Node.name]
#     Vang_c = gld_node_angs_C_df[t_ind,Node.name]
#     Vph_gld[1,nd_ind] = Vmag_a*exp(1im*Vang_a)/Node.Vbase
#     Vph_gld[2,nd_ind] = Vmag_b*exp(1im*Vang_b)/Node.Vbase
#     Vph_gld[3,nd_ind] = Vmag_c*exp(1im*Vang_c)/Node.Vbase
# end

# # correct for missing phases
# Vph_opt = value.(Vph)
# for (nd_ind,Node) in enumerate(psm.Nodes)
#     if ~occursin("A",Node.phases)
#         Vph_opt[1,nd_ind] = 0.0
#     end
#     if ~occursin("B",Node.phases)
#         Vph_opt[2,nd_ind] = 0.0
#     end
#     if ~occursin("C",Node.phases)
#         Vph_opt[3,nd_ind] = 0.0
#     end
# end

# mag_err = abs.(Vph_opt)-abs.(Vph_gld)
# p2 = plot(1:n_nodes,transpose(mag_err),label=["Phase A" "Phase B" "Phase C"])
# xlabel!("Node Index")
# ylabel!("Absolute Voltage Magnitude Error (pu)")
# display(p2)

# ang_err = angle.(Vph_opt)-angle.(Vph_gld)
# p3 = plot(1:n_nodes,transpose(ang_err),label=["Phase A" "Phase B" "Phase C"])
# xlabel!("Node Index")
# ylabel!("Absolute Voltage Angle Error (rad)")
# display(p3)

# visualize feeder
node_Xcoords = [Node.X_coord for Node in psm.Nodes]
node_Ycoords = [Node.Y_coord for Node in psm.Nodes]
node_colormap = cgrad(:turbo)
Vph_mag = abs.(value.(Vph))
Vmin = minimum(Vph_mag)
Vmax = maximum(Vph_mag)
# Vmin = 0.95
# Vmax = 1.05
Vmag_out_a = Vph_mag[1,:]
Vmag_out_b = Vph_mag[2,:]
Vmag_out_c = Vph_mag[3,:]
norm_Vmag_out_a = (Vmag_out_a .- Vmin) ./ (Vmax-Vmin)
norm_Vmag_out_b = (Vmag_out_b .- Vmin) ./ (Vmax-Vmin)
norm_Vmag_out_c = (Vmag_out_c .- Vmin) ./ (Vmax-Vmin)
node_colors_a = [get(node_colormap,val) for val in norm_Vmag_out_a]
node_colors_b = [get(node_colormap,val) for val in norm_Vmag_out_b]
node_colors_c = [get(node_colormap,val) for val in norm_Vmag_out_c]

# vis_plt = plot(layout=grid(1,4, widths=(0.3,0.3,0.3,0.1)), size=(1200,300))
vis_plt = plot(layout=grid(1,3, widths=(1/3,1/3,1/3)), size=(1200,300))
for (br_ind,Branch) in enumerate(psm.Branches)
    plot!([Branch.X_coord,Branch.X2_coord],[Branch.Y_coord,Branch.Y2_coord],color=:black,subplot=1)
    plot!([Branch.X_coord,Branch.X2_coord],[Branch.Y_coord,Branch.Y2_coord],color=:black,subplot=2)
    plot!([Branch.X_coord,Branch.X2_coord],[Branch.Y_coord,Branch.Y2_coord],color=:black,subplot=3)
end
for (nd_ind,Node) in enumerate(psm.Nodes)
    if occursin("A",Node.phases)
        plot!([Node.X_coord],[Node.Y_coord],seriestype=:scatter,color=node_colors_a[nd_ind],markersize=3,subplot=1)
    end
    if occursin("B",Node.phases)
        plot!([Node.X_coord],[Node.Y_coord],seriestype=:scatter,color=node_colors_b[nd_ind],markersize=3,subplot=2)
    end
    if occursin("C",Node.phases)
        plot!([Node.X_coord],[Node.Y_coord],seriestype=:scatter,color=node_colors_c[nd_ind],markersize=3,subplot=3)
    end
end
plot!(title="Phase A",xformatter=:none,yformatter=:none,legend=:false,subplot=1)
plot!(title="Phase B",xformatter=:none,yformatter=:none,legend=:false,subplot=2)
plot!(title="Phase C",xformatter=:none,yformatter=:none,legend=:false,subplot=3)
# heatmap!(rand(2,2), clims=(Vmin,Vmax),  right_margin = 10Plots.mm, framestyle=:none, c=node_colormap, cbar=true, lims=(-1,0),colorbar_title = " \nVoltage Magnitude (pu)",subplot=4)
display(vis_plt)

# colorbar plot - axis labels are cut off... 
dummy_vals = reshape(range(Vmin, Vmax, length=100), :, 1)
colorbar_plot = heatmap(
    dummy_vals,
    c=cgrad(:turbo),
    axis=nothing,
    colorbar=true,
    size=(150, 400),
    ticks=:native
)
display(colorbar_plot)

# # single phase plot with colorbar - not currently working... :(
# single_phase = "B"
# show_single_phase = true
# if show_single_phase
#     # choose vars for specific phase
#     if single_phase == "A"
#         node_colors_sp = node_colors_a
#     elseif single_phase == "B"
#         Vmag = Vph_mag[2,:]
#         # node_colors_sp = node_colors_b
#     elseif single_phase == "C"
#         node_colors_sp = node_colors_c
#     else
#         error("Invalid single phase selection: $single_phase")
#     end
#     # gather data to scatter
#     x_coords_sp = Float64[]
#     y_coords_sp = Float64[]
#     # colors_sp = RGBA{Float64}[]
#     for (nd_ind,Node) in enumerate(psm.Nodes)
#         if occursin(single_phase,Node.phases)
#             push!(x_coords_sp, Node.X_coord)
#             push!(y_coords_sp, Node.Y_coord)
#             # push!(colors_sp, node_colors_sp[nd_ind])
#         end
#     end
#     # create plot
#     vs_plt_sp = plot(layout=grid(1,1), size=(600,600))
#     for (br_ind,Branch) in enumerate(psm.Branches)
#         plot!([Branch.X_coord,Branch.X2_coord],[Branch.Y_coord,Branch.Y2_coord],color=:black)
#     end
#     scatter!(vs_plt_sp, x_coords_sp, y_coords_sp, 
#         # color=Vmag, 
#         markersize=3, legend=false, title="Phase $single_phase",
#         xlabel="X Coordinate (m)", ylabel="Y Coordinate (m)",colorbar=true
#     )
#     # plot!(title="Phase $single_phase",xformatter=:none,yformatter=:none,legend=:false)
# end

# cb_plt = heatmap(rand(2,2), clims=(Vmin,Vmax),  right_margin = 10Plots.mm, framestyle=:none, c=node_colormap, cbar=true, lims=(-1,0),colorbar_title = " \nVoltage Magnitude (pu)")
# display(cb_plt)