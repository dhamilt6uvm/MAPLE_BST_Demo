## Optimize k-values using ip opt
##############################################################################
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
@pyimport matplotlib.pyplot as plt
hasattr = pyimport("builtins").hasattr
pickle = pyimport("pickle")
pyopen = pyimport("builtins").open
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")


show_plot = true
show_hists = false
check_global_opt = false

## Load the feeder ######################################################
substation_name = "Burton_Hill_small02"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model.pkl"
pkl_file = pyopen(fname, "rb")
psm = pickle.load(pkl_file)
pkl_file.close()


## Import data ##########################################################
# jacobians for day, night, dP and dQ
dVdP_day, dVdQ_day, dVdP_night, dVdQ_night = deserialize("djr-all/BH_small02_Jacobians01.jls")
# cases to test: x worst voltage deviations
tbl = CSV.File("djr-all/BHsmall02-8760-V-results/Vdiff_idx_BHsmall02.csv")           # extract Sload from worst voltage cases using the saved csv
worst_V_idx = collect(tbl.A)
cases_to_get = 20
# Average initial voltages
tbl = CSV.File("djr-all/exported-csvs/Vph0_day01_BHsmall02.csv")
V0_day = collect(tbl.A)
tbl = CSV.File("djr-all/exported-csvs/Vph0_night01_BHsmall02.csv")
V0_night = collect(tbl.A)
# Average load values
df = CSV.read("djr-all/exported-csvs/S0_day01_BHsmall02.csv", DataFrame)
P0_day = df.x1
Q0_day = df.x2
df = CSV.read("djr-all/exported-csvs/S0_night01_BHsmall02.csv", DataFrame)
P0_night = df.x1
Q0_night = df.x2


## Functions ###########################################################
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


## Extract data ########################################################
# get nodes indices
nodes, n_nodes = get_loadgen_nodes_LO(psm)
ph_col = 2
# Pull load and is_day data from x worst cases
day_hours = 12:24
night_hours = 1:11
is_day_worst = Bool[]
Sload_worst = zeros(ComplexF64, cases_to_get, n_nodes)
for (ii,t_ind) in enumerate(worst_V_idx[1:cases_to_get])
    Sload_worst[ii,:] = get_netload_onetime(psm, nodes, t_ind)
    hour = mod1(t_ind,24)
    if hour in day_hours
        push!(is_day_worst, true)
    elseif hour in night_hours
        push!(is_day_worst, false)
    end
end
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
notgen_idx_in_209 = setdiff(1:n_nodes, gen_idx_in_209)


## Compute relevant V-tildes ###########################################
Vtil_all = zeros(Float64, n_nodes, cases_to_get)
for (ii,is_day) in enumerate(is_day_worst)
    Pvec = real(Sload_worst[ii,:])
    if is_day
        Vtil_all[:,ii] = V0_day + dVdP_day * (Pvec - P0_day) - dVdQ_day * Q0_day
    else
        Vtil_all[:,ii] = V0_night + dVdP_night * (Pvec - P0_night) - dVdQ_night * Q0_night
    end
end


## Optimization ###########################################################
# set up
sftmrg = 1e-3       # safety margin
k0 = -10*ones(n_nodes,1)
k0[notgen_idx_in_209] .= 0
# determine if day or night, set generic vars
ii = 1
if is_day_worst[ii]
    dVdQ = dVdQ_day
else
    dVdQ = dVdQ_night
end    
v_til = Vtil_all[:,ii]
# Define the optimization model
model = Model(Ipopt.Optimizer)
# define variables
@variable(model, kinv[1:n_nodes] <= 0)            # k must be negative
kallD = kinv .* I(n_nodes)                        # diagonal matrix of k values   
@variable(model, vstar[1:n_nodes])                # steady state voltage
@variable(model, x_unit[1:n_nodes])               # unit vector - value not important - needed for 2-norm constraint
# set starting values
set_start_value.(kinv, k0)
set_start_value.(vstar, 1)
set_start_value.(x_unit, 1 ./ sqrt(n_nodes))
# define constraints
@constraint(model, (I(n_nodes)-dVdQ*kallD)*vstar == -dVdQ*kallD*ones(n_nodes,1) + v_til)    # steady state voltage constraint
@constraint(model, kinv[notgen_idx_in_209] .== 0)                                       # set k for non-generator nodes to 0
@constraint(model, (dVdQ*kallD * x_unit)' * (dVdQ*kallD * x_unit) <= 1)         # (XK * x )^T  *  (XK * x)   (basically says if 2-norm squared < 1 then 2-norm less than 1)
@constraint(model, x_unit' * x_unit == 1)                                       # (x)^T  * (x) = 1   (norm of x must be 1)
# define the objective function
@objective(model, Min, sum((vstar - ones(n_nodes)).^2))
# Solve the optimization problem
optimize!(model)

kvals = value.(kinv)
println(kvals[gen_idx_in_209])


## Evaluate steady state voltage ##########################################
function get_final_voltage(k,X,vtil,vref)
    n = size(vref,1)
    K = diagm(k)
    vstar = (I(n) - X*K) \ (vtil - X*K*vref)
    return vstar
end

vstar_all = zeros(Float64, n_nodes, cases_to_get)
for (ii, is_day) in enumerate(is_day_worst)
    vtil = Vtil_all[:,ii]
    if is_day
        vstar_all[:,ii] = get_final_voltage(kvals, dVdQ_day, vtil, ones(n_nodes,1))
    else
        vstar_all[:,ii] = get_final_voltage(kvals, dVdQ_night, vtil, ones(n_nodes,1))
    end
end


## Import the before-vvc steady state voltage ###########################
df = CSV.read("djr-all/BHsmall02-8760-V-results/V_allAMI_BHsmall02.csv", DataFrame)
Vallall = Matrix(df)
V_worst = Vallall[worst_V_idx[1:cases_to_get], nodes]


## Compare #############################################################
VVC_norm = zeros(Float64, cases_to_get, 2)
OG_norm = zeros(Float64, cases_to_get, 2)
vref = ones(n_nodes)
for ii = 1:cases_to_get
    V_vvc = vstar_all[:,ii]
    V_og = V_worst[ii,:]
    VVC_norm[ii, 1] = norm(vref - V_vvc) / sqrt(n_nodes)
    OG_norm[ii, 1] = norm(vref - V_og) / sqrt(n_nodes)
    VVC_norm[ii, 2] = norm(vref - V_vvc, Inf)
    OG_norm[ii, 2] = norm(vref - V_og, Inf)
end
# make plots
n_disp = 10
bar_width = 0.35
x = 1:n_disp
ymax = maximum([maximum(OG_norm), maximum(VVC_norm)])
# plot grouped bars
if show_plot
    # 2-norm
    plt.bar(x .- bar_width/2, OG_norm[1:n_disp,1], width=bar_width, label="No VVC")
    plt.bar(x .+ bar_width/2, VVC_norm[1:n_disp,1], width=bar_width, label="With VVC")
    plt.xticks(x, string.(worst_V_idx[1:n_disp]))                     # position x-ticks at group centers
    plt.xlabel("Hour of Year")
    plt.ylabel("\$\\frac{1}{\\sqrt{n}} \\|1-v\\|_2\$")
    plt.ylim(0,ymax)
    plt.legend()
    plt.title("2-norm")
    plt.show()
    # infinity norm
    plt.bar(x .- bar_width/2, OG_norm[1:n_disp,2], width=bar_width, label="No VVC")
    plt.bar(x .+ bar_width/2, VVC_norm[1:n_disp,2], width=bar_width, label="With VVC")
    plt.xticks(x, string.(worst_V_idx[1:n_disp]))                     # position x-ticks at group centers
    plt.xlabel("Hour of Year")
    plt.ylabel("\$\\|1-v\\|_\\infty\$")
    plt.ylim(0,ymax)
    plt.legend()
    plt.title("\$\\infty\$-norm")
    plt.show()

    # make histogram of voltage profiles ################
    if show_hists
        vmin = min(minimum(V_worst),minimum(vstar_all))
        vmax = max(maximum(V_worst),maximum(vstar_all))
        for jj = 1:n_disp
            v_b4 = V_worst[jj,:]
            v_vvc = vstar_all[:,jj]
            fig, axs = plt.subplots(2, 1, figsize=(6, 6)) 
            axs[1].hist(v_b4, bins=10, alpha=0.7, label="No VVC", color="blue")
            axs[1].set_title("Dist. of Voltages with no VVC")
            axs[1].set_xlabel("Voltage (pu)")
            axs[1].set_ylabel("Frequency")
            axs[1].legend()
            axs[1].set_xlim(vmin, vmax)
            axs[2].hist(v_vvc, bins=10, alpha=0.7, label="VVC", color="orange")
            axs[2].set_title("Dist. of Voltages with VVC")
            axs[2].set_xlabel("Value")
            axs[2].set_ylabel("Frequency")
            axs[2].legend()
            axs[2].set_xlim(vmin, vmax)
            plt.tight_layout()
            plt.show()
        end
    end

    # plot voltage profiles ########
    colors = plt.cm.tab10.colors            # tuple of RGBA from matplotlib colormap
    for jj in 1:n_disp
        c = colors[(jj - 1) % length(colors) + 1]  # pick a color
        x = collect(1:n_nodes)
        v_b4 = V_worst[jj, :]
        v_vvc = vstar_all[:, jj]
        plt.plot(x, v_b4, color=c, label="Samp $jj")
        plt.plot(x, v_vvc, color=c, linestyle="--")
    end
    for idx in gen_idx_in_209
        bar_height = abs(kvals[idx]) / maximum(abs.(kvals)) / 50
        plt.plot([idx,idx],[0.92,0.92+bar_height],linewidth=4, color="red")
    end
    plt.xlabel("Node Index")
    plt.ylabel("Voltage (pu)")
    plt.grid(true)
    plt.show()
    # find nodes with high final voltage
    idx_sus = findall(row -> any(>(1), row), eachrow(vstar_all))
    # repeat plot but with only these nodes
    for jj in 1:n_disp
        c = colors[(jj - 1) % length(colors) + 1]  # pick a color
        x = collect(1:length(idx_sus))
        v_b4 = V_worst[jj, idx_sus]
        v_vvc = vstar_all[idx_sus, jj]
        plt.plot(x, v_b4, color=c, label="Samp $jj")
        plt.plot(x, v_vvc, color=c, linestyle="--")
    end
    plt.xlabel("Node Index")
    plt.ylabel("Voltage (pu)")
    plt.show()
    # Find the actual node numbers that are sus
    nodes_sus = nodes[idx_sus]
end


## check global optimality #########################################
if check_global_opt
    n_test = 10
    kvals_all = zeros(Float64, n_nodes,n_test)
    objvals_all = zeros(Float64, n_test)
    for ii = 1:n_test
        # randomize starting values: k0
        k0 = -2 .* randn(n_nodes)
        k0[notgen_idx_in_209] .= 0

        # Define the optimization model
        model = Model(Ipopt.Optimizer)
        # define variables
        @variable(model, kinv[1:n_nodes] <= 0)            # k must be negative
        kallD = kinv .* I(n_nodes)                        # diagonal matrix of k values   
        @variable(model, vstar[1:n_nodes])                # steady state voltage
        @variable(model, x_unit[1:n_nodes])               # unit vector - value not important - needed for 2-norm constraint
        # set starting values
        set_start_value.(kinv, k0)
        set_start_value.(vstar, 1)
        set_start_value.(x_unit, 1 ./ sqrt(n_nodes))
        # define constraints
        @constraint(model, (I(n_nodes)-dVdQ*kallD)*vstar == -dVdQ*kallD*ones(n_nodes,1) + v_til)    # steady state voltage constraint
        @constraint(model, kinv[notgen_idx_in_209] .== 0)                                       # set k for non-generator nodes to 0
        @constraint(model, (dVdQ*kallD * x_unit)' * (dVdQ*kallD * x_unit) <= 1)         # (XK * x )^T  *  (XK * x)   (basically says if 2-norm squared < 1 then 2-norm less than 1)
        @constraint(model, x_unit' * x_unit == 1)                                       # (x)^T  * (x) = 1   (norm of x must be 1)
        # define the objective function
        @objective(model, Min, sum((vstar - ones(n_nodes)).^2))
        # Solve the optimization problem
        optimize!(model)
        kvals_all[:,ii] = value.(kinv)
        objvals_all[ii] = value(objective_value(model))
    end
    # plot the solutions
    kvals_gens = kvals_all[gen_idx_in_209,:]
    heatmap(
        kvals_gens;
        xlabel = "Rand Init Condition Trial",
        ylabel = "Generator Number",
        title = "IPopt 2-norm constraint globally optimal?",
        clims = (minimum(kvals_gens), maximum(kvals_gens)),          # color limits
        colorbar = true,         # show colorbar
        cbar_title = "slope value",    # colorbar label
        color = :viridis          # color scheme
    )
    println("The objective function values are: ")
    for (ii,val) in enumerate(objvals_all)
        valr = round(val,digits=5)
        println("$ii: $valr")
    end
end

## Check out singular values... #####################################
svals_day = svdvals(dVdQ_day)               # compute singular values of the matrices
svals_night = svdvals(dVdQ_night)