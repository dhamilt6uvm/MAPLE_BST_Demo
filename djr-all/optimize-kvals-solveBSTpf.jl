## Process psm data, find the optimal k value, simulate with BST to find final voltages
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
pickle = pyimport("pickle")
pyopen = pyimport("builtins").open
plt = pyimport("matplotlib.pyplot")
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")
using Revise
include("misc-funcs.jl")
using .AllFuncs
## Set up for function use: 
V0_mag = 1                          # substation voltage
V0_ref = V0_mag*[1,exp(-im*2*pi/3),exp(im*2*pi/3)]
linear_solver = "mumps"


## Modify these things as needed: 
n_scenario = 10


## Load the feeders ######################################################
ph_col = 2          # phase B only - specific to this feeder
substation_name = "Burton_Hill_small02"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model.pkl"
pkl_file = pyopen(fname, "rb")
psm = pickle.load(pkl_file)         # true model with AMI
pkl_file.close()
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model_DAY1.pkl"
pkl_file = pyopen(fname, "rb")
psm_day = pickle.load(pkl_file)     # fake data for day averages jacobian calculating
pkl_file.close()
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model_NIGHT1.pkl"
pkl_file = pyopen(fname, "rb")
psm_night = pickle.load(pkl_file)   # fake data for night averages jacobian
pkl_file.close()

## Shift PSMs for day and night so that t_ind=1 is the average
# determine epsilon (which was used to calculate jacobian numerically)
epsilon = abs(psm_day.Loads[1].Sload[1,ph_col] - psm_day.Loads[1].Sload[2,ph_col])
# subtract epsilon from first load in both
psm_day.Loads[1].Sload[1,ph_col] -= epsilon
psm_night.Loads[1].Sload[1,ph_col] -= epsilon


## Import data ##########################################################
# jacobians for day, night, dP and dQ
dVdP_day, dVdQ_day, dVdP_night, dVdQ_night = deserialize("djr-all/BH_small02_Jacobians01.jls")
# cases to test: x worst voltage deviations
tbl = CSV.File("djr-all/BHsmall02-8760-V-results/Vdiff_idx_BHsmall02.csv")           # extract Sload from worst voltage cases using the saved csv
worst_V_idx = collect(tbl.A)


## Get network info #####################################################
nodes, n_nodes = get_loadgen_nodes_LO(psm_day)
# find initial conditions of injections (day/night averages) from psm's - will be used in linearized calculation
Sload0_day = get_netload_onetime(psm_day, nodes, 1, ph_col)
Sload0_night = get_netload_onetime(psm_night, nodes, 1, ph_col)
# find initial voltages at these averaged conditions 
Vph0_day = solve_pf(psm_day, V0_ref, 1, linear_solver)
Vph0_night = solve_pf(psm_night, V0_ref, 1, linear_solver)
# get list of generators
gen_idx_in_209 = get_gen_idx(psm, nodes)
notgen_idx_in_209 = setdiff(1:n_nodes, gen_idx_in_209)
# convert things to single phase and magnitudes
V0_day = abs.(Vph0_day[ph_col,nodes])
V0_night = abs.(Vph0_night[ph_col,nodes])
P0_day = real(Sload0_day)
P0_night = real(Sload0_night)
Q0_day = imag(Sload0_day)
Q0_night = imag(Sload0_night)



## Deal with the test cases ###########################################
# determine if they're day or night
day_hours = 12:24
night_hours = 1:11
is_day = get_is_day(worst_V_idx[1:n_scenario], day_hours, night_hours)
# find the load values
Sload_worst = zeros(ComplexF64, n_scenario, n_nodes)
for (ii,t_ind) in enumerate(worst_V_idx[1:n_scenario])
    Sload_worst[ii,:] = get_netload_onetime(psm, nodes, t_ind, ph_col)
end


## Compute V-tilde ###################################################
# use only 1 scenario to find only one set of k's
xth_worst = 1
Pvec = real(Sload_worst[xth_worst,:])
if is_day[xth_worst]
    Vtil = V0_day + dVdP_day * (Pvec - P0_day) - dVdQ_day * Q0_day
    dVdQ = dVdQ_day
else
    Vtil = V0_night + dVdP_night * (Pvec - P0_night) - dVdQ_night * Q0_night
    dVdQ = dVdQ_night
end


## Optimization ###########################################################
# set up
sftmrg = 1e-3       # safety margin
# set initial condition - largest(ish) possible while being feasible
factors = -0.1:-0.1:-3
for fac in factors
    k0_atmpt = fac*ones(n_nodes)
    k0_atmpt[notgen_idx_in_209] .= 0
    XKmat = dVdQ * (Diagonal(k0_atmpt))
    if norm(XKmat) < 1
        k0 = k0_atmpt
    else
        break
    end
end 
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
@constraint(model, (I(n_nodes)-dVdQ*kallD)*vstar == -dVdQ*kallD*ones(n_nodes,1) + Vtil)    # steady state voltage constraint
@constraint(model, kinv[notgen_idx_in_209] .== 0)                                       # set k for non-generator nodes to 0
@constraint(model, (dVdQ*kallD * x_unit)' * (dVdQ*kallD * x_unit) <= 1)         # (XK * x )^T  *  (XK * x)   (basically says if 2-norm squared < 1 then 2-norm less than 1)
@constraint(model, x_unit' * x_unit == 1)                                       # (x)^T  * (x) = 1   (norm of x must be 1)
# define the objective function
@objective(model, Min, sum((vstar - ones(n_nodes)).^2))
# Solve the optimization problem
optimize!(model)
kvals = value.(kinv)
kvals = round.(kvals, digits=5)
maxk = maximum(abs.(kvals))
obj_val = objective_value(model)   # function value at optimum
println("Found optimal k values with max $maxk")
println("Objective function value: $obj_val")



#############################################################################
## Simulate the system with BST and VVC #####################################
#############################################################################

## Get basic info ###########################################################
n_nodes_psm = length(psm.Nodes)

## VVC function ############################################################
function VVC_output(k, v, nodes)
    v_nodes = v[nodes]
    Q_out = k .* v_nodes - k
    return Q_out
end

## Modify scenario to match net-load convention ############################
# converts any duplicate loads and gen values at the specified t_ind to 
# be the net load (all loads at a node except the first are 0 and all
# gens are 0)
write_netload_onetime(psm, nodes, worst_V_idx[1], Sload_worst[1,:])
# solve it for initial voltage
V_tmp = solve_pf(psm, V0_ref, worst_V_idx[1], linear_solver)
V_out = abs.(V_tmp[ph_col,:])
V_init = V_out          # need to hold onto it for later


## Iterative simulation #####################################################
# track voltage and Q
Q_track = Array{Float64}(undef, n_nodes, 0)
V_track = Array{Float64}(undef, n_nodes_psm, 0)
# set up loop
V_diff = 1e6
iter_max = 1e2
thresh = 1e-4
iter = 0
V_last = zeros(n_nodes_psm)
# loop
while iter < iter_max && V_diff > thresh
    # VVC - compute Q output
    Q_out = VVC_output(kvals, V_out, nodes)
    # write Q into psm
    for (ii,node) in enumerate(psm.Nodes[nodes])
        load_ind = node.loads[1] + 1                    # FIX THIS PLUS ONE BULLLLLLLLLLLL IT'S EVERYHWERE AWWEEAHHHHHASDLFKLHAHHHAHAAAAA
        psm.Loads[load_ind].Sload[worst_V_idx[1],ph_col] -= Q_out[ii]*1im
    end
    # resolve psm for voltage
    V_tmp = solve_pf(psm, V0_ref, worst_V_idx[1], linear_solver)
    V_out = abs.(V_tmp[ph_col,:])
    # update exit condtions
    V_diff = norm(V_out .- V_last)
    V_last = V_out
    iter += 1
    # store tracking vars
    Q_track = hcat(Q_track, Q_out)
    V_track = hcat(V_track, V_out)
end
# tack initial conditions to tracking
V_track = hcat(V_init, V_track)
Q_track = hcat(zeros(n_nodes), Q_track)


## Plot the voltages and Q outputs ##################################
iters = axes(Q_track, 2)   # iterations
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
# Plot voltages
for node in nodes
    axs[1].plot(iters, V_track[node, :])
end
axs[1].set_title("Voltage at each node")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Voltage")
# Plot Q outputs
for i in axes(Q_track, 1)
    axs[2].plot(iters, Q_track[i, :])
end
axs[2].set_title("Reactive power at each node")
axs[2].set_xlabel("Iteration")
axs[2].set_ylabel("Q output")
plt.tight_layout()
plt.show()


# next steps - look for bugs and figure out why the response is so small


# """
# Mads suggestions:

# Hot-starting - maybe different than an initial guess (or warm start)

# Extract IP opt flags 

# Also do hot start in IP opt, may also need the dual, can extract from a solution

# -C-o-n-f-i-r-m- -t-hat it is actually USING the initial conditions

# -M-a-k-e- -s-u-r-e- -t-hat nothing except the starting point is changing from trial to trial

# -P-l-u-g- -i-n- -t-r-i-al 9 and see fmincon should go FAST

# -T-r-y- -a- -0- initial condition

# -S-e-t- -a-l-l- -i-n-it conditions to -10, -20, -30… 

# What does fmincon do when fed with infeasible solution"""