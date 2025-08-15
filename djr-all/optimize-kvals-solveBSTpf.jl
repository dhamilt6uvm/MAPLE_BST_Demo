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

# next: trash this repl and start over to make sure funcs work
# think mostly done with test-linear... keep moving through optimize kvals


"""
Mads suggestions:

Hot-starting - maybe different than an initial guess (or warm start)

Extract IP opt flags 

Also do hot start in IP opt, may also need the dual, can extract from a solution

Confirm that it is actually USING the initial conditions

Make sure that nothing except the starting point is changing from trial to trial

Plug in trial 9 and see fmincon should go FAST

Try a 0 initial condition

Set all init conditions to -10, -20, -30… 

What does fmincon do when fed with infeasible solution"""