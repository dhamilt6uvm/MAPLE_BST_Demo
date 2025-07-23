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
using Graphs

# Import Python modules
pickle = pyimport("pickle")
pushfirst!(pyimport("sys")."path", "")
pyimport("GLM_Tools")

# Load the .pkl file 
substation_name = "South_Alburgh"
# "South_Alburgh"
fname = "Feeder_Data/$(substation_name)/Python_Model/$(substation_name)_Model.pkl"
pkl_file = open(fname, "r")
psm = pickle.load(pkl_file)
close(pkl_file)


selected_phase = "A"                    # change this to select different phases
if selected_phase == "A"
    col = 1
elseif  selected_phase == "B"
    col = 2
else
    col = 3
end

# --------------------------------------------- #
#               extracting loads                #
# --------------------------------------------- #
loads = psm.Loads

loads_phase = [load.phases for load in loads]

load_selected_phase = [loads[i] for (i, phase) in enumerate(loads_phase) if occursin(selected_phase, phase)]
n_loads = length(load_selected_phase)
load_sample= pybuiltin("vars")(load_selected_phase[1])
n_sample = size(load_sample["Sload"],1)
load_keys = sort(collect(keys(load_sample)))

loading = spzeros(ComplexF64,8387,n_sample)
for i in 1:n_loads
    load = pybuiltin("vars")(load_selected_phase[i])
    id = load["parent_node_ind"] + 1
    if haskey(load, "Sload")
        sload = load["Sload"]
        loading[id, :] .+= sload[:,col]
    elseif haskey(load, "constant_power_A")
        if abs.(load["constant_power_A"]) > 1e-6
            println( abs.(load.constant_power_A))
            println( abs.(load.constant_power_B))
            println( abs.(load.constant_power_C))
        end
    end
end


# --------------------------------------------- #
#               extracting genrations           #
# --------------------------------------------- #
generators = psm.Generators
gens_phase = [gen.phases for gen in generators]
gen_selected_phase = [generators[i] for (i, phase) in enumerate(gens_phase) if occursin(selected_phase, phase)]

n_gens = length(gen_selected_phase)
gen_sample= pybuiltin("vars")(gen_selected_phase[1])
gen_keys = sort(collect(keys(gen_sample)))

generation = spzeros(ComplexF64,8387,n_sample)
for i in 1:n_gens
    gen = pybuiltin("vars")(gen_selected_phase[i])
    id = gen["parent_node_ind"] + 1
    if haskey(gen, "Sgen")
        sgen = gen["Sgen"]
        generation[id, :] .+= sgen[:,col]
    elseif haskey(gen, "constant_power_A")
        if abs.(gen["constant_power_A"]) > 1e-6
            n_sample( abs.(gen.constant_power_A))
            println( abs.(gen.constant_power_B))
            println( abs.(gen.constant_power_C))
        end
    end
end


# --------------------------------------------- #
#               extracting nodes                #
# --------------------------------------------- #
nodes = psm.Nodes               # starts from 0
nodes_phase = [node.phases for node in nodes]
node_selected_phase = [nodes[i] for (i, phase) in enumerate(nodes_phase) if occursin(selected_phase, phase)]
n_nodes = length(node_selected_phase)
node_sample = pybuiltin("vars")(node_selected_phase[1])
node_keys = sort(collect(keys(node_sample)))

X_coords = [node.X_coord for node in node_selected_phase] 
Y_coords = [node.Y_coord for node in node_selected_phase] 

node_ID = [node.index for node in node_selected_phase] .+ 1  # starts from 1, while Dakota's starts from 0
nni = copy(node_ID)

# # --------------------------------------------- #
# #               extracting branches             #
# # --------------------------------------------- #
branches = psm.Branches
branches_phase = [branch.phases for branch in branches]
branch_selected_phase = [branches[i] for (i, phase) in enumerate(branches_phase) if occursin(selected_phase, phase)]

n_branches = length(branch_selected_phase)
branch_sample = pybuiltin("vars")(branch_selected_phase[1])
branch_keys = keys(branch_sample)

Z_single_phase = zeros(ComplexF64,n_branches)
for ii in 1:n_branches
    branch = pybuiltin("vars")(branch_selected_phase[ii])
    phase = branch["phases"]
    if branch["type"] == "switch" && branch["status"] == "OPEN"
        continue
    elseif haskey(branch,"Z_pu_3ph")
        z = branch["Z_pu_3ph"]
        Z_single_phase[ii] = z[col,col]
    else
        println("Branch $ii does not have impedance matrix")
    end
end

branch_types = [branch.type for branch in branch_selected_phase]
from_node    = [branch.from_node_ind for branch in branch_selected_phase] .+1
to_node      = [branch.to_node_ind for branch in branch_selected_phase] .+1

adj  = spzeros(Int64, length(nodes), length(nodes))         # Adjacency
Ybus = spzeros(ComplexF64, length(nodes), length(nodes))    # Admittance matrix 
for ii = 1:n_branches
    
    adj[from_node[ii], to_node[ii]] += 1
    adj[to_node[ii], from_node[ii]] += 1

    if abs(Z_single_phase[ii]) != 0
        Ybus[from_node[ii], to_node[ii]] +=-1/Z_single_phase[ii]
        Ybus[to_node[ii], from_node[ii]] +=-1/Z_single_phase[ii]
    else    # small values for fake-branches                            #### what's a fake branch?
        Ybus[from_node[ii], to_node[ii]] +=-1/(1e-6im)
        Ybus[to_node[ii], from_node[ii]] +=-1/(1e-6im)
    end
end

adj = adj[node_ID, node_ID]
Ybus = Ybus[node_ID, node_ID]
Ybus[diagind(Ybus)] .= -sum(Ybus,dims=2) #############

# Check network connectivity
Degree = sum(adj,dims=2)
islanded = findall(Degree.==0)
parallel_lines = findall(adj .>1)
g = Graph(adj) 
Connection = connected_components(g)
if length(Connection) != 1
    println("The network of phase $(selected_phase) is not connected!!!!!!!")
end

# --------------------------------------------- #
#                    Shunts                     #
# --------------------------------------------- #
shunts = psm.Shunts
shunts_phase = [shunt.phases for shunt in shunts]
shunt_selected_phase = [shunts[i] for (i,phase) in enumerate(shunts_phase) if occursin(selected_phase,phase)]

n_shunts = length(shunt_selected_phase)
shunt_sample = pybuiltin("vars")(shunt_selected_phase[1])
shunt_keys = keys(shunt_sample)

for ii in 1:n_shunts
    shunt = shunt_selected_phase[ii]
    if shunt.type == "capacitor"
        if (shunt.switchA == "CLOSED" && selected_phase == "A") || (shunt.switchB == "CLOSED" && selected_phase == "B")  || (shunt.switchC == "CLOSED" && selected_phase == "C")
            id = findfirst(node_ID .== shunt["parent_node_ind"]+1 )
            Ybus[id, id] += shunt.Ycap[1,1]
        end
    end
end


# --------------------------------------------- #
#                     model                     #
# --------------------------------------------- #
S_inj = generation[node_ID,:] .- loading[node_ID,:]


p1 = plot(abs.(S_inj))

# Load Selection
Load_sum = real(sum(S_inj, dims=1))
high_loads = argmax(Load_sum)[2]
println("High load bus: ", high_loads)
low_loads = argmin(Load_sum)[2]
println("Low load bus: ", low_loads)


## Commented vv
using Statistics
using Plots
using Colors
correlation = cor(real.(S_inj))
p2 = heatmap(correlation, color=:viridis, xlabel="Bus ID", ylabel="Bus ID", title="Load correlation")
display(p2)
least_cor = Tuple.(argmin(correlation))

highlight_points1 = [high_loads, low_loads]
highlight_points2 = [least_cor[1], least_cor[2]]
p3 = plot(Load_sum', label="Load sum")
scatter!(highlight_points1, Load_sum[highlight_points1], color=:red, marker=:circle, markersize=8, label="max and min loads")
scatter!(highlight_points2, Load_sum[highlight_points2], color=:blue, marker=:circle, markersize=8, label="least correlated")
display(p3)


using MAT
data = matopen("coords_phase$selected_phase.mat","w")
write(data,"xcoor",X_coords)
write(data,"ycoor",Y_coords)
close(data)

using GraphPlot
# Plot the graph
edge_cols = [i % 2 == 0 ? colorant"blue" : colorant"red" for i in 1:ne(g)]
gplot(g, X_coords, Y_coords,
    NODESIZE = 0.01,
    EDGELINEWIDTH = 0.1,
    edgestrokec = colorant"red",
    nodefillc   = colorant"blue",
    plot_size        = (10cm, 10cm)
)



## Questions:
# what is the output of this? how do I turn in into a matpower case file? - save variables from in this script to something
# how can I see what the network that I have created looks like? 
# what are fake branches?
# you're extracting loads, is this one particular loading condition at some time stamp? or is there a set? - hourly loads at every node over 1 week
# i will ideally want a set of loads that I can use as a data set to evaluate my model performance - does this have that already?
# Is this already radial and so a substation exists? - always node 1
# how can I count (and then modify) the generators? - there are generators but they're fixed to 0 most likley
# if what I want is a matpower case file, is the easiest path to just extract the raw data and then work in matpower land? 