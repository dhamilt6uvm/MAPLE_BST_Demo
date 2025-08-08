using PyCall

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

for node in psm.Nodes
    println(node.name, ": ", node.loads)
    for load_ind in node.loads
        println(psm.Loads[load_ind+1])
    end
end