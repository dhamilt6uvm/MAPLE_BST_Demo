import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Get the directory one level up from the script
sys.path.insert(0, parent_dir)  # Add the parent directory to the system path 
import GLM_Tools.PowerSystemModel as psm
import pickle

substation_name = "Burton_Hill_small02"
target_coords = (1694258.0, 794284.0)

# Open pkl file
pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
with open(pkl_file, 'rb') as file:
    pkl_model = pickle.load(file)

min_dist = 1e8
for Node in pkl_model.Nodes:
    node_coords = (Node.X_coord, Node.Y_coord)
    dist = ((node_coords[0] - target_coords[0])**2 + (node_coords[1] - target_coords[1])**2)**0.5
    if dist < min_dist:
        min_dist = dist
        closest_node = Node

print(f"Closest node is {closest_node.name} at coordinates ({closest_node.X_coord}, {closest_node.Y_coord})")