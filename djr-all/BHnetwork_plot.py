import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Get the directory one level up from the script
sys.path.insert(0, parent_dir)  # Add the parent directory to the system path 
import GLM_Tools.PowerSystemModel as psm
import pickle

def plot_feeder(substation_name, save_it):

    # Define nicer colors (RGB triplets)
    col_red  = (0.85, 0.33, 0.10)
    col_blue  = (0.00, 0.45, 0.74)
    col_green = (0.47, 0.67, 0.19)

    # Use a nicer font (Computer Modern is LaTeX-like)
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['mathtext.fontset'] = 'cm'   # Computer Modern
    mpl.rcParams['mathtext.rm'] = 'serif'     # use serif inside mathtext

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Open pkl file
    pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
    with open(pkl_file, 'rb') as file:
        pkl_model = pickle.load(file)

    # Plot branches (black thin lines)
    for Branch in pkl_model.Branches:
        ax.plot([Branch.X_coord, Branch.X2_coord],
                [Branch.Y_coord, Branch.Y2_coord],
                color='black', linewidth=0.2)

    # Plot nodes
    for Node in pkl_model.Nodes:
        ax.plot(Node.X_coord, Node.Y_coord, '.', color=col_blue, markersize=3)

    # Plot loads
    for ld_ind, Load in enumerate(pkl_model.Loads):
        ax.plot(Load.X_coord, Load.Y_coord, 'o', color=col_red,
                markersize=3)

    # Plot generators
    for gen_ind, Generator in enumerate(pkl_model.Generators):
        ax.plot(Generator.X_coord, Generator.Y_coord, 'o',
                 color=col_green, markersize=6, alpha=0.4)
        
    # Head node (first node in list, larger black dot)
    head_node = pkl_model.Nodes[0]
    ax.plot(head_node.X_coord, head_node.Y_coord, '^', color='black', markersize=6)
           
    # Custom legend handles
    node_handle      = mlines.Line2D([], [], color=col_blue, marker='.', linestyle='None', markersize=6, label='Nodes')
    load_handle      = mlines.Line2D([], [], color=col_red, marker='o', linestyle='None', markersize=8, label='Loads')
    gen_handle       = mlines.Line2D([], [], color=col_green, marker='o', linestyle='None', markersize=8, alpha=0.4, label='Generators')
    head_handle      = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=8, label='Head Node')

    plt.legend(handles=[node_handle, load_handle, gen_handle, head_handle], loc='best')

    # remove axes
    ax.set_axis_off()

    # Show figure
    fig.canvas.draw()

    if save_it:
        fig.savefig(f"{substation_name}_feeder_plot.svg", bbox_inches='tight', pad_inches=0)


subsname = "Burton_Hill_small02"

plot_feeder(subsname, save_it=True)