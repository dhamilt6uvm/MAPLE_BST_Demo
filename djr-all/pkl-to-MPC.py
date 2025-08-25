# Convert pkl power-system model to a matpower case file that is a text file
import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Get the directory one level up from the script
sys.path.insert(0, parent_dir)  # Add the parent directory to the system path 
import GLM_Tools.PowerSystemModel as psm
import pickle


# set up
t_ind = 8561        # time index to grab (0-indexed)
ph_col = 1          # phase column to grab (0-indexed)
vmin = 0.8          # min voltage at buses
vmax = 1.2          # max
Pmax = 100          # max active p for generators
Qmax = 100          # max Q
Qmin = -100         # min
Sbase_MPmw = 1      # units of MW!!!
Sbase_MP = Sbase_MPmw * 1e6     # units of watts

# matpower case file name
feeder_name = "BH_small02"
MPC_file = f"case_{feeder_name}_{t_ind+1}"

#pkl file name
substation_name = "Burton_Hill_small02"
pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"

export_casefile = True
export_loads = True


# load pickle model
with open(pkl_file, 'rb') as file:
        pkl_model = pickle.load(file)

# Model info needed later
Sbase_BST_1ph = pkl_model.Sbase_1ph
ngens = len(pkl_model.Generators) + 1

if export_casefile:
    ## Write the top of the file
    with open(MPC_file, 'w') as file:
        file.write(f"function mpc = {MPC_file}\n")
        file.write("%% MATPOWER Case Format : Version 2\n")
        file.write("mpc.version = '2';\n\n")
        file.write("%%-----  Power Flow Data  -----%%\n")
        file.write("%% system MVA base\n")
        file.write(f"mpc.baseMVA = {Sbase_MPmw};\n\n")

        ## Bus matrix header
        file.write("%% bus data\n")
        file.write("%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin\n")
        file.write("mpc.bus = [\n")
        ## Bus matrix data
        for node in pkl_model.Nodes:
            idx = node.index + 1            # node index (+1 because of substation at 1 and 0-indx python)
            if idx == 1: 
                node_type = 3
            else:
                node_type = 1 
            Pd = 0                          # active power demand (init to 0, if has loads then add to it)
            Qd = 0                          # reactive
            if len(node.loads) != 0:     
                for load_ind in node.loads:
                    load = pkl_model.Loads[load_ind]
                    if hasattr(load, "Sload"):
                        Pd += np.real(load.Sload[t_ind, ph_col])
                        Qd += np.imag(load.Sload[t_ind, ph_col])
            # Convert Pd and Qd to MW and round
            # Pd = round(Pd * Sbase_BST_1ph / 1e6, 8)     # units of megawatts
            # Qd = round(Qd * Sbase_BST_1ph / 1e6, 8)     # units of megawatts
            Pd *= Sbase_BST_1ph / 1e6     # units of megawatts
            Qd *= Sbase_BST_1ph / 1e6     # units of megawatts
            kVbase = node.Vbase / 1000 * np.sqrt(3)     # base voltage in units KV (l2l) (node.Vbase is l2n)
            file.write(f"{idx}\t{node_type}\t{Pd}\t{Qd}\t0\t0\t1\t1\t0\t{kVbase}\t1\t{vmax}\t{vmin};\n")
        ## end of bus matrix
        file.write("];\n\n")

        ## Generator Matrix header
        file.write("%% generator data\n")
        file.write("%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf\n")
        file.write("mpc.gen = [\n")
        str_12zeros = "0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0"
        # first line is the "substation" - need to artificially add first node as a generator
        file.write(f"1\t0\t0\t{Qmax}\t{Qmin}\t1\t0\t1\t{Pmax}\t{str_12zeros};\n")
        ## Generator Matrix data
        for node in pkl_model.Nodes:
            idx = node.index + 1
            Pg = 0                          # active power generation (init to 0, if has loads then add to it)
            Qg = 0                          # reactive
            if len(node.gens) != 0:     
                for gen_ind in node.gens:
                    gen = pkl_model.Generators[gen_ind]
                    if hasattr(gen, "Sgen"):
                        Pg += np.real(gen.Sgen[t_ind, ph_col])     # negative sign because generation sign convention
                        Qg += np.imag(gen.Sgen[t_ind, ph_col])
                # Convert Pg and Qg to MVA and round
                # Pg = round(Pg * Sbase_BST_1ph / 1e6, 8)     # units of megawatts
                # Qg = round(Qg * Sbase_BST_1ph / 1e6, 8)     # units of megawatts
                Pg *= Sbase_BST_1ph / 1e6     # units of megawatts
                Qg *= Sbase_BST_1ph / 1e6     # units of megawatts
                # only write if node has gens
                file.write(f"{idx}\t{Pg}\t{Qg}\t{Qmax}\t{Qmin}\t1\t0\t1\t{Pmax}\t{str_12zeros};\n")
        # end of gens matrix
        file.write("];\n\n")

        ## Branch data matrix header
        file.write("%% branch data\n")
        file.write("%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax\n")
        file.write("mpc.branch = [\n")
        str_branch_end = "0\t0\t0\t0\t0\t0\t1\t-360\t360"
        ## Branch matrix data
        for branch in pkl_model.Branches:
            fbus = branch.from_node_ind + 1
            tbus = branch.to_node_ind + 1
            r_pu_BST = np.real(branch.Z.item())     # resistance in bst_pu
            x_pu_BST = np.imag(branch.Z.item())     # reactance in bst_pu
            # r_pu_MP = round(Sbase_MP / (Sbase_BST_1ph) * r_pu_BST, 8)           # originally had (3*Sbase_BST_1ph) but needed a *3 somewhere, not sure why
            # x_pu_MP = round(Sbase_MP / (Sbase_BST_1ph) * x_pu_BST, 8)
            r_pu_MP = Sbase_MP / (Sbase_BST_1ph) * r_pu_BST           # originally had (3*Sbase_BST_1ph) but needed a *3 somewhere, not sure why
            x_pu_MP = Sbase_MP / (Sbase_BST_1ph) * x_pu_BST
            file.write(f"{fbus}\t{tbus}\t{r_pu_MP}\t{x_pu_MP}\t{str_branch_end};\n")
        ## end of branch matrix
        file.write("];\n\n")

        ## GenCost Matrix header
        file.write("%%-----  OPF Data  -----%%\n")
        file.write("%% generator cost data\n")
        file.write("%	1	startup	shutdown	n	x1	y1	...	xn	yn\n")
        file.write("%	2	startup	shutdown	n	c(n-1)	...	c0\n")
        file.write("mpc.gencost = [\n")
        ## Gencost matrix data
        for ii in range(ngens):
            file.write("2\t0\t0\t3\t0\t1\t0;\n")
        ## end of gencost matrix
        file.write("];\n\n")


## If need exported load files
if export_loads:
    # get info from model
    n_time = pkl_model.Loads[1].Sload.shape[0]
    n_nodes = len(pkl_model.Nodes)
    n_gens = len(pkl_model.Generators)
    gen_ct = 0
    # init matrices
    Pd_all = np.zeros([n_nodes, n_time])
    Qd_all = np.zeros([n_nodes, n_time])
    Pg_all = np.zeros([n_gens, n_time])
    Qg_all = np.zeros([n_gens, n_time])

    # loop through nodes and extract data
    for (ii,node) in enumerate(pkl_model.Nodes):

        # Build loads matrices
        Pd, Qd = np.zeros(n_time), np.zeros(n_time)      # (init to 0, if has loads then add to it)
        if len(node.loads) != 0:     
            for load_ind in node.loads:
                load = pkl_model.Loads[load_ind]
                if hasattr(load, "Sload"):
                    Pd += np.real(load.Sload[:, ph_col])
                    Qd += np.imag(load.Sload[:, ph_col])
        Pd_all[ii, :] = Pd.ravel()
        Qd_all[ii, :] = Qd.ravel()

        # Build Gens matrices
        Pg, Qg = np.zeros(n_time), np.zeros(n_time)      # (init to 0, if has loads then add to it)
        if len(node.gens) != 0:     
            for gen_ind in node.gens:
                gen = pkl_model.Generators[gen_ind]
                if hasattr(gen, "Sgen"):
                    Pg += np.real(gen.Sgen[:, ph_col])     # negative sign because generation sign convention
                    Qg += np.imag(gen.Sgen[:, ph_col])
            Pg_all[gen_ct,:] = Pg.ravel()
            Qg_all[gen_ct,:] = Qg.ravel()
            gen_ct += 1


    # add row of zeros for substation
    Pg_all = np.vstack([np.zeros((1,n_time)), Pg_all])
    Qg_all = np.vstack([np.zeros((1,n_time)), Qg_all])

    # convert everything to MW / MVAr
    Pd_all *= Sbase_BST_1ph / 1e6
    Qd_all *= Sbase_BST_1ph / 1e6
    Pg_all *= Sbase_BST_1ph / 1e6
    Qg_all *= Sbase_BST_1ph / 1e6

    # export matrices
    np.savetxt(f"{feeder_name}_Pd_all.csv", Pd_all, delimiter=",")
    np.savetxt(f"{feeder_name}_Qd_all.csv", Qd_all, delimiter=",")
    np.savetxt(f"{feeder_name}_Pg_all.csv", Pg_all, delimiter=",")
    np.savetxt(f"{feeder_name}_Qg_all.csv", Qg_all, delimiter=",")
