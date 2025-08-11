import numpy as np

def convert_phases(val,from_phases,to_phases):
    if ('N' in from_phases) or ('D' in from_phases):
        from_phases = from_phases[:-1] # remove N or D
    if ('N' in to_phases) or ('D' in to_phases):
        to_phases = to_phases[:-1] # remove N or D
    M = np.zeros((len(to_phases),len(from_phases)))
    for t_ind, char in enumerate(to_phases):
        if char in from_phases:
            f_ind = from_phases.find(char)
            M[t_ind,f_ind] = 1.0
    if val.ndim == 1: # if a vector
        return M @ val
    elif val.ndim == 2: # if a matrix
        return M @ val @ M.T
    else:
        raise ValueError(f"convert_phases not supported for values of dimension {val.ndim}")
    
def find_node_parent(node_name,Node_Dict,Shunt_Dict):
    max_find_parent_tries = 10
    find_parent_tries = 0
    while (node_name not in Node_Dict) and (find_parent_tries < max_find_parent_tries):
        if node_name in Shunt_Dict:
            shunt_obj = Shunt_Dict[node_name]
            node_name = shunt_obj.parent
        else:
            raise ValueError(f"Could not find node object: {node_name}")
        find_parent_tries += 1
    if find_parent_tries == max_find_parent_tries:
        raise ValueError(f"Failed to find parent node of {node_name} in {max_find_parent_tries} tries.")
    return node_name


class PowerSystemModel:
    def __init__(self, Nodes, Branches, Loads, Generators, Shunts, Configs):
        self.Sbase_3ph = 1e6 # in VA
        self.Sbase_1ph = self.Sbase_3ph/3 # in VA
        self.default_resistance = 1e-4

        self.Nodes = Nodes
        self.Branches = Branches
        self.Loads = Loads
        self.Generators = Generators
        self.Shunts = Shunts
        self.Configs = Configs

        self.glm_header = ""
        self.glm_helics_obj = ""
        self.glm_misc_objs = ""

        for ind, node in enumerate(self.Nodes):
            node.index = ind
            node.outgoing_branches = []
            node.incoming_branches = []
            node.child_nodes = []
            node.loads = []
            node.gens = []
            node.shunts = []
        

        self.Node_Dict = {obj.name: obj for obj in self.Nodes}

        # check for parented nodes, and create fake branches for them
        for ind, node in enumerate(self.Nodes):
            if hasattr(node,"parent"):
                parent_node = self.Node_Dict[node.parent]
                parent_index = parent_node.index
                node.parent_node_ind = parent_index
                self.Nodes[parent_index].child_nodes.append(node.index)
                # update Node Dict
                self.Node_Dict[parent_node.name] = self.Nodes[parent_index]
                self.Node_Dict[node.name] = node
                # add a fake branch
                fake_branch = Branch("fake",f"fake_branch_{node.name}",parent_node.name,node.name,node.phases,[],"")
                self.Branches.append(fake_branch)

        for ind, branch in enumerate(self.Branches):
            branch.index = ind
        for ind, load in enumerate(self.Loads):
            load.index = ind
        for ind, gen in enumerate(self.Generators):
            gen.index = ind
        for ind, shunt in enumerate(self.Shunts):
            shunt.index = ind
        for ind, config in enumerate(self.Configs):
            config.index = ind

        for load in self.Loads:
            parent_name = load.parent
            parent_node = self.Node_Dict[parent_name]
            load.parent_node_ind = parent_node.index
            # update load list in parent node
            parent_node.loads.append(load.index)
            # convert Sload to per unit
            # load.Sload = load.Sload/self.Sbase_1ph

        for gen in self.Generators:
            parent_name = gen.parent
            parent_node = self.Node_Dict[parent_name]
            gen.parent_node_ind = parent_node.index
            # update gen list in parent node
            parent_node.gens.append(gen.index)
            # convert Sgen to per unit
            # load.Sgen = load.Sgen/self.Sbase_1ph

        for shunt in self.Shunts:
            parent_name = shunt.parent
            parent_node = self.Node_Dict[parent_name]
            shunt.parent_node_ind = parent_node.index
            # update shunt list in parent node
            parent_node.shunts.append(shunt.index)

        self.Load_Dict = {obj.name: obj for obj in self.Loads}
        self.Generator_Dict = {obj.name: obj for obj in self.Generators}
        self.Shunt_Dict = {obj.name: obj for obj in self.Shunts}
        self.Config_Dict = {obj.name: obj for obj in self.Configs}

        for branch in self.Branches:
            from_node_name = branch.from_node
            to_node_name = branch.to_node
            if from_node_name not in self.Node_Dict:
                from_node_name = find_node_parent(from_node_name,self.Node_Dict,self.Shunt_Dict)
            if to_node_name not in self.Node_Dict:
                to_node_name = find_node_parent(to_node_name,self.Node_Dict,self.Shunt_Dict)
            from_node = self.Node_Dict[from_node_name]
            to_node = self.Node_Dict[to_node_name]
            branch.from_node_ind = from_node.index
            branch.to_node_ind = to_node.index
            from_node.outgoing_branches.append(branch.index)
            to_node.incoming_branches.append(branch.index)

        self.Branch_Dict = {obj.name: obj for obj in self.Branches}


    def compute_impedances(self):
        for branch in self.Branches:
            if branch.type in ["overhead_line", "underground_line"]:
                config_name = branch.config
                if config_name not in self.Config_Dict:
                    raise ValueError(f"Could not find line config object: {config_name}")
                branch_config = self.Config_Dict[config_name] 
                Z_ohms_per_mi = np.array([[branch_config.z11,branch_config.z12,branch_config.z13],
                                              [branch_config.z21,branch_config.z22,branch_config.z23],
                                              [branch_config.z31,branch_config.z32,branch_config.z33]])
                mi_per_foot = 1/5280
                branch.Z_ohms_3ph = branch.length*mi_per_foot*Z_ohms_per_mi 
                # convert to pu
                from_node_name = branch.from_node
                to_node_name = branch.to_node
                if from_node_name not in self.Node_Dict:
                    from_node_name = find_node_parent(from_node_name,self.Node_Dict,self.Shunt_Dict)
                if to_node_name not in self.Node_Dict:
                    to_node_name = find_node_parent(to_node_name,self.Node_Dict,self.Shunt_Dict)
                from_node = self.Node_Dict[from_node_name]
                to_node = self.Node_Dict[to_node_name]
                if from_node.Vbase != to_node.Vbase:
                    raise ValueError(f"Nominal voltages for nodes {from_node_name} ({from_node.Vbase}) and {to_node_name} ({to_node.Vbase}) are not equal, but line {branch.name} connects them!")
                Vbase_ln = from_node.Vbase
                branch.Ibase = self.Sbase_1ph/Vbase_ln
                branch.Zbase = Vbase_ln/branch.Ibase
                branch.Z_pu_3ph = branch.Z_ohms_3ph/branch.Zbase
                branch.Z = convert_phases(branch.Z_pu_3ph,"ABCN",branch.phases)
                # calculate A, B, C, D matrices from Kersting
                # V_ABC = A*V_abc + B*I_abc
                # I_ABC = C*V_abc + D*I_abc
                branch.A_br = np.eye(3)
                branch.B_br = branch.Z_pu_3ph
                branch.C_br = np.zeros(3)
                branch.D_br = np.eye(3)
            if branch.type in ["transformer"]:
                config_name = branch.config
                if config_name not in self.Config_Dict:
                    raise ValueError(f"Could not find line config object: {config_name}")
                branch_config = self.Config_Dict[config_name] 
                # get turns ratio
                from_node_name = branch.from_node
                to_node_name = branch.to_node
                if from_node_name not in self.Node_Dict:
                    from_node_name = find_node_parent(from_node_name,self.Node_Dict,self.Shunt_Dict)
                if to_node_name not in self.Node_Dict:
                    to_node_name = find_node_parent(to_node_name,self.Node_Dict,self.Shunt_Dict)
                from_node = self.Node_Dict[from_node_name]
                to_node = self.Node_Dict[to_node_name]
                branch.Vbp_ln = from_node.Vbase
                branch.Vbs_ln = to_node.Vbase
                branch.Vp_rated = branch_config.primary_voltage
                branch.Vs_rated = branch_config.secondary_voltage
                branch.nt_pu = (branch.Vp_rated/branch.Vs_rated)/(branch.Vbp_ln/branch.Vbs_ln)
                # convert impedance to pu
                branch.ratedKVA = branch_config.power_rating
                r_pu = (self.Sbase_1ph/(1000*branch.ratedKVA))*branch_config.resistance
                x_pu = (self.Sbase_1ph/(1000*branch.ratedKVA))*branch_config.reactance
                branch.Z_pu_3ph = np.zeros((3,3),dtype=complex)
                for ph in branch.phases:
                    if ph == 'A':
                        branch.Z_pu_3ph[0,0] = complex(r_pu,x_pu)
                    if ph == 'B':
                        branch.Z_pu_3ph[1,1] = complex(r_pu,x_pu)
                    if ph == 'C':
                        branch.Z_pu_3ph[2,2] = complex(r_pu,x_pu)
                branch.Z = convert_phases(branch.Z_pu_3ph,"ABCN",branch.phases)
                # calculate A, B, C, D matrices from Kersting
                # V_ABC = A*V_abc + B*I_abc
                # I_ABC = C*V_abc + D*I_abc
                if branch_config.connect_type in ["SINGLE_PHASE","WYE_WYE"]:
                    branch.A_br = branch.nt_pu*np.eye(3)
                    branch.B_br = branch.nt_pu*branch.Z_pu_3ph
                    branch.C_br = np.zeros(3)
                    branch.D_br = np.eye(3)/branch.nt_pu
                elif branch_config.connect_type in ["DELTA_GWYE"]:
                    branch.A_br = -branch.nt_pu*np.array([[0,2,1],[1,0,2],[2,1,0]])/np.sqrt(3)
                    branch.B_br = branch.nt_pu*np.dot(branch.A_br,branch.Z_pu_3ph)
                    branch.C_br = np.zeros(3)
                    branch.D_br = np.array([[1,-1,0],[0,1,-1],[-1,0,1]])/(np.sqrt(3)*branch.nt_pu)
                elif branch_config.connect_type in ["DELTA_DELTA"]:
                    # branch.A_br = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])/3
                    branch.A_br = branch.nt_pu*np.eye(3)
                    branch.B_br = branch.nt_pu*complex(r_pu,x_pu)*np.array([[1,0,0],[0,1,0],[-1,-1,0]])
                    branch.C_br = np.zeros(3)
                    branch.D_br = np.eye(3)/branch.nt_pu
                else:
                    raise ValueError(f"Connection type {branch_config.connect_type} not yet supported for branch {branch.name}.")
            if branch.type in ["fuse", "switch", "sectionalizer", "recloser", "regulator"]:
                branch.Z_ohms_3ph = np.zeros((3,3),dtype=complex)
                for ph in branch.phases:
                    if ph == 'A':
                        branch.Z_ohms_3ph[0,0] = complex(self.default_resistance,0.0)
                    if ph == 'B':
                        branch.Z_ohms_3ph[1,1] = complex(self.default_resistance,0.0)
                    if ph == 'C':
                        branch.Z_ohms_3ph[2,2] = complex(self.default_resistance,0.0)
                # convert to pu
                from_node_name = branch.from_node
                to_node_name = branch.to_node
                if from_node_name not in self.Node_Dict:
                    if from_node_name in self.Shunt_Dict:
                        shunt_obj = self.Shunt_Dict[from_node_name]
                        from_node_name = shunt_obj.parent
                    else:
                        raise ValueError(f"Could not find node object: {from_node_name}")
                if to_node_name not in self.Node_Dict:
                    if to_node_name in self.Shunt_Dict:
                        shunt_obj = self.Shunt_Dict[to_node_name]
                        to_node_name = shunt_obj.parent
                    else:
                        raise ValueError(f"Could not find node object: {from_node_name}")
                from_node = self.Node_Dict[from_node_name]
                to_node = self.Node_Dict[to_node_name]
                if from_node.Vbase != to_node.Vbase:
                    raise ValueError(f"Nominal voltages for nodes {from_node_name} ({from_node.Vbase}) and {to_node_name} ({to_node.Vbase}) are not equal, but line {branch.name} connects them!")
                Vbase_ln = from_node.Vbase
                branch.Ibase = self.Sbase_1ph/Vbase_ln
                branch.Zbase = Vbase_ln/branch.Ibase
                branch.Z_pu_3ph = branch.Z_ohms_3ph/branch.Zbase
                branch.Z = convert_phases(branch.Z_pu_3ph,"ABCN",branch.phases)
                # calculate A, B, C, D matrices from Kersting
                # V_ABC = A*V_abc + B*I_abc
                # I_ABC = C*V_abc + D*I_abc
                branch.A_br = np.eye(3)
                branch.B_br = branch.Z_pu_3ph
                branch.C_br = np.zeros(3)
                branch.D_br = np.eye(3)
            if branch.type in ["fake"]:
                branch.Z_ohms_3ph = np.zeros((3,3),dtype=complex)
                branch.Z_pu_3ph = np.zeros((3,3),dtype=complex)
                branch.Z = convert_phases(branch.Z_pu_3ph,"ABCN",branch.phases)
                # calculate A, B, C, D matrices from Kersting
                # V_ABC = A*V_abc + B*I_abc
                # I_ABC = C*V_abc + D*I_abc
                branch.A_br = np.eye(3)
                branch.B_br = branch.Z_pu_3ph
                branch.C_br = np.zeros(3)
                branch.D_br = np.eye(3)
        for shunt in self.Shunts:
            if shunt.type in ["capacitor"]:
                Vbase = shunt.base_voltage
                shunt.Ibase = self.Sbase_1ph/Vbase
                shunt.Ybase = shunt.Ibase/Vbase
                shunt.Ycap = np.zeros((3,3),dtype=complex)
                if shunt.phases_connected[-1] == "N":
                    for ph in shunt.phases:
                        if ph == 'A':
                            shunt.Ycap[0,0] = complex(0.0,(shunt.capacitor_A/(Vbase**2))/shunt.Ybase)
                        if ph == 'B':
                            shunt.Ycap[1,1] = complex(0.0,(shunt.capacitor_B/(Vbase**2))/shunt.Ybase)
                        if ph == 'C':
                            shunt.Ycap[2,2] = complex(0.0,(shunt.capacitor_C/(Vbase**2))/shunt.Ybase)
                else:
                    raise ValueError(f"Delta-connected cap banks not currently supported. ({shunt.name})")

    def __repr__(self):
        return f"PowerSystemModel(Nodes={len(self.Nodes)},Branches={len(self.Branches)},Loads={len(self.Loads)},Generators={len(self.Generators)},Shunts={len(self.Shunts)})"

class Node:
    def __init__(self, name, phases, nom_volt, node_type, parent, glm_string):
        self.name = name
        self.phases = phases
        self.Vbase = nom_volt # line-to-neutral, in V
        self.node_type = node_type
        if parent is not None:
            self.parent = parent
        self.glm_string = glm_string

    def __repr__(self):
        return f"Node(name={self.name},phases={self.phases},Vbase={self.Vbase},node_type={self.node_type})"
    
class Branch:
    def __init__(self, type, name, from_node, to_node, phases, params, glm_string):
        self.type = type
        self.name = name
        self.from_node = from_node
        self.to_node = to_node
        self.phases = phases
        self.glm_string = glm_string

        if self.type in ["overhead_line","underground_line"]:
            self.config = params[0] 
            self.length = params[1] # in ft
        elif self.type in ["transformer"]:
            self.config = params[0] 
        elif self.type in ["fuse"]:
            self.current_limit = params[0] 
            self.mean_replacement_time = params[1] 
            self.repair_dist_type = params[2] 
        elif self.type in ["switch"]:
            self.status = params[0] 
        elif self.type in ["recloser"]:
            self.max_number_of_tries = params[0] 
        elif self.type in ["regulator"]:
            self.config = params[0] 
            self.sense_node = params[1]

    def __repr__(self):
        return f"Branch(type={self.type},name={self.name},from_node={self.from_node},to_node={self.to_node},phases={self.phases})"
    
class Load:
    def __init__(self, name, parent, phases, nom_volt, params, glm_string):
        self.name = name
        self.parent = parent
        self.phases = phases
        self.base_voltage = nom_volt
        self.constant_power_A = params[0]
        self.constant_power_B = params[1]
        self.constant_power_C = params[2]
        self.glm_string = glm_string

        # self.Sload = np.array([[self.constant_power_A, self.constant_power_B, self.constant_power_C]])

    def __repr__(self):
        return f"Load(name={self.name},parent={self.parent},phases={self.phases},base_voltage={self.base_voltage})"
    
class Generator:
    def __init__(self, name, parent, phases, nom_volt, params, glm_string):
        self.name = name
        self.parent = parent
        self.phases = phases
        self.base_voltage = nom_volt
        self.constant_power_A = params[0]
        self.constant_power_B = params[1]
        self.constant_power_C = params[2]
        self.glm_string = glm_string

        # self.Sgen = np.array([[self.constant_power_A, self.constant_power_B, self.constant_power_C]])

    def __repr__(self):
        return f"Gen(name={self.name},parent={self.parent},phases={self.phases},base_voltage={self.base_voltage})"
    
class Shunt:
    def __init__(self, type, name, parent, phases, nom_volt, params, glm_string):
        self.type = type
        self.name = name
        self.parent = parent
        self.phases = phases
        self.base_voltage = nom_volt
        self.glm_string = glm_string

        if self.type in ["capacitor"]:
            self.phases_connected = params[0] 
            self.capacitor_A = params[1] 
            self.capacitor_B = params[2] 
            self.capacitor_C = params[3] 
            self.control_level = params[4] 
            self.control = params[5] 
            self.pt_phase = params[6] 
            self.switchA = params[7] 
            self.switchB = params[8] 
            self.switchC = params[9] 

    def __repr__(self):
        return f"Shunt(type={self.type},name={self.name},parent={self.parent},phases={self.phases},base_voltage={self.base_voltage})"
    
class Config:
    def __init__(self, type, name, params, glm_string):
        self.type = type
        self.name = name
        self.glm_string = glm_string

        if self.type in ["line_configuration"]:
            self.z11 = params[0] # in Ohms/mi
            self.z12 = params[1] 
            self.z13 = params[2] 
            self.z21 = params[3] 
            self.z22 = params[4] 
            self.z23 = params[5] 
            self.z31 = params[6] 
            self.z32 = params[7] 
            self.z33 = params[8] 

        elif self.type in ["transformer_configuration"]:
            self.connect_type = params[0] 
            self.install_type = params[1] 
            self.power_rating = params[2] # in kVA, total for transformer
            self.primary_voltage = params[3] 
            self.secondary_voltage = params[4] 
            self.resistance = params[5] # in pu
            self.reactance = params[6] # in pu
            

    def __repr__(self):
        return f"Config(type={self.type},name={self.name})"