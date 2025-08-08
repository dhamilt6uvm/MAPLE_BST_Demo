import re
import json
import os
import csv
import pandas as pd
import numpy as np
import GLM_Tools.PowerSystemModel as psm
import pickle
import matplotlib.pyplot as plt
import warnings

def parse_node(node_string):

    name_match = re.search(r"name\s+([^\s][^;]*);", node_string, re.S)
    if name_match:
        name = name_match.group(1)
    else:
        raise ValueError(f"Could not find name of node object: {node_string}")
    
    phases_match = re.search(r"phases\s+([ABCDN]*);", node_string, re.S)
    if phases_match:
        phases = phases_match.group(1)
    else:
        raise ValueError(f"Could not find phases of node object: {node_string}")
    
    nom_volt_match = re.search(r"nominal_voltage\s+(\d+(\.\d+)?);", node_string, re.S)
    if nom_volt_match:
        nom_volt = float(nom_volt_match.group(1))
    else:
        raise ValueError(f"Could not find phases of node object: {node_string}")
    
    bus_type_match = re.search(r"bustype\s+([A-Z]*);", node_string, re.S)
    if bus_type_match:
        bus_type = bus_type_match.group(1)
    else:
        bus_type = ""

    parent_match = re.search(r"parent\s+([^\s][^;]*);", node_string, re.S)
    if parent_match:
        parent = parent_match.group(1)
    else:
        parent = None

    return psm.Node(name,phases,nom_volt,bus_type,parent,node_string)


def parse_branch(branch_type,branch_string):

    name_match = re.search(r"name\s+([^\s][^;]*);", branch_string, re.S)
    if name_match:
        name = name_match.group(1)
    else:
        raise ValueError(f"Could not find name of branch object: {branch_string}")
    
    from_bus_match = re.search(r"from\s+([^\s][^;]*);", branch_string, re.S)
    if from_bus_match:
        from_bus = from_bus_match.group(1)
    else:
        raise ValueError(f"Could not find from bus of branch object: {branch_string}")
    
    to_bus_match = re.search(r"to\s+([^\s][^;]*);", branch_string, re.S)
    if to_bus_match:
        to_bus = to_bus_match.group(1)
    else:
        raise ValueError(f"Could not find to bus of branch object: {branch_string}")
    
    phases_match = re.search(r"phases\s+([ABCDN]*);", branch_string, re.S)
    if phases_match:
        phases = phases_match.group(1)
    else:
        raise ValueError(f"Could not find phases of branch object: {branch_string}")
    
    branch_params = []
    
    if branch_type in ["overhead_line","underground_line","transformer","regulator"]:
        config_match = re.search(r"configuration\s+([^\s][^;]*);", branch_string, re.S)
        if config_match:
            config = config_match.group(1)
        else:
            raise ValueError(f"Could not find configuration of branch object: {branch_string}")
        branch_params.append(config)
    
    if branch_type in ["overhead_line","underground_line"]:
        length_match = re.search(r"length\s+(\d+(\.\d+)?);", branch_string, re.S)
        if length_match:
            length = float(length_match.group(1))
        else:
            raise ValueError(f"Could not find length of branch object: {branch_string}")
        branch_params.append(length)

    if branch_type in ["fuse"]:
        current_limit_match = re.search(r"current_limit\s+(\d+(\.\d+)?)(\s?A)?;", branch_string, re.S)
        if current_limit_match:
            current_limit = float(current_limit_match.group(1))
        else:
            raise ValueError(f"Could not find current limit of branch object: {branch_string}")
        branch_params.append(current_limit)

        mean_replacement_time_match = re.search(r"mean_replacement_time\s+(\d+(\.\d+)?)\s*;", branch_string, re.S)
        if mean_replacement_time_match:
            mean_replacement_time = float(mean_replacement_time_match.group(1))
        else:
            raise ValueError(f"Could not find mean replacement time of branch object: {branch_string}")
        branch_params.append(mean_replacement_time)

        repair_dist_type_match = re.search(r"repair_dist_type\s+([A-Z]*);", branch_string, re.S)
        if repair_dist_type_match:
            repair_dist_type = repair_dist_type_match.group(1)
        else:
            raise ValueError(f"Could not find repair dist type of branch object: {branch_string}")
        branch_params.append(repair_dist_type)

    if branch_type in ["switch"]:
        status_match = re.search(r"status\s+([A-Z]*);", branch_string, re.S)
        if status_match:
            status = status_match.group(1)
        else:
            raise ValueError(f"Could not find status of branch object: {branch_string}")
        branch_params.append(status)

    if branch_type in ["recloser"]:
        max_number_of_tries_match = re.search(r"max_number_of_tries\s+([0-9]*);", branch_string, re.S)
        if max_number_of_tries_match:
            max_number_of_tries = int(max_number_of_tries_match.group(1))
        else:
            raise ValueError(f"Could not find max number of tries of branch object: {branch_string}")
        branch_params.append(max_number_of_tries)

    if branch_type in ["regulator"]:
        sense_node_match = re.search(r"sense_node\s+([^\s][^;]*);", branch_string, re.S)
        if sense_node_match:
            sense_node = sense_node_match.group(1)
        else:
            raise ValueError(f"Could not find sense node of branch object: {branch_string}")
        branch_params.append(sense_node)
    
    return psm.Branch(branch_type,name,from_bus,to_bus,phases,branch_params,branch_string)

def parse_load(load_string):

    name_match = re.search(r"name\s+([^\s][^;]*);", load_string, re.S)
    if name_match:
        name = name_match.group(1)
    else:
        raise ValueError(f"Could not find name of load object: {load_string}")
    
    parent_match = re.search(r"parent\s+([^\s][^;]*);", load_string, re.S)
    if parent_match:
        parent = parent_match.group(1)
    else:
        raise ValueError(f"Could not find parent of load object: {load_string}")
    
    phases_match = re.search(r"phases\s+([ABCDN]*);", load_string, re.S)
    if phases_match:
        phases = phases_match.group(1)
    else:
        raise ValueError(f"Could not find phases of load object: {load_string}")
    
    nom_volt_match = re.search(r"nominal_voltage\s+(\d+(\.\d+)?);", load_string, re.S)
    if nom_volt_match:
        nom_volt = float(nom_volt_match.group(1))
    else:
        raise ValueError(f"Could not find phases of load object: {load_string}")
    
    load_params = []

    for ph in ["A","B","C"]:
        constant_power_match = re.search(fr"constant_power_{ph}\s+([+-]?\d*\.\d+[+-]?\d*\.\d+j);", load_string, re.S)
        if constant_power_match:
            constant_power = complex(constant_power_match.group(1))
        else:
            constant_power = complex(0,0)
        load_params.append(constant_power)

    return psm.Load(name,parent,phases,nom_volt,load_params,load_string)

def parse_generator(gen_string):

    name_match = re.search(r"name\s+([^\s][^;]*);", gen_string, re.S)
    if name_match:
        name = name_match.group(1)
    else:
        raise ValueError(f"Could not find name of gen object: {gen_string}")
    
    parent_match = re.search(r"parent\s+([^\s][^;]*);", gen_string, re.S)
    if parent_match:
        parent = parent_match.group(1)
    else:
        raise ValueError(f"Could not find parent of gen object: {gen_string}")
    
    phases_match = re.search(r"phases\s+([ABCDN]*);", gen_string, re.S)
    if phases_match:
        phases = phases_match.group(1)
    else:
        raise ValueError(f"Could not find phases of gen object: {gen_string}")
    
    nom_volt_match = re.search(r"nominal_voltage\s+(\d+(\.\d+)?);", gen_string, re.S)
    if nom_volt_match:
        nom_volt = float(nom_volt_match.group(1))
    else:
        raise ValueError(f"Could not find phases of gen object: {gen_string}")
    
    gen_params = []

    for ph in ["A","B","C"]:
        constant_power_match = re.search(fr"constant_power_{ph}\s+([+-]?\d*\.\d+[+-]?\d*\.\d+j);", gen_string, re.S)
        if constant_power_match:
            constant_power = complex(constant_power_match.group(1))
        else:
            constant_power = complex(0,0)
        gen_params.append(constant_power)

    return psm.Generator(name,parent,phases,nom_volt,gen_params,gen_string)

def parse_shunt(shunt_type,shunt_string):

    name_match = re.search(r"name\s+([^\s][^;]*);", shunt_string, re.S)
    if name_match:
        name = name_match.group(1)
    else:
        raise ValueError(f"Could not find name of shunt object: {shunt_string}")
    
    parent_match = re.search(r"parent\s+([^\s][^;]*);", shunt_string, re.S)
    if parent_match:
        parent = parent_match.group(1)
    else:
        raise ValueError(f"Could not find parent of shunt object: {shunt_string}")
    
    phases_match = re.search(r"phases\s+([ABCDN]*);", shunt_string, re.S)
    if phases_match:
        phases = phases_match.group(1)
    else:
        raise ValueError(f"Could not find phases of shunt object: {shunt_string}")
    
    nom_volt_match = re.search(r"nominal_voltage\s+(\d+(\.\d+)?);", shunt_string, re.S)
    if nom_volt_match:
        nom_volt = float(nom_volt_match.group(1))
    else:
        raise ValueError(f"Could not find phases of shunt object: {shunt_string}")
    
    shunt_params = []

    if shunt_type in ["capacitor"]:
        phases_connected_match = re.search(r"phases_connected\s+([ABCDN]*);", shunt_string, re.S)
        if phases_connected_match:
            phases_connected = phases_connected_match.group(1)
        else:
            raise ValueError(f"Could not find phases_connected of shunt object: {shunt_string}")
        shunt_params.append(phases_connected)

        for ph in ["A","B","C"]:
            capacitor_match = re.search(fr"capacitor_{ph}\s+(\d+(\.\d+)?);", shunt_string, re.S)
            if capacitor_match:
                capacitor = float(capacitor_match.group(1))
            else:
                raise ValueError(f"Could not find capacitor_{ph} of shunt object: {shunt_string}")
            shunt_params.append(capacitor)

        control_level_match = re.search(r"control_level\s+([A-Z]*);", shunt_string, re.S)
        if control_level_match:
            control_level = control_level_match.group(1)
        else:
            raise ValueError(f"Could not find control_level of shunt object: {shunt_string}")
        shunt_params.append(control_level)

        control_match = re.search(r"control\s+([A-Z]*);", shunt_string, re.S)
        if control_match:
            control = control_match.group(1)
        else:
            raise ValueError(f"Could not find control of shunt object: {shunt_string}")
        shunt_params.append(control)

        pt_phase_match = re.search(r"pt_phase\s+([ABC]*);", shunt_string, re.S)
        if pt_phase_match:
            pt_phase = pt_phase_match.group(1)
        else:
            raise ValueError(f"Could not find pt_phase of shunt object: {shunt_string}")
        shunt_params.append(pt_phase)

        for ph in ["A","B","C"]:
            switch_match = re.search(fr"switch{ph}\s+([A-Z]*);", shunt_string, re.S)
            if switch_match:
                switch = switch_match.group(1)
            else:
                raise ValueError(f"Could not find switch{ph} of shunt object: {shunt_string}")
            shunt_params.append(switch)

    return psm.Shunt(shunt_type,name,parent,phases,nom_volt,shunt_params,shunt_string)

def parse_config(config_type,config_string):

    name_match = re.search(r"name\s+([^\s][^;]*);", config_string, re.S)
    if name_match:
        name = name_match.group(1)
    else:
        raise ValueError(f"Could not find name of config object: {config_string}")
    
    config_params = []

    if config_type in ["line_configuration"]:
        z_strs = ['11','12','13','21','22','23','31','32','33']
        for z_str in z_strs:
            z_match = re.search(fr"z{z_str}\s+([+-]?\d*\.\d+[+-]?\d*\.\d+j);", config_string, re.S)
            if z_match:
                z = complex(z_match.group(1))
            else:
                z = complex(1e-4,1e-4)
                warnings.warn("WARNING: Need to implement conductor and line spacing parsing!")
                # raise ValueError(f"Could not find z{z_str} of line config object: {config_string}")
            config_params.append(z)

    elif config_type in ["transformer_configuration"]:
        connect_type_match = re.search(r"connect_type\s+([^\s][^;]*);", config_string, re.S)
        if connect_type_match:
            connect_type = connect_type_match.group(1)
        else:
            raise ValueError(f"Could not find connect_type of transformer config object: {config_string}")
        config_params.append(connect_type)

        install_type_match = re.search(r"install_type\s+([^\s][^;]*);", config_string, re.S)
        if install_type_match:
            install_type = install_type_match.group(1)
        else:
            raise ValueError(f"Could not find install_type of transformer config object: {config_string}")
        config_params.append(install_type)

        power_rating_match = re.search(r"power_rating\s+(\d+(\.\d+)?);", config_string, re.S)
        if power_rating_match:
            power_rating = float(power_rating_match.group(1))
        else:
            raise ValueError(f"Could not find power_rating of transformer config object: {config_string}")
        config_params.append(power_rating)

        primary_voltage_match = re.search(r"primary_voltage\s+(\d+(\.\d+)?);", config_string, re.S)
        if primary_voltage_match:
            primary_voltage = float(primary_voltage_match.group(1))
        else:
            raise ValueError(f"Could not find primary_voltage of transformer config object: {config_string}")
        config_params.append(primary_voltage)

        secondary_voltage_match = re.search(r"secondary_voltage\s+(\d+(\.\d+)?);", config_string, re.S)
        if secondary_voltage_match:
            secondary_voltage = float(secondary_voltage_match.group(1))
        else:
            raise ValueError(f"Could not find secondary_voltage of transformer config object: {config_string}")
        config_params.append(secondary_voltage)

        resistance_match = re.search(r"resistance\s+(\d+(\.\d+)?);", config_string, re.S)
        if resistance_match:
            resistance = float(resistance_match.group(1))
        else:
            raise ValueError(f"Could not find resistance of transformer config object: {config_string}")
        config_params.append(resistance)

        reactance_match = re.search(r"reactance\s+(\d+(\.\d+)?);", config_string, re.S)
        if reactance_match:
            reactance = float(reactance_match.group(1))
        else:
            raise ValueError(f"Could not find reactance of transformer config object: {config_string}")
        config_params.append(reactance)

    elif config_type in ["regulator_configuration"]:
        connect_type_match = re.search(r"connect_type\s+([^\s][^;]*);", config_string, re.S)
        if connect_type_match:
            connect_type = connect_type_match.group(1)
        else:
            raise ValueError(f"Could not find connect_type of regulator config object: {config_string}")
        config_params.append(connect_type)

        band_center_match = re.search(r"band_center\s+(\d+(\.\d+)?);", config_string, re.S)
        if band_center_match:
            band_center = float(band_center_match.group(1))
        else:
            raise ValueError(f"Could not find band_center of regulator config object: {config_string}")
        config_params.append(band_center)

        band_width_match = re.search(r"band_width\s+(\d+(\.\d+)?);", config_string, re.S)
        if band_width_match:
            band_width = float(band_width_match.group(1))
        else:
            raise ValueError(f"Could not find band_width of regulator config object: {config_string}")
        config_params.append(band_width)

        regulation_match = re.search(r"regulation\s+(\d+(\.\d+)?);", config_string, re.S)
        if regulation_match:
            regulation = float(regulation_match.group(1))
        else:
            raise ValueError(f"Could not find regulation of regulator config object: {config_string}")
        config_params.append(regulation)

        raise_taps_match = re.search(r"raise_taps\s+(\d+);", config_string, re.S)
        if raise_taps_match:
            raise_taps = int(raise_taps_match.group(1))
        else:
            raise ValueError(f"Could not find raise_taps of regulator config object: {config_string}")
        config_params.append(raise_taps)

        lower_taps_match = re.search(r"lower_taps\s+(\d+);", config_string, re.S)
        if lower_taps_match:
            lower_taps = int(lower_taps_match.group(1))
        else:
            raise ValueError(f"Could not find lower_taps of regulator config object: {config_string}")
        config_params.append(lower_taps)

        CT_phase_match = re.search(r"CT_phase\s+([ABCDN]*);", config_string, re.S)
        if CT_phase_match:
            CT_phase = CT_phase_match.group(1)
        else:
            raise ValueError(f"Could not find CT_phase of regulator config object: {config_string}")
        config_params.append(CT_phase)

        PT_phase_match = re.search(r"PT_phase\s+([ABCDN]*);", config_string, re.S)
        if PT_phase_match:
            PT_phase = PT_phase_match.group(1)
        else:
            raise ValueError(f"Could not find PT_phase of regulator config object: {config_string}")
        config_params.append(PT_phase)

        Type_match = re.search(r"Type\s+([AB]);", config_string, re.S)
        if Type_match:
            Type = Type_match.group(1)
        else:
            raise ValueError(f"Could not find Type of regulator config object: {config_string}")
        config_params.append(Type)

        Control_match = re.search(r"Control\s+([^\s][^;]*);", config_string, re.S)
        if Control_match:
            Control = Control_match.group(1)
        else:
            raise ValueError(f"Could not find Control of regulator config object: {config_string}")
        config_params.append(Control)

        control_level_match = re.search(r"control_level\s+([^\s][^;]*);", config_string, re.S)
        if control_level_match:
            control_level = control_level_match.group(1)
        else:
            raise ValueError(f"Could not find control_level of regulator config object: {config_string}")
        config_params.append(control_level)

        for ph in ["A","B","C"]:
            tap_pos_match = re.search(fr"tap_pos_{ph}\s+(\d+);", config_string, re.S)
            if tap_pos_match:
                tap_pos = int(tap_pos_match.group(1))
            else:
                raise ValueError(f"Could not find tap_pos_{ph} of regulator config object: {config_string}")
            config_params.append(tap_pos)

    return psm.Config(config_type,name,config_params,config_string)


def parse_glm_to_pkl(substation_name):

    glm_file_dir = f"Feeder_Data/{substation_name}/Input_Data/"
    glm_file_name = f"{substation_name}_Helics.glm"
    glm_file = os.path.join(glm_file_dir,glm_file_name)

    pkl_file_dir = f"Feeder_Data/{substation_name}/Python_Model/"
    pkl_file_name = f"{substation_name}_Model.pkl"
    pkl_file = os.path.join(pkl_file_dir,pkl_file_name)

    print(f"Parsing {glm_file_name}...")

    with open(glm_file, 'r') as file:
        glm_data = file.read()

    # Parse GLM file
    first_obj = re.search(r"object (\S*) \{[^{}]*\}", glm_data, re.S)
    header = glm_data[:first_obj.start()]

    objects = []
    node_objs = []
    branch_objs = []
    load_objs = []
    shunt_objs = []
    config_objs = []
    helics_objs = []
    misc_objs = []

    Nodes = []
    Branches = []
    Loads = []
    Generators = []
    Shunts = []
    Configs = []

    for obj in re.finditer(r"object (\S*) \{[^{}]*\}", glm_data, re.S):
        objects.append(obj.group(0))
        obj_type = obj.group(1).strip('"')
        if obj_type in ["node","meter"]:
            node_string = obj.group(0)
            node_objs.append(node_string)
            Nodes.append(parse_node(node_string))
        elif obj_type in ["overhead_line", "underground_line", "transformer", "fuse", "switch", "sectionalizer", "recloser", "regulator"]:
            branch_string = obj.group(0)
            branch_objs.append(branch_string)
            Branches.append(parse_branch(obj_type,branch_string))
        elif obj_type in ["load"]:
            load_string = obj.group(0)
            load_objs.append(load_string)
            name_match = re.search(r"name\s+([^\s][^;]*);", load_string, re.S)
            if name_match:
                name = name_match.group(1)
            else:
                raise ValueError(f"Could not find name of load object: {load_string}")
            if "negLdGen" in name:
                Generators.append(parse_generator(load_string))
            else:
                Loads.append(parse_load(load_string))
        elif obj_type in ["capacitor"]:
            shunt_string = obj.group(0)
            shunt_objs.append(shunt_string)
            Shunts.append(parse_shunt(obj_type,shunt_string))
        elif obj_type in ["regulator_configuration", "transformer_configuration", "line_configuration"]:
            config_string = obj.group(0)
            config_objs.append(config_string)
            Configs.append(parse_config(obj_type,config_string))
        elif obj_type in ["helics_msg"]:
            helics_objs.append(obj.group(0))
        elif obj_type in ["voltdump", "currdump", "impedance_dump", "group_recorder", "recorder"]:
            misc_objs.append(obj.group(0))
        else:
            print(f"Unrecognized object type: {obj_type}")

    miss_objs = len(objects) - (len(node_objs) + len(branch_objs) + len(load_objs) + len(shunt_objs) + len(config_objs) + len(helics_objs) + len(misc_objs))
    if miss_objs > 0:
        print(f"Missing {miss_objs} objects.")
    else:
        print(f"  Found all {len(objects)} objects.")
        print(f"  Found {len(node_objs)} node objects.")
        print(f"  Found {len(branch_objs)} branch objects.")
        print(f"  Found {len(load_objs)} load objects.")
        print(f"  Found {len(shunt_objs)} shunt objects.")
        print(f"  Found {len(config_objs)} config objects.")
        print(f"  Found {len(helics_objs)} helics_msg objects.")
        print(f"  Found {len(misc_objs)} misc objects.")

    # Create a model from the components
    Model = psm.PowerSystemModel(Nodes,Branches,Loads,Generators,Shunts,Configs)

    Model.compute_impedances()

    Model.glm_header = header
    if len(helics_objs) > 0:
        Model.glm_helics_obj = helics_objs[0]
    Model.glm_misc_objs = misc_objs

    # Save the power system model
    if not os.path.exists(pkl_file_dir):
        os.makedirs(pkl_file_dir)
    with open(pkl_file, 'wb') as file:
        pickle.dump(Model, file)

    print(f"Done parsing {glm_file_name}. Python model saved to {pkl_file}.")

def populate_ami_loads_pkl(substation_name, start_date, end_date, load_fixed_pf):

    # Open pkl file
    pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
    with open(pkl_file, 'rb') as file:
        pkl_model = pickle.load(file)

    Sbase_1ph = pkl_model.Sbase_1ph

    # Generate the list of hourly datetimes
    datetimes_list = pd.date_range(start=start_date, end=end_date, freq='h', tz='US/Eastern')

    tsim_hr = len(datetimes_list) # in hours
    tstep_hr = 1 # in hours

    # Load meter databases
    load_dict_file = f"Feeder_Data/{substation_name}/meter_number_data.csv"
    gen_dict_file = f"Feeder_Data/{substation_name}/gen_meter_number_data.csv"

    load_dict = pd.read_csv(load_dict_file)
    gen_dict = pd.read_csv(gen_dict_file)

    # Load the AMI data
    load_ami_data_file = f"Feeder_Data/{substation_name}/AMI_Data/{substation_name}_True_Load_AMI_Data.csv"
    gen_ami_data_file = f"Feeder_Data/{substation_name}/AMI_Data/{substation_name}_True_Gen_AMI_Data.csv"

    load_ami_data = pd.read_csv(load_ami_data_file)
    gen_ami_data = pd.read_csv(gen_ami_data_file)

    # Convert times to datetime
    load_ami_data['start_date_time'] = pd.to_datetime(load_ami_data['start_date_time'], utc=True)
    gen_ami_data['start_date_time'] = pd.to_datetime(gen_ami_data['start_date_time'], utc=True)

    # Loads to skip: these loads are in GIS but not in WindMil/GLD, need to investigate further...    
    loads_to_skip_file = f"Feeder_Data/{substation_name}/loads_in_gis_but_not_glm.csv"
    gens_to_skip_file = f"Feeder_Data/{substation_name}/gens_in_gis_but_not_glm.csv"

    loads_to_skip_data = pd.read_csv(loads_to_skip_file)
    gens_to_skip_data = pd.read_csv(gens_to_skip_file)

    loads_to_skip = [str(load) for load in loads_to_skip_data['Service Number'].tolist()]
    gens_to_skip = [str(gen) for gen in gens_to_skip_data['Object ID'].tolist()]

    # Filter AMI data between relevant dates
    load_ami_filt = load_ami_data[(load_ami_data['start_date_time']>=datetimes_list[0]) & (load_ami_data['start_date_time']<=datetimes_list[-1])]
    gen_ami_filt = gen_ami_data[(gen_ami_data['start_date_time']>=datetimes_list[0]) & (gen_ami_data['start_date_time']<=datetimes_list[-1])]

    # Update start and end dates in PowerSystemModel object
    pkl_model.ami_datetimes = load_ami_filt['start_date_time'].values
    num_datetimes = len(pkl_model.ami_datetimes)

    # Loop through AMI loads
    for index, row in load_dict.iterrows():
        service_num = row['Service Number']
        meter_num = row['Meter Number']
        if service_num not in loads_to_skip:
            load_name = f"_{service_num}_cons"
            load_obj_ind = pkl_model.Load_Dict[load_name].index
            load_phase = row['Phase']
            num_phases = len(load_phase)
            load_value_vect = np.zeros((num_datetimes,3),dtype=complex)
            for ph_ind in range(num_phases):
                ph = load_phase[ph_ind]
                if str(meter_num) not in load_ami_filt.columns[1:]:
                    load_value_P = 0.0
                else:
                    load_value_P = 1000*(load_ami_filt[str(meter_num)].values)
                fixed_pf = load_fixed_pf
                load_value_Q = load_value_P*np.sign(fixed_pf)*np.sqrt(1/(fixed_pf**2)-1)
                load_value = np.array(load_value_P+1j*load_value_Q)/num_phases
                load_value_pu = load_value/Sbase_1ph
                if ph == "A":
                    load_value_vect[:,0] = load_value_pu
                elif ph == "B":
                    load_value_vect[:,1] = load_value_pu
                elif ph == "C":
                    load_value_vect[:,2] = load_value_pu
                else:
                    ValueError(f"Phase {ph} not recognized in load: {load_name}")
            pkl_model.Loads[load_obj_ind].Sload = load_value_vect

    # Loop through AMI Generation
    for index, row in gen_dict.iterrows():
        object_id = row['Object ID']
        meter_num = row['Meter Number']
        if str(object_id) not in gens_to_skip:
            gen_name = f"gene_{object_id}_negLdGen"
            gen_obj_ind = pkl_model.Generator_Dict[gen_name].index
            gen_phase = row['Phase']
            num_phases = len(gen_phase)
            gen_value_vect = np.zeros((num_datetimes,3),dtype=complex)
            for ph_ind in range(num_phases):
                ph = gen_phase[ph_ind]
                if str(meter_num) not in gen_ami_filt.columns[1:]:
                    gen_value_P = 0.0
                else:
                    gen_value_P = 1000*(gen_ami_filt[str(meter_num)].values)
                fixed_pf = 1.00
                gen_value_Q = gen_value_P*np.sign(fixed_pf)*np.sqrt(1/(fixed_pf**2)-1)
                gen_value = np.array(gen_value_P+1j*gen_value_Q)/num_phases
                gen_value_pu = gen_value/Sbase_1ph
                if ph == "A":
                    gen_value_vect[:,0] = gen_value_pu
                elif ph == "B":
                    gen_value_vect[:,1] = gen_value_pu
                elif ph == "C":
                    gen_value_vect[:,2] = gen_value_pu
                else:
                    ValueError(f"Phase {ph} not recognized in load: {gen_name}")
            pkl_model.Generators[gen_obj_ind].Sgen = gen_value_vect
    
    # Save updated pkl file
    with open(pkl_file, 'wb') as file:
        pickle.dump(pkl_model, file)

def populate_inv_ratings_pkl(substation_name):

    substation_dict = {"South_Alburgh":"28",
                       "South_Hero":"29",
                       "Burton_Hill":"43"}

    if substation_name in substation_dict:
        substation_number = substation_dict[substation_name]
    else:
        raise ValueError(f"Substation name \"{substation_name}\" not recognized. Valid substation names are: {list(substation_dict.keys())}")
    
    # Open Generator GIS data
    gis_gen_fname = f"Feeder_Data/VEC_GIS_Data/gs_generator073025.json"
    with open(gis_gen_fname, 'r') as file:
        gis_gen_data = json.load(file)

    # Parse GIS generator data and build dictionary
    gen_object_ids = []
    gen_system_types = []
    gen_sizes = []
    gen_phases = []

    for feature in gis_gen_data["features"]:
        attributes = feature["attributes"]
        gs_substation = attributes["gs_substation"]
        if gs_substation == substation_number:
            object_id = attributes["OBJECTID"]
            gs_system_type = attributes["gs_system_type"]
            gs_generator_system_size = attributes["gs_generator_system_size"]
            gs_phase = attributes["gs_phase"]

            gen_object_ids.append(object_id)
            gen_system_types.append(gs_system_type)
            gen_sizes.append(gs_generator_system_size)
            gen_phases.append(gs_phase)

    # Create DataFrame and write to CSV
    gen_data = pd.DataFrame({
        "Object ID": gen_object_ids,
        "System Type": gen_system_types,
        "Phase": gen_phases,
        "System Size (kVA)": gen_sizes
    })

    gen_data.to_csv(f"Feeder_Data/{substation_name}/generator_sizes.csv", index=False)

    # Open pkl file
    pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
    with open(pkl_file, 'rb') as file:
        pkl_model = pickle.load(file)

    for Gen in pkl_model.Generators:
        object_id_match = re.search(r"gene_([0-9]*)_negLdGen", Gen.name, re.S)
        if object_id_match:
            object_id = int(object_id_match.group(1))
            if object_id in gen_object_ids:
                ind = gen_object_ids.index(object_id)
                gen_size = gen_sizes[ind]
                if gen_size is not None:
                    Gen.Srated = float(gen_sizes[ind])
                else:
                    Gen.Srated = None
            else:
                raise ValueError(f"Could not find generator in GIS: {object_id}")
        else:
            raise ValueError(f"Could not find object_id of generator object: {Gen.name}")


    # Save updated pkl file
    with open(pkl_file, 'wb') as file:
        pickle.dump(pkl_model, file)


def add_coords_to_pkl(substation_name):

    # Open pkl file
    pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
    with open(pkl_file, 'rb') as file:
        pkl_model = pickle.load(file)

    # Open coordinate data files (from WindMil)
    branch_coord_file = f"Feeder_Data/{substation_name}/Coordinate_Data/{substation_name}_Branch_Coords.xls"
    branch_coord_df = pd.read_excel(branch_coord_file)

    # find branch coordinates
    fake_branches = []
    for branch in pkl_model.Branches:
        # skip fake branches for now
        if branch.type == "fake":
            fake_branches.append(branch.index)
            continue
        branch_name = branch.name
        # catch special cases
        if branch_name[0] == "_":
            branch_name = branch_name[1:]  
        if branch_name[-13:] == "_FakePhasesBC":
            branch_name = branch_name[:-13]           
        reg_match = re.search(r".*(volt_[0-9]*).*", branch_name, re.S)
        if reg_match is not None:
            branch_name = reg_match.group(1)
        # find matching row in dataframe
        match_df = branch_coord_df.loc[branch_coord_df['Element Name'] == branch_name]
        if match_df.empty:
            raise ValueError(f"Couldn't find match for branch: {branch_name}")
        branch.X_coord = match_df['X Coordinate'].values[0]
        branch.Y_coord = match_df['Y Coordinate'].values[0]
        branch.X2_coord = match_df['X2 Coordinate'].values[0]
        branch.Y2_coord = match_df['Y2 Coordinate'].values[0]

        # get node coordinates
        from_node = pkl_model.Nodes[branch.from_node_ind]
        to_node = pkl_model.Nodes[branch.to_node_ind]
        from_node.X_coord = branch.X2_coord
        from_node.Y_coord = branch.Y2_coord
        to_node.X_coord = branch.X_coord
        to_node.Y_coord = branch.Y_coord

    for branch_ind in fake_branches:
        branch = pkl_model.Branches[branch_ind]
        from_node = pkl_model.Nodes[branch.from_node_ind]
        to_node = pkl_model.Nodes[branch.to_node_ind]
        if hasattr(to_node,'X_coord'):
            branch.X_coord = to_node.X_coord
            branch.Y_coord = to_node.Y_coord
        else:
            branch.X_coord = from_node.X_coord
            branch.Y_coord = from_node.Y_coord
            to_node.X_coord = branch.X_coord
            to_node.Y_coord = branch.Y_coord
        branch.X2_coord = from_node.X_coord
        branch.Y2_coord = from_node.Y_coord


    for load in pkl_model.Loads:
        parent_node = pkl_model.Nodes[load.parent_node_ind]
        load.X_coord = parent_node.X_coord
        load.Y_coord = parent_node.Y_coord

    for gen in pkl_model.Generators:
        parent_node = pkl_model.Nodes[gen.parent_node_ind]
        gen.X_coord = parent_node.X_coord
        gen.Y_coord = parent_node.Y_coord
    
    # Save updated pkl file
    with open(pkl_file, 'wb') as file:
        pickle.dump(pkl_model, file)

def plot_feeder(substation_name):

    # Open pkl file
    pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
    with open(pkl_file, 'rb') as file:
        pkl_model = pickle.load(file)

    for Branch in pkl_model.Branches:
        plt.plot([Branch.X_coord,Branch.X2_coord],[Branch.Y_coord,Branch.Y2_coord],color='black')
    for Node in pkl_model.Nodes:
        # print(f"{Node.name}, {Node.X_coord}, {Node.Y_coord}")
        plt.plot(Node.X_coord,Node.Y_coord,'b.',markersize=5)
    for ld_ind, Load in enumerate(pkl_model.Loads):
        plt.plot(Load.X_coord,Load.Y_coord,'ro',markersize=5,markerfacecolor='none')
    for gen_ind, Generator in enumerate(pkl_model.Generators):
        plt.plot(Generator.X_coord,Generator.Y_coord,'go',markersize=5,markerfacecolor='none')
    plt.show()

def create_glm_from_pkl(pkl_file,glm_file):

    with open(pkl_file, 'rb') as file:
        pkl_model = pickle.load(file)

    with open(glm_file, 'w') as file:
        file.write(pkl_model.glm_header)
        file.write(pkl_model.glm_helics_obj)
        file.write("\n\n")
        for Node in pkl_model.Nodes:
            file.write(Node.glm_string)
            file.write("\n\n")
        for Branch in pkl_model.Branches:
            file.write(Branch.glm_string)
            file.write("\n\n")
        for Load in pkl_model.Loads:
            file.write(Load.glm_string)
            file.write("\n\n")
        for Generator in pkl_model.Generators:
            file.write(Generator.glm_string)
            file.write("\n\n")
        for Shunt in pkl_model.Shunts:
            file.write(Shunt.glm_string)
            file.write("\n\n")
        for Config in pkl_model.Configs:
            file.write(Config.glm_string)
            file.write("\n\n")
        for misc_obj in pkl_model.glm_misc_objs:
            file.write(misc_obj)
            file.write("\n\n")
