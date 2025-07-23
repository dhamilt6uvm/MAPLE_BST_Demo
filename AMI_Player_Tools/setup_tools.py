import re
import json
import os
import csv
import pandas as pd
import glob
import pickle
import GLM_Tools.PowerSystemModel as psm

def get_meter_numbers(substation_name):

    substation_dict = {"South_Alburgh":"28",
                       "South_Hero":"29",
                       "Burton_Hill":"43"}

    if substation_name in substation_dict:
        substation_number = substation_dict[substation_name]
    else:
        raise ValueError(f"Substation name \"{substation_name}\" not recognized. Valid substation names are: {list(substation_dict.keys())}")

    gis_sp_fname = f"Feeder_Data/VEC_GIS_Data/gs_service_point050224.json"
    with open(gis_sp_fname, 'r') as file:
        gis_sp_data = json.load(file)

    gis_gen_fname = f"Feeder_Data/VEC_GIS_Data/gs_generator050224.json"
    with open(gis_gen_fname, 'r') as file:
        gis_gen_data = json.load(file)

    # Parse GIS service point data and build service -> meter number dictionary
    object_ids = []
    service_nums = []
    meter_nums = []
    null_meters = []
    phases = []
    x_coords = []
    y_coords = []

    for feature in gis_sp_data["features"]:
        attributes = feature["attributes"]
        geometry = feature["geometry"]
        gs_substation = attributes["gs_substation"]
        if gs_substation == substation_number:
            gs_service_number = attributes["gs_service_number"]
            gs_meter_number = attributes["gs_meter_number"]
            gs_phase = attributes["gs_phase"]
            x_coord = geometry["x"]
            y_coord = geometry["y"]
            if gs_meter_number is None:
                null_meters.append(gs_service_number) 
            elif gs_meter_number == "":
                null_meters.append(gs_service_number) 
            else:
                service_nums.append(gs_service_number)
                meter_nums.append(gs_meter_number)
                phases.append(gs_phase)
                x_coords.append(x_coord)
                y_coords.append(y_coord)
                
    # Parse GIS generator data and build service -> meter number dictionary
    gen_object_ids = []
    gen_service_nums = []
    gen_meter_nums = []
    gen_null_meters = []
    gen_phases = []
    gen_x_coords = []
    gen_y_coords = []
    gen_net_meter_sw = []

    for feature in gis_gen_data["features"]:
        attributes = feature["attributes"]
        geometry = feature["geometry"]
        gs_substation = attributes["gs_substation"]
        if gs_substation == substation_number:
            object_id = attributes["OBJECTID"]
            gs_service_number = attributes["gs_service_number"]
            gs_meter_number = attributes["gs_meter_number"]
            gs_phase = attributes["gs_phase"]
            x_coord = geometry["x"]
            y_coord = geometry["y"]
            gs_net_meter_sw = attributes["gs_net_meter_sw"]
            if gs_meter_number is None:
                gen_null_meters.append(gs_service_number) 
            elif gs_meter_number == "":
                gen_null_meters.append(gs_service_number) 
            else:
                gen_object_ids.append(object_id)
                gen_service_nums.append(gs_service_number)
                gen_meter_nums.append(gs_meter_number)
                gen_phases.append(gs_phase)
                gen_x_coords.append(x_coord)
                gen_y_coords.append(y_coord)
                gen_net_meter_sw.append(gs_net_meter_sw)

    has_gen = []
    gen_meter = []
    for serv_num in service_nums:
        if serv_num in gen_service_nums:
            has_gen.append('Y')
            ind = gen_service_nums.index(serv_num)
            gen_meter.append(gen_meter_nums[ind])
        else:
            has_gen.append('N')
            gen_meter.append("")



    # Create DataFrame and write to CSV
    out_data = pd.DataFrame({
        "Service Number": service_nums,
        "Meter Number": meter_nums,
        "Phase": phases,
        "X": x_coords,
        "Y": y_coords,
        "Has Separate Gen Meter": has_gen,
        "Gen Meter": gen_meter
    })

    null_meters_data = pd.DataFrame({
        "Service Number": null_meters,
    })

    gen_out_data = pd.DataFrame({
        "Object ID": gen_object_ids,
        "Service Number": gen_service_nums,
        "Meter Number": gen_meter_nums,
        "Phase": gen_phases,
        "X": gen_x_coords,
        "Y": gen_y_coords,
        "Net Meter Switched": gen_net_meter_sw
    })

    gen_null_meters_data = pd.DataFrame({
        "Service Number": gen_null_meters,
    })

    if not os.path.exists(f"Feeder_Data/{substation_name}/"):
        os.makedirs(f"Feeder_Data/{substation_name}/")

    out_data.to_csv(f"Feeder_Data/{substation_name}/meter_number_data.csv", index=False)
    null_meters_data.to_csv(f"Feeder_Data/{substation_name}/null_meters.csv", index=False)

    gen_out_data.to_csv(f"Feeder_Data/{substation_name}/gen_meter_number_data.csv", index=False)
    gen_null_meters_data.to_csv(f"Feeder_Data/{substation_name}/gen_null_meters.csv", index=False)

    print(f"Created CSV dictionary files for load and generator meters in Feeder_Data/{substation_name}/.")

def query_writer(substation_name,start_time,end_time,max_k=2000):

    fdir = f"Feeder_Data/{substation_name}/MySQL_Queries/"
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # Get SQL host name from user for security reasons
    host_name = input("Please enter VEC MySQL server address (XX.XX.X.XXX): ")

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(f"Feeder_Data/{substation_name}/meter_number_data.csv")
    gen_data = pd.read_csv(f"Feeder_Data/{substation_name}/gen_meter_number_data.csv")

    # Extract the 'Meter Number' column
    meter_nums = data['Meter Number']
    gen_meter_nums = gen_data['Meter Number']

    # Loop to process meter numbers in chunks
    k_start = 0
    n = 1
    while k_start < len(meter_nums):
        # Determine the end index for this chunk
        k_end = min(len(meter_nums), (n * max_k))

        # Create the output file name
        fname = os.path.join(fdir,f"{substation_name}_Load_MySQL_Query_{n}.txt")
        output_file = f"{substation_name}_Load_AMI_Data_{n}.txt"

        # Write to the file
        with open(fname, "w") as file:
            # Write the initial part of the MySQL query
            file.write(f"mysql --host {host_name} --user UVM -p -e 'select * from MDM.meter_reads_interval where `asset_id` in (")
            
            # Write the meter numbers for this chunk
            for i in range(k_start, k_end):
                file.write(f"\"{meter_nums[i]}\"")
                if i < k_end - 1:
                    file.write(", ")
            
            # Finish the MySQL query and output redirection
            file.write(f") and `start_date_time` between \"{start_time}\" and \"{end_time}\";' > {output_file}")

        # Update the start index and iteration variables
        k_start = k_end
        n += 1

    print(f"Created MySQL queries for loads in {substation_name}. Located in Feeder_Data/{substation_name}/MySQL_Queries/ folder.")


    # Do it again for generator meters
    k_start = 0
    n = 1
    while k_start < len(gen_meter_nums):
        # Determine the end index for this chunk
        k_end = min(len(gen_meter_nums), (n * max_k))

        # Create the output file name
        fname = os.path.join(fdir,f"{substation_name}_Gen_MySQL_Query_{n}.txt")
        output_file = f"{substation_name}_Gen_AMI_Data_{n}.txt"

        # Write to the file
        with open(fname, "w") as file:
            # Write the initial part of the MySQL query
            file.write(f"mysql --host {host_name} --user UVM -p -e 'select * from MDM.meter_reads_interval where `asset_id` in (")
            
            # Write the meter numbers for this chunk
            for i in range(k_start, k_end):
                file.write(f"\"{gen_meter_nums[i]}\"")
                if i < k_end - 1:
                    file.write(", ")
            
            # Finish the MySQL query and output redirection
            file.write(f") and `start_date_time` between \"{start_time}\" and \"{end_time}\";' > {output_file}")

        # Update the start index and iteration variables
        k_start = k_end
        n += 1

    print(f"Created MySQL queries for generators in {substation_name}. Located in Feeder_Data/{substation_name}/MySQL_Queries/ folder.")

    # Create a directory for AMI data to go in
    ami_fdir = f"Feeder_Data/{substation_name}/AMI_Data/"
    if not os.path.exists(ami_fdir):
        os.makedirs(ami_fdir)
    print(f"Created folder for {substation_name} AMI data. Located in {ami_fdir} folder.")


def parse_ami_data(substation_name,ami_type="Load",save_15min=False):

    valid_ami_types = ["Load","Gen"]
    if ami_type not in valid_ami_types:
        raise ValueError(f"AMI type \"{ami_type}\" not recognized. Valid inputs are: {valid_ami_types}.")

    # List of file paths or use glob to match files
    file_paths = glob.glob(f"Feeder_Data/{substation_name}/AMI_Data/{substation_name}_{ami_type}_AMI_Data_*.txt")  # Adjust pattern as needed

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through the list of file paths and read each file
    for file_path in file_paths:
        # Read the current file into a DataFrame
        df = pd.read_csv(file_path, sep='\t')
        
        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames into one
    concatenated_df = pd.concat(dfs, ignore_index=True)

    # Convert 'start_date_time' to datetime if needed
    concatenated_df['start_date_time'] = pd.to_datetime(concatenated_df['start_date_time'])

    # Create a pivot table
    pivot_df = concatenated_df.pivot_table(index='start_date_time', 
                            columns='asset_id',
                            values='value',
                            aggfunc='first', 
                            fill_value=None)
    
    # Make sure the asset ids are integers
    pivot_df.columns = pivot_df.columns.astype(int)
    
    if ami_type =="Gen":
        # Need to check if net meter direction is switched
        meter_data_file = f"Feeder_Data/{substation_name}/gen_meter_number_data.csv"
        meter_data = pd.read_csv(meter_data_file)
        switch_meters_df = meter_data[meter_data['Net Meter Switched'] == 'Y']
        meters_to_switch = switch_meters_df['Meter Number'].tolist()

        # Loop through each meter number and negate the corresponding column in pivot_df
        for meter_number in meters_to_switch:
            if meter_number in pivot_df.columns:
                # Negate all values in the column where asset_id = meter_number
                pivot_df[meter_number] = -pivot_df[meter_number]

    hourly_data = pivot_df.resample('h').sum()

    # Save the pivot table to a CSV file
    hourly_data.to_csv(f"Feeder_Data/{substation_name}/AMI_Data/{substation_name}_{ami_type}_AMI_Data.csv", index=True)  # Set index=False if you don't want to save the index

    if save_15min:
        pivot_df.to_csv(f"Feeder_Data/{substation_name}/AMI_Data/{substation_name}_{ami_type}_AMI_Data_15_min.csv", index=True)  # Set index=False if you don't want to save the index

    print(f"Parsed AMI {ami_type} data for {substation_name} into a CSV file. Located in Feeder_Data/{substation_name}/AMI_Data/ folder.")


def calculate_true_load(substation_name): # This function should likely be combined with parse_ami_data() eventually...

    net_load_file = f"Feeder_Data/{substation_name}/AMI_Data/{substation_name}_Load_AMI_Data.csv"
    gen_ami_file = f"Feeder_Data/{substation_name}/AMI_Data/{substation_name}_Gen_AMI_Data.csv"

    sp_meter_data_file = f"Feeder_Data/{substation_name}/meter_number_data.csv"

    # Read the tab-delimited file into a DataFrame
    net_load_df = pd.read_csv(net_load_file)
    gen_ami_df = pd.read_csv(gen_ami_file)
    sp_meter_df = pd.read_csv(sp_meter_data_file)

    # Get meters with BTM gen
    meters_with_btm_df = sp_meter_df[sp_meter_df['Has Separate Gen Meter'] == 'Y']
    meters_with_btm = meters_with_btm_df['Meter Number'].tolist()
    btm_meters = list(map(str,list(map(int,meters_with_btm_df['Gen Meter'].tolist()))))

    true_load_df = net_load_df
    missing_loads_with_btm = []
    for meter_ind, meter in enumerate(meters_with_btm):
        if meter in btm_meters: # FIX ME: there are some meters that match gen meter nums but have Has Seperate Gen Meter = N
            gen_ami_df[btm_meters[meter_ind]] = 0
        else:
            if str(meter) in net_load_df.columns:
                true_load_df[str(meter)] = true_load_df[str(meter)] + gen_ami_df[str(int(btm_meters[meter_ind]))]
            else:
                missing_loads_with_btm.append(meter)

    # Save the pivot table to a CSV file
    true_load_df.to_csv(f"Feeder_Data/{substation_name}/AMI_Data/{substation_name}_True_Load_AMI_Data.csv", index=True)  # Set index=False if you don't want to save the index
    gen_ami_df.to_csv(f"Feeder_Data/{substation_name}/AMI_Data/{substation_name}_True_Gen_AMI_Data.csv", index=True)  # Set index=False if you don't want to save the index

    # Store list of loads with btm generation that are missing from AMI data
    missing_loads_df = pd.DataFrame(missing_loads_with_btm, columns=['Meter Number'])
    missing_loads_df.to_csv(f"Feeder_Data/{substation_name}/AMI_Data/{substation_name}_Missing_Loads_With_BTM.csv", index=False)

    print(f"Calculated \"true\" AMI data for loads and generation in {substation_name}. Located in Feeder_Data/{substation_name}/AMI_Data/ folder.")

def modify_runner_json(substation_name,start_time,end_time,ami_load_fixed_pf,include_hc):

    if include_hc:
        template_runner_file = f"Runner_Files/Templates/hc_cosim_runner.json"
    else:
        template_runner_file = f"Runner_Files/Templates/cosim_runner.json"

    new_runner_file_name = f"{substation_name}_cosim_runner.json"
    new_runner_dir = f"Runner_Files\\{substation_name}\\"
    new_runner_file = os.path.join(new_runner_dir,new_runner_file_name)

    # Open and read the existing JSON file
    with open(template_runner_file, 'r') as infile:
        data = json.load(infile)
    
    # Check if the {Feeder_Name} exists in the data and replace it with the substation_Name
    for federate in data.get('federates', []):
        if 'exec' in federate:
            federate['exec'] = federate['exec'].replace('Feeder_Name', substation_name)
            if 'name' in federate:
                if federate['name'] == "AMI Player Federate":
                    federate['exec'] = federate['exec'].replace('Start_Time', start_time)
                    federate['exec'] = federate['exec'].replace('End_Time', end_time)
                    federate['exec'] = federate['exec'].replace('Load_Fixed_PF', str(ami_load_fixed_pf))
        
    # Write the updated data to a new JSON file
    if not os.path.exists(new_runner_dir):
        os.makedirs(new_runner_dir)

    with open(new_runner_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    print(f"Updated data written to {new_runner_file}")

def modify_gridlabd_config_json(substation_name):

    template_runner_file = f"Runner_Files/Templates/glm_fed_config.json"

    new_runner_file_name = f"{substation_name}_glm_fed_config.json"
    new_runner_dir = f"Feeder_Data\\{substation_name}\\Config_Files"
    new_runner_file = os.path.join(new_runner_dir,new_runner_file_name)

    # Open and read the existing JSON file
    with open(template_runner_file, 'r') as infile:
        data = json.load(infile)
    
    # Check if the {Feeder_Name} exists in the data and replace it with the substation_name
    if 'logfile' in data:
        data['logfile'] = data['logfile'].replace('Feeder_Name', substation_name)
    
    # Write the updated data to a new JSON file
    if not os.path.exists(new_runner_dir):
        os.makedirs(new_runner_dir)

    with open(new_runner_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    print(f"Updated data written to {new_runner_file}")

def modify_ami_config_json(substation_name):

    template_runner_file = f"Runner_Files/Templates/ami_player_config.json"

    new_runner_file_name = f"{substation_name}_ami_player_config.json"
    new_runner_dir = f"Feeder_Data\\{substation_name}\\Config_Files"
    new_runner_file = os.path.join(new_runner_dir,new_runner_file_name)

    # Open and read the existing JSON file
    with open(template_runner_file, 'r') as infile:
        data = json.load(infile)
    
    # Check if the {Feeder_Name} exists in the data and replace it with the substation_name
    if 'logfile' in data:
        data['logfile'] = data['logfile'].replace('Feeder_Name', substation_name)
    
    # Write the updated data to a new JSON file
    if not os.path.exists(new_runner_dir):
        os.makedirs(new_runner_dir)

    with open(new_runner_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    print(f"Updated data written to {new_runner_file}")

def create_runner_files(substation_name, sim_start_time, sim_end_time, ami_load_fixed_pf,include_hc):
    modify_runner_json(substation_name, sim_start_time, sim_end_time, ami_load_fixed_pf,include_hc)
    modify_gridlabd_config_json(substation_name)
    modify_ami_config_json(substation_name)

def find_diff_GIS_GLM(substation_name):

    # open the parsed GLM model (in .pkl format)
    pkl_file_dir = f"Feeder_Data/{substation_name}/Python_Model/"
    pkl_file_name = f"{substation_name}_Model.pkl"
    pkl_file = os.path.join(pkl_file_dir,pkl_file_name)

    with open(pkl_file, 'rb') as file:
        Model = pickle.load(file)

    # get list of service numbers in GLM loads
    sn_list = []
    gen_list = []
    for load in Model.Loads:
        name = load.name
        sn_match = re.search(r"^_([0-9]*)_cons", name, re.S)
        sn_int_match = re.search(r"^_([0-9]*_[0-9])_cons", name, re.S)
        gen_match = re.search(r"gene_([0-9]*)_negLdGen", name, re.S)
        if sn_match:
            sn_list.append(sn_match.group(1))
        elif sn_int_match:
            sn_list.append(sn_int_match.group(1))
    for gen in Model.Generators:
        name = gen.name
        gen_match = re.search(r"gene_([0-9]*)_negLdGen", name, re.S)
        if gen_match:
            gen_list.append(gen_match.group(1))

    # get list of service numbers in GIS data
    load_dict = pd.read_csv(f"Feeder_Data/{substation_name}/meter_number_data.csv")
    gis_list = load_dict['Service Number'].tolist()
    gen_dict = pd.read_csv(f"Feeder_Data/{substation_name}/gen_meter_number_data.csv")
    gis_gen_list = [str(gen) for gen in gen_dict['Object ID'].tolist()]

    # find service numbers in GIS that are not in GLM
    diff_gis_glm = [sn for sn in gis_list if sn not in sn_list]
    diff_gis_glm_data = pd.DataFrame({
        "Service Number": diff_gis_glm,
    })
    diff_gen_gis_glm = [sn for sn in gis_gen_list if sn not in gen_list]
    diff_gen_gis_glm_data = pd.DataFrame({
        "Object ID": diff_gen_gis_glm,
    })

    diff_gis_glm_data.to_csv(f"Feeder_Data/{substation_name}/loads_in_gis_but_not_glm.csv", index=False)
    diff_gen_gis_glm_data.to_csv(f"Feeder_Data/{substation_name}/gens_in_gis_but_not_glm.csv", index=False)