import AMI_Player_Tools.setup_tools as setup_tools
import GLM_Tools.parsing_tools as glm_parser
import GLM_Tools.modif_tools as glm_modif_tools
import os

substation_name = "Burton_Hill_small02"

# Meter Number Dictionary Settings
build_meter_dicts_flag = False # Set this to False if you already built these CSV files

# MySQL Query Writer Settings
generate_MySQL_query_flag = False # Set this to False if you already have AMI data
mysql_query_start_time = "2024-01-01 00:00:00"
mysql_query_end_time = "2025-01-01 00:00:00"
max_meters_per_query = 1000 # Set this to avoid massive queries that time out. This will break up into multiple queries.

# AMI Data Parsing Settings
parse_ami_data_flag = False
save_15min_flag = False

# Parse GLM Settings
parse_glm_flag = True

# Create Simulation Settings
create_new_sim_flag = True
sim_start_time = '2024-01-01 00:00:00'
sim_end_time = '2025-01-01 00:00:00'
ami_load_fixed_pf = 0.98
include_hc = False
regulator_control = "MANUAL" # use "DEFAULT" to not change regulator controls

# Optimization Settings
add_ami_to_pkl_flag = True

# Visualization Settings
add_coords_to_pkl_flag = True

#############################################################################################################

if build_meter_dicts_flag:
    setup_tools.get_meter_numbers(substation_name)

if generate_MySQL_query_flag:
    setup_tools.query_writer(substation_name, mysql_query_start_time, mysql_query_end_time, max_meters_per_query)
    print("Please perform MySQL queries and add AMI data to the proper folder. Set generate_MySQL_query_flag to False and re-run this script to continue.")
    exit()

if parse_ami_data_flag:
    setup_tools.parse_ami_data(substation_name, "Load", save_15min_flag)
    setup_tools.parse_ami_data(substation_name, "Gen", save_15min_flag)
    setup_tools.calculate_true_load(substation_name)

if parse_glm_flag:
    glm_parser.parse_glm_to_pkl(substation_name)

if create_new_sim_flag:
    # make sure the appropriate Output_Data folder exists
    sim_output_dir = f"Feeder_Data/{substation_name}/Output_Data"
    if not os.path.exists(sim_output_dir):
        os.makedirs(sim_output_dir)
    # modify glm clock to correct datetimes and build new runner files
    glm_modif_tools.modify_glm_clock(substation_name, sim_start_time, sim_end_time)
    # modify regulators (if desired)
    if not (regulator_control == "DEFAULT"):
        glm_modif_tools.modify_reg_controls(substation_name,regulator_control)
    # set up HELICS runner files
    setup_tools.find_diff_GIS_GLM(substation_name)
    setup_tools.create_runner_files(substation_name, sim_start_time, sim_end_time, ami_load_fixed_pf, include_hc)
    print(f"Co-simulation runner files have been created. To run the co-simulation, copy and paste the following command into the command line:")
    print(f"helics run --path Runner_Files/{substation_name}/{substation_name}_cosim_runner.json")

if add_ami_to_pkl_flag:
    print(f"Populating pkl file with AMI data...")
    glm_parser.populate_ami_loads_pkl(substation_name, sim_start_time, sim_end_time, ami_load_fixed_pf)
    print(f"Done populating pkl file with AMI data.")

if add_coords_to_pkl_flag:    
    print(f"Populating pkl file with coordinate data...")
    glm_parser.add_coords_to_pkl(substation_name)
    print(f"Done populating pkl file with coordinate data.")
    print(f"Plotting feeder...")
    glm_parser.plot_feeder(substation_name)
    print(f"Done plotting feeder.")
