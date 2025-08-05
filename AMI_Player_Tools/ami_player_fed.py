import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import helics as h
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Note that this is a bad fix. Should create a seperate python environment
import argparse

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)



def destroy_federate(fed):
    grantedtime = h.helicsFederateRequestTime(fed, h.HELICS_TIME_MAXTIME)
    status = h.helicsFederateDisconnect(fed)
    h.helicsFederateDestroy(fed)
    logger.info("Federate finalized")

def update_pub_json(curr_time,load_ami_data,gen_ami_data,load_dict,loads_to_skip,gen_dict,gens_to_skip,args): 
    load_ami_snap = load_ami_data[load_ami_data['start_date_time']==curr_time]
    gen_ami_snap = gen_ami_data[gen_ami_data['start_date_time']==curr_time]

    pub_json = "{\n"

    # Loop through AMI loads
    for index, row in load_dict.iterrows():
        service_num = row['Service Number']
        meter_num = row['Meter Number']
        if service_num not in loads_to_skip:
            load_name = f"_{service_num}_cons"
            pub_json += "\t\"{name}\": {{\n".format(name=load_name)
            load_phase = row['Phase']
            num_phases = len(load_phase)
            for ph_ind in range(num_phases):
                ph = load_phase[ph_ind]
                if str(meter_num) not in load_ami_snap.columns[1:]:
                    load_value_P = 0.0
                else:
                    load_value_P = 1000*(load_ami_snap[str(meter_num)].values[0])
                fixed_pf = args.load_fixed_pf
                load_value_Q = load_value_P*np.sign(fixed_pf)*np.sqrt(1/(fixed_pf**2)-1)
                load_value_str = f"{load_value_P/num_phases}{load_value_Q/num_phases:+}j" # For multi-phase loads this divides load evenly across all phases.
                load_string = "\t\t\"constant_power_{phase}\": \"{value}\"".format(phase=ph,value=load_value_str)
                pub_json += load_string
                if ph_ind < num_phases-1:
                    pub_json += ",\n"
            pub_json += "\n\t}"
            pub_json += ",\n"

    # # check if there is generation AMI
    # if len(gen_dict) == len(gens_to_skip):
    #     # no generation AMI, so remove last two characters (,\n)
    #     pub_json = pub_json[:-2]

    # Loop through AMI Generation
    for index, row in gen_dict.iterrows():
        object_id = row['Object ID']
        meter_num = row['Meter Number']
        if str(object_id) not in gens_to_skip:
            gen_name = f"gene_{object_id}_negLdGen"
            pub_json += "\t\"{name}\": {{\n".format(name=gen_name)
            gen_phase = row['Phase']
            num_phases = len(gen_phase)
            for ph_ind in range(num_phases):
                ph = gen_phase[ph_ind]
                if str(meter_num) not in gen_ami_snap.columns[1:]:
                    gen_value_P = 0.0
                else:
                    gen_value_P = -1000*(gen_ami_snap[str(meter_num)].values[0])
                fixed_pf = 1.00
                gen_value_Q = gen_value_P*np.sign(fixed_pf)*np.sqrt(1/(fixed_pf**2)-1)
                gen_value_str = f"{gen_value_P/num_phases}{gen_value_Q/num_phases:+}j" # For multi-phase gens this divides gen evenly across all phases.
                gen_string = "\t\t\"constant_power_{phase}\": \"{value}\"".format(phase=ph,value=gen_value_str)
                pub_json += gen_string
                if ph_ind < num_phases-1:
                    pub_json += ",\n"
            pub_json += "\n\t}"
            pub_json += ",\n"

    # Remove last two characters (,\n)
    pub_json = pub_json[:-2]
    pub_json += "\n}"

    # Uncomment this line to print json to log
    logger.info(pub_json)

    logger.info(f"Net load [kW]: {load_ami_snap.iloc[:,2:].sum(axis=1)-gen_ami_snap.iloc[:,2:].sum(axis=1)}")

    return pub_json


if __name__ == "__main__":

    ################################# Parse Command Line Arguments ############################################
    parser = argparse.ArgumentParser()
    parser.add_argument('substation_name', type=str, help='Substation_Name')
    parser.add_argument('start_time', type=str, help='Start time in YYYY-MM-DD hh:mm:ss format')
    parser.add_argument('end_time', type=str, help='End time in YYYY-MM-DD hh:mm:ss format')
    parser.add_argument('load_fixed_pf', type=float, help='Fixed power factor for AMI loads')
    args = parser.parse_args()

    substation_name = args.substation_name

    #################################  Registering  federate from json  ########################################
    fed = h.helicsCreateValueFederateFromConfig(f"Feeder_Data/{substation_name}/Config_Files/{substation_name}_ami_player_config.json")
    federate_name = h.helicsFederateGetName(fed)
    logger.info("HELICS Version: {}".format(h.helicsGetVersion()))
    logger.info("{}: Federate {} has been registered".format(federate_name, federate_name))
    pubkeys_count = h.helicsFederateGetPublicationCount(fed)
    subkeys_count = h.helicsFederateGetInputCount(fed)

    ######################   Reference to Publications and Subscription from index  #############################
    pubid = {}
    subid = {}
    for i in range(0, pubkeys_count):
        pubid["m{}".format(i)] = h.helicsFederateGetPublicationByIndex(fed, i)
        pubtype = h.helicsPublicationGetType(pubid["m{}".format(i)])
        pubname = h.helicsPublicationGetName(pubid["m{}".format(i)])
        logger.info("{}: Registered Publication ---> {}".format(federate_name, pubname))
    for i in range(0, subkeys_count):
        subid["m{}".format(i)] = h.helicsFederateGetInputByIndex(fed, i)
        h.helicsInputSetDefaultComplex(subid["m{}".format(i)], 0, 0)
        sub_key = h.helicsSubscriptionGetTarget(subid["m{}".format(i)])
        logger.info("{}: Registered Subscription ---> {}".format(federate_name, sub_key))

    ######################   AMI Reading Setup  ######################################

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

    ######################   Co-Simulation Timing Setup  ######################################

    # Define the start and end dates as strings
    start_date = args.start_time
    end_date = args.end_time

    # Generate the list of hourly datetimes
    datetimes_list = pd.date_range(start=start_date, end=end_date, freq='h', tz='US/Eastern')

    tsim_hr = len(datetimes_list) # in hours
    tstep_hr = 1 # in hours
    tsim = 60*60*tsim_hr # in seconds
    tstep = 60*60*tstep_hr # in seconds

    ######################   Entering Intializing Mode  ##########################################################
    h.helicsFederateEnterInitializingMode(fed)

    curr_time = datetimes_list[0]
    pub_json = update_pub_json(curr_time,load_ami_data,gen_ami_data,load_dict,loads_to_skip,gen_dict,gens_to_skip,args)

    for i in range(0, pubkeys_count):
        pub = pubid["m{}".format(i)]
        status = h.helicsPublicationPublishString(pub, pub_json)

    ######################   Entering Execution Mode  ##########################################################
    status = h.helicsFederateEnterExecutingMode(fed)

    #########################################   Starting Co-simulation  ####################################################

    grantedtime = -1

    for t in range(1, tsim_hr, tstep_hr):

        t_sec = 60*60*t
        curr_time = datetimes_list[t]
        logger.info(f"Processing {curr_time}...")
        
        ############################   Publishing Load and Gen to GridLAB-D #######################################################
        
        pub_json = update_pub_json(curr_time,load_ami_data,gen_ami_data,load_dict,loads_to_skip,gen_dict,gens_to_skip,args)

        for i in range(0, pubkeys_count):
            pub = pubid["m{}".format(i)]
            status = h.helicsPublicationPublishString(pub, pub_json)

        # logger.info("{} - {}".format(grantedtime, t_sec))
        while grantedtime < t_sec:
            grantedtime = h.helicsFederateRequestTime(fed, t_sec)
            logger.info("{} - {}".format(grantedtime, t_sec))

    ##############################   Terminating Federate   ########################################################

    while grantedtime < tsim:
        grantedtime = h.helicsFederateRequestTime(fed, tsim)
    logger.info("{}: Destroying federate".format(federate_name))
    destroy_federate(fed)
    logger.info("{}: Done!".format(federate_name))

