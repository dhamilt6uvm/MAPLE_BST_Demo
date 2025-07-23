import GLM_Tools.parsing_tools as glm_parser
import GLM_Tools.modif_tools as glm_modif_tools
import os

substation_name = "Burton_Hill"

# subfeeder_name = "3A"
# start_node = "_43-7-3A_n"
# branches_to_remove = ["_43-7-3A"]

subfeeder_name = "small00"
start_node = "span_51271_n"
branches_to_remove = ["span_51271"]

glm_modif_tools.create_subfeeder(substation_name, subfeeder_name, start_node, branches_to_remove)


