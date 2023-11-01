import numpy as np
import pandas as pd
import json
import os

def read_configuration_file(ConfigFile):
  cwd = os.getcwd()
  print(cwd)
  pathConfigFile = cwd + "\\" + ConfigFile
  with open(pathConfigFile, 'r') as f:
    simulations = json.load(f)
  return simulations

def create_sims_folders(sims):
  """Creates the simulation folders.
  Args: sims, the json configuration file
  
  Returns: folders, list of lists.
      Each list contains the folder name and a dictionary of the parameters.
  """
  imprint_every = sims["imprint_every"]
  max_imprints = sims["max_imprints"]
  charges = sims["vortex_charge"]
  repetitive = sims["repetitive"]
  vortex_position_x = sims["vortex_position_x"]
  vortex_position_y = sims["vortex_position_y"]
  assert repetitive==0 or repetitive==1
  if repetitive:
    max_imprints = sims['max_imprints']
    assert len(max_imprints) >= 1, "max_imprints is not correct in configuration file"
    assert len(charges) == len(imprint_every)
    if len(max_imprints) == 1:
      max_imprints = [max_imprints[0]]*len(imprint_every)
    
    parameters_repetitive = []
    for i in range(len(charges)):
      parameters_repetitive.append({"vortex_charge":charges[i], "vortex_position_x":vortex_position_x[i], "vortex_position_y":vortex_position_y[i], "max_imprints":max_imprints[i], "imprint_every":imprint_every[i], "repetitive":repetitive})
    folders = folders_repetitive(parameters_repetitive)
  else:
    parameters_multi_vortex = []
    for i in range(len(charges)):
      parameters_multi_vortex.append({"vortex_charge":charges[i], "vortex_position_x":vortex_position_x[i], "vortex_position_y":vortex_position_y[i], "max_imprints":max_imprints[i], "imprint_every":imprint_every[i], "repetitive":repetitive})
    folders = folders_multi_vortex(parameters_multi_vortex)
  return folders

def folders_repetitive(parameters_list):
  """create folders of simulations and returns their names in a list.
  parameters_list is a list of dictionaries."""
  simulations = []
  for parameters in parameters_list:
    charges = parameters["vortex_charge"]
    max_imprints = parameters["max_imprints"]
    imprint_every = parameters["imprint_every"]
    simulation_name = f'1vortex__batch{charges}__total_imprints{max_imprints}__every{imprint_every}_fps10'
    simulations.append([simulation_name, parameters])
    print("creating folder: ", simulation_name)
    if not os.path.isdir(simulation_name):
       os.mkdir(simulation_name)
    else:
      print(f"{simulation_name} folder already exists...skipping")
  return simulations

def folders_multi_vortex(parameters_list):
  pass
