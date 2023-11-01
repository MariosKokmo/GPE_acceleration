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
  """Creates the simulation folders"""
  imprint_again_every = sims['period_of_reimprint']
  max_imprints = sims['max_imprints']
  charges = sims['vortex_charge']
  repetitive = sims['repetitive']
  assert repetitive==0 or repetitive==1
  if repetitive:
    max_imprints = sims['max_imprints']
    assert len(max_imprints) >= 1, "max_imprints is not correct in configuration file"
    assert len(charges) == len(imprint_again_every)
    if len(max_imprints) == 1:
      max_imprints = [max_imprints[0]]*len(imprint_again_every)
    parameters_repetitive = zip(max_imprints, imprint_again_every, charges)
    folders_repetitive(parameters_repetitive)
  else:
    parameters_multi_vortex = zip(max_imprints, imprint_again_every, charges)
    folders_multi_vortex(parameters_multi_vortex)

def folders_repetitive(parameters_list):
  """create folders of simulations and returns their names in a list"""
  simulations = []
  for parameters in parameters_list:
    imprints, every, charge = parameters
    simulation_name = f'1vortex__batch{charge}__total_imprints{imprints}__every{every}_fps10'
    simulations.append(simulation_name)
    print("creating folder: ", simulation_name)
    if not os.path.isdir(simulation_name):
       os.mkdir(simulation_name)
    else:
      print(f"{simulation_name} folder already exists...skipping")
  return simulations

def folders_multi_vortex(parameters_list):
  pass