import numpy as np
import pandas as pd
import json
import os
import math
from gpe_library import CONSTANTS

def read_configuration_file(ConfigFile):
  cwd = os.getcwd()
  print(cwd)
  pathConfigFile = cwd + "\\" + ConfigFile
  with open(pathConfigFile, 'r') as f:
    simulations = json.load(f)
  return simulations

def get_simulations(sims):
  """
  Creates the distinct simulations to be run.
  Assigns names and parameters to the simulations.

  Args: sims, the dictionary of all the simulation parameters
  
  Returns: simulations, list of lists.
      Each list contains the folder name and a dictionary of the parameters for the simulation.
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
      parameters_repetitive.append({"vortex_charge":charges[i], "vortex_position_x":vortex_position_x[i],\
                                     "vortex_position_y":vortex_position_y[i], "max_imprints":max_imprints[i],\
                                        "imprint_every":imprint_every[i], "repetitive":repetitive})
    simulations = _simulations_repetitive(parameters_repetitive)
  else:
    parameters_multi_vortex = []
    for i in range(len(charges)):
      parameters_multi_vortex.append({"vortex_charge":charges[i], "vortex_position_x":vortex_position_x[i],\
                                       "vortex_position_y":vortex_position_y[i], "max_imprints":max_imprints[i],\
                                          "imprint_every":imprint_every[i], "repetitive":repetitive})
    simulations = _simulations_multi_vortex(parameters_multi_vortex)
  return simulations

def _simulations_repetitive(parameters_list):
  """
  Creates folders of simulations and returns their names in a list.
  Args: parameters_list, list of dictionaries.
  """
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

def _simulations_multi_vortex(parameters_list):
  pass

def get_simulation_parameters(ConfigFilePath):
  """
  Returns simulation parameters as read from the configuration file
  after adding some more
  """
  pi = CONSTANTS.pi
  sim_params = read_configuration_file(ConfigFilePath)
  # Grid
  n1, n2, n3 =sim_params["Grid_resolution"]
  dim = np.array([n1,n2,n3], dtype=np.float64)
  x_min = np.array(sim_params["Grid_negative_limits"])
  x_max = np.array(sim_params["Grid_positive_limits"])

  # frequencies
  fx, fz, fy = sim_params["Trapping_frequencies"]
  wx = 2*pi*float(fx)
  wy = 2*pi*float(fy)
  wz = 2*pi*float(fz)
  w = np.array([wx,wz,wy])
  omega_ho = (wx*wy*wz)**(1/3)

  # Time steps
  t_evol = sim_params["Total_simulation_time"]
  dt = sim_params["dt"]
  shots = sim_params["snapshots"]
  dtau = omega_ho*dt
  kmax = int(t_evol//dt)

  # Vortex
  vortex_charge = sim_params["vortex_charge"]
  vortex_position_x = sim_params["vortex_position_x"]
  vortex_position_y = sim_params["vortex_position_y"]

  # Re-imprint
  imprint_every = sim_params["imprint_every"]
  max_imprints = sim_params["max_imprints"]
  repetitive = sim_params["repetitive"]

  # harmonic potential length scale in meters
  a_ho = math.sqrt(CONSTANTS.hbar/CONSTANTS.m1/omega_ho)  

  # interaction strength
  u = 4.* CONSTANTS.pi * CONSTANTS.nat * CONSTANTS.ascat/a_ho 

  # normalizations
  w = w/omega_ho
  x_max = x_max * 1e-6/a_ho
  x_min = x_min * 1e-6/a_ho

  dx = (x_max - x_min)/dim
  dp = 2*pi/(x_max - x_min)
  d_x = np.prod(dx)

  # sim params will be used throughout
  simulation_params = {
      "grid":[n1, n2, n3],
      "x_min":x_min,
      "x_max":x_max,
      "frequencies":sim_params["Trapping_frequencies"],
      "w":w,
      "dx":dx,
      "dp":dp,
      "dtau":dtau,
      "kmax":kmax,
      "u":u,
      "a_ho":a_ho,
      "omega_ho":omega_ho,
      "d_x":d_x,
      "shots":shots,
      "dt":dt,
      "vortex_charge":vortex_charge,
      "vortex_position_x":vortex_position_x,
      "vortex_position_y":vortex_position_y,
      "imprint_every":imprint_every,
      "max_imprints":max_imprints,
      "repetitive":repetitive
  }
  ok = _check_simulation_parameters(simulation_params)
  if not ok:
    raise Exception("there is an error in the configuration")
  return simulation_params

def _check_simulation_parameters(simulation_params):
  return True