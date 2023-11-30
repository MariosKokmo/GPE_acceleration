import numpy as np
import pandas as pd
import json
import os
import math
from library.gpe_library import CONSTANTS


def _read_configuration_file(ConfigFile):
  """Reads the configuration file
  
  Args: str, the path to the file
  Returns: dictionary, the contents of the configuration file
  """
  cwd = os.getcwd()
  print(cwd)
  pathConfigFile = cwd + "/" + ConfigFile
  with open(pathConfigFile, 'r') as f:
    simulations = json.load(f)
  return simulations

def get_simulation_combinations(sims):
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
  imprinting_charge = sims["imprinting_charge"]
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
                                        "imprint_every":imprint_every[i], "repetitive":repetitive, "imprinting_charge":imprinting_charge[i]})
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

  Returns: List[List[str,dictionary]], List of Lists where the first
          item of evey inner list is the name of the simulation and 
          the second item is a dictionary of the simulation parameters
  """
  simulations = []
  for parameters in parameters_list:
    charges = parameters["vortex_charge"]
    imprinting_charge = parameters["imprinting_charge"]
    max_imprints = parameters["max_imprints"]
    imprint_every = parameters["imprint_every"]
    simulation_name = f"1vortex__initial{charges}_repetitive{imprinting_charge}__total_imprints{max_imprints}__every{imprint_every}_fps10"
    simulations.append([simulation_name, parameters])
    print("creating folder: ", simulation_name)
    if not os.path.isdir(simulation_name):
       os.mkdir(simulation_name)
    else:
      print(f"{simulation_name} folder already exists...skipping")
  return simulations

def _simulations_multi_vortex(parameters_list):
  # TODO: Create the simulations for experiments with
  # multiple vortices
  pass

def get_simulation_parameters(ConfigFilePath):
  """
  Returns simulation parameters as read from the configuration file
  after adding some more.

  Args: str, the configuration file path

  Returns: dict, dictionary of the simulation parameters.
  """
  pi = CONSTANTS.pi
  # read the file
  sim_params = _read_configuration_file(ConfigFilePath)
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
  vortex_excitation = sim_params["vortex_excitation"]
  vortex_charge = sim_params["vortex_charge"]
  imprinting_charge = sim_params["imprinting_charge"]
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
      "Grid_resolution":[n1, n2, n3],
      "x_min":x_min,
      "x_max":x_max,
      "Trapping_frequencies":sim_params["Trapping_frequencies"],
      "Potential_type":sim_params["Potential_type"],
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
      "vortex_excitation":vortex_excitation,
      "vortex_charge":vortex_charge,
      "imprinting_charge":imprinting_charge,
      "vortex_position_x":vortex_position_x,
      "vortex_position_y":vortex_position_y,
      "imprint_every":imprint_every,
      "max_imprints":max_imprints,
      "repetitive":repetitive
  }
  ok, msg = _check_simulation_parameters(simulation_params)
  if not ok:
    print(msg)
  return simulation_params, msg

def _check_simulation_parameters(simulation_params):
  """Performs checks of the simulation parameters
  Args: dict, the simulation parameters
  Returns: 
    ok: bool, True if the checks pass, otherwise False
    msg: str, the fault that was detected
  """
  ###################################
  ####### Perform grid checks #######
  ###################################
  # the minimums should be negative
  for index, x in enumerate(simulation_params["x_min"]):
        if x > 0:
          msg = f"x_min for axis {index+1} is not negative. Grid is assumed symmetric."
          return False, msg
  # the grid should be symmetric
  xmins = simulation_params["x_min"]
  xmaxs = simulation_params["x_max"]
  for index, x_min in enumerate(xmins):
      if abs(x_min) != abs(xmaxs[index]):
          msg = f"{index+1} max and min are not symmetric. Grid is assumed symmetric."
          return False, msg
  
  ##################################
  #### Perform frequency checks ####
  ##################################
  for index, freq in enumerate(simulation_params["Trapping_frequencies"]):
      if freq <= 0:
          msg = f"Frequency {index+1} is negative or zero. Frequencies are assumed positives."
          return False, msg
  
  ##################################
  #### Perform vortex checks #######
  ##################################
  # Check the number of combinations
  if len(simulation_params["vortex_charge"]) != len(simulation_params["vortex_position_x"]):
      msg = f"The number of vortex charges doesn't agree with the number of x positions"
      return False, msg
  if len(simulation_params["vortex_position_y"]) != len(simulation_params["vortex_position_x"]):
      msg = f"The number of x positions doesn't agree with the number of y positions"
      return False, msg
  if len(simulation_params["vortex_charge"]) != len(simulation_params["imprinting_charge"]):
      msg = f"The number of initial vortex charges doesn't agree with the number of imprinted charges"
      return False, msg
  for index, charges in enumerate(simulation_params["vortex_charge"]):
      if isinstance(charges, list):
          if len(charges) != len(simulation_params["vortex_position_x"][index]):
              msg = f"The number of charges doesn't agree with the number of x positions at index {index}"
              return False, msg
          if len(charges) != len(simulation_params["vortex_position_y"][index]):
              msg = f"The number of charges doesn't agree with the number of y positions at index {index}"
              return False, msg
          if len(charges) != len(simulation_params["imprinting_charge"][index]):
              msg = f"The number of initial charges {len(charges)}, doesn't agree with the number of imprinted {len(simulation_params['imprinting_charge'][index])}"
              return False, msg
           
  return True, None