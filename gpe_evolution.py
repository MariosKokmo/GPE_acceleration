# -*- coding: utf-8 -*-
"""GPE evolution. 
Implements a split-step Fourier algorithm for simulating the evolution of a BEC using the Gross-Pitaevskii equation.
The code automatically utilises CUDA acceleration (GPU) if it is available.

Current imprinting objects are only vortices.
There is the possibility of repetitively imprinting vortices into larger and larger objects.

'''
Created on Sat September 9 21:38:24 2023

@author: Marios Kokmotos
'''
"""

##############################################################################
################   IMPORT LIBRARIES    #######################################
##############################################################################
import math
import numpy as np
import torch
import os
from pathlib import Path
from gpe_library import *

def run_simulation(max_imprints,\
                   imprint_again_every,\
                   charge,\
                   vort_x,\
                   vort_y,\
                   delay_to_first_reimprint,\
                   repetitive,\
                   ground_state,\
                   device,\
                   sim_params
                  ):
  """
  Runs one simulation with specific parameters.
  """
  # imprint the batch every this number of snapshots
  imprint_again_every = imprint_again_every 
  max_imprints = max_imprints
  # snapshots until first imprint, excludes initial vortex
  delay_to_first_reimprint = delay_to_first_reimprint 
  repetitive = repetitive
  
  grid = sim_params["grid"]
  n1, n2, n3 = grid
  ##############################################################################
  ##############    VORTEX DATA (to be modified)   #############################
  ##############################################################################
  vort_x = np.array([vort_x])
  vort_y = np.array([vort_y])
  vort_charge = np.array([charge])
  vortices = np.vstack((vort_x,vort_y,vort_charge))

  ##############################################################################
  ##############    EMPTY MATRICES TO FIT DATA    ##############################
  ##############################################################################
  phase = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=device)
  uext1 = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=device)
  psi1 = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=device)
  x1 = torch.zeros((1,n1), dtype=torch.float64, device=device)
  x2 = torch.zeros((1,n2), dtype=torch.float64, device=device)
  x3 = torch.zeros((1,n3), dtype=torch.float64, device=device)
  p1 = torch.zeros((1,n1), dtype=torch.float64, device=device)
  p2 = torch.zeros((1,n2), dtype=torch.float64, device=device)
  p3 = torch.zeros((1,n3), dtype=torch.float64, device=device)
  p_sq = torch.zeros((n1,n2,n3), dtype=torch.float64, device=device)

  ##############################################################################
  ##############    SET UP THE SIMULATION    ###################################
  ##############################################################################
  # initialize grids and external potential
  x_min = sim_params["x_min"]
  x_max = sim_params["x_max"]
  dx = sim_params["dx"]
  dp = sim_params["dp"]
  w = sim_params["w"]
  kmax = sim_params["kmax"]
  omega_ho = sim_params["omega_ho"]
  a_ho = sim_params["a_ho"]
  shots = sim_params["shots"]
  dtau = sim_params["dtau"]
  d_x = sim_params["d_x"]
  dt = sim_params["dt"]
  u = sim_params["u"]

  uext1, x1, x2, x3, p1, p2, p3, p_sq = init_state(x1, x2, x3, p1, p2, p3, x_min, x_max, dx, dp, w, n1, n2, n3, uext1)
  uext1 = uext1.to(device=device)

  # read the ground state
  psi1 = read_ground_state(ground_state, psi1, n1, n2, n3) # shape (n1,n2,n3) complex
  psi1 = torch.tensor(psi1, device=device) # dtype complex128

  # calculate the phase needed to imprint the vortices on the ground state (takes a while because of the for loops)
  phase = imprint_vortices(vortices, phase, x1, x2, x3, n1, n2, n3)
  init_phase = phase.detach()

  # imprint the vortices. Initial imprint
  psi1 = update_phase(psi1, phase, n1, n2, n3)

  ##############################################################################
  ##############    MAIN LOOP OF SIMULATION    #################################
  ##############################################################################

  num_imprints = 0 # there has already been 1 imprint, the initial one
  count = 0
  
  if repetitive:
    print(f"Will imprint every {imprint_again_every} snapshots for {max_imprints} times")

  for iteration in range(kmax):
    t = dt*iteration*omega_ho
    utot1 = u*torch.abs(psi1)**2 + uext1 # Total potential shape (n1,n2,n3)

    if (iteration%(kmax/shots) == 0):
        write_data(psi1, count, x1, x3, n1, n3, a_ho)
        count += 1
        print('t = ', t/omega_ho)

    # Repetitive imprinting
    if (iteration%((kmax//shots)*imprint_again_every) == 0) and (num_imprints < max_imprints) and repetitive and (count > delay_to_first_reimprint):
      num_imprints += 1
      print("Imprinting again...")
      # extract current phase of psi1
      cur_phase = extract_phase(psi1)
      # add the new vortices (init_phase)
      new_phase = add_phase(cur_phase, init_phase)
      # update the phase
      psi1 = update_phase(psi1, new_phase, n1, n2, n3)

    # split-step evolution
    psi1 = x_evolution(psi1, utot1, dtau)
    psi1 = p_evolution(psi1, dtau, p_sq)
    psi1 = x_evolution(psi1, utot1, dtau)
    psi1 = normalize(psi1, d_x)

##########################################################################################
##########################################################################################
