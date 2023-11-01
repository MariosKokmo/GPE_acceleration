# -*- coding: utf-8 -*-
"""GPE evolution. 
Implements a split-step Fourier algorithm for simulating the evolution of a BEC using the Gross-Pitaevskii equation.
The code automatically utilises CUDA acceleration (GPU) if it is available.

Current imprinting objects are only vortices.
There is the possibility of repetitively imprinting vortices into larger and larger objects.

# -*- coding: utf-8 -*-
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

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = 'cpu'
print('Running on ', DEVICE)

##############################################################################
#####################   GRID PARAMETERS      #################################
##############################################################################
ground_state = os.getcwd() + "\\" + '3d-psi_60.dat'

# Grid dimensions
n1=512
n2=16
n3=512
dim = np.array([n1,n2,n3], dtype=np.float64)

x_min = np.array([-60, -1.5, -60])
x_max = np.array([60, 1.5, 60])

fx=20
fz=300
fy=20

##############################################################################
##############################################################################
##############################################################################
wx = 2*pi*float(fx)
wy = 2*pi*float(fy)
wz = 2*pi*float(fz)
w = np.array([wx,wz,wy])

omega_ho = (wx*wy*wz)**(1/3)
a_ho = math.sqrt(hbar/m1/omega_ho)  # harmonic potential length scale in meters
ascat = 99*a_bohr	# scattering length
nat = 5e+4		# number of atoms

u = 4.*pi*nat*ascat/a_ho # interaction strength

t_evol = 150e-3
dt = 5e-7
shots = 150
dtau = omega_ho*dt
kmax = int(t_evol//dt)


w = w/omega_ho
x_max = x_max * 1e-6/a_ho
x_min = x_min * 1e-6/a_ho

dx = (x_max - x_min)/dim
dp = 2*pi/(x_max - x_min)

d_x = np.prod(dx)

def run_simulation(max_imprints, imprint_again_every, \
                    charge, vort_x=0, vort_y=0, \
                      delay_to_first_reimprint=3, repetitive=True):
  """
  Runs one simulation with specific parameters.
  """
  # imprint the batch every this number of snapshots
  imprint_again_every = imprint_again_every 
  max_imprints = max_imprints
  # snapshots until first imprint, excludes initial vortex
  delay_to_first_reimprint = delay_to_first_reimprint 
  repetitive = repetitive

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
  phase = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=DEVICE)
  uext1 = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=DEVICE)
  psi1 = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=DEVICE)
  x1 = torch.zeros((1,n1), dtype=torch.float64, device=DEVICE)
  x2 = torch.zeros((1,n2), dtype=torch.float64, device=DEVICE)
  x3 = torch.zeros((1,n3), dtype=torch.float64, device=DEVICE)
  p1 = torch.zeros((1,n1), dtype=torch.float64, device=DEVICE)
  p2 = torch.zeros((1,n2), dtype=torch.float64, device=DEVICE)
  p3 = torch.zeros((1,n3), dtype=torch.float64, device=DEVICE)
  p_sq = torch.zeros((n1,n2,n3), dtype=torch.float64, device=DEVICE)

  ##############################################################################
  ##############    SET UP THE SIMULATION    ###################################
  ##############################################################################
  # initialize grids and external potential
  uext1, x1, x2, x3, p1, p2, p3, p_sq = init_state(x1, x2, x3, p1, p2, p3, x_min, x_max, dx, dp, w, n1, n2, n3, uext1)
  uext1 = uext1.to(device=DEVICE)

  # read the ground state
  psi1 = read_ground_state(ground_state, psi1, n1, n2, n3) # shape (n1,n2,n3) complex
  psi1 = torch.tensor(psi1, device=DEVICE) # dtype complex128

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

if __name__ == '__main__':
  # imprint the batch every this number of snapshots
  imprint_again_every = [15, 20, 40, 50, 40, 50] 
  max_imprints = [2, 5, 2, 2, 1, 2]
  charges = [2, 2, 5, 5, 10, 10]
  parameters_list = zip(max_imprints, imprint_again_every, charges)
  # create folders of simulations
  simulations = []
  for parameters in parameters_list:
    imprints, every, charge = parameters
    simulation_name = f'1vortex__batch{charge}__total_imprints{imprints}__every{every}_fps10'
    simulations.append(simulation_name)
    print("creating folder: ", simulation_name)
    if not os.path.isdir(simulation_name):
       os.mkdir(simulation_name)
    
    os.chdir(os.getcwd() + "\\" + simulation_name)
    print(os.getcwd())
    print()
    print("Running: ", simulation_name)
    run_simulation(imprints, every, charge)
    path = Path(os.getcwd())
    parent_path = path.parent.absolute()
    os.chdir(parent_path)