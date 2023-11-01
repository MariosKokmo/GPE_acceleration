"""Entry point script for running the simulations"""
import gpe_evolution
import setup_simulations
import ground_state
from gpe_library import CONSTANTS
import os
from pathlib import Path
import torch
import math
import numpy as np

logfile = open("log.txt", "w")
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = 'cpu'
print('Running on ', DEVICE)
logfile.write(f'Running on {DEVICE}\n')

pi = CONSTANTS.pi
hbar = CONSTANTS.hbar
m1 = CONSTANTS.m1
a_bohr = CONSTANTS.a_bohr
ascat = CONSTANTS.ascat
nat = CONSTANTS.nat

##############################################################################
#####################   SETUP SIMULATIONS   ##################################
##############################################################################
simulations = setup_simulations.read_configuration_file("configuration_file.json")
simulation_details = setup_simulations.create_sims_folders(simulations)

# Grid
n1, n2, n3 =simulations["Grid_resolution"]
dim = np.array([n1,n2,n3], dtype=np.float64)
x_min = np.array(simulations["Grid_negative_limits"])
x_max = np.array(simulations["Grid_positive_limits"])

# frequencies
fx, fz, fy = simulations["Trapping_frequencies"]
wx = 2*pi*float(fx)
wy = 2*pi*float(fy)
wz = 2*pi*float(fz)
w = np.array([wx,wz,wy])
omega_ho = (wx*wy*wz)**(1/3)

# Time steps
t_evol = simulations["Total_simulation_time"]
dt = simulations["dt"]
shots = simulations["snapshots"]
dtau = omega_ho*dt
kmax = int(t_evol//dt)

# Vortices
charges = simulations["vortex_charge"]
vort_x = simulations["vortex_position_x"]
vort_y = simulations["vortex_position_y"]

# Repetitive imprinting
repetitive = simulations["repetitive"]
max_imprints = simulations["max_imprints"]
imprint_every = simulations["imprint_every"]


# find ground state for the specific grid and potential if it doesn't exist
gs_path = f"{n1}x{n2}_{fx}-{fz}_ground_state.dat"
if not os.path.exists(gs_path):
    gs_path = ground_state.find_ground_state()

gs = os.getcwd() + "\\" + gs_path

##############################################################################
##############################################################################
##############################################################################

a_ho = math.sqrt(hbar/m1/omega_ho)  # harmonic potential length scale in meters

u = 4.*pi*nat*ascat/a_ho # interaction strength

w = w/omega_ho
x_max = x_max * 1e-6/a_ho
x_min = x_min * 1e-6/a_ho

dx = (x_max - x_min)/dim
dp = 2*pi/(x_max - x_min)
d_x = np.prod(dx)

# sim params will be used throughout
sim_params = {
    "grid":[n1, n2, n3],
    "x_min":x_min,
    "x_max":x_max,
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
    "dt":dt
}

for combination in simulation_details:
    simulation_name, parameters = combination
    if not os.path.isdir(simulation_name):
        logfile.write(f"The simulation folder {simulation_name} does not exist. Simulation was not run.")
        print(f"The simulation folder {simulation_name} does not exist. Simulation was not run.")

    imprint_every = parameters["imprint_every"]
    max_imprints = parameters["max_imprints"]
    charges = parameters["vortex_charge"]
    repetitive = parameters["repetitive"]
    vortex_position_x = parameters["vortex_position_x"]
    vortex_position_y = parameters["vortex_position_y"]

    os.chdir(os.getcwd() + "\\" + simulation_name)
    print()
    print("Currently in: ",os.getcwd())
    logfile.write(f"Currently in: {os.getcwd()}\n")
    print()
    print("Running: ", simulation_name)
    logfile.write(f"Running: {simulation_name}")
    gpe_evolution.run_simulation(
                    max_imprints=max_imprints,\
                    imprint_again_every=imprint_every,\
                    charge=charges,\
                    vort_x=vortex_position_x,\
                    vort_y=vortex_position_y,\
                    delay_to_first_reimprint=3,\
                    repetitive=repetitive,\
                    ground_state=gs,\
                    device=DEVICE,\
                    sim_params=sim_params)
    path = Path(os.getcwd())
    parent_path = path.parent.absolute()
    os.chdir(parent_path)
logfile.close()