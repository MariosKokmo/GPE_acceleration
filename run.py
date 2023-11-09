"""Entry point script for running the simulations.
It sets up the folders and all simulations described in the
JSON configuration file. Then proceeds on running these simulations in sequence.

Note that all simulations for a specific configuration file must be on the same grid size
and the same external potential.
"""
import library.gpe_evolution as gpe_evolution
import utils.setup_simulations as setup_simulations
import library.ground_state as ground_state
from library.gpe_library import CONSTANTS
import os
from pathlib import Path
import torch
from datetime import datetime

logfile = open("log.txt", "w")
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = 'cpu'
print('Running on ', DEVICE)
logfile.write(f'Running on {DEVICE}. Started at {datetime.now()}\n')

pi = CONSTANTS.pi
hbar = CONSTANTS.hbar
m1 = CONSTANTS.m1
a_bohr = CONSTANTS.a_bohr
ascat = CONSTANTS.ascat
nat = CONSTANTS.nat

##############################################################################
#####################   SETUP ALL SIMULATIONS   ##############################
##############################################################################
simulation_parameters = setup_simulations.get_simulation_parameters("configuration_file.json")
simulation_combinations = setup_simulations.get_simulations(simulation_parameters)
print("----------------------------------------")
logfile.write("----------------------------------------\n")

##############################################################################
# grid and frequencies
n1,n2,n3 = simulation_parameters["grid"]
fx,fy,fz = simulation_parameters["frequencies"]

##############################################################################
# find ground state for the specific grid and potential if it doesn't exist
gs_file = f"{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat"
if not os.path.exists(gs_file):
    print("Calculating ground state...")
    logfile.write("Calculating ground state...\n")
    _ = ground_state.find_ground_state(simulation_parameters, gs_file, device=DEVICE)
logfile.write(f"Ground state file: {gs_file}\n")
gs_path = os.getcwd() + "/" + gs_file

##############################################################################
# Run the simulations
for combination in simulation_combinations:
    simulation_name, parameters = combination
    if not os.path.isdir(simulation_name):
        logfile.write(f"The simulation folder {simulation_name} does not exist. Creating now...")
        print(f"The simulation folder {simulation_name} does not exist. Creating now...")
        os.mkdir(simulation_name)
    imprint_every = parameters["imprint_every"]
    max_imprints = parameters["max_imprints"]
    charges = parameters["vortex_charge"]
    repetitive = parameters["repetitive"]
    vortex_position_x = parameters["vortex_position_x"]
    vortex_position_y = parameters["vortex_position_y"]

    # change the working folder
    os.chdir(os.getcwd() + "/" + simulation_name)
    print()
    print("Currently in: ",os.getcwd())
    logfile.write(f"Currently in: {os.getcwd()}\n")
    print()
    print("Running: ", simulation_name)
    logfile.write(f"Running: {simulation_name}, started at {datetime.now()}\n")
    gpe_evolution.run_simulation(
                    max_imprints=max_imprints,\
                    imprint_again_every=imprint_every,\
                    charge=charges,\
                    vort_x=vortex_position_x,\
                    vort_y=vortex_position_y,\
                    delay_to_first_reimprint=3,\
                    repetitive=repetitive,\
                    ground_state=gs_path,\
                    device=DEVICE,\
                    sim_params=simulation_parameters,\
                    logfile=logfile
                    )
    # go back to the parent directory to run the next sim
    path = Path(os.getcwd())
    parent_path = path.parent.absolute()
    os.chdir(parent_path)

###############################################################################
# Close the log file. END
logfile.write(f"Finished all simulations at {datetime.now()}")
logfile.close()
