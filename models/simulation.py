"""Provides the class for the simulation"""
from utils.setup_simulations import get_simulation_parameters
import library.gpe_evolution as gpe_evolution
import os
from pathlib import Path

class Simulation:
    def __init__(self, parameters, app):
        self.parameters
        self.simulation_combinations
        self.logger = app.logger
        self.time = app.time
        self.device = app.device

    def read_parameters(self, configFile):
        pass

    def run_simulations(self):
        # Run the simulations
        for combination in simulation_combinations:
            simulation_name, parameters = combination
            if not os.path.isdir(simulation_name):
                self.logger.write(f"[INFO] {self.time} -- The simulation folder {simulation_name} does not exist. Creating now...")
                os.mkdir(simulation_name)
            
            # Simulation parameters
            imprint_every = parameters["imprint_every"]
            max_imprints = parameters["max_imprints"]
            charges = parameters["vortex_charge"]
            imprinting_charge = parameters["imprinting_charge"]
            repetitive = parameters["repetitive"]
            vortex_position_x = parameters["vortex_position_x"]
            vortex_position_y = parameters["vortex_position_y"]

            # change the working folder and run the simulation
            os.chdir(os.getcwd() + "/" + simulation_name)
            self.logger.write(f"[INFO] {self.time} -- Currently in: {os.getcwd()}\n\n")
            self.logger.write(f"[INFO] {self.time} -- Running: {simulation_name}, started at {self.time}\n")
            gpe_evolution.run_simulation(
                            max_imprints=max_imprints,\
                            imprint_again_every=imprint_every,\
                            charge=charges,\
                            imprinting_charge = imprinting_charge,\
                            vort_x=vortex_position_x,\
                            vort_y=vortex_position_y,\
                            delay_to_first_reimprint=3,\
                            repetitive=repetitive,\
                            ground_state=gs_path,\
                            device=self.device,\
                            sim_params=simulation_parameters,\
                            logfile=self.logger
                            )
            # go back to the parent directory to run the next sim
            path = Path(os.getcwd())
            parent_path = path.parent.absolute()
            os.chdir(parent_path)