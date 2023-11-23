"""Provides the class for the simulation"""
from utils.setup_simulations import set_up_simulations
import library.gpe_evolution as gpe_evolution
from models.BEC import BEC
import os
from pathlib import Path

class Simulation:
    def __init__(self, system, app):
        self.simulation_combinations
        self.app = app
        self.logger = self.app.logger
        self.time = self.app.time
        self.device = self.app.device
        self.BEC
        self.system = system
    
    def setup_simulations(self, configFile):
        """
        Creates the simulation combinations to be run.
        """
        self.simulation_combinations = set_up_simulations(configFile)

    def run_simulations(self):
        """
        Sets up the simulation and the BEC.
        """
        # Run the simulations
        for combination in self.simulation_combinations:
            simulation_name, parameters = combination
            if not os.path.isdir(simulation_name):
                self.logger.write(f"[INFO] {self.time} -- The simulation folder {simulation_name} does not exist. Creating now...")
                os.mkdir(simulation_name)

            # change the working folder and run the simulation
            os.chdir(os.getcwd() + "/" + simulation_name)
            self.logger.write(f"[INFO] {self.time} -- Currently in: {os.getcwd()}\n\n")
            self.logger.write(f"[INFO] {self.time} -- Running: {simulation_name}, started at {self.time}\n")
            
            # define the BEC
            self.BEC = BEC(parameters, self.app)
            self.BEC.evolve()

            # go back to the parent directory to run the next sim
            path = Path(os.getcwd())
            parent_path = path.parent.absolute()
            os.chdir(parent_path)