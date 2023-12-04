"""Provides the class for the simulation"""
import utils.setup_simulations as setup_simulations
from models.BEC import BEC
import os
from pathlib import Path

class Simulations:
    """
    Class that holds all simulations to be run.
    For every simulation, a new BEC is created and initialised.
    Then it is let to evolve.
    """
    def __init__(self, system, app):
        self.simulation_combinations = None
        self.app = app
        self.logger = self.app.logger
        self.time = self.app.time()
        self.device = self.app.device
        self.BEC = None
        self.system = system
        # set up simulations
        self._setup_simulations()
    
    def _setup_simulations(self):
        """
        Creates the simulation combinations to be run.
        """
        self.simulation_combinations = setup_simulations.get_simulation_combinations(self.system.simulation_parameters)

    def run_simulations(self):
        """
        For every simulation, it creates the BEC.
        Runs the simulation.
        """
        # Run the simulations
        for combination in self.simulation_combinations:
            simulation_name, parameters = combination
            if not os.path.isdir(simulation_name):
                self.logger.write(f"[INFO]: {self.time} -- The simulation folder {simulation_name} does not exist. Creating now...")
                os.mkdir(simulation_name)

            # change the working folder and run the simulation
            os.chdir(os.getcwd() + "/" + simulation_name)
            self.logger.write(f"[INFO]: {self.time} -- Currently in: {os.getcwd()}\n\n")
            self.logger.write(f"[INFO]: {self.time} -- Running: {simulation_name}, started at {self.time}\n")
            
            # define the BEC
            self.BEC = BEC(parameters, self.system,  self.app)
            self.BEC.evolve()

            # go back to the parent directory to run the next sim
            path = Path(os.getcwd())
            parent_path = path.parent.absolute()
            os.chdir(parent_path)