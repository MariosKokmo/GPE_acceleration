"""Provides the class for the simulation. The simulations need a system i.e. laboratory setup
and a BEC model i.e. condensate that will evolve."""
import src.utils.setup_simulations as setup_simulations
import os
from pathlib import Path
import torch


def _get_bec_class(model_type: str):
    """
    Return the BEC model class for the given model_type string.

    model_type is set in the configuration file and controls which physics
    model is used for the condensate evolution:

        "BEC"           → src.models.BEC.BEC
                          Zero-temperature GPE with vortex / soliton imprinting.

        "FiniteTempBEC" → src.models.finite_temp_BEC.FiniteTempBEC
                          Stochastic Projected GPE (SGPE): damped GPE + thermal
                          noise.  Requires "temperature" and optionally
                          "damping_coefficient" in the config.

        "ZNGBEC"        → src.experimental.zng.zng_BEC.ZNGBEC
                          Full Zaremba-Nikuni-Griffin two-component framework:
                          condensate GPE coupled to a Monte Carlo thermal cloud.
                          Requires "temperature", "n_test_particles", "gamma_12".

    Imports are deferred so that the experimental ZNG module is only loaded
    when explicitly requested.

    Args:
        model_type (str): One of "BEC", "FiniteTempBEC", "ZNGBEC".

    Returns:
        type: The BEC model class.

    Raises:
        ValueError: If model_type is not a recognised string.
    """
    if model_type == "FiniteTempBEC":
        from src.models.finite_temp_BEC import FiniteTempBEC
        return FiniteTempBEC
    if model_type == "ZNGBEC":
        from src.experimental.zng.zng_BEC import ZNGBEC
        return ZNGBEC
    if model_type == "BEC":
        from src.models.BEC import BEC
        return BEC
    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        f"Choose from: 'BEC', 'FiniteTempBEC', 'ZNGBEC'."
    )

class Simulations:
    """
    Class that holds all simulations to be run.
    For every simulation, a new BEC is created and initialised.
    Then it is let to evolve.
    """
    def __init__(self, system, app):
        self.simulation_combinations = None
        self.app = app
        self.logger = None
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
        For every simulation, it creates a new BEC.
        Then runs the simulation.
        """
        # Run the simulations sequentially. For each simulation, create a new BEC and evolve it.
        for combination in self.simulation_combinations:
            simulation_name, parameters = combination

            if not os.path.isdir(simulation_name):
                self.app.logger.info(f"The simulation folder {simulation_name} does not exist. Creating now...")
                os.mkdir(simulation_name)

            # change the working folder and run the simulation
            os.chdir(os.getcwd() + "/" + simulation_name)
            logfile  = f"{simulation_name}_log.txt"
            self.app.set_logger(logfile)
            self.logger = self.app.logger
            
            # create the new log file for the simulation
            self.logger.info(f"Currently in: {os.getcwd()}")
            self.logger.info(f"Running: {simulation_name}, started at {self.time}")
            
            # save the simulation parameters as a json file
            setup_simulations.save_parameters_to_json(self.system.simulation_parameters)

            # Instantiate the correct model class based on model_type in config.
            # Defaults to "BEC" (zero-temperature GPE) when the key is absent.
            BECClass = _get_bec_class(parameters.get("model_type", "BEC"))
            self.BEC = BECClass(parameters, self.system, self.app, simulation_name)
            self.BEC.evolve()
            
            # go back to the parent directory to prepare to run the next sim
            path = Path(os.getcwd())
            parent_path = path.parent.absolute()
            os.chdir(parent_path)
            
            # free unused memory
            torch.cuda.empty_cache()