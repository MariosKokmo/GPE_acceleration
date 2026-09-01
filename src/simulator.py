import torch
import src.application as ap
from src.models.system import System
from src.models.simulation import Simulations


class Simulator:
    """
    Object-oriented interface for running GPE simulations.

    Usage:
        sim = Simulator("configuration_file.json")
        sim.run()

    After construction, sub-objects are accessible:
        sim.app        -- application config / logger / device
        sim.system     -- grid, potential, simulation parameters
        sim.simulations -- simulation combinations and last BEC
    """

    def __init__(self, config_file="configuration_file.json", app_config_file="appConfig.json"):
        self.app = ap.application(app_config_file)
        if config_file:
            self.app.configFile = config_file

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.app.set_device(device)
        self.app.logger.info(f"Running on {device}.")

        self.system = System(self.app)
        self.simulations = Simulations(self.system, self.app)

    def run(self):
        """Run all simulation combinations defined in the config file."""
        self.simulations.run_simulations()
        self.app.reset_logger()
        self.app.logger.info("Finished all simulations.")
