import src.application as ap
import torch
from src.models.system import System
from src.models.simulation import Simulations

def main(configFile="configuration_file.json"):
    # Set up the application
    app = ap.application()
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    app.set_device(DEVICE)

    app.logger.info(f"Running on {DEVICE}.")

    # Set up the external system
    system = System(app)

    # set up simulations
    simulations = Simulations(system, app)

    # run simulations 
    simulations.run_simulations()
    
    # close log file and exit
    app.reset_logger()
    app.logger.info("Finished all simulations.")

if __name__ == "__main__":
    main()