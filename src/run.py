import src.application as ap
import torch
from src.models.system import System
from src.models.simulation import Simulations

def main():
    # Set up the application
    app = ap.application()
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    app.set_device(DEVICE)

    app.open_logger()
    app.logger.write(f"[INFO]: {app.time()} -- Running on {DEVICE}.\n")
    app.close_logger()

    # Set up the external system
    system = System(app)

    # set up simulations
    simulations = Simulations(system, app)

    # run simulations 
    simulations.run_simulations()
    
    # close log file and exit
    app.reset_logger()
    app.open_logger()
    app.logger.write(f"[INFO]: {app.time()} -- Finished all simulations.\n")
    app.close_logger()

if __name__ == "__main__":
    main()