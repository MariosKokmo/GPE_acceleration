import application
import torch
from models.system import System
from models.simulation import Simulations

def main():
    # Set up the application
    app = application.application()
    logfile = open("log.txt", "w")
    app.set_logger(logfile)
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    app.set_device(DEVICE)
    app.logger.write(f"[INFO]: {app.time()} -- Running on {DEVICE}.\n")

    # Set up the external system
    system = System(app)

    # set up simulations
    simulations = Simulations(system, app)

    # run simulations 
    simulations.run_simulations()
    
    # close log file and exit
    app.logger.write(f"[INFO]: {app.time()} -- Finished all simulations.\n")
    app.logger.close()

if __name__ == "__main__":
    main()