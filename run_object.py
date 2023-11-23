import application
from models.BEC import BEC
from models.system import System
from models.simulation import Simulation
from library.potentials import select_potential

def main():
    # Set up the application
    app = application.application()
    logfile = open("log.txt", "w")
    app.set_logger(logfile)

    # Set up the external system
    system = System(grid, frequencies, app)

    # define the external potential
    new_potential = select_potential(potentialType)
    assert new_potential, "Potential was not given"
    uext = new_potential()

    # set up simulations
    simulations = Simulation(system)

    # run simulations 
    simulations.run_simulations()
    
    # close log file and exit
    app.logger.write(f"[INFO] {app.time} -- Finished all simulations.")
    app.logger.close()

if __name__ == "__main__":
    main()