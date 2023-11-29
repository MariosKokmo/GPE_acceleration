"""Provides the laboratory system of external potential"""
import library.gpe_library as gpe
import utils.setup_simulations as setup_simulations
from library.potentials import select_potential

class System:
    def __init__(self, app):
        self.app = app
        self.device = self.app.device
        self.logger = self.app.logger
        self.time = app.time
        self.space_axes = None
        self.momentum_axes = None
        # squared momentum grid (3D)
        self.p_sq = None
        # real space grid (3D)
        self.space_grid = None
        self.simulation_parameters = None
        # the selected external potential
        self.uext = None
        # call the parameter initialisation
        self._initialise_parameters()
        
    def _initialise_parameters(self):
        """
        Initialises the parameters of the system, grid, frequencies etc
        """
        if not self.device:
            raise Exception("Device needs to be set before the system")
        if not self.logger:
            raise Exception("Logger needs to be set before the system")
        self.simulation_parameters = setup_simulations.get_simulation_parameters("configuration_file.json")
        # define the external potential
        potentialType = self.simulation_parameters["Potential_type"]
        new_potential = select_potential(potentialType)
        assert new_potential, "Potential was not given"
        self.uext = new_potential(app=self.app, **self.simulation_parameters)

        # initialise the grid
        self._initialise_grid()
        

    def _initialise_grid(self):
        assert self.simulation_parameters, "System simulation parameters not initialised"
        n1, n2, n3 = self.simulation_parameters["Grid_resolution"]
        x_min = self.simulation_parameters["x_min"]
        x_max = self.simulation_parameters["x_max"]
        dx = self.simulation_parameters["dx"]
        dp = self.simulation_parameters["dp"]
        w = self.simulation_parameters["w"]
        x1, x2, x3, p1, p2, p3, p_sq, space_grid = gpe.init_grid(x_min, x_max,\
                                                        dx, dp, w,\
                                                        n1, n2, n3, self.device)
        self.space_axes = [x1, x2, x3]
        self.momentum_axes = [p1, p2, p3]
        self.p_sq = p_sq
        self.space_grid = space_grid
