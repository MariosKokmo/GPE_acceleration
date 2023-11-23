"""Provides the laboratory system of external potential"""
import library.gpe_library as gpe
import library.ground_state as gs
import os

class System:
    def __init__(self, app):
        self.device = app.device
        self.logger = app.logger
        self.time = app.time
        self.gs_path
        self.grid
        self.frequencies
        self.x_min
        self.x_max
        self.dx
        self.dp
        self.w
        
    def initialise_parameters(self):
        """
        Initialises the parameters of the system, grid, frequencies etc
        """
        if not self.device:
            raise Exception("Device needs to be set before the system")
        if not self.logger:
            raise Exception("Logger needs to be set before the system")

    def initialise_grid(self):
        n1, n2, n3 = self.grid
        x1, x2, x3, p1, p2, p3, p_sq = gpe.init_grid(self.x_min, self.x_max,\
                                                        self.dx, self.dp, self.w,\
                                                        n1, n2, n3, self.device)