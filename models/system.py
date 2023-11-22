"""Provides the laboratory system of external potential"""
import library.gpe_library as gpe
import library.ground_state as gs
import os

class System:
    def __init__(self, grid, frequencies, app):
        self.grid = grid
        self.frequencies = frequencies
        self.device = app.device
        self.logger = app.logger
        self.time = app.time
        self.gs_path
        self.x_min
        self.x_max
        self.dx
        self.dp
        self.w
        
    def initialise_parameters(self):
        if not self.device:
            raise Exception("Device needs to be set before the system")
        if not self.logger:
            raise Exception("Logger needs to be set before the system")

    def find_ground_state(self):
        n1, n2, n3 = self.grid
        fx, fy, fz = self.frequencies
        # find ground state for the specific grid and potential if it doesn't exist
        gs_file = f"{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat"
        if not os.path.exists(gs_file):
            self.logger.write(f"[INFO] {self.time} -- Calculating ground state...\n")
            _ = gs.find_ground_state(simulation_parameters, gs_file, device=self.device)
        self.logger.write(f"[INFO] Ground state file: {gs_file}\n")
        self.gs_path = os.getcwd() + "/" + gs_file


    def initialise_grid(self):
        n1, n2, n3 = self.grid
        x1, x2, x3, p1, p2, p3, p_sq = gpe.init_grid(self.x_min, self.x_max,\
                                                        self.dx, self.dp, self.w,\
                                                        n1, n2, n3, self.device)