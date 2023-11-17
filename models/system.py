"""Provides the laboratory system of external potential"""
import library.gpe_library as gpe

class System:
    def __init__(self, grid, device):
        self.grid = grid
        self.device = device
        self.x_min
        self.x_max
        self.dx
        self.dp
        self.w
        
    def initialise_parameters(self):
        pass

    def find_ground_state(self):
        pass

    def initialise_grid(self):
        n1, n2, n3 = self.grid
        uext1, x1, x2, x3, p1, p2, p3, p_sq = gpe.init_grid(self.x_min, self.x_max,\
                                                            self.dx, self.dp, self.w,\
                                                            n1, n2, n3, self.device)