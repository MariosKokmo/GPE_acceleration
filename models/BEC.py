"""Provides the BCE class"""
import library.gpe_library as gpe
import torch

class BEC:
    def __init__(self, app):
        self.psi = None
        self.device = app.device
        self.logger = app.logger
        self.time = app.time

    def initialise(self, ground_state_path, grid):
        """Initialises the ground state"""
        if self.psi:
            self.logger.write(f"{self.time} [WARN] Trying to initialise an already initialised BEC. It will skip.")
            return
        n1, n2, n3 = grid
        self.psi = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=self.device)
        self.psi = gpe.read_ground_state(ground_state_path, self.psi, n1, n2, n3)
    
    def step(self, utot, dtau, p_sq, d_x):
        """Performs a step evolution of the BEC"""
        self.psi = gpe.split_step_step(self.psi, utot, dtau, p_sq, d_x)
    
    def extract_phase(self):
        return gpe.extract_phase(self.psi)

    def imprint_vortices(self, vortices, axes, grid):
        """
        Args:
        -----
            vortices: torch.Tensor
            axes: torch.Tensor, array of 1D tensors. Each 1D tensor is an axis
            grid: list[int], the grid size
        Returns:
        --------
            torch.Tensor, the new phase to be imprinted
        """
        x1, x2, x3 = axes
        n1, n2, n3 = grid
        phase = self.extract_phase()
        new_phase = gpe.imprint_vortices(vortices, phase, x1, x2, x3, n1, n2, n3, self.device)
        return new_phase

    def repetitive_imprint(self, repetitive_phase):
        """Performs the repetitive imprint. This could be any imprint after the initial one"""
        # extract current phase of psi1
        cur_phase = self.extract_phase()
        # add the new vortices (init_phase)
        new_phase = gpe.add_phase(cur_phase, repetitive_phase)
        # update the phase of the wavefunction
        self.psi = gpe.update_phase(new_phase)
        