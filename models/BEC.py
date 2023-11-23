"""Provides the BCE class"""
import library.gpe_library as gpe
import library.ground_state as gs
import os
import torch

class BEC:
    def __init__(self, simulation_parameters, app):
        self.psi = None
        self.device = app.device
        self.logger = app.logger
        self.time = app.time
        self.simulation_parameters
        self.gs_path = None

    def _find_ground_state(self):
        n1, n2, n3 = self.simulation_parameters["Grid_resolution"]
        fx, fy, fz = self.simulation_parameters["Trapping_frequencies"]
        # find ground state for the specific grid and potential if it doesn't exist
        gs_file = f"{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat"
        if not os.path.exists(gs_file):
            self.logger.write(f"[INFO] {self.time} -- Calculating ground state...\n")
            _ = gs.find_ground_state(self.simulation_parameters, gs_file, device=self.device)
        self.logger.write(f"[INFO] Ground state file: {gs_file}\n")
        self.gs_path = os.getcwd() + "/" + gs_file
    
    def _initialise(self):
        """Initialises the ground state"""
        if self.psi:
            self.logger.write(f"{self.time} [WARN] Trying to initialise an already initialised BEC. It will skip.")
            return
        self._find_ground_state()
        n1, n2, n3 = self.simulation_parameters["Grid_resolution"]
        self.psi = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=self.device)
        self.psi = gpe.read_ground_state(self.gs_path, self.psi, n1, n2, n3)
    
    def _step(self, utot, dtau, p_sq, d_x):
        """Performs a step evolution of the BEC"""
        self.psi = gpe.split_step_step(self.psi, utot, dtau, p_sq, d_x)
    
    def _extract_phase(self):
        return gpe.extract_phase(self.psi)

    def _imprint_vortices(self, vortices, axes, grid):
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

    def _repetitive_imprint(self, repetitive_phase):
        """Performs the repetitive imprint. This could be any imprint after the initial one"""
        # extract current phase of psi1
        cur_phase = self.extract_phase()
        # add the new vortices (init_phase)
        new_phase = gpe.add_phase(cur_phase, repetitive_phase)
        # update the phase of the wavefunction
        self.psi = gpe.update_phase(new_phase)

    def evolve(self):
        # initialise the BEC on the ground state
        self._initialise()
        # imprint the topological excitation

        # evolve the BEC and perform re-imprint
        