"""Provides the BCE class"""
import library.gpe_library as gpe
import library.ground_state as gs
import numpy as np
import os
import torch

class BEC:
    def __init__(self, parameters, system, app):
        self.psi = None
        self.device = app.device
        self.logger = app.logger
        self.time = app.time
        self.parameters = parameters
        self.system = system
        self.gs_path = None

    def _find_ground_state(self):
        """
        Finds the ground state for the BEC in the system.
        If it exists, it just reads the file.
        """
        n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
        fx, fy, fz = self.system.simulation_parameters["Trapping_frequencies"]
        # find ground state for the specific grid and potential if it doesn't exist
        gs_file = f"{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat"
        if not os.path.exists(gs_file):
            self.logger.write(f"[INFO] {self.time} -- Calculating ground state...\n")
            _ = gs.find_ground_state(self.parameters, gs_file, device=self.device)
        self.logger.write(f"[INFO] Ground state file: {gs_file}\n")
        self.gs_path = os.getcwd() + "/" + gs_file
    
    def _initialise(self):
        """
        Reads the ground state file and initialises the wavefunction to ground state. 
        """
        if self.psi:
            self.logger.write(f"{self.time} [WARN] Trying to initialise an already initialised BEC. It will skip.")
            return
        self._find_ground_state()
        n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
        self.psi = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=self.device)
        self.psi = gpe.read_ground_state(self.gs_path, self.psi, n1, n2, n3)
    
    def _step(self, utot, dtau, p_sq, d_x):
        """
        Performs a step evolution of the BEC
        following the split-step Fourier method
        """
        self.psi = gpe.split_step_step(self.psi, utot, dtau, p_sq, d_x)
    
    def _extract_phase(self):
        """"
        Returns the phase of the condensate.
        Returns:
        --------
            torch.Tensor, the phase of the wavefunction
        """
        return gpe.extract_phase(self.psi)

    def _create_vortices(self, vortices):
        """
        Creates the initial phase distribution of the vortices to be imprinted
        Args:
        -----
            vortices: torch.Tensor
            axes: torch.Tensor, array of 1D tensors. Each 1D tensor is an axis
            grid: list[int], the grid size
        Returns:
        --------
            torch.Tensor, the new phase to be imprinted
        """
        assert self.system.space_axes, "the system has no axes initialised"
        x1, x2, x3 = self.system.space_axes
        n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
        phase = self._extract_phase()
        new_phase = gpe.imprint_vortices(vortices, phase, x1, x2, x3, n1, n2, n3, self.device)
        return new_phase
    
    def _imprint_vortices(self, vortices):
        """
        Imprints the initial vortices.
        Updates the wavefunction.
        """
        new_phase = self._create_vortices(vortices)
        self.psi = gpe.update_phase(new_phase)

    def _repetitive_imprint(self, repetitive_phase):
        """
        Performs the repetitive imprint.
        This could be any imprint after the initial one.
        """
        # extract current phase of psi1
        cur_phase = self._extract_phase()
        # add the new vortices (init_phase)
        new_phase = gpe.add_phase(cur_phase, repetitive_phase)
        # update the phase of the wavefunction
        self.psi = gpe.update_phase(new_phase)
        
    def evolve(self):
        # get the parameters for easy access
        imprint_every = self.parameters["imprint_every"]
        max_imprints = self.parameters["max_imprints"]
        charges = self.parameters["vortex_charge"]
        imprinting_charge = self.parameters["imprinting_charge"]
        repetitive = self.parameters["repetitive"]
        vort_x = self.parameters["vortex_position_x"]
        vort_y = self.parameters["vortex_position_y"]
        vort_x = np.array([vort_x])
        vort_y = np.array([vort_y])
        vort_charge = np.array([charges])
        imprinting_charge = np.array([imprinting_charge])
        vortices = np.vstack((vort_x, vort_y, vort_charge))
        imprinting_vortices = np.vstack((vort_x, vort_y, imprinting_charge))

        # initialise the BEC on the ground state
        self._initialise()

        # imprint the topological excitation
        self._imprint_vortices(self, vortices)
        
        # evolve the BEC and perform re-imprint
        