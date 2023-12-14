"""Provides the BCE class"""
import library.gpe_library as gpe
import library.ground_state as gs
import utils.video_creation
import numpy as np
import os
import torch
from pathlib import Path
from sys import platform

class BEC:
    def __init__(self, parameters, system, app):
        self.psi = None
        self.write_velocity = app.write_velocity
        self.device = app.device
        self.logger = app.logger
        self.time = app.time
        self.parameters = parameters
        self.system = system
        self.gs_path = None
        self.repetitive_phase = None

    def _find_ground_state(self):
        """
        Finds the ground state for the BEC in the system.
        If it exists, it just reads the file.
        The required format is `{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat`
        """
        n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
        fx, fy, fz = self.system.simulation_parameters["Trapping_frequencies"]
        # find ground state for the specific grid and potential if it doesn't exist
        cur_path = Path(os.getcwd())
        parent_path = str(cur_path.parent.absolute())
        os.chdir(parent_path)
        gs_file = f"{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat"
        if not os.path.exists(gs_file):
            self.logger.write(f"[INFO]: {self.time()} -- Calculating ground state...\n")
            _ = gs.find_ground_state(self.parameters, self.system, gs_file, device=self.device)
        self.logger.write(f"[INFO]: {self.time()} -- Ground state file: {gs_file}\n")
        if platform == "win32":
            self.gs_path = os.getcwd() + "\\" + gs_file
        else:
            self.gs_path = os.getcwd() + "/" + gs_file
        os.chdir(cur_path)
    
    def _initialise(self):
        """
        Reads the ground state file and initialises the wavefunction to ground state. 
        """
        if self.psi:
            self.logger.write(f"[WARN]: {self.time()} -- Trying to initialise an already initialised BEC. It will overwrite.")
        self._find_ground_state()
        n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
        self.psi = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=self.device)
        self.psi = gpe.read_ground_state(self.gs_path, n1, n2, n3)
        self.psi = self.psi.to(self.device)
    
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
        assert self.system.space_axes, self.logger.write(f"[FATAL]: {self.time()}-- the system has no axes initialised")
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
        self.psi = gpe.update_phase(self.psi, new_phase)

    def _calculate_repetitive_phase(self, imprinting_vortices):
        """
        Creates the phase to be added at each repetitive step.
        """
        x1, x2, x3 = self.system.space_axes
        n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
        self.repetitive_phase = gpe.create_additive_phase(imprinting_vortices, x1, x2, x3, n1, n2, n3, self.device)

    def _repetitive_imprint(self):
        """
        Performs the repetitive imprint.
        This could be any imprint after the initial one.
        """
        # extract current phase of psi1
        cur_phase = self._extract_phase()
        # add the new vortices (init_phase)
        new_phase = gpe.add_phase(cur_phase, self.repetitive_phase)
        # update the phase of the wavefunction
        self.psi = gpe.update_phase(self.psi, new_phase)
        
    def evolve(self):
        # get the parameters for easy access
        kmax = self.system.simulation_parameters["kmax"]
        dt = self.system.simulation_parameters["dt"]
        omega_ho = self.system.simulation_parameters["omega_ho"]
        shots = self.system.simulation_parameters["shots"]
        dtau = self.system.simulation_parameters["dtau"]
        d_x = self.system.simulation_parameters["d_x"]
        a_ho = self.system.simulation_parameters["a_ho"]
        p_sq = self.system.p_sq
        p_grid = self.system.p_grid
        x1, x2, x3 = self.system.space_axes
        p1, p2, p3 = self.system.momentum_axes
        uext = self.system.uext.potential
        n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
        u = self.system.simulation_parameters["u"]
        rms_measurements = {}

        if self.parameters["vortex_excitation"]:
            # Vortex related
            imprint_every = self.parameters["imprint_every"]
            max_imprints = self.parameters["max_imprints"]
            charges = self.parameters["vortex_charge"]
            imprinting_charge = self.parameters["imprinting_charge"]
            repetitive = self.parameters["repetitive"]
            vort_x = self.parameters["vortex_position_x"]
            vort_y = self.parameters["vortex_position_y"]
            imprint_position_x = self.parameters["imprint_position_x"]
            imprint_position_y = self.parameters["imprint_position_y"]
            vort_x = np.array([vort_x])
            vort_y = np.array([vort_y])
            imprint_position_x = np.array([imprint_position_x])
            imprint_position_y = np.array([imprint_position_y])
            vort_charge = np.array([charges])
            imprinting_charge = np.array([imprinting_charge])
            vortices = np.vstack((vort_x, vort_y, vort_charge))
            imprinting_vortices = np.vstack((imprint_position_x, imprint_position_y, imprinting_charge))
            

        # initialise the BEC on the ground state
        self._initialise()

        # calculate the repetitive imprinting phase
        self._calculate_repetitive_phase(imprinting_vortices)

        # imprint the topological excitation
        self._imprint_vortices(vortices)
        
        # evolve the BEC and perform re-imprint
        ##############################################################################
        ##############    MAIN LOOP OF SIMULATION    #################################
        ##############################################################################

        num_imprints = 0 # there has already been 1 imprint, the initial one
        count = 0
        wait = 5
        if repetitive:
            self.logger.write(f"[INFO]: {self.time()} -- Will imprint every {imprint_every} snapshots for {max_imprints} times\n")
            SimulationName=f'{len(vort_x)}vortex__initCharge{vort_charge[0]}__imprintCharge{imprinting_charge[0]}_total_imprints{max_imprints}__every{imprint_every}'
        else:
            SimulationName=f'{len(vort_x)}_{d}_{D}'

        for iteration in range(kmax):
            t = dt*iteration*omega_ho
            utot = u*torch.abs(self.psi)**2 + uext # Total potential shape (n1,n2,n3)

            if (iteration%(kmax/shots) == 0):
                # Write some data
                gpe.write_data(self.psi, count, x1, x3, n1, n3, a_ho)
                cur_phase = self._extract_phase()
                if self.write_velocity:
                    gpe.write_velocity2D(cur_phase, count, x1, x3, n1, n2, n3, a_ho, p_grid)
                rms = gpe.rms_radius(self.psi, self.system.center, self.system.space_grid)
                rms_measurements[count] = rms
                
                count += 1
                self.logger.write(f"t = {t/omega_ho}\n")
                if count%40==0:
                # create the video in the current folder
                    utils.video_creation.create_video(count,\
                        repetitive=repetitive,\
                        max_imprints=max_imprints,\
                        imprint_every=imprint_every,\
                        vort_x=vort_x,\
                        vort_charge=vort_charge,
                        n1=n1,n3=n3
                        )
                    if self.write_velocity:
                        utils.video_creation.create_velocity_video(count,\
                                                               SimulationName,\
                                                               n1,n3)

            # Repetitive imprinting
            if (iteration%((kmax//shots)*imprint_every) == 0) and (num_imprints < max_imprints) and (count>wait) and repetitive:
                num_imprints += 1
                print("Imprinting again...")
                self.logger.write(f"[INFO]: {self.time()} -- Imprinting again...\n")
                self._repetitive_imprint()

            # split-step evolution
            self._step(utot, dtau, p_sq, d_x)
        

        # Write the RMS measurements in a file
        gpe.write_rms(rms_measurements, SimulationName)

        # create the full video
        utils.video_creation.create_video(count,\
                        repetitive=repetitive,\
                        max_imprints=max_imprints,\
                        imprint_every=imprint_every,\
                        vort_x=vort_x,\
                        vort_charge=vort_charge,
                        n1=n1,n3=n3
                        )
        if self.write_velocity:
            utils.video_creation.create_velocity_video(count,\
                                                    SimulationName,\
                                                    n1,n3)
        