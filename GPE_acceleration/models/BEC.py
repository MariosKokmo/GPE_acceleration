"""
Provides the BEC class.
Once a BEC object is initialised, its evolution can be run.
"""
import GPE_acceleration.library.gpe_library as gpe
import GPE_acceleration.library.ground_state as gs
import GPE_acceleration.utils.read_write_utils as rw
from GPE_acceleration.utils import video_creation
import numpy as np
import os
import torch
from pathlib import Path
from sys import platform

class BEC:
    def __init__(self, parameters, system, app, simulation_name):
        self.psi = None
        self.app = app
        self.device = app.device
        self.logger = app.logger
        self.time = app.time
        self.parameters = parameters
        self.simulation_name = simulation_name
        self.system = system
        self.gs_path = None
        self.repetitive_phase = None
        self.all_phases = {}

    def _find_ground_state(self):
        """
        Finds the ground state for the BEC in the system.
        If it exists, it just reads the file.
        The required format is `{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat`
        If a ground state file does not exist, it is computed.
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
        self.psi = gs.read_ground_state(self.gs_path, n1, n2, n3)
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
        
        new_phase = gpe.create_vortices(vortices, x1, x2, x3, n1, n2, n3, self.device)
        return new_phase
    
    def _imprint_vortices(self, vortices):
        """
        Imprints the initial vortices.
        Updates the wavefunction.
        """
        vortex_phase = self._create_vortices(vortices)
        self.psi = gpe.update_phase(self.psi, vortex_phase)

    def _calculate_repetitive_phase(self, imprinting_vortices):
        """
        Creates the phase to be added at each repetitive step.
        """
        self.repetitive_phase = self._create_vortices(imprinting_vortices)

    def _calculate_all_phases(self, imprinting_vortices):
        """
        Calculates a special phase. This could be a huge anti-charge etc.

        Input:
            imprinting_vortices: np.array, shape=(no. imprints, 3, no.vortices)
            each element is a collection
            of vortex positions along with their charges

        """
        for _, imprint in imprinting_vortices.items():
            x = tuple(imprint[0])
            y = tuple(imprint[1])
            charge = tuple(imprint[2])
            key = (x,y,charge)
            if key in self.all_phases.keys():
                self.logger.write(f"[INFO]: {self.time()} -- Phase for {key} already calculated.\n")
                continue
            else:
                self.logger.write(f"[INFO]: {self.time()} -- Calculating phase for {key}...\n")
                self.all_phases[key] = self._create_vortices(imprint)

    def _create_vortex_list(self, imprint_position_x, imprint_position_y, imprinting_charge, imprint_times):
        """
        Takes as input 3 arrays. Each array is for one simulation and
        can contain multiple sub-arrays. Each sub-array contains the vortex-related
        parameters for a differetn imprint time.
        
        Returns:
        --------
            np.array, 3D numpy array. 3 x #of vortices x #of imprints
        Example:
        --------
        imprinting_charge = [ [1,2], [3,4], [5,6] ]
        There are 3 imprint times. In the first imprint we imprint 2 vortices
        one with charge 1 and the other with charge 2. In the second imprint the
        two vortices that are imprinted are of charge 3 and 4 respectively etc.
        """
        assert len(imprint_position_x) == len(imprint_position_y)
        assert len(imprint_position_x) == len(imprinting_charge)
        vortex_list_all_iterations = {}
        
        imprint_position_x = imprint_position_x[0]
        imprint_position_y = imprint_position_y[0]
        imprinting_charge = imprinting_charge[0]

        for idx, time in enumerate(imprint_times):
            vort_x = imprint_position_x[idx]
            vort_y = imprint_position_y[idx]
            vort_charge = imprinting_charge[idx]
            vort = np.vstack((vort_x, vort_y, vort_charge))
            vortex_list_all_iterations[time] = vort
            
        return vortex_list_all_iterations

    def _repetitive_imprint(self, phase):
        """
        Performs the repetitive imprint.
        This could be any imprint after the initial one.
        """
        # update the phase of the wavefunction
        self.psi = gpe.update_phase(self.psi, phase)
        
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
            vort_x = np.array([vort_x])
            vort_y = np.array([vort_y])
            vort_charge = np.array([charges])
            vortices = np.vstack((vort_x, vort_y, vort_charge))
            initial_imprint_time = self.parameters["initial_imprint_time"]
            if repetitive:
                imprinting_charge = np.array([imprinting_charge], dtype=object)
                imprint_times = self.parameters["imprint_times"]
                imprint_position_x = self.parameters["imprint_position_x"]
                imprint_position_y = self.parameters["imprint_position_y"]
                imprint_position_x = np.array([imprint_position_x], dtype=object)
                imprint_position_y = np.array([imprint_position_y], dtype=object)
                imprinting_vortices_dictionary = self._create_vortex_list(imprint_position_x, imprint_position_y, imprinting_charge, imprint_times)

        # initialise the BEC on the ground state
        self._initialise()

        if repetitive:
            # calculate the repetitive imprinting phase
            self._calculate_all_phases(imprinting_vortices_dictionary)
        
        # evolve the BEC and perform re-imprint
        ##############################################################################
        ##############    MAIN LOOP OF SIMULATION    #################################
        ##############################################################################

        num_imprints = 0 # counts the additional imprints beyond the initial one
        count = 0
        initial_imprint_occured = False
        shots_per_ms = shots // (self.system.simulation_parameters["Total_simulation_time"]*1000)
        if repetitive:
            imprintTime = imprint_times[num_imprints]
            imprint_times_str = "_".join([str(round(time / shots_per_ms,2)) for time in imprint_times])
            vortices_to_imprint = "_".join([str(key) for key in self.all_phases.keys()])
            imprinting_charge_str = "_".join([str(charge) for charge in imprinting_charge]).replace("\n","_")
            self.logger.write(f"[INFO]: {self.time()} -- Will imprint at the following times : {imprint_times_str}\n")
            self.logger.write(f"[INFO]: {self.time()} -- Will imprint the following (x,y,charges) : {vortices_to_imprint}\n")
            SimulationName=f'{len(vort_x)}vortex__initCharge{vort_charge[0]}__imprintCharge{imprinting_charge_str}__times{imprint_times_str}'
        else:
            SimulationName=self.simulation_name

        # create storage for the line cross sections
        cross_line = torch.zeros(shots, n1)

        for iteration in range(kmax):
            t = dt*iteration*omega_ho
            utot = u*torch.abs(self.psi)**2 + uext # Total potential shape (n1,n2,n3)

            # write data file
            if (iteration%(kmax//shots) == 0):
                # Write some data
                rw.write_data(self.psi, count, x1, x3, n1, n3, a_ho)
                if self.app.phase_imaging:
                    cur_phase = self._extract_phase()
                    rw.save_figure_phase(cur_phase, count)
                rms = gpe.rms_radius(self.psi, self.system.center, self.system.space_grid)
                rms_measurements[count] = rms

                cross_line[count,:] = gpe.calculate_cross_section_line(self.psi)

                count += 1
                self.logger.write(f"t = {t/omega_ho}\n")
            
            # Perform the initial imprint
            if (iteration == ((kmax//shots)*initial_imprint_time)) and (not initial_imprint_occured):
                # imprint the topological excitation for the first time
                self.logger.write(f"[INFO]: {self.time()} -- Imprinting the initial topological object\n")
                self._imprint_vortices(vortices)
                initial_imprint_occured = True

            # repetitive imprinting with possibly different phases
            if repetitive and (iteration == (kmax//shots)*imprintTime) and (num_imprints < max_imprints):
                vortex_array = imprinting_vortices_dictionary[imprintTime]
                x = tuple(vortex_array[0])
                y = tuple(vortex_array[1])
                charge = tuple(vortex_array[2])
                key_for_phase = (x,y,charge)
                # extract the phase to imprint
                phaseImp = self.all_phases[key_for_phase]
                print("Imprinting again...")
                self.logger.write(f"[INFO]: {self.time()} -- Imprinting again...{key_for_phase}\n")
                self._repetitive_imprint(phaseImp)
                num_imprints += 1
                if (num_imprints < max_imprints) and (num_imprints < len(imprint_times)):# to avoid out-of-bounds index
                    imprintTime = imprint_times[num_imprints]

            # split-step evolution
            self._step(utot, dtau, p_sq, d_x)
        

        # Write the RMS measurements in a file
        rw.write_rms(rms_measurements, SimulationName)
        rw.save_rms_figure(f'{SimulationName}_RMS_meas.txt')

        # save the cross sections in figure and file
        rw.save_cross_section_line_figure(cross_line)
        rw.save_tensor_to_csv(cross_line, "cross_line_density.csv")

        # create the full video
        video_creation.create_video(count=count,\
                        simulation_name=SimulationName,\
                        n1=n1,n3=n3
                        )
        if self.app.write_velocity:
            video_creation.create_velocity_video(count,\
                                                    SimulationName,\
                                                    n1,n3)
