"""
Provides the BEC class.
Once a BEC object is initialised, its evolution can be run.
"""
from src.library.gpe_library import GPELibrary as gpe
from src.library.gpe_library import GPE2DLibrary as gpe2d
import src.library.ground_state as gs
import src.utils.read_write_utils as rw
from src.utils import video_creation
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
        self.reset_potential = False

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
            self.logger.info("Calculating ground state...")
            _ = gs.find_ground_state(self.parameters, self.system, gs_file, device=self.device)
        self.logger.info(f"Ground state file: {gs_file}")
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
            self.logger.warning("Trying to initialise an already initialised BEC. It will overwrite.")
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
        assert self.system.space_axes, self.logger.critical("the system has no axes initialised")
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
                self.logger.info(f"Phase for {key} already calculated.")
                continue
            else:
                self.logger.info(f"Calculating phase for {key}...")
                self.all_phases[key] = self._create_vortices(imprint)

    def _create_vortex_list(self, imprint_position_x, imprint_position_y, imprinting_charge, imprint_times):
        """
        Takes as input 3 arrays. Each array is for one simulation and
        can contain multiple sub-arrays. Each sub-array contains the vortex-related
        parameters for a different imprint time.
        
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
        """
        Evolves the BEC system over time.
        """
        # get the parameters for easy access
        self._initialize_simulation_parameters()

        # initialise the BEC on the ground state
        self._initialise()

        if self.parameters["repetitive"]:
            # calculate the repetitive imprinting phase
            self._calculate_all_phases(self.imprinting_vortices_dictionary)

        # evolve the BEC and perform re-imprint
        self._main_simulation_loop()

        # write various output files
        self._write_simulation_outputs()

    def _initialize_simulation_parameters(self):
        """
        Initializes simulation parameters for easy access.
        """
        params = self.system.simulation_parameters
        self.kmax = params["kmax"]
        self.dt = params["dt"]
        self.omega_ho = params["omega_ho"]
        self.shots = params["shots"]
        self.dtau = params["dtau"]
        self.d_x = params["d_x"]
        self.a_ho = params["a_ho"]
        self.p_sq = self.system.p_sq
        self.p_grid = self.system.p_grid
        self.x1, self.x2, self.x3 = self.system.space_axes
        self.p1, self.p2, self.p3 = self.system.momentum_axes
        self.uext = self.system.uext.potential
        self.n1, self.n2, self.n3 = params["Grid_resolution"]
        self.u = params["u"]
        self.rms_measurements = {}
        self.cross_line = torch.zeros(self.shots, self.n1)
        self.energies = []

        if self.parameters["vortex_excitation"]:
            self._initialize_vortex_parameters()

    def _initialize_vortex_parameters(self):
        """
        Initializes vortex-related parameters.
        """
        self.imprint_every = self.parameters["imprint_every"]
        self.max_imprints = self.parameters["max_imprints"]
        self.charges = self.parameters["vortex_charge"]
        self.imprinting_charge = self.parameters["imprinting_charge"]
        self.repetitive = self.parameters["repetitive"]
        self.vort_x = np.array([self.parameters["vortex_position_x"]])
        self.vort_y = np.array([self.parameters["vortex_position_y"]])
        self.vort_charge = np.array([self.charges])
        self.vortices = np.vstack((self.vort_x, self.vort_y, self.vort_charge))
        self.initial_imprint_time = self.parameters["initial_imprint_time"]

        if self.repetitive:
            self.imprinting_charge = np.array([self.imprinting_charge], dtype=object)
            self.imprint_times = self.parameters["imprint_times"]
            imprint_position_x = np.array([self.parameters["imprint_position_x"]], dtype=object)
            imprint_position_y = np.array([self.parameters["imprint_position_y"]], dtype=object)
            self.imprinting_vortices_dictionary = self._create_vortex_list(
                imprint_position_x, imprint_position_y, self.imprinting_charge, self.imprint_times
            )

    def _main_simulation_loop(self):
        """
        Main loop for evolving the BEC system.
        """
        num_imprints = 0
        count = 0
        initial_imprint_occured = False
        shots_per_ms = self.shots // (self.system.simulation_parameters["Total_simulation_time"] * 1000)

        # Open density evolution file
        density_file = open("density_evolution.txt", "w")
        density_file.write("count\ttime\tmax_density\tpeak_indices\n")

        if self.repetitive:
            imprintTime = self.imprint_times[num_imprints]
            self._log_repetitive_imprint_info(shots_per_ms)

        for iteration in range(self.kmax):
            t = self.dt * iteration * self.omega_ho
            utot = self.u * torch.abs(self.psi) ** 2 + self.uext

            if iteration % (self.kmax // self.shots) == 0:
                self._write_iteration_data(count, t)
                count += 1

            if iteration == (self.kmax // self.shots) * self.initial_imprint_time and not initial_imprint_occured:
                self._perform_initial_imprint()
                initial_imprint_occured = True

            if self.repetitive and iteration == (self.kmax // self.shots) * imprintTime and num_imprints < self.max_imprints:
                self._perform_repetitive_imprint(num_imprints)
                num_imprints += 1
                if num_imprints < self.max_imprints and num_imprints < len(self.imprint_times):
                    imprintTime = self.imprint_times[num_imprints]

            if iteration >= (self.kmax // self.shots) * self.system.uext.switchOff_time and not self.reset_potential:
                self._turn_off_potential()

            # Calculate density peak between count 148 and 190
            if 148 <= count <= 190:
                max_density, peak_indices = gpe.calculate_density_peak(self.psi)
                density_file.write(f"{count}\t{t / self.omega_ho}\t{max_density.item():.6e}\t{peak_indices}\n")

            self._step(utot, self.dtau, self.p_sq, self.d_x)

        # Close density evolution file
        density_file.close()

    def _log_repetitive_imprint_info(self, shots_per_ms):
        """
        Logs information about repetitive imprinting.
        """
        imprint_times_str = "_".join([str(round(time / shots_per_ms, 2)) for time in self.imprint_times])
        vortices_to_imprint = "_".join([str(key) for key in self.all_phases.keys()])
        imprinting_charge_str = "_".join([str(charge) for charge in self.imprinting_charge]).replace("\n", "_")
        self.logger.info(f"Will imprint at the following times : {imprint_times_str}")
        self.logger.info(f"Will imprint the following (x,y,charges) : {vortices_to_imprint}")
        self.simulation_name=f'{len(self.vort_x)}vortex__initCharge{self.vort_charge[0]}__imprintCharge{imprinting_charge_str}__times{imprint_times_str}'

    def _write_iteration_data(self, count, t):
        """
        Writes data for the current iteration.
        """
        rw.write_data(self.psi, count, self.x1, self.x3, self.n1, self.n3, self.a_ho)
        if self.app.phase_imaging:
            cur_phase = self._extract_phase()
            rw.save_figure_phase(cur_phase, count)
        rms = gpe.rms_radius(self.psi, self.system.center, self.system.space_grid)
        self.rms_measurements[count] = rms
        self.cross_line[count, :] = gpe.calculate_cross_section_line(self.psi)
        self.energies.append(gpe.calculate_energy_allocation(self.psi, self.uext, self.p_grid, {"u": self.u}))
        self.logger.info(f"t = {t / self.omega_ho}")

    def _perform_initial_imprint(self):
        """
        Performs the initial imprint of vortices.
        """
        self.logger.info("Imprinting the initial topological object")
        self._imprint_vortices(self.vortices)

    def _perform_repetitive_imprint(self, num_imprints):
        """
        Performs repetitive imprinting with possibly different phases.
        """
        vortex_array = self.imprinting_vortices_dictionary[self.imprint_times[num_imprints]]
        x = tuple(vortex_array[0])
        y = tuple(vortex_array[1])
        charge = tuple(vortex_array[2])
        key_for_phase = (x, y, charge)
        phaseImp = self.all_phases[key_for_phase]
        self.logger.info(f"Imprinting again...{key_for_phase}")
        self._repetitive_imprint(phaseImp)

    def _turn_off_potential(self):
        """
        Turns off the external potential.
        """
        self.logger.info("External potential set to zero")
        self.uext = self.system.uext.zero()
        self.reset_potential = True

    def _write_simulation_outputs(self):
        """
        Writes various output files after the simulation.
        """
        SimulationName = self.simulation_name
        rw.write_rms(self.rms_measurements, SimulationName)
        rw.save_rms_figure(f"{SimulationName}_RMS_meas.txt")
        rw.save_cross_section_line_figure(self.cross_line)
        rw.save_tensor_to_csv(self.cross_line, "cross_line_density.csv")
        rw.write_energy_terms(self.energies, "energies.txt")
        video_creation.create_video(count=len(self.rms_measurements), simulation_name=SimulationName, n1=self.n1, n3=self.n3)
        if self.app.write_velocity:
            video_creation.create_velocity_video(len(self.rms_measurements), SimulationName, self.n1, self.n3)
