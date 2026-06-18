"""
Provides the BEC class.
Once a BEC object is initialised, its evolution can be run.
"""
from typing import Optional, List, Tuple, Dict, Any, Union
from src.library.gpe_library import GPELibrary as gpe
from src.library.gpe_library import GPE2DLibrary as gpe2d
from src.library.common_utils import CommonUtils as cu
import src.library.ground_state as gs
import src.utils.read_write_utils as rw
from src.utils import video_creation
import numpy as np
import os
import torch
from pathlib import Path
from sys import platform

class BEC:
    def __init__(self, parameters: Dict[str, Any], system: Any, app: Any, simulation_name: str) -> None:
        self.psi: Optional[torch.Tensor] = None
        self.app = app
        self.device = app.device
        self.logger = app.logger
        self.time = app.time
        self.parameters = parameters # simulation configuration parameters e.g. vortices
        self.simulation_name = simulation_name
        self.system = system # system parameters are in here e.g. grid, potential etc.
        self.gs_path: Optional[str] = None
        self.repetitive_phase: Optional[torch.Tensor] = None
        self.all_phases: Dict[Tuple, torch.Tensor] = {}
        self.reset_potential: bool = False

    def _find_ground_state(self) -> None:
        """
        Finds the ground state for the BEC in the system.
        If it exists, it just reads the file.
        The required format is `{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat`
        If a ground state file does not exist, it is computed.
        """
        try:
            n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
            fx, fy, fz = self.system.simulation_parameters["Trapping_frequencies"]
        except KeyError as e:
            self.logger.error(f"Missing simulation parameter: {e}")
            raise

        # find ground state for the specific grid and potential if it doesn't exist
        cur_path = Path(os.getcwd())
        parent_path = str(cur_path.parent.absolute())
        
        try:
            os.chdir(parent_path)
            gs_file = f"{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat"
            if not os.path.exists(gs_file):
                self.logger.info("Calculating ground state...")
                try:
                    _ = gs.GroundState.find_ground_state(self.parameters, self.system, gs_file, device=self.device)
                except Exception as e:
                    self.logger.error(f"Failed to calculate ground state: {e}")
                    raise
            
            self.logger.info(f"Ground state file: {gs_file}")
            if platform == "win32":
                self.gs_path = os.getcwd() + "\\" + gs_file
            else:
                self.gs_path = os.getcwd() + "/" + gs_file
        except OSError as e:
            self.logger.error(f"File system error in _find_ground_state: {e}")
            raise
        finally:
            os.chdir(cur_path)
    
    def _initialise(self) -> None:
        """
        Reads the ground state file and initialises the wavefunction to ground state. 
        """
        if self.psi is not None:
            self.logger.warning("Trying to initialise an already initialised BEC. It will overwrite.")
        
        try:
            self._find_ground_state()
            n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
            self.psi = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=self.device)
            if self.gs_path is None:
                raise ValueError("Ground state path is None after _find_ground_state")
            self.psi = gs.GroundState.read_ground_state(self.gs_path, n1, n2, n3)
            self.psi = self.psi.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to initialise BEC: {e}")
            raise

    def _step(self, utot: torch.Tensor, dtau: float, p_sq: torch.Tensor, d_x: float) -> None:
        """
        Performs a step evolution of the BEC
        following the split-step Fourier method
        """
        if self.psi is None:
            raise RuntimeError("BEC wavefunction (psi) is not initialized.")
        try:
            self.psi = gpe.split_step_step(self.psi, utot, dtau, p_sq, d_x)
        except Exception as e:
            self.logger.error(f"Error during step evolution: {e}")
            raise
    
    def _extract_phase(self) -> torch.Tensor:
        """"
        Returns the phase of the condensate.
        Returns:
        --------
            torch.Tensor, the phase of the wavefunction
        """
        if self.psi is None:
            raise RuntimeError("BEC wavefunction (psi) is not initialized.")
        return cu.extract_phase(self.psi)

    def _create_vortices(self, vortices: np.ndarray) -> torch.Tensor:
        """
        Creates the initial phase distribution of the vortices to be imprinted
        Args:
        -----
            vortices: np.ndarray, array of vortex positions and charges
            axes: torch.Tensor, array of 1D tensors. Each 1D tensor is an axis
            grid: list[int], the grid size
        Returns:
        --------
            torch.Tensor, the new phase to be imprinted
        """
        if not self.system.space_axes:
            self.logger.critical("the system has no axes initialised")
            raise ValueError("System space axes not initialized")
        
        x1, x2, x3 = self.system.space_axes
        n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
        
        try:
            new_phase = gpe2d.create_vortices(vortices, x1, x2, x3, n1, n2, n3, self.device)
            return new_phase
        except Exception as e:
            self.logger.error(f"Error creating vortices: {e}")
            raise
    
    def _imprint_vortices(self, vortices: np.ndarray) -> None:
        """
        Imprints the initial vortices.
        Updates the wavefunction.
        """
        try:
            vortex_phase = self._create_vortices(vortices)
            if self.psi is None:
                raise RuntimeError("BEC wavefunction (psi) is not initialized.")
            self.psi = cu.update_phase(self.psi, vortex_phase)
        except Exception as e:
            self.logger.error(f"Error imprinting vortices: {e}")
            raise

    def _calculate_repetitive_phase(self, imprinting_vortices: Dict) -> None:
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
        try:
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
        except Exception as e:
            self.logger.error(f"Error calculating all phases: {e}")
            raise

    def _create_vortex_list(self, imprint_position_x: np.ndarray, imprint_position_y: np.ndarray, 
                            imprinting_charge: np.ndarray, imprint_times: Union[List[float], np.ndarray]) -> Dict[float, np.ndarray]:
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
        if len(imprint_position_x) != len(imprint_position_y):
            raise ValueError(f"Length mismatch: imprint_position_x ({len(imprint_position_x)}) vs imprint_position_y ({len(imprint_position_y)})")
        if len(imprint_position_x) != len(imprinting_charge):
            raise ValueError(f"Length mismatch: imprint_position_x ({len(imprint_position_x)}) vs imprinting_charge ({len(imprinting_charge)})")
        
        vortex_list_all_iterations = {}

        try:
            # Safely access the first element, assuming these are arrays of arrays/lists
            if len(imprint_position_x) == 0:
                self.logger.warning("Empty imprint position arrays provided.")
                return {}

            imprint_position_x_0 = imprint_position_x[0]
            imprint_position_y_0 = imprint_position_y[0]
            imprinting_charge_0 = imprinting_charge[0]

            for idx, time in enumerate(imprint_times):
                if idx >= len(imprint_position_x_0):
                    break 
                vort_x = imprint_position_x_0[idx]
                vort_y = imprint_position_y_0[idx]
                vort_charge = imprinting_charge_0[idx]
                vort = np.vstack((vort_x, vort_y, vort_charge))
                vortex_list_all_iterations[time] = vort
            
            return vortex_list_all_iterations
        except IndexError as e:
            self.logger.error(f"Index error in creating vortex list: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in creating vortex list: {e}")
            raise

    def _repetitive_imprint(self, phase: torch.Tensor) -> None:
        """
        Performs the repetitive imprint.
        This could be any imprint after the initial one.
        """
        # update the phase of the wavefunction
        if self.psi is None:
            raise RuntimeError("BEC wavefunction (psi) is not initialized.")
        try:
            self.psi = cu.update_phase(self.psi, phase)
        except Exception as e:
            self.logger.error(f"Error during repetitive imprint: {e}")
            raise
        
    def evolve(self) -> None:
        """
        Evolves the BEC system over time.
        """
        try:
            # get the parameters for easy access
            self._initialize_simulation_parameters()

            # initialise the BEC on the ground state
            self._initialise()

            if self.parameters.get("repetitive", False):
                # calculate the repetitive imprinting phase
                if not hasattr(self, 'imprinting_vortices_dictionary'):
                    self.logger.warning("imprinting_vortices_dictionary not found, skipping phase calculation.")
                else:
                    self._calculate_all_phases(self.imprinting_vortices_dictionary)

            # evolve the BEC and perform re-imprint
            self._main_simulation_loop()

            # write various output files
            self._write_simulation_outputs()
        except Exception as e:
            self.logger.critical(f"Simulation failed in evolve(): {e}")
            raise

    def _initialize_simulation_parameters(self) -> None:
        """
        Initializes simulation parameters for easy access.
        """
        try:
            params = self.system.simulation_parameters
            self.kmax = params["kmax"]
            self.dt = params["dt"]
            self.dx = params["dx"]
            self.omega_ho = params["omega_ho"]
            self.shots = params["shots"]
            self.dtau = params["dtau"]
            self.d_x = params["d_x"]
            self.a_ho = params["a_ho"]
            self.p_sq = self.system.p_sq
            self.p_grid = self.system.p_grid
            self.x1, self.x2, self.x3 = self.system.space_axes
            self.p1, self.p2, self.p3 = self.system.momentum_axes
            if self.system.uext is None:
                raise ValueError("External potential (uext) is None")
            self.uext = self.system.uext.potential
            self.n1, self.n2, self.n3 = params["Grid_resolution"]
            self.u = params["u"]
            self.k3 = params["k3"]
            self.rms_measurements = {}
            self.cross_line = torch.zeros(self.shots, self.n1)
            self.energies = []

            # Defaults so the evolution loop is safe when vortices are disabled
            # (e.g. dark-soliton-only runs). Overwritten by the vortex
            # initialisation below when vortex excitation is enabled.
            self.vortex_excitation = bool(self.parameters.get("vortex_excitation", False))
            self.repetitive = False
            self.imprint_times = []
            self.initial_imprint_time = None

            if self.vortex_excitation:
                self._initialize_vortex_parameters()

            if self.parameters.get("dark_soliton", False):
                self._initialize_dark_soliton_parameters()
        except KeyError as e:
            self.logger.error(f"Missing simulation parameter: {e}")
            raise
        except AttributeError as e:
            self.logger.error(f"System attribute missing: {e}")
            raise

    def _initialize_vortex_parameters(self) -> None:
        """
        Initializes vortex-related parameters.
        """
        try:
            self.imprint_every = self.parameters["imprint_every"]
            self.max_imprints = self.parameters["max_imprints"]
            self.charges = self.parameters["vortex_charge"]
            self.imprinting_charge = self.parameters["imprinting_charge"]
            self.repetitive = self.parameters.get("repetitive", False)
            self.vort_x = np.array([self.parameters["vortex_position_x"]])
            self.vort_y = np.array([self.parameters["vortex_position_y"]])
            self.vort_charge = np.array([self.charges])
            self.vortices = np.vstack((self.vort_x, self.vort_y, self.vort_charge)) # type: ignore
            self.initial_imprint_time = self.parameters["initial_imprint_time"]

            if self.repetitive:
                self.imprinting_charge = np.array([self.imprinting_charge], dtype=object)
                self.imprint_times = self.parameters["imprint_times"]
                # Assuming these parameters are lists if repetitive is True
                imprint_position_x = np.array([self.parameters["imprint_position_x"]], dtype=object)
                imprint_position_y = np.array([self.parameters["imprint_position_y"]], dtype=object)
                self.imprinting_vortices_dictionary = self._create_vortex_list(
                    imprint_position_x, imprint_position_y, self.imprinting_charge, self.imprint_times
                )
        except KeyError as e:
            self.logger.error(f"Missing vortex parameter: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing vortex parameters: {e}")
            raise

    def _initialize_dark_soliton_parameters(self) -> None:
        """
        Initializes dark-soliton parameters from the simulation configuration.

        Expected keys (all lists of the same length):
          - ``soliton_positions``: centre positions in dimensionless units.
          - ``soliton_widths``: characteristic widths (healing length scale).
          - ``soliton_axes``: which axis each soliton is perpendicular to (1 or 3).
          - ``soliton_greyness`` (optional): grey-soliton angle in radians per soliton.
          - ``soliton_imprint_time``: snapshot index at which the soliton is imprinted.
        """
        try:
            self.soliton_positions = self.parameters["soliton_positions"]
            self.soliton_widths = self.parameters["soliton_widths"]
            self.soliton_axes = self.parameters["soliton_axes"]
            self.soliton_greyness = self.parameters.get("soliton_greyness", None)
            self.soliton_imprint_time = self.parameters.get("soliton_imprint_time", 0)
            self.soliton_imprinted = False
            self.logger.info(
                f"Dark soliton configured: {len(self.soliton_positions)} soliton(s), "
                f"imprint at snapshot {self.soliton_imprint_time}"
            )
        except KeyError as e:
            self.logger.error(f"Missing dark soliton parameter: {e}")
            raise

    def _imprint_dark_solitons(self) -> None:
        """
        Creates and applies the dark-soliton mask to the wavefunction.
        """
        if self.psi is None:
            raise RuntimeError("BEC wavefunction (psi) is not initialized.")
        try:
            x1, _, x3 = self.system.space_axes
            mask = gpe2d.create_dark_soliton(
                x1, x3,
                self.n1, self.n2, self.n3,
                positions=self.soliton_positions,
                widths=self.soliton_widths,
                axes=self.soliton_axes,
                greyness=self.soliton_greyness,
                device=self.device,
            )
            self.psi = gpe2d.imprint_dark_soliton(self.psi, mask)
            self.soliton_imprinted = True
            self.logger.info("Dark soliton(s) imprinted successfully.")
        except Exception as e:
            self.logger.error(f"Error imprinting dark solitons: {e}")
            raise

    def _main_simulation_loop(self) -> None:
        """
        Main loop for evolving the BEC system.
        """
        if self.psi is None:
            raise RuntimeError("BEC wavefunction (psi) is not initialized.")
            
        try:
            num_imprints = 0
            count = 0
            initial_imprint_occured = False
            
            total_time = self.system.simulation_parameters.get("Total_simulation_time")
            if total_time is None or total_time == 0:
                self.logger.warning("Total_simulation_time is 0 or None. Defaulting to 1.") # prevent div by zero
                total_time = 1
            
            shots_per_ms = self.shots // (total_time * 1000)
            if shots_per_ms == 0:
                shots_per_ms = 1

            imprintTime = 0
            if self.repetitive and self.imprint_times:
                imprintTime = self.imprint_times[num_imprints]
                self._log_repetitive_imprint_info(shots_per_ms)

            for iteration in range(self.kmax):
                t = self.dt * iteration * self.omega_ho
                loss_term = - self.k3 * torch.abs(self.psi)**4
                utot = (self.u * torch.abs(self.psi) ** 2 + self.uext) + (1j * loss_term)

                if self.shots > 0 and iteration % (self.kmax // self.shots) == 0:
                    self._write_iteration_data(count, t)
                    count += 1

                if self.vortex_excitation and not initial_imprint_occured and \
                        iteration == (self.kmax // self.shots) * self.initial_imprint_time:
                    self._perform_initial_imprint()
                    initial_imprint_occured = True

                if hasattr(self, 'soliton_positions') and not self.soliton_imprinted:
                    if iteration == (self.kmax // self.shots) * self.soliton_imprint_time:
                        self._imprint_dark_solitons()

                if self.repetitive and self.imprint_times and iteration == (self.kmax // self.shots) * imprintTime and num_imprints < self.max_imprints:
                    self._perform_repetitive_imprint(num_imprints)
                    num_imprints += 1
                    if num_imprints < self.max_imprints and num_imprints < len(self.imprint_times):
                        imprintTime = self.imprint_times[num_imprints]

                uext_switch_off = self.system.uext.switchOff_time if hasattr(self.system.uext, 'switchOff_time') else None
                if uext_switch_off is not None and iteration >= (self.kmax // self.shots) * uext_switch_off and not self.reset_potential:
                    self._turn_off_potential()

                self._step(utot, self.dtau, self.p_sq, self.d_x)
                
                if hasattr(self.system.uext, 'evol'):
                    self.uext = self.system.uext.evol(t)
        except Exception as e:
            self.logger.error(f"Error in main simulation loop at iteration {iteration if 'iteration' in locals() else 'unknown'}: {e}")
            raise

    def _log_repetitive_imprint_info(self, shots_per_ms: float) -> None:
        """
        Logs information about repetitive imprinting.
        """
        try:
            if not self.imprint_times:
                self.logger.warning("No imprint times to log.")
                return

            imprint_times_str = "_".join([str(round(time / shots_per_ms, 2)) for time in self.imprint_times])
            vortices_to_imprint = "_".join([str(key) for key in self.all_phases.keys()])
            imprinting_charge_str = "_".join([str(charge) for charge in self.imprinting_charge]).replace("\n", "_")
            self.logger.info(f"Will imprint at the following times : {imprint_times_str}")
            self.logger.info(f"Will imprint the following (x,y,charges) : {vortices_to_imprint}")
            self.simulation_name=f'{len(self.vort_x)}vortex__initCharge{self.vort_charge[0]}__imprintCharge{imprinting_charge_str}__times{imprint_times_str}'
        except Exception as e:
            self.logger.error(f"Error logging repetitive imprint info: {e}")

    def _write_iteration_data(self, count: int, t: float) -> None:
        """
        Writes data for the current iteration.
        """
        if self.psi is None:
            self.logger.error("Skipping write_iteration_data because psi is None.")
            return

        try:
            rw.write_data(self.psi, count, self.x1, self.x3, self.n1, self.n3, self.a_ho, self.dx)
            
            if hasattr(self.app, 'phase_imaging') and self.app.phase_imaging:
                cur_phase = self._extract_phase()
                rw.save_figure_phase(cur_phase, count)
                
            rms = gpe2d.rms_radius(self.psi, self.system.center, self.system.space_grid)
            self.rms_measurements[count] = rms
            self.cross_line[count, :] = gpe2d.calculate_cross_section_line(self.psi)
            self.energies.append(gpe.calculate_energy_allocation(self.psi, self.uext, (self.p1, self.p2, self.p3), u=self.u))
            
            self.logger.info(f"t = {t / self.omega_ho}")
        except Exception as e:
            self.logger.error(f"Error writing iteration data at step {count}: {e}")
            # Do not raise here to avoid stopping simulation for a single write failure, unless critical.
            # Usually logging is enough.

    def _perform_initial_imprint(self) -> None:
        """
        Performs the initial imprint of vortices.
        """
        try:
            self.logger.info("Imprinting the initial topological object")
            self._imprint_vortices(self.vortices)
        except Exception as e:
            self.logger.error(f"Failed to perform initial imprint: {e}")
            raise

    def _perform_repetitive_imprint(self, num_imprints: int) -> None:
        """
        Performs repetitive imprinting with possibly different phases.
        """
        try:
            if not self.imprint_times or num_imprints >= len(self.imprint_times):
                self.logger.warning(f"Invalid repetitive imprint request: num_imprints={num_imprints}")
                return

            vortex_array = self.imprinting_vortices_dictionary[self.imprint_times[num_imprints]]
            x = tuple(vortex_array[0])
            y = tuple(vortex_array[1])
            charge = tuple(vortex_array[2])
            key_for_phase = (x, y, charge)
            
            if key_for_phase not in self.all_phases:
                self.logger.error(f"Phase for key {key_for_phase} not calculated.")
                return

            phaseImp = self.all_phases[key_for_phase]
            self.logger.info(f"Imprinting again...{key_for_phase}")
            self._repetitive_imprint(phaseImp)
        except Exception as e:
            self.logger.error(f"Error performing repetitive imprint: {e}")
            raise

    def _turn_off_potential(self) -> None:
        """
        Turns off the external potential.
        """
        try:
            self.logger.info("External potential set to zero")
            if hasattr(self.system.uext, 'zero'):
                self.uext = self.system.uext.zero()
                self.reset_potential = True
            else:
                self.logger.warning("External potential does not support 'zero()' method.")
        except Exception as e:
            self.logger.error(f"Error turning off potential: {e}")

    def _write_simulation_outputs(self) -> None:
        """
        Writes various output files after the simulation.
        """
        try:
            SimulationName = self.simulation_name
            rw.write_rms(self.rms_measurements, SimulationName)
            rw.save_rms_figure(f"{SimulationName}_RMS_meas.txt")
            rw.save_cross_section_line_figure(self.cross_line)
            rw.save_tensor_to_csv(self.cross_line, "cross_line_density.csv")
            rw.write_energy_terms(self.energies, "energies.txt")
            
            video_creation.create_video(count=len(self.rms_measurements), simulation_name=SimulationName, n1=self.n1, n3=self.n3)
            
            if hasattr(self.app, 'write_velocity') and self.app.write_velocity:
                video_creation.create_velocity_video(len(self.rms_measurements), SimulationName, self.n1, self.n3)
        except Exception as e:
            self.logger.error(f"Error writing final simulation outputs: {e}")
