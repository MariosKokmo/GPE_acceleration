"""
Base BEC class with common functionality.
This class provides the core functionality for BEC simulations.
Extend this class and override methods as needed for custom simulations.
"""
from typing import Optional, Dict, Any
import logging
from src.library.gpe_library import GPELibrary as gpe
from src.library.gpe_library import GPE2DLibrary as gpe2d
from src.library.ground_state import GroundState as gs
import src.utils.read_write_utils as rw
from src.utils import video_creation
import os
import torch
from pathlib import Path
from sys import platform


class BaseBEC:
    """
    Base class for Bose-Einstein Condensate simulations.
    
    This class provides common functionality including:
    - Ground state initialization
    - Time evolution using split-step Fourier method
    - Parameter management
    - Data output and logging
    
    To create a custom simulation:
    1. Inherit from this class
    2. Override _main_simulation_loop() to implement custom physics
    3. Override _initialize_custom_parameters() for simulation-specific parameters
    4. Override _write_custom_outputs() for additional output files
    """
    
    def __init__(self, parameters: Dict[str, Any], system: Any, app: Any, simulation_name: str) -> None:
        """
        Initialize the BEC simulation.
        
        Args:
            parameters: dict, simulation-specific parameters
            system: System object containing grid, potential, etc.
            app: Application object with device, logger, etc.
            simulation_name: str, name of this simulation
        """
        if app is None:
            raise ValueError("app must not be None")
        if system is None:
            raise ValueError("system must not be None")

        self.psi: Optional[torch.Tensor] = None  # Wavefunction
        self.app = app
        self.device = app.device
        self.logger = app.logger if getattr(app, "logger", None) else logging.getLogger(__name__)
        self.time = app.time
        self.parameters = parameters
        self.simulation_name = simulation_name
        self.system = system
        self.gs_path: Optional[str] = None
        
        # TODO: Add your custom instance variables here
        # Example:
        # self.custom_data = []
        # self.analysis_results = {}

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
            self.logger.exception("Missing simulation parameter while finding ground state.")
            raise

        # Find ground state for the specific grid and potential if it doesn't exist
        cur_path = Path(os.getcwd())
        parent_path = str(cur_path.parent.absolute())
        
        try:
            os.chdir(parent_path)
            
            gs_file = f"{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat"
            if not os.path.exists(gs_file):
                self.logger.info("Calculating ground state...")
                try:
                    _ = gs.find_ground_state(self.parameters, self.system, gs_file, device=self.device)
                except Exception as e:
                    self.logger.exception("Failed to calculate ground state.")
                    raise
            self.logger.info(f"Ground state file: {gs_file}")
            
            if platform == "win32":
                self.gs_path = os.getcwd() + "\\" + gs_file
            else:
                self.gs_path = os.getcwd() + "/" + gs_file
        except OSError as e:
            self.logger.exception("File system error in _find_ground_state.")
            raise
        finally:
            os.chdir(cur_path)
    
    def _initialise(self) -> None:
        """
        Reads the ground state file and initialises the wavefunction to ground state.
        
        Override this method if you need custom initialization logic.
        """
        if self.psi is not None:
            self.logger.warning("Trying to initialise an already initialised BEC. It will overwrite.")
        
        try:
            self._find_ground_state()
            n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
            if self.gs_path is None:
                raise ValueError("Ground state path is None after _find_ground_state")
            self.psi = gs.read_ground_state(self.gs_path, n1, n2, n3)
            self.psi = self.psi.to(self.device)
        except Exception as e:
            self.logger.exception("Failed to initialise BEC.")
            raise
        
        # TODO: Add custom initialization if needed
        # Example: apply initial phase imprinting, modify amplitude, etc.
    
    def _step(self, utot: torch.Tensor, dtau: float, p_sq: torch.Tensor, d_x: float) -> None:
        """
        Performs a single time step evolution of the BEC
        following the split-step Fourier method.
        
        Args:
            utot: torch.Tensor, total potential (interaction + external)
            dtau: float, time step size
            p_sq: torch.Tensor, squared momentum grid
            d_x: float, spatial grid spacing
        """
        if self.psi is None:
            raise RuntimeError("BEC wavefunction (psi) is not initialized.")
        self.psi = gpe.split_step_step(self.psi, utot, dtau, p_sq, d_x)
    
    def _extract_phase(self) -> torch.Tensor:
        """
        Returns the phase of the condensate wavefunction.
        
        Returns:
            torch.Tensor, the phase of the wavefunction
        """
        if self.psi is None:
             raise RuntimeError("BEC wavefunction (psi) is not initialized.")
        return gpe.extract_phase(self.psi)

    def evolve(self) -> None:
        """
        Main evolution method. This orchestrates the entire simulation.
        
        The typical flow is:
        1. Initialize parameters
        2. Initialize wavefunction to ground state
        3. Run the main simulation loop
        4. Write outputs
        
        Override this if you need a completely different simulation structure.
        """
        try:
            # Get the parameters for easy access
            self._initialize_simulation_parameters()

            # Initialise the BEC on the ground state
            self._initialise()

            # TODO: Add any pre-evolution setup here
            # Example: calculate special phases, set up diagnostics, etc.
            
            # Evolve the BEC
            self._main_simulation_loop()

            # Write various output files
            self._write_simulation_outputs()
        except Exception as e:
            self.logger.exception("Simulation failed in evolve().")
            raise

    def _get_snapshot_interval(self) -> Optional[int]:
        """
        Returns the iteration interval used for snapshot writes.

        Returns None when snapshots are disabled.
        """
        if self.shots <= 0:
            return None
        return max(1, self.kmax // self.shots)

    def _initialize_simulation_parameters(self) -> None:
        """
        Initializes simulation parameters for easy access.
        
        This extracts commonly used parameters from the system and stores them
        as instance variables for faster access during the simulation loop.
        
        Override or extend this method to add custom parameters.
        """
        try:
            params = self.system.simulation_parameters
            
            # Time evolution parameters
            self.kmax = params["kmax"]
            self.dt = params["dt"]
            self.omega_ho = params["omega_ho"]
            self.shots = params["shots"]
            self.dtau = params["dtau"]
            self.d_x = params["d_x"]
            self.a_ho = params["a_ho"]
            
            # Grid parameters
            self.p_sq = self.system.p_sq
            self.p_grid = self.system.p_grid
            self.x1, self.x2, self.x3 = self.system.space_axes
            self.p1, self.p2, self.p3 = self.system.momentum_axes
            self.n1, self.n2, self.n3 = params["Grid_resolution"]
            
            # Potential and interaction
            if self.system.uext is None:
                raise ValueError("External potential (uext) is None")
            self.uext = self.system.uext.potential
            self.u = params["u"]

            if self.kmax <= 0:
                raise ValueError("kmax must be > 0")
            if self.shots < 0:
                raise ValueError("shots must be >= 0")
            if self.n1 <= 0 or self.n2 <= 0 or self.n3 <= 0:
                raise ValueError("Grid_resolution values must all be > 0")
            
            # Measurement arrays
            self.rms_measurements = {}
            self.cross_line = torch.zeros(self.shots, self.n1)
            self.energies = []
            
            # TODO: Call custom parameter initialization
            self._initialize_custom_parameters()
        except KeyError as e:
            self.logger.exception("Missing simulation parameter during initialization.")
            raise
        except AttributeError as e:
            self.logger.exception("System attribute missing during initialization.")
            raise
        except ValueError:
            self.logger.exception("Invalid simulation parameter values.")
            raise
            
    def _initialize_custom_parameters(self) -> None:
        """
        Initialize custom simulation-specific parameters.
        
        Override this method in your derived class to set up additional parameters.
        
        Example:
            def _initialize_custom_parameters(self):
                self.my_custom_param = self.parameters.get("my_param", default_value)
                self.special_measurements = []
        """
        # TODO: Add your custom parameter initialization here
        pass

    def _main_simulation_loop(self) -> None:
        """
        Main loop for evolving the BEC system.
        
        This is a TEMPLATE method that should be overridden for custom simulations.
        The default implementation provides a basic time evolution with measurements.
        
        Common modifications:
        - Add custom physics (e.g., vortex imprinting, stirring, etc.)
        - Implement time-dependent potentials
        - Add special measurement/diagnostic routines
        - Implement adaptive time stepping
        """
        if self.psi is None:
            raise RuntimeError("BEC wavefunction (psi) is not initialized.")
        count = 0
        iteration = 0
        snapshot_interval = self._get_snapshot_interval()
        
        # TODO: Add any pre-loop initialization
        # Example: open diagnostic files, initialize counters, etc.
        
        self.logger.info("Starting main simulation loop...")
        
        try:
            for iteration in range(self.kmax):
                # Current time
                t = self.dt * iteration * self.omega_ho
                
                # Total potential: interaction + external
                utot = self.u * torch.abs(self.psi) ** 2 + self.uext
                
                # Save data at regular intervals
                if (
                    snapshot_interval is not None
                    and iteration % snapshot_interval == 0
                    and count < self.shots
                ):
                    self._write_iteration_data(count, t)
                    count += 1
                
                # TODO: Add custom physics here
                # Examples:
                # - if iteration == special_time:
                #     self._apply_custom_operation()
                # - if some_condition(t):
                #     self._modify_potential()
                # - Adaptive measurements based on system state
                
                # Perform time step
                self._step(utot, self.dtau, self.p_sq, self.d_x)
                if hasattr(self.system.uext, 'evol'):
                    self.uext = self.system.uext.evol(t)
            
            self.logger.info(f"Simulation loop completed. Total iterations: {self.kmax}")
        except Exception as e:
            self.logger.exception(f"Error in main simulation loop at iteration {iteration}.")
            raise
        
        # TODO: Add any post-loop cleanup or final measurements

    def _write_iteration_data(self, count: int, t: float) -> None:
        """
        Writes data for the current iteration.
        
        This is called at regular intervals during the simulation to save
        snapshots of the system state.
        
        Override or extend this to add custom measurements or outputs.
        
        Args:
            count: int, snapshot counter
            t: float, current time
        """
        if self.psi is None:
             self.logger.error("Skipping write_iteration_data because psi is None.")
             return

        if count < 0 or count >= self.shots:
            self.logger.warning(
                f"Skipping write_iteration_data for out-of-range snapshot index {count}."
            )
            return

        try:
            # Write density data
            rw.write_data(self.psi, count, self.x1, self.x3, self.n1, self.n3, self.a_ho)
            
            # Optional: save phase imaging
            if hasattr(self.app, 'phase_imaging') and self.app.phase_imaging:
                cur_phase = self._extract_phase()
                rw.save_figure_phase(cur_phase, count)
            
            # Calculate and store RMS radius
            rms = gpe.rms_radius(self.psi, self.system.center, self.system.space_grid)
            self.rms_measurements[count] = rms
            
            # Calculate cross-section line density
            self.cross_line[count, :] = gpe2d.calculate_cross_section_line(self.psi)
            
            # Calculate energy allocation
            self.energies.append(
                gpe.calculate_energy_allocation(self.psi, self.uext, self.p_grid, {"u": self.u})
            )
            
            # Log progress
            self.logger.info(f"t = {t / self.omega_ho}")
        except Exception as e:
            self.logger.exception(f"Error writing iteration data at step {count}.")
            raise
        
        # TODO: Add custom measurements here
        # Examples:
        # - self.custom_observable[count] = self._calculate_custom_quantity()
        # - self._check_convergence_criteria()
        # - self._save_special_diagnostic(count)

    def _write_simulation_outputs(self) -> None:
        """
        Writes various output files after the simulation.
        
        This is called once at the end of the simulation to generate
        final output files, plots, and videos.
        
        Override or extend this to add custom outputs.
        """
        try:
            # Write RMS measurements
            rw.write_rms(self.rms_measurements, self.simulation_name)
            rw.save_rms_figure(f"{self.simulation_name}_RMS_meas.txt")
            
            # Write cross-section data
            rw.save_cross_section_line_figure(self.cross_line)
            rw.save_tensor_to_csv(self.cross_line, "cross_line_density.csv")
            
            # Write energy data
            rw.write_energy_terms(self.energies, "energies.txt")
            
            # Create visualization videos
            video_creation.create_video(
                count=len(self.rms_measurements), 
                simulation_name=self.simulation_name, 
                n1=self.n1, 
                n3=self.n3
            )
            
            if hasattr(self.app, 'write_velocity') and self.app.write_velocity:
                video_creation.create_velocity_video(
                    len(self.rms_measurements), 
                    self.simulation_name, 
                    self.n1, 
                    self.n3
                )
            
            # TODO: Write custom outputs
            self._write_custom_outputs()
            
            self.logger.info(f"All outputs written for simulation: {self.simulation_name}")
        except Exception as e:
            self.logger.exception("Error writing final simulation outputs.")
            raise
    
    def _write_custom_outputs(self) -> None:
        """
        Write custom simulation-specific outputs.
        
        Override this method in your derived class to save additional data.
        
        Example:
            def _write_custom_outputs(self):
                np.save('my_data.npy', self.custom_data)
                with open('analysis.txt', 'w') as f:
                    f.write(str(self.analysis_results))
        """
        # TODO: Add your custom output writing here
        pass

    # Utility methods that might be useful
    
    def get_density(self) -> torch.Tensor:
        """
        Returns the density |psi|^2 of the condensate.
        
        Returns:
            torch.Tensor, density of the wavefunction
        """
        if self.psi is None:
            raise RuntimeError("BEC wavefunction (psi) is not initialized.")
        return torch.abs(self.psi) ** 2

