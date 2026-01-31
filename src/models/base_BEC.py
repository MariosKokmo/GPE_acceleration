"""
Base BEC class with common functionality.
This class provides the core functionality for BEC simulations.
Extend this class and override methods as needed for custom simulations.
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
    
    def __init__(self, parameters, system, app, simulation_name):
        """
        Initialize the BEC simulation.
        
        Args:
            parameters: dict, simulation-specific parameters
            system: System object containing grid, potential, etc.
            app: Application object with device, logger, etc.
            simulation_name: str, name of this simulation
        """
        self.psi = None  # Wavefunction
        self.app = app
        self.device = app.device
        self.logger = app.logger
        self.time = app.time
        self.parameters = parameters
        self.simulation_name = simulation_name
        self.system = system
        self.gs_path = None
        
        # TODO: Add your custom instance variables here
        # Example:
        # self.custom_data = []
        # self.analysis_results = {}

    def _find_ground_state(self):
        """
        Finds the ground state for the BEC in the system.
        If it exists, it just reads the file.
        The required format is `{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat`
        If a ground state file does not exist, it is computed.
        """
        n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
        fx, fy, fz = self.system.simulation_parameters["Trapping_frequencies"]
        
        # Find ground state for the specific grid and potential if it doesn't exist
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
        
        Override this method if you need custom initialization logic.
        """
        if self.psi is not None:
            self.logger.warning("Trying to initialise an already initialised BEC. It will overwrite.")
        
        self._find_ground_state()
        n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
        self.psi = torch.zeros((n1, n2, n3), dtype=torch.cdouble, device=self.device)
        self.psi = gs.read_ground_state(self.gs_path, n1, n2, n3)
        self.psi = self.psi.to(self.device)
        
        # TODO: Add custom initialization if needed
        # Example: apply initial phase imprinting, modify amplitude, etc.
    
    def _step(self, utot, dtau, p_sq, d_x):
        """
        Performs a single time step evolution of the BEC
        following the split-step Fourier method.
        
        Args:
            utot: torch.Tensor, total potential (interaction + external)
            dtau: float, time step size
            p_sq: torch.Tensor, squared momentum grid
            d_x: float, spatial grid spacing
        """
        self.psi = gpe.split_step_step(self.psi, utot, dtau, p_sq, d_x)
    
    def _extract_phase(self):
        """
        Returns the phase of the condensate wavefunction.
        
        Returns:
            torch.Tensor, the phase of the wavefunction
        """
        return gpe.extract_phase(self.psi)

    def evolve(self):
        """
        Main evolution method. This orchestrates the entire simulation.
        
        The typical flow is:
        1. Initialize parameters
        2. Initialize wavefunction to ground state
        3. Run the main simulation loop
        4. Write outputs
        
        Override this if you need a completely different simulation structure.
        """
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

    def _initialize_simulation_parameters(self):
        """
        Initializes simulation parameters for easy access.
        
        This extracts commonly used parameters from the system and stores them
        as instance variables for faster access during the simulation loop.
        
        Override or extend this method to add custom parameters.
        """
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
        self.uext = self.system.uext.potential
        self.u = params["u"]
        
        # Measurement arrays
        self.rms_measurements = {}
        self.cross_line = torch.zeros(self.shots, self.n1)
        self.energies = []
        
        # TODO: Call custom parameter initialization
        self._initialize_custom_parameters()
    
    def _initialize_custom_parameters(self):
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

    def _main_simulation_loop(self):
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
        count = 0
        
        # TODO: Add any pre-loop initialization
        # Example: open diagnostic files, initialize counters, etc.
        
        self.logger.info("Starting main simulation loop...")
        
        for iteration in range(self.kmax):
            # Current time
            t = self.dt * iteration * self.omega_ho
            
            # Total potential: interaction + external
            utot = self.u * torch.abs(self.psi) ** 2 + self.uext
            
            # Save data at regular intervals
            if iteration % (self.kmax // self.shots) == 0:
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
        
        self.logger.info(f"Simulation loop completed. Total iterations: {self.kmax}")
        
        # TODO: Add any post-loop cleanup or final measurements

    def _write_iteration_data(self, count, t):
        """
        Writes data for the current iteration.
        
        This is called at regular intervals during the simulation to save
        snapshots of the system state.
        
        Override or extend this to add custom measurements or outputs.
        
        Args:
            count: int, snapshot counter
            t: float, current time
        """
        # Write density data
        rw.write_data(self.psi, count, self.x1, self.x3, self.n1, self.n3, self.a_ho)
        
        # Optional: save phase imaging
        if self.app.phase_imaging:
            cur_phase = self._extract_phase()
            rw.save_figure_phase(cur_phase, count)
        
        # Calculate and store RMS radius
        rms = gpe.rms_radius(self.psi, self.system.center, self.system.space_grid)
        self.rms_measurements[count] = rms
        
        # Calculate cross-section
        self.cross_line[count, :] = gpe.calculate_cross_section_line(self.psi)
        
        # Calculate energy allocation
        self.energies.append(
            gpe.calculate_energy_allocation(self.psi, self.uext, self.p_grid, {"u": self.u})
        )
        
        # Log progress
        self.logger.info(f"t = {t / self.omega_ho}")
        
        # TODO: Add custom measurements here
        # Examples:
        # - self.custom_observable[count] = self._calculate_custom_quantity()
        # - self._check_convergence_criteria()
        # - self._save_special_diagnostic(count)

    def _write_simulation_outputs(self):
        """
        Writes various output files after the simulation.
        
        This is called once at the end of the simulation to generate
        final output files, plots, and videos.
        
        Override or extend this to add custom outputs.
        """
        SimulationName = self.simulation_name
        
        # Write RMS measurements
        rw.write_rms(self.rms_measurements, SimulationName)
        rw.save_rms_figure(f"{SimulationName}_RMS_meas.txt")
        
        # Write cross-section data
        rw.save_cross_section_line_figure(self.cross_line)
        rw.save_tensor_to_csv(self.cross_line, "cross_line_density.csv")
        
        # Write energy data
        rw.write_energy_terms(self.energies, "energies.txt")
        
        # Create visualization videos
        video_creation.create_video(
            count=len(self.rms_measurements), 
            simulation_name=SimulationName, 
            n1=self.n1, 
            n3=self.n3
        )
        
        if self.app.write_velocity:
            video_creation.create_velocity_video(
                len(self.rms_measurements), 
                SimulationName, 
                self.n1, 
                self.n3
            )
        
        # TODO: Write custom outputs
        self._write_custom_outputs()
        
        self.logger.info(f"All outputs written for simulation: {SimulationName}")
    
    def _write_custom_outputs(self):
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
    
    def get_density(self):
        """
        Returns the density |psi|^2 of the condensate.
        
        Returns:
            torch.Tensor, density of the wavefunction
        """
        return torch.abs(self.psi) ** 2

