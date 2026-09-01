"""
Base BEC class with common functionality.
This class provides the core functionality for BEC simulations.
Extend this class and override methods as needed for custom simulations.
"""
from typing import Optional, Dict, Any
import logging
from src.library.common_utils import CommonUtils as cu
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

        # Coordinate system: "cylindrical" when r_max is present in the config,
        # "cartesian" otherwise.  All coordinate-dependent branches key off this.
        self._coord = self.system._coord
        if not self._coord:
            raise ValueError("Coordinate system is not set in the system object.")

        if self._coord == "cylindrical":
            from src.library.gpe_cylindrical_library import GPECylindricalLibrary as _cyl
            from src.library.gpe_cylindrical_library import GPE2DCylindricalLibrary as _cyl2d
            from src.library.ground_state_cylindrical import CylindricalGroundState as _cgs
            self._lib = _cyl
            # Diagnostics (rms_radius, column densities) live on the 2-D
            # subclass in both coordinate systems.
            self._gpe2d_lib = _cyl2d
            self._gs_lib = _cgs
        elif self._coord == "cartesian":
            from src.library.gpe_library import GPELibrary as gpe
            from src.library.gpe_library import GPE2DLibrary as gpe2d
            from src.library.ground_state import GroundState as gs
            self._lib = gpe
            self._gpe2d_lib = gpe2d
            self._gs_lib = gs

        # TODO: Add your custom instance variables here
        # Example:
        # self.custom_data = []
        # self.analysis_results = {}

    def _find_ground_state(self) -> None:
        """
        Finds the ground state for the BEC in the system.
        If the file already exists it is loaded directly; otherwise it is computed.

        Cartesian filename: ``{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat``
        Cylindrical filename: ``{n_r}x{n_phi}x{n_z}_{fr}_{fz}Hz_ground_state_cyl.dat``
        """
        try:
            n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
            freqs = self.system.simulation_parameters["Trapping_frequencies"]
        except KeyError:
            self.logger.exception("Missing simulation parameter while finding ground state.")
            raise

        cur_path = Path(os.getcwd())
        parent_path = str(cur_path.parent.absolute())

        try:
            os.chdir(parent_path)

            if self._coord == "cylindrical":
                fr = freqs[0]
                fz = freqs[-1]
                gs_file = f"{n1}x{n2}x{n3}_{fr}_{fz}Hz_ground_state_cyl.dat"
            else:
                fx, fy, fz = freqs
                gs_file = f"{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat"

            if not os.path.exists(gs_file):
                self.logger.info("Calculating ground state...")
                try:
                    self._gs_lib.find_ground_state(self.parameters, self.system, gs_file, device=self.device)
                except Exception:
                    self.logger.exception("Failed to calculate ground state.")
                    raise

            self.logger.info(f"Ground state file: {gs_file}")
            sep = "\\" if platform == "win32" else "/"
            self.gs_path = os.getcwd() + sep + gs_file

        except OSError:
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
            self.psi = self._gs_lib.read_ground_state(self.gs_path, n1, n2, n3)
            self.psi = self.psi.to(self.device)
        except Exception:
            self.logger.exception("Failed to initialise BEC.")
            raise
        
        # TODO: Add custom initialization if needed
        # Example: apply initial phase imprinting, modify amplitude, etc.
    
    def _step(self, utot: torch.Tensor, dtau: float,
              p_sq: torch.Tensor = None, d_x: float = None) -> None:
        """
        Performs a single time step using the split-step Fourier method.

        Cartesian: ``_step(utot, dtau, p_sq, d_x)`` — explicit momentum grid and
        volume element are required.

        Cylindrical: ``_step(utot, dtau)`` — the kinetic operator is applied via
        the precomputed radial eigendecomposition stored on ``self`` (set up by
        ``_initialize_simulation_parameters``).

        Args:
            utot  : total potential (interaction + external).
            dtau  : time step size.
            p_sq  : squared momentum grid (Cartesian only).
            d_x   : flat volume element (Cartesian only).
        """
        if self.psi is None:
            raise RuntimeError("BEC wavefunction (psi) is not initialized.")
        if self._coord == "cylindrical":
            self.psi = self._lib.split_step_step(
                self.psi, utot, dtau,
                self.kz, self.m_modes, self.r,
                self.eigvecs_dict, self.eigvals_dict,
                self.dr, self.dphi, self.dz,
            )
        else:
            self.psi = self._lib.split_step_step(self.psi, utot, dtau, p_sq, d_x)
    
    def _apply_three_body_loss(self, dtau: float) -> None:
        """
        Apply the three-body loss factor exp(-k3 |psi|^4 dtau) in place.

        This is the loss written as its own operator rather than as an
        imaginary part of the potential. The template loop folds it into
        ``utot`` instead, which is exact there because every other factor in
        the split step is unitary. Inside the SGPE it is not: the propagator
        carries an ``(i + gamma)`` prefactor, so an imaginary ``utot`` would
        give the right decay multiplied by a spurious phase of order
        ``gamma * k3 * |psi|^4 * dtau``. Applying the loss separately keeps the
        SGPE in its standard form,

            dpsi/dt = -(i + gamma)(H_mf - mu) psi - k3 |psi|^4 psi + noise

        Args:
            dtau: time increment to apply the loss over. Strang-split callers
                pass half a step before and half after the propagator.
        """
        if not self.k3 or self.psi is None:
            return
        self.psi = torch.exp(-self.k3 * torch.abs(self.psi) ** 4 * dtau) * self.psi

    def _extract_phase(self) -> torch.Tensor:
        """
        Returns the phase of the condensate wavefunction.
        
        Returns:
            torch.Tensor, the phase of the wavefunction
        """
        if self.psi is None:
             raise RuntimeError("BEC wavefunction (psi) is not initialized.")
        return cu.extract_phase(self.psi)

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

            # Time evolution parameters (shared by both coordinate systems)
            self.kmax    = params["kmax"]
            self.dt      = params["dt"]
            self.omega_ho = params["omega_ho"]
            self.shots   = params["shots"]
            self.dtau    = params["dtau"]
            self.a_ho    = params["a_ho"]
            self.n1, self.n2, self.n3 = params["Grid_resolution"]

            # Potential and interaction (shared)
            if self.system.uext is None:
                raise ValueError("External potential (uext) is None")
            self.uext = self.system.uext.potential
            self.u = params["u"]
            # Three-body loss rate. Read with .get so a hand-built System or an
            # older configuration without the key still runs (lossless).
            self.k3 = params.get("k3", 0.0)

            # --- Coordinate-specific grid parameters ---
            if self._coord == "cylindrical":
                self._init_cylindrical_grid_params(params)
            elif self._coord == "cartesian":
                self._init_cartesian_grid_params(params)

            if self.kmax <= 0:
                raise ValueError("kmax must be > 0")
            if self.shots < 0:
                raise ValueError("shots must be >= 0")
            if self.n1 <= 0 or self.n2 <= 0 or self.n3 <= 0:
                raise ValueError("Grid_resolution values must all be > 0")

            # Measurement arrays
            # cross_line stores a 1-D density profile per snapshot:
            #   Cartesian  → n(x) at fixed y,z  (length n1)
            #   Cylindrical → n(r) at fixed z   (length n_r = n1)
            self.rms_measurements = {}
            self.cross_line = torch.zeros(self.shots, self.n1)
            self.energies = []

            # TODO: Call custom parameter initialization
            self._initialize_custom_parameters()

            # Dark-soliton parameters (shared by all subclasses; Cartesian only).
            self._initialize_dark_soliton_parameters()
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

    # ------------------------------------------------------------------
    # Dark-soliton imprinting — shared by every BaseBEC subclass
    # (FiniteTempBEC, ZNGBEC, ...). Cartesian only; cylindrical geometry
    # has no soliton library. BEC has its own standalone implementation.
    # ------------------------------------------------------------------
    def _initialize_dark_soliton_parameters(self) -> None:
        """
        Read this simulation's dark-soliton parameters from ``self.parameters``.

        Dark solitons are configured per simulation (one flat list of values for
        this simulation), enabled by the global ``dark_soliton`` flag. Sets
        ``self._soliton_enabled`` / ``self._soliton_imprinted``, which the
        per-iteration hook :meth:`_maybe_imprint_solitons` keys off. Supported
        only for Cartesian grids.
        """
        self._soliton_enabled = False
        self._soliton_imprinted = False

        if not self.parameters.get("dark_soliton", False):
            return
        if self._coord != "cartesian":
            self.logger.warning(
                "Dark solitons are only supported in Cartesian coordinates; "
                "skipping soliton imprinting for this simulation."
            )
            return

        positions = self.parameters.get("soliton_positions")
        if not positions:
            return  # this simulation imprints no soliton

        self.soliton_positions = positions
        self.soliton_widths = self.parameters.get("soliton_widths")
        self.soliton_axes = self.parameters.get("soliton_axes")
        self.soliton_greyness = self.parameters.get("soliton_greyness", None)
        self.soliton_imprint_time = self.parameters.get("soliton_imprint_time", 0)
        self._soliton_enabled = True
        self.logger.info(
            f"Dark soliton configured: {len(self.soliton_positions)} soliton(s), "
            f"imprint at snapshot {self.soliton_imprint_time}"
        )

    def _imprint_dark_solitons(self) -> None:
        """Create and apply this simulation's dark-soliton mask to the wavefunction."""
        if self.psi is None:
            raise RuntimeError("BEC wavefunction (psi) is not initialized.")
        try:
            x1, _, x3 = self.system.space_axes
            mask = self._gpe2d_lib.create_dark_soliton(
                x1, x3,
                self.n1, self.n2, self.n3,
                positions=self.soliton_positions,
                widths=self.soliton_widths,
                axes=self.soliton_axes,
                greyness=self.soliton_greyness,
                device=self.device,
            )
            self.psi = self._gpe2d_lib.imprint_dark_soliton(self.psi, mask)
            self._soliton_imprinted = True
            self.logger.info("Dark soliton(s) imprinted successfully.")
        except Exception:
            self.logger.exception("Error imprinting dark solitons.")
            raise

    def _maybe_imprint_solitons(self, iteration: int) -> None:
        """
        Imprint this simulation's solitons when the loop reaches the configured
        snapshot. Safe no-op when solitons are disabled or already imprinted.

        Subclasses that override :meth:`_main_simulation_loop` should call this
        once per iteration to inherit dark-soliton support.
        """
        if not getattr(self, "_soliton_enabled", False) or self._soliton_imprinted:
            return
        interval = self._get_snapshot_interval()
        if interval is None:
            return
        if iteration == interval * self.soliton_imprint_time:
            self._imprint_dark_solitons()

    # ------------------------------------------------------------------
    # Grid helpers called from _initialize_simulation_parameters
    # ------------------------------------------------------------------

    def _init_cartesian_grid_params(self, params: dict) -> None:
        """Load Cartesian grid objects from the system object."""
        self.p_sq  = self.system.p_sq
        self.p_grid = self.system.p_grid
        self.x1, self.x2, self.x3 = self.system.space_axes
        self.p1, self.p2, self.p3 = self.system.momentum_axes
        self.dx  = params["dx"]
        self.d_x = params["d_x"]

    def _init_cylindrical_grid_params(self, params: dict) -> None:
        """
        Build or load cylindrical grid objects.

        If the system object already carries pre-built cylindrical grid
        attributes (``system.r``, ``system.kz``, etc.) they are reused directly.
        Otherwise the grid is built from the scalar parameters in ``params``.

        Sets the following instance attributes:
            r, phi, z          – 1-D coordinate arrays
            kz, m_modes        – spectral grids
            dr, dphi, dz       – spacings
            eigvecs_dict, eigvals_dict – precomputed radial kinetic eigendecomp
            n_r, n_phi, n_z    – dimensional aliases (same objects as n1, n2, n3)
        """
        self.n_r   = self.n1
        self.n_phi = self.n2
        self.n_z   = self.n3

        if getattr(self.system, "r", None) is not None:
            # System has pre-built cylindrical grid
            self.r       = self.system.r
            self.phi     = self.system.phi
            self.z       = self.system.z
            self.kz      = self.system.kz
            self.m_modes = self.system.m_modes
            self.dr      = self.system.dr
            self.dphi    = self.system.dphi
            self.dz      = self.system.dz
            self.eigvecs_dict = self.system.eigvecs_dict
            self.eigvals_dict = self.system.eigvals_dict
        else:
            # Build from scalar params
            (self.r, self.phi, self.z,
             self.kz, self.m_modes,
             self.dr, self.dphi, self.dz, _) = self._lib.init_grid(
                params["r_max"], params["z_min"], params["z_max"],
                self.n_r, self.n_phi, self.n_z, self.device,
            )
            self.eigvecs_dict, self.eigvals_dict = self._lib.build_radial_operators(
                self.r, self.dr, self.m_modes, self.device
            )

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

                # Recompute total potential from the current psi and uext.
                # A non-zero three-body rate enters as an imaginary part: the
                # real-space factor exp(-i dtau/2 (V + iL)) splits exactly into
                # a unitary phase times the amplitude decay exp(dtau L / 2), so
                # atoms are genuinely removed rather than just reshuffled.
                utot = self.u * torch.abs(self.psi) ** 2 + self.uext
                if self.k3:
                    utot = utot + 1j * (-self.k3 * torch.abs(self.psi) ** 4)

                # Save data at regular intervals
                if (
                    snapshot_interval is not None
                    and iteration % snapshot_interval == 0
                    and count < self.shots
                ):
                    self._write_iteration_data(count, t)
                    count += 1

                # Imprint dark solitons at the configured snapshot (no-op if disabled)
                self._maybe_imprint_solitons(iteration)

                # TODO: Add custom physics here
                # Examples:
                # - if iteration == special_time:
                #     self._apply_custom_operation()
                # - if some_condition(t):
                #     self._modify_potential()
                # - Adaptive measurements based on system state

                # Perform time step
                # Cartesian: passes p_sq / d_x explicitly.
                # Cylindrical: _step reads the kinetic operators from self.*.
                if self._coord == "cylindrical":
                    self._step(utot, self.dtau)
                else:
                    self._step(utot, self.dtau, self.p_sq, self.d_x)

                # Advance time-dependent external potential so next iteration
                # picks up the updated uext when computing utot.
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
            if self._coord == "cylindrical":
                self._write_iteration_data_cylindrical(count, t)
            elif self._coord == "cartesian":
                self._write_iteration_data_cartesian(count, t)
        except Exception as e:
            self.logger.exception(f"Error writing iteration data at step {count}.")
            raise
        
        # TODO: Add custom measurements here
        # Examples:
        # - self.custom_observable[count] = self._calculate_custom_quantity()
        # - self._check_convergence_criteria()
        # - self._save_special_diagnostic(count)

    def _write_iteration_data_cartesian(self, count: int, t: float) -> None:
        """Snapshot diagnostics for a Cartesian (x, y, z) grid."""
        rw.write_data(self.psi, count, self.x1, self.x3, self.n1, self.n3, self.a_ho, self.dx)

        if hasattr(self.app, 'phase_imaging') and self.app.phase_imaging:
            rw.save_figure_phase(self._extract_phase(), count)

        rms = self._gpe2d_lib.rms_radius(self.psi, self.system.center, self.system.space_grid)
        self.rms_measurements[count] = rms

        self.cross_line[count, :] = self._gpe2d_lib.calculate_cross_section_line(self.psi)

        self.energies.append(
            self._lib.calculate_energy_allocation(
                self.psi, self.uext, (self.p1, self.p2, self.p3), self.d_x, u=self.u
            )
        )
        self.logger.info(f"t = {t / self.omega_ho}")

    def _write_iteration_data_cylindrical(self, count: int, t: float) -> None:
        """Snapshot diagnostics for a cylindrical (r, φ, z) grid."""
        # Column density n(r, z) = ∫|ψ|² dφ
        rw.write_data_cylindrical(
            self.psi, count, self.r, self.phi, self.n_r, self.n_phi, self.a_ho, self.dz
        )

        # Optional phase imaging at the φ = 0 cross-section
        if hasattr(self.app, 'phase_imaging') and self.app.phase_imaging:
            rw.save_figure_phase_cylindrical(self._extract_phase(), count)

        # RMS radius ⟨r²⟩^½ with cylindrical volume element r dr dφ dz
        rms = self._gpe2d_lib.rms_radius(self.psi, self.r, self.dr, self.dphi, self.dz)
        self.rms_measurements[count] = rms

        # Radial density profile at z-midplane: n(r) = Σ_φ |ψ[:, :, n_z//2]|² · dφ
        self.cross_line[count, :] = (
            torch.sum(torch.abs(self.psi[:, :, self.n_z // 2]) ** 2, dim=1) * self.dphi
        )

        # Energy with cylindrical volume element
        self.energies.append(
            self._lib.calculate_energy_allocation(
                self.psi, self.uext,
                self.r, self.dr, self.dphi, self.dz,
                self.kz, self.m_modes,
                u=self.u,
            )
        )
        self.logger.info(f"t = {t / self.omega_ho}")

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
            n_frames = len(self.rms_measurements)
            if self._coord == "cylindrical":
                video_creation.create_video_cylindrical(
                    count=n_frames,
                    simulation_name=self.simulation_name,
                    n_r=self.n_r,
                    n_phi=self.n_phi,
                )
            else:
                video_creation.create_video(
                    count=n_frames,
                    simulation_name=self.simulation_name,
                    n1=self.n1,
                    n3=self.n3,
                )

            if hasattr(self.app, 'write_velocity') and self.app.write_velocity:
                if self._coord == "cylindrical":
                    video_creation.create_velocity_video_cylindrical(
                        n_frames,
                        self.simulation_name,
                        self.n_r,
                        self.n_phi,
                    )
                else:
                    video_creation.create_velocity_video(
                        n_frames,
                        self.simulation_name,
                        self.n1,
                        self.n3,
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

