r"""
Laboratory system: grid and external potential.

The :class:`System` object is built once per run, before any BEC exists. It
parses the configuration file, decides whether the simulation is Cartesian
:math:`(x, y, z)` or cylindrical :math:`(r, \varphi, z)`, builds the
corresponding spatial and momentum grids, and constructs the external potential
:math:`V_\mathrm{ext}` that every model then evolves in.
"""
import src.utils.setup_simulations as setup_simulations
from src.library.potentials import select_potential
from src.library.potentials_cylindrical import select_potential_cylindrical

class System:
    r"""
    Laboratory setup shared by every simulation of a run.

    Construction parses the configuration file, validates it, selects the
    external potential and builds the grid. Only the attributes of the detected
    coordinate system are populated; those of the other one stay ``None``.

    Args:
        app: Application object providing the device, the logger, the
            configuration file name and the run timestamp. Both the device and
            the logger must already be set.

    Raises:
        Exception: If the device or the logger is not set, if the coordinate
            system is invalid, if the configuration file contains an error, or
            if no potential could be selected.

    Attributes:
        simulation_parameters (dict): Parsed and validated simulation
            parameters.
        uext: The external potential object; ``uext.potential`` holds
            :math:`V_\mathrm{ext}` on the device.
        space_axes (list): Cartesian coordinate axes ``[x1, x2, x3]``.
        momentum_axes (list): Cartesian momentum axes ``[p1, p2, p3]``.
        p_sq (torch.Tensor): Squared momentum grid :math:`p^{2}` used by the
            kinetic half-steps of the split-step propagator.
        p_grid (tuple): Momentum meshgrids ``(px, py, pz)``.
        space_grid: Coordinate meshgrids of the spatial grid.
        center (tuple): Coordinates of the grid centre.
        r (torch.Tensor): Cylindrical radial axis :math:`r`.
        phi (torch.Tensor): Cylindrical azimuthal axis :math:`\varphi`.
        z (torch.Tensor): Cylindrical axial axis :math:`z`.
        kz (torch.Tensor): Axial momentum grid :math:`k_z`.
        m_modes (torch.Tensor): Azimuthal mode numbers :math:`m`.
        dr (float): Radial grid spacing.
        dphi (float): Azimuthal grid spacing.
        dz (float): Axial grid spacing.
        eigvecs_dict (dict): Eigenvectors of the radial kinetic operator, one
            entry per azimuthal mode :math:`m`.
        eigvals_dict (dict): Corresponding eigenvalues.
    """
    def __init__(self, app):
        self.app = app
        self.device = self.app.device
        self.logger = self.app.logger
        self.configFile = self.app.configFile
        self.time = app.time

        # Coordinate system — set during _initialise_parameters
        self._coord = None

        # Cartesian grid objects
        self.space_axes = None
        self.momentum_axes = None
        self.p_sq = None
        self.p_grid = None
        self.space_grid = None
        self.center = None

        # Cylindrical grid objects (remain None for Cartesian simulations)
        self.r = None
        self.phi = None
        self.z = None
        self.kz = None
        self.m_modes = None
        self.dr = None
        self.dphi = None
        self.dz = None
        self.eigvecs_dict = None
        self.eigvals_dict = None

        self.simulation_parameters = None
        self.uext = None

        self._initialise_parameters()

    def _initialise_parameters(self):
        r"""
        Initialise the simulation parameters, the external potential and the
        grid.

        The coordinate system is detected in this priority order:

        1. An explicit ``"coordinates"`` key in the configuration file,
           ``"coordinates": "cylindrical"`` or ``"coordinates": "cartesian"``.
        2. Auto-detection fallback: cylindrical when ``"r_max"`` is present,
           Cartesian otherwise.

        Raises:
            Exception: If the device or logger is not set before the system, if
                the coordinate system is invalid, if the configuration file is
                faulty, or if no potential could be selected.
        """
        if not self.device:
            raise Exception("Device needs to be set before the system")
        if not self.logger:
            raise Exception("Logger needs to be set before the system")

        # Detect coordinate system without fully parsing the config
        raw = setup_simulations._load_json_from_cwd(self.configFile)
        coords = raw.get("coordinates", "").lower()
        if coords == "cylindrical":
            self._coord = "cylindrical"
        elif coords == "cartesian":
            self._coord = "cartesian"
        else:
            self._coord = "cylindrical" if "r_max" in raw  and raw["r_max"] is not None else "cartesian"

        if self._coord == "cylindrical":
            self.logger.info("Detected cylindrical coordinate system based on config file.")
            import src.sims_setup.cylindrical_setup as setup_sims
        elif self._coord == "cartesian":
            self.logger.info("Detected Cartesian coordinate system based on config file.")
            import src.sims_setup.cartesian_setup as setup_sims
        else:
            self.logger.critical("Invalid coordinate system specified in config file.")
            raise Exception("Invalid coordinate system specified in config file. See log for details.")

        if self._coord == "cylindrical":
            self.simulation_parameters, fault = \
                setup_sims.get_simulation_parameters_cylindrical(self.configFile)
        elif self._coord == "cartesian":
            self.simulation_parameters, fault = \
                setup_sims.get_simulation_parameters_cartesian(self.configFile)

        if fault and fault.strip():
            self.logger.critical(fault)
            raise Exception("there is an error in the configuration file. See Log.")

        potentialType = self.simulation_parameters["Potential_type"]
        if self._coord == "cylindrical":
            new_potential = select_potential_cylindrical(
                potentialType, self.app, **self.simulation_parameters
            )
        else:
            new_potential = select_potential(
                potentialType, self.app, **self.simulation_parameters
            )

        if not new_potential:
            self.logger.critical("Potential was not selected")
            raise Exception("there is an error in the configuration file. See log.")

        self.uext = new_potential
        self.logger.info(f"Potential on device {self.uext.potential.device}")

        self._initialise_grid()


    def _initialise_grid(self):
        r"""
        Build the grid for the detected coordinate system.

        Dispatches to :meth:`_initialise_grid_cylindrical` or
        :meth:`_initialise_grid_cartesian`.
        """
        assert self.simulation_parameters, "System simulation parameters not initialised"
        if self._coord == "cylindrical":
            self._initialise_grid_cylindrical()
        else:
            self._initialise_grid_cartesian()

    def _initialise_grid_cartesian(self):
        r"""
        Build the Cartesian :math:`(x, y, z)` grid objects.

        Sets the coordinate and momentum axes, the squared momentum grid
        :math:`p^{2}` used by the kinetic half-steps, the meshgrids, and the
        grid centre.
        """
        import src.library.gpe_library as gpe
        n1, n2, n3 = self.simulation_parameters["Grid_resolution"]
        x_min = self.simulation_parameters["x_min"]
        dx = self.simulation_parameters["dx"]
        dp = self.simulation_parameters["dp"]
        x1, x2, x3, p1, p2, p3, p_sq, space_grid, p_grid = gpe.GPELibrary.init_grid(
            x_min, dx, dp, n1, n2, n3, self.device
        )
        self.space_axes = [x1, x2, x3]
        self.momentum_axes = [p1, p2, p3]
        self.p_sq = p_sq
        self.p_grid = p_grid
        self.space_grid = space_grid
        self.center = (x1[n1 // 2], x2[n2 // 2], x3[n3 // 2])

    def _initialise_grid_cylindrical(self):
        r"""
        Build the cylindrical :math:`(r, \varphi, z)` grid and precompute the
        radial operators.

        Alongside the axes and spacings, this diagonalises the radial part of
        the kinetic operator once per azimuthal mode :math:`m` and stores the
        eigenvectors and eigenvalues, so that the propagator can apply
        :math:`e^{-i T \Delta t / 2}` as a matrix product instead of an FFT in
        :math:`r`.
        """
        from src.library.gpe_cylindrical_library import GPECylindricalLibrary as cyl
        params = self.simulation_parameters
        n_r, n_phi, n_z = params["Grid_resolution"]
        (self.r, self.phi, self.z,
         self.kz, self.m_modes,
         self.dr, self.dphi, self.dz,
         self.space_grid) = cyl.init_grid(
            params["r_max"], params["z_min"], params["z_max"],
            n_r, n_phi, n_z, self.device,
        )
        self.center = (self.r[n_r // 2], 0.0, self.z[n_z // 2])
        self.eigvecs_dict, self.eigvals_dict = cyl.build_radial_operators(
            self.r, self.dr, self.m_modes, self.device
        )
        self.logger.info(
            f"Cylindrical grid: n_r={n_r}, n_phi={n_phi}, n_z={n_z}, "
            f"r_max={params['r_max']:.3f}, z=[{params['z_min']:.3f}, {params['z_max']:.3f}]"
        )
