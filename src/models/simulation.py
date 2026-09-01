r"""
Driver for the set of simulations to be run.

A simulation needs two ingredients: a :class:`~src.models.system.System`, i.e.
the laboratory setup that fixes the grid and the external potential, and a BEC
model, i.e. the condensate that evolves in it. :class:`Simulations` expands the
configuration file into one run per parameter combination, creates a fresh BEC
for each of them and lets it evolve.
"""
import src.utils.setup_simulations as setup_simulations
import os
from pathlib import Path
import torch


def _get_bec_class(model_type: str):
    r"""
    Return the BEC model class for a given ``model_type`` string.

    ``model_type`` is set in the configuration file and selects the physics
    model used for the condensate evolution:

    ``"BEC"``
        :class:`src.models.BEC.BEC` — zero-temperature GPE with vortex and
        soliton imprinting.
    ``"FiniteTempBEC"``
        :class:`src.models.finite_temp_BEC.FiniteTempBEC` — Stochastic
        Projected GPE (SGPE), i.e. a damped GPE plus thermal noise. Requires
        ``"temperature"`` and optionally ``"damping_coefficient"`` in the
        config.
    ``"ZNGBEC"``
        :class:`src.experimental.zng.zng_BEC.ZNGBEC` — full
        Zaremba-Nikuni-Griffin two-component framework, a condensate GPE
        coupled to a Monte Carlo thermal cloud. Requires ``"temperature"``,
        ``"n_test_particles"`` and ``"gamma_12"``.

    The imports are deferred so that the experimental ZNG module is loaded only
    when it is explicitly requested.

    Args:
        model_type (str): One of ``"BEC"``, ``"FiniteTempBEC"`` or
            ``"ZNGBEC"``.

    Returns:
        type: The BEC model class, ready to be instantiated.

    Raises:
        ValueError: If ``model_type`` is not one of the recognised strings.
    """
    if model_type == "FiniteTempBEC":
        from src.models.finite_temp_BEC import FiniteTempBEC
        return FiniteTempBEC
    if model_type == "ZNGBEC":
        from src.experimental.zng.zng_BEC import ZNGBEC
        return ZNGBEC
    if model_type == "BEC":
        from src.models.BEC import BEC
        return BEC
    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        f"Choose from: 'BEC', 'FiniteTempBEC', 'ZNGBEC'."
    )

class Simulations:
    r"""
    Container for every simulation to be run.

    The configuration file may specify several values for one or more
    parameters; each combination becomes its own simulation, with its own
    folder, log file and BEC instance. The runs are executed sequentially by
    :meth:`run_simulations`.

    Args:
        system (System): Laboratory setup providing the grid, the external
            potential and the parsed simulation parameters.
        app: Application object carrying the device, the logger and the run
            timestamp.

    Attributes:
        simulation_combinations (list): Pairs ``(simulation_name, parameters)``,
            one per run, built by :meth:`_setup_simulations`.
        BEC: The model instance of the simulation currently running, or
            ``None`` before the first run.
        device (torch.device): Device the simulations run on.
        logger (logging.Logger): Logger of the simulation currently running.
    """
    def __init__(self, system, app):
        self.simulation_combinations = None
        self.app = app
        self.logger = None
        self.time = self.app.time()
        self.device = self.app.device
        self.BEC = None
        self.system = system
        # set up simulations
        self._setup_simulations()

    def _setup_simulations(self):
        r"""
        Build the list of simulation combinations to be run.

        The parameter sets are expanded from
        ``system.simulation_parameters`` and stored in
        :attr:`simulation_combinations`.
        """
        self.simulation_combinations = setup_simulations.get_simulation_combinations(self.system.simulation_parameters)

    def run_simulations(self):
        r"""
        Run every simulation combination in sequence.

        For each combination, a dedicated folder and log file are created, the
        parameters are saved to JSON, a fresh BEC of the configured
        ``model_type`` is instantiated and evolved, and the CUDA cache is freed
        before moving on to the next run. The working directory is restored to
        the parent folder after each simulation.
        """
        # Run the simulations sequentially. For each simulation, create a new BEC and evolve it.
        for combination in self.simulation_combinations:
            simulation_name, parameters = combination

            if not os.path.isdir(simulation_name):
                self.app.logger.info(f"The simulation folder {simulation_name} does not exist. Creating now...")
                os.mkdir(simulation_name)

            # change the working folder and run the simulation
            os.chdir(os.getcwd() + "/" + simulation_name)
            logfile  = f"{simulation_name}_log.txt"
            self.app.set_logger(logfile)
            self.logger = self.app.logger

            # create the new log file for the simulation
            self.logger.info(f"Currently in: {os.getcwd()}")
            self.logger.info(f"Running: {simulation_name}, started at {self.time}")

            # save the simulation parameters as a json file
            setup_simulations.save_parameters_to_json(self.system.simulation_parameters)

            # Instantiate the correct model class based on model_type in config.
            # Defaults to "BEC" (zero-temperature GPE) when the key is absent.
            BECClass = _get_bec_class(parameters.get("model_type", "BEC"))
            self.BEC = BECClass(parameters, self.system, self.app, simulation_name)
            self.BEC.evolve()

            # go back to the parent directory to prepare to run the next sim
            path = Path(os.getcwd())
            parent_path = path.parent.absolute()
            os.chdir(parent_path)

            # free unused memory
            torch.cuda.empty_cache()
