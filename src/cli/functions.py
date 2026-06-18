import os
import logging
from src.utils.setup_simulations import _load_json_from_cwd, _perform_frequency_checks
from src.sims_setup.cartesian_setup import get_simulation_parameters_cartesian
from src.sims_setup.cylindrical_setup import get_simulation_parameters_cylindrical
from src.run import main as run_code

def _set_logger_verbosity(logger, verbose):
    """Set console/file handler levels according to CLI verbosity."""
    if logger is None:
        return

    if verbose <= 0:
        target_level = logging.WARNING
    elif verbose == 1:
        target_level = logging.INFO
    else:
        target_level = logging.DEBUG

    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(target_level)


def validate_config(config_path, verbose, logger=None):
    """
    Validates the configuration file for the simulation.

    Parameters
    ----------
    verbose : int
        Verbosity level (0: silent, 1: info, 2: debug).
    config_path : str
        Path to the configuration file.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration file is invalid or missing required fields.
    """
    _set_logger_verbosity(logger, verbose)

    # Check if the configuration file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' does not exist.")

    if logger is not None:
        logger.info(f"Found configuration file: {config_path}")
        logger.debug(f"Loaded configuration path: {config_path}")

    # Peek at the raw config to pick the coordinate system (explicit
    # "coordinates" key, else auto-detect from the presence of "r_max").
    raw = _load_json_from_cwd(config_path)
    if not isinstance(raw, dict):
        raise ValueError("Configuration parameters must be a dictionary.")
    coords = str(raw.get("coordinates", "")).strip().lower()
    if coords not in ("cartesian", "cylindrical"):
        coords = "cylindrical" if "r_max" in raw else "cartesian"

    # Frequencies must be positive *before* the loader derives omega_ho / a_ho
    # (a non-positive product yields a complex omega_ho and crashes the sqrt).
    # Checking the raw values first gives a clean message instead of a TypeError.
    freqs = raw.get("Trapping_frequencies")
    if isinstance(freqs, list):
        freq_ok, freq_msg = _perform_frequency_checks({"Trapping_frequencies": freqs})
        if not freq_ok:
            if logger is not None:
                logger.error(freq_msg)
            return False, freq_msg

    # Run the full coordinate-specific loader. It derives the simulation
    # parameters and runs every validation check (grid, frequency, vortex,
    # re-imprinting), returning (params, msg); params is None on a fatal error.
    try:
        if coords == "cylindrical":
            params, msg = get_simulation_parameters_cylindrical(config_path)
        else:
            params, msg = get_simulation_parameters_cartesian(config_path)
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f"Configuration validation error: {e}")

    ok = params is not None and not msg

    if logger is not None:
        if ok:
            logger.info("Configuration file validation passed.")
        else:
            logger.error(msg or "Configuration is missing required parameters.")

    return ok, msg


def check_args(args, logger=None):
    """
    Validates the command-line arguments.
    """
    # Check if config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file '{args.config}' does not exist in the specified path.")

    # Check if app file exists
    if not os.path.exists(args.app):
        raise FileNotFoundError(f"Application file '{args.app}' does not exist in the specified path.")

    # Ensure --run is only executed if --check is passed
    if args.run and not args.check:
        raise ValueError("The '--run' flag requires '--check' to be executed first.")

    if logger is not None:
        logger.debug("Arguments validated successfully.")


def run_simulations(config_path=None, app_config_path=None, logger=None):
    if logger is not None:
        logger.info("Launching simulation run.")
    run_code(config_path, app_config_path)