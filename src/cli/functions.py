import os
import logging
from src.utils.setup_simulations import _check_simulation_parameters, get_simulation_parameters
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

    if logger is not None:
        logger.debug(f"Loaded configuration path: {config_path}")

    params, msg = get_simulation_parameters(config_path)
    if logger is not None and msg:
        logger.info(msg)

    # Ensure params is a dictionary
    if not isinstance(params, dict):
        raise ValueError("Configuration parameters must be a dictionary.")

    # Validate the configuration using functions from setup_simulations.py
    try:
        ok, msg = _check_simulation_parameters(params)
    except ValueError as e:
        raise ValueError(f"Configuration validation error: {e}")

    if logger is not None:
        if ok:
            logger.info("Configuration file validation passed.")
        else:
            logger.error(msg)

    return ok, msg


def check_args(args, logger=None):
    """
    Validates the command-line arguments.
    """
    # Check if config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file '{args.config}' does not exist.")

    # Check if app file exists
    if not os.path.exists(args.app):
        raise FileNotFoundError(f"Application file '{args.app}' does not exist.")

    # Ensure --run is only executed if --check is passed
    if args.run and not args.check:
        raise ValueError("The '--run' flag requires '--check' to be executed first.")

    if logger is not None:
        logger.debug("Arguments validated successfully.")


def run_simulations(config_path=None, app_config_path=None, logger=None):
    if logger is not None:
        logger.info("Launching simulation run.")
    run_code(config_path, app_config_path)