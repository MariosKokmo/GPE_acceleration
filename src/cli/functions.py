import os
import json
from src.utils.setup_simulations import _check_simulation_parameters, get_simulation_parameters
from src.run import main as run_code

def validate_config(config_path, verbose):
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
    # Check if the configuration file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' does not exist.")

    if verbose > 0:
        print(f"[INFO] Found configuration file: {config_path}")

    if verbose > 1:
        print(f"[DEBUG] Loaded configuration: {config_path}")

    params, msg = get_simulation_parameters(config_path)
    if verbose > 0:
        print(f"[INFO] {msg}")

    # Ensure params is a dictionary
    if not isinstance(params, dict):
        raise ValueError("Configuration parameters must be a dictionary.")

    # Validate the configuration using functions from setup_simulations.py
    try:
        ok, msg =_check_simulation_parameters(params)
    except ValueError as e:
        raise ValueError(f"Configuration validation error: {e}")

    if verbose > 0:
        if ok:
            print("[INFO] Configuration file validation passed.")
        else:
            print(msg)

def check_args(args):
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

    if args.verbose > 1:
        print("[DEBUG] Arguments validated successfully.")

def run_simulations():
    run_code()