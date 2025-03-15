import argparse
import os
from functions import validate_config, run_simulations

def main():
    parser = argparse.ArgumentParser(description="GPU accelerated code for GPE systems.")
    parser.add_argument("config", type=str, help="The path to the configuration file for the simulations relative to the current directory.")
    parser.add_argument("app", type=str, help="The path to the application file for the simulations.")
    parser.add_argument("-c", "--check", action="store_true", help="Performs the validation of the configuration files.")
    parser.add_argument("--run", action="store_true", help="Runs the simulations. Requires checks to have passed.")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Set verbosity level (0: silent, 1: info, 2: debug).")

    args = parser.parse_args()

    # Validate arguments
    check_args(args)

    # Perform configuration validation
    if args.check:
        if args.verbose > 0:
            print("[INFO] Validating configuration files...")
        validate_config(args.verbose)

    # Run simulations
    if args.run:
        if args.verbose > 0:
            print("[INFO] Running simulations...")
        run_simulations()

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

if __name__ == "__main__":
    main()