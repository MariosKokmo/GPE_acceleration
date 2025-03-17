import argparse
import os
from src.cli.functions import validate_config, run_simulations, check_args

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
        validate_config(args.config, args.verbose)

    # Run simulations
    if args.run:
        if args.verbose > 0:
            print("[INFO] Running simulations...")
        run_simulations()


if __name__ == "__main__":
    main()