import argparse
from src.cli.functions import validate_config, run_simulations, check_args
from src.application import application

def main():
    parser = argparse.ArgumentParser(description="GPU accelerated code for GPE systems.")
    parser.add_argument("config", type=str, help="The path to the configuration file for the simulations relative to the current directory.")
    parser.add_argument("app", type=str, help="The path to the application configuration file for the simulations relative to the current directory.")
    parser.add_argument("-c", "--check", action="store_true", help="Performs the validation of the configuration files.")
    parser.add_argument("--run", action="store_true", help="Runs the simulations. Requires checks to have passed.")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Set verbosity level (0: silent, 1: info, 2: debug).")

    args = parser.parse_args()

    # Validate arguments
    check_args(args)
    app = application(args.app)
    logger = app.logger

    # Perform configuration validation
    if args.check:
        logger.info("Validating configuration files...")
        validate_config(args.config, args.verbose, logger=logger)

    # Run simulations
    if args.run:
        logger.info("Running simulations...")
        run_simulations(args.config, args.app, logger=logger)


if __name__ == "__main__":
    main()