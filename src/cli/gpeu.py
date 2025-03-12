import argparse
from functions import validate_config, run_simulations

def main():
    parser = argparse.ArgumentParser(description="GPU accelerated code for GPE systems.")
    parser.add_argument("config", type=str, help="The path to the configuration file for the simulations relative to the current directory")
    parser.add_argument("app", type=str, help="")
    parser.add_argument("-v", "--verbose", action="store_true", help="Search recursively in subdirectories")
    parser.add_argument("-c", "--check", default=False, help="Performs the validation of the configuration files.")
    parser.add_argument("--run", help="Runs the simulations. Requires checks to have passed.")

    args = parser.parse_args()
    
    if not args.config:
        parser.error("configuration file was not specified")
    if not args.app:
        parser.error("app file was not specified")
    
    check_args(args)
    if args.check:
        validate_config(args.verbose)
    

    run_simulations()


def check_args(*args):
    pass

if __name__ == "__main__":
    main()