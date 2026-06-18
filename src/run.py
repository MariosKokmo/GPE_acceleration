import os
import sys

# Allow running this file directly (python src/run.py) by exposing project root.
if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from src.simulator import Simulator


def main(configFile="configuration_file.json", appConfigFile="appConfig.json"):
    sim = Simulator(config_file=configFile, app_config_file=appConfigFile)
    sim.run()


if __name__ == "__main__":
    main()