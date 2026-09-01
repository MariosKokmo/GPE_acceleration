"""Filesystem locations used by the GUI.

The GUI lives in ``<project_root>/gui``. The simulation config files and the
runnable script live in the project root, so we derive everything from there.
"""
import os

GUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../gui
PROJECT_ROOT = os.path.dirname(GUI_DIR)

# Canonical config filenames the solver expects in its working directory.
CONFIG_FILENAME = "configuration_file.json"
APP_CONFIG_FILENAME = "appConfig.json"

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, CONFIG_FILENAME)
DEFAULT_APP_CONFIG_PATH = os.path.join(PROJECT_ROOT, APP_CONFIG_FILENAME)

# Script that the runner launches in "script" mode (relative to the working dir).
RUN_SCRIPT_REL = os.path.join("src", "run.py")
