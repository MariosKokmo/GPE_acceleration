import unittest
import os
import sys
sys.path.append('.')
from unittest.mock import patch
from src.cli.gpeu import main

class TestGPEU(unittest.TestCase):

    def test_main_with_valid_args(self):
        """
        Test the main function with valid command-line arguments.
        Checks only that the configuration file passes the checks
        """
        # Use existing configuration_file.json and appConfig.json
        config_file = "configuration_file.json"
        app_file = "appConfig.json"

        # Mock command-line arguments
        test_args = [
            "gpeu.py",
            config_file,
            app_file,
            "--check",
            "-v",
            "1"
        ]
        with patch.object(sys, 'argv', test_args):
            main()

    def test_main_missing_config(self):
        # Mock command-line arguments with a missing config file
        test_args = [
            "gpeu.py",
            "missing_config.json",
            "app.py",
            "--check"
        ]
        with patch.object(sys, 'argv', test_args), self.assertRaises(FileNotFoundError):
            main()

    def test_main_missing_app(self):
        # Create a temporary config file
        config_file = "config.json"
        with open(config_file, "w") as f:
            f.write("{}")

        try:
            # Mock command-line arguments with a missing app file
            test_args = [
                "gpeu.py",
                config_file,
                "missing_app.py",
                "--check"
            ]
            with patch.object(sys, 'argv', test_args), self.assertRaises(FileNotFoundError):
                main()
        finally:
            # Cleanup
            os.remove(config_file)

    def test_main_run_without_check(self):
        # Create temporary config and app files
        config_file = "config.json"
        app_file = "app.py"
        with open(config_file, "w") as f:
            f.write("{}")
        with open(app_file, "w") as f:
            f.write("print('App running')")

        try:
            # Mock command-line arguments with --run but without --check
            test_args = [
                "gpeu.py",
                config_file,
                app_file,
                "--run"
            ]
            with patch.object(sys, 'argv', test_args), self.assertRaises(ValueError):
                main()
        finally:
            # Cleanup
            os.remove(config_file)
            os.remove(app_file)

if __name__ == '__main__':
    unittest.main()
