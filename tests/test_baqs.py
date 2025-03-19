import unittest
import os
import sys
sys.path.append('.')
from unittest.mock import patch
from src.cli.baqs import main
from unittest.mock import MagicMock
from src.cli.functions import check_args

class TestBAQS(unittest.TestCase):

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
            "baqs.py",
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
            "baqs.py",
            "missing_config.json",
            "app.json",
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
                "baqs.py",
                config_file,
                "missing_app.json",
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
        app_file = "app.json"
        with open(config_file, "w") as f:
            f.write("{}")
        with open(app_file, "w") as f:
            f.write("print('App running')")

        try:
            # Mock command-line arguments with --run but without --check
            test_args = [
                "baqs.py",
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

class TestCheckArgs(unittest.TestCase):

    def setUp(self):
        # Create temporary files for testing
        self.config_file = "config.json"
        self.app_file = "app.json"
        with open(self.config_file, "w") as f:
            f.write("{}")
        with open(self.app_file, "w") as f:
            f.write("print('App running')")

    def tearDown(self):
        # Cleanup temporary files
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        if os.path.exists(self.app_file):
            os.remove(self.app_file)

    def test_check_args_missing_config(self):
        args = MagicMock()
        args.config = "missing_config.json"
        args.app = self.app_file
        args.check = True
        args.run = False
        args.verbose = 0

        with self.assertRaises(FileNotFoundError):
            check_args(args)

    def test_check_args_missing_app(self):
        args = MagicMock()
        args.config = self.config_file
        args.app = "missing_app.json"
        args.check = True
        args.run = False
        args.verbose = 0

        with self.assertRaises(FileNotFoundError):
            check_args(args)

    def test_check_args_run_without_check(self):
        args = MagicMock()
        args.config = self.config_file
        args.app = self.app_file
        args.check = False
        args.run = True
        args.verbose = 0

        with self.assertRaises(ValueError):
            check_args(args)

    def test_check_args_success(self):
        args = MagicMock()
        args.config = self.config_file
        args.app = self.app_file
        args.check = True
        args.run = False
        args.verbose = 1

        # Should not raise any exceptions
        check_args(args)

if __name__ == '__main__':
    unittest.main()
