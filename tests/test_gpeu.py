import unittest
import os
import sys
sys.path.append('.')
from unittest.mock import patch
from src.cli.gpeu import main

class TestGPEU(unittest.TestCase):

    def test_main_with_valid_args(self):
        with unittest.mock.patch('tempfile.TemporaryDirectory') as tmp_path:
            # Create temporary config and app files
            config_file = os.path.join(tmp_path, "config.json")
            app_file = os.path.join(tmp_path, "app.py")
            config_file.write_text("{}")
            app_file.write_text("print('App running')")

            try:
                # Mock command-line arguments
                test_args = [
                    "gpeu.py",
                    str(config_file),
                    str(app_file),
                    "--check",
                    "--run",
                    "-v",
                    "1"
                ]
                with patch.object(sys, 'argv', test_args):
                    main()
            finally:
                # Cleanup
                config_file.unlink(missing_ok=True)
                app_file.unlink(missing_ok=True)

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
        with unittest.mock.patch('tempfile.TemporaryDirectory') as tmp_path:
            # Create a temporary config file
            config_file = os.path.join(tmp_path, "config.json")
            config_file.write_text("{}")

            try:
                # Mock command-line arguments with a missing app file
                test_args = [
                    "gpeu.py",
                    str(config_file),
                    "missing_app.py",
                    "--check"
                ]
                with patch.object(sys, 'argv', test_args), self.assertRaises(FileNotFoundError):
                    main()
            finally:
                # Cleanup
                config_file.unlink(missing_ok=True)

    def test_main_run_without_check(self):
        with unittest.mock.patch('tempfile.TemporaryDirectory') as tmp_path:
            # Create temporary config and app files
            config_file = os.path.join(tmp_path, "config.json")
            app_file = os.path.join(tmp_path, "app.py")
            config_file.write_text("{}")
            app_file.write_text("print('App running')")

            try:
                # Mock command-line arguments with --run but without --check
                test_args = [
                    "gpeu.py",
                    str(config_file),
                    str(app_file),
                    "--run"
                ]
                with patch.object(sys, 'argv', test_args), self.assertRaises(ValueError):
                    main()
            finally:
                # Cleanup
                config_file.unlink(missing_ok=True)
                app_file.unlink(missing_ok=True)

if __name__ == '__main__':
    unittest.main()
