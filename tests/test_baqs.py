"""Tests for the ``baqs`` command line entry point and its argument checks.

``main`` parses ``sys.argv``, validates the two file arguments through
``check_args`` and then either checks or runs the configuration. These tests
cover the validation contract only — that missing files and an unchecked
``--run`` are refused — plus one end-to-end ``--check`` against the repository's
own configuration.

The scratch files are created in a temporary directory rather than the working
directory, so a run cannot overwrite a real ``config.json`` or ``app.json``.
Their *contents* are irrelevant to every test but the first: validation refuses
the arguments before anything tries to parse them.
"""
import unittest
import os
import sys
import tempfile
sys.path.append('.')
from unittest.mock import patch
from src.cli.baqs import main
from unittest.mock import MagicMock
from src.cli.functions import check_args


class ScratchFilesCase(unittest.TestCase):
    """Provides ``self.config_file`` and ``self.app_file`` in a temp directory.

    Both exist and are readable; nothing here depends on what is inside them.
    The directory and its contents are removed after each test.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="baqs_cli_")
        self.config_file = os.path.join(self._tmp.name, "config.json")
        self.app_file = os.path.join(self._tmp.name, "app.json")
        with open(self.config_file, "w") as f:
            f.write("{}")
        with open(self.app_file, "w") as f:
            f.write("{}")
        self.missing = os.path.join(self._tmp.name, "does_not_exist.json")

    def tearDown(self):
        self._tmp.cleanup()


class TestBAQS(ScratchFilesCase):

    def test_main_with_valid_args(self):
        """
        Test the main function with valid command-line arguments.
        Checks only that the configuration file passes the checks

        This one does use the repository's real ``configuration_file.json`` and
        ``appConfig.json``, so it doubles as a check that the committed
        configuration is still accepted by the validator. It therefore has to
        run from the repository root.
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
        """A config path that does not exist is refused before any parsing."""
        test_args = [
            "baqs.py",
            self.missing,
            self.app_file,
            "--check"
        ]
        with patch.object(sys, 'argv', test_args), self.assertRaises(FileNotFoundError):
            main()

    def test_main_missing_app(self):
        """An app-config path that does not exist is refused the same way."""
        test_args = [
            "baqs.py",
            self.config_file,
            self.missing,
            "--check"
        ]
        with patch.object(sys, 'argv', test_args), self.assertRaises(FileNotFoundError):
            main()

    def test_main_run_without_check(self):
        """``--run`` without ``--check`` is refused: a run always validates first."""
        test_args = [
            "baqs.py",
            self.config_file,
            self.app_file,
            "--run"
        ]
        with patch.object(sys, 'argv', test_args), self.assertRaises(ValueError):
            main()


class TestCheckArgs(ScratchFilesCase):
    """The same contract, exercised directly on ``check_args``.

    ``main`` reaches these paths through argparse; here the parsed namespace is
    faked, so a failure points at the validator rather than at the parsing.
    """

    def test_check_args_missing_config(self):
        """A missing config file raises FileNotFoundError."""
        args = MagicMock()
        args.config = self.missing
        args.app = self.app_file
        args.check = True
        args.run = False
        args.verbose = 0

        with self.assertRaises(FileNotFoundError):
            check_args(args)

    def test_check_args_missing_app(self):
        """A missing app-config file is rejected too."""
        args = MagicMock()
        args.config = self.config_file
        args.app = self.missing
        args.check = True
        args.run = False
        args.verbose = 0

        with self.assertRaises(FileNotFoundError):
            check_args(args)

    def test_check_args_run_without_check(self):
        """Asking to run without checking raises ValueError."""
        args = MagicMock()
        args.config = self.config_file
        args.app = self.app_file
        args.check = False
        args.run = True
        args.verbose = 0

        with self.assertRaises(ValueError):
            check_args(args)

    def test_check_args_success(self):
        """Two existing files with ``--check`` pass validation.

        There is nothing to assert beyond the absence of an exception:
        ``check_args`` validates and returns nothing.
        """
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
