"""
Tests for src.utils.setup_simulations
======================================

Covers every public function and all critical private helpers in both the
Cartesian and cylindrical coordinate sections of the module.

Structure
---------
Section 1 – Shared helpers (_require_keys, _load_json_from_cwd, get_application_config,
             _format_name_component, _ensure_simulation_directory, save_parameters_to_json)
Section 2 – Cartesian validators (_perform_grid_checks, _perform_frequency_checks,
             _perform_vortex_checks, _perform_reimprint_checks, _check_simulation_parameters)
Section 3 – Cartesian simulation builders (_simulations_repetitive, _simulations_multi_vortex,
             get_simulation_combinations)
Section 4 – Cartesian end-to-end (get_simulation_parameters)
Section 5 – Cylindrical validators (_perform_grid_checks_cylindrical,
             _perform_vortex_checks_cylindrical, _check_simulation_parameters_cylindrical)
Section 6 – Cylindrical end-to-end (get_simulation_parameters_cylindrical)

Note on broken legacy tests
----------------------------
The original test file patched ``_read_configuration_file`` which was renamed to
``_load_json_from_cwd``.  All such tests have been rewritten here to patch the
correct target.
"""

import json
import math
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.append(".")

from src.utils.setup_simulations import (
    # helpers
    _require_keys,
    _load_json_from_cwd,
    get_application_config,
    _format_name_component,
    _ensure_simulation_directory,
    save_parameters_to_json,
    # shared validators and builders
    _perform_frequency_checks,
    _perform_reimprint_checks,
    _simulations_repetitive,
    _simulations_multi_vortex,
    get_simulation_combinations,
)
from src.sims_setup.cartesian_setup import (
    _perform_grid_checks,
    _perform_vortex_checks,
    _check_simulation_parameters,
    get_simulation_parameters_cartesian,
)
from src.sims_setup.cylindrical_setup import (
    _perform_grid_checks_cylindrical,
    _perform_vortex_checks_cylindrical,
    _check_simulation_parameters_cylindrical,
    get_simulation_parameters_cylindrical,
)

# ---------------------------------------------------------------------------
# Shared minimal configs used across multiple test classes
# ---------------------------------------------------------------------------

# Minimal on-disk JSON that satisfies every key read by get_simulation_parameters.
# vortex_excitation=False so no folder-creation side effects occur.
_CART_CONFIG = {
    "Grid_resolution": [32, 32, 32],
    "Grid_negative_limits": [-10, -10, -10],
    "Grid_positive_limits": [10, 10, 10],
    "Trapping_frequencies": [100.0, 100.0, 10.0],
    "Total_simulation_time": 0.001,
    "dt": 1e-5,
    "snapshots": 5,
    "Potential_type": "harmonic",
    "SwitchOff_time": None,
    "vortex_excitation": False,
    "vortex_charge": [[1]],
    "imprinting_charge": [[1]],
    "vortex_position_x": [[0]],
    "vortex_position_y": [[0]],
    "imprint_position_x": [[0]],
    "imprint_position_y": [[0]],
    "initial_imprint_time": [0],
    "repetitive": 0,
    "imprint_every": [],
    "imprint_times": [],
    "max_imprints": [],
}

# A dict shaped like what _check_simulation_parameters expects (post-derivation).
_CART_PARAMS = {
    "x_min": [-6.3, -6.3, -6.3],
    "x_max": [6.3, 6.3, 6.3],
    "Grid_resolution": [32, 32, 32],
    "Trapping_frequencies": [100.0, 100.0, 10.0],
    "vortex_charge": [[1]],
    "vortex_position_x": [[0]],
    "vortex_position_y": [[0]],
    "imprinting_charge": [[1]],
    "repetitive": 0,
    "Total_simulation_time": 1.0,
    "shots": 100,
    "imprint_times": [],
    "imprint_position_x": [[0]],
    "imprint_position_y": [[0]],
}

# Minimal cylindrical on-disk JSON.
_CYL_CONFIG = {
    "Grid_resolution": [32, 8, 32],
    "r_max": 10.0,
    "z_min": -10.0,
    "z_max": 10.0,
    "Trapping_frequencies": [100.0, 10.0],
    "Total_simulation_time": 0.001,
    "dt": 1e-5,
    "snapshots": 5,
    "Potential_type": "harmonic",
    "SwitchOff_time": None,
    "vortex_excitation": False,
    "vortex_charge": [[1]],
    "imprinting_charge": [[1]],
    "vortex_position_x": [[0]],
    "vortex_position_y": [[0]],
    "imprint_position_x": [[0]],
    "imprint_position_y": [[0]],
    "initial_imprint_time": [0],
    "repetitive": 0,
    "imprint_every": [],
    "imprint_times": [],
    "max_imprints": [],
}

# Dict shaped like what _check_simulation_parameters_cylindrical expects.
_CYL_PARAMS = {
    "r_max": 5.0,
    "z_min": -5.0,
    "z_max": 5.0,
    "Grid_resolution": [32, 8, 32],
    "Trapping_frequencies": [100.0, 10.0],
    "vortex_charge": [[1]],
    "vortex_position_x": [[0]],
    "vortex_position_y": [[0]],
    "imprinting_charge": [[1]],
    "repetitive": 0,
    "Total_simulation_time": 1.0,
    "shots": 100,
    "imprint_times": [],
    "imprint_position_x": [[0]],
    "imprint_position_y": [[0]],
}


def _write_temp_config(data):
    """
    Write *data* as JSON to a NamedTemporaryFile inside the cwd.

    Returns the basename (not the full path) so callers can pass it directly
    to ``_load_json_from_cwd`` / ``get_simulation_parameters``.

    The caller is responsible for deleting the file.
    """
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=os.getcwd()
    )
    json.dump(data, f)
    f.close()
    return os.path.basename(f.name)


# =============================================================================
# Section 1 – Shared helpers
# =============================================================================


class TestRequireKeys(unittest.TestCase):
    """Unit tests for the _require_keys guard."""

    def test_all_keys_present_no_exception(self):
        """No exception when every required key is present."""
        _require_keys({"a": 1, "b": 2}, ["a", "b"], "test context")

    def test_single_missing_key_raises(self):
        """ValueError names the missing key."""
        with self.assertRaises(ValueError) as ctx:
            _require_keys({"a": 1}, ["a", "b"], "ctx")
        self.assertIn("b", str(ctx.exception))

    def test_multiple_missing_keys_all_named(self):
        """ValueError names every missing key when several are absent."""
        with self.assertRaises(ValueError) as ctx:
            _require_keys({}, ["x", "y", "z"], "ctx")
        msg = str(ctx.exception)
        for key in ("x", "y", "z"):
            self.assertIn(key, msg)

    def test_extra_keys_are_ignored(self):
        """Extra keys in the data dict do not trigger an error."""
        _require_keys({"a": 1, "b": 2, "extra": 99}, ["a", "b"], "ctx")

    def test_empty_required_list_never_raises(self):
        """Empty required-keys list always passes."""
        _require_keys({}, [], "ctx")


class TestLoadJsonFromCwd(unittest.TestCase):
    """Unit tests for _load_json_from_cwd (reads relative to cwd)."""

    def setUp(self):
        """Create a temp JSON file in the current directory."""
        # Initialise to None first so tearDown is always safe even if
        # _write_temp_config raises before the assignment completes.
        self._tmp = None
        self._tmp = _write_temp_config({"key": "value", "num": 42})

    def tearDown(self):
        """Delete the temp JSON file created in setUp."""
        if self._tmp and os.path.exists(self._tmp):
            os.unlink(self._tmp)

    def test_valid_file_returns_dict(self):
        """A well-formed JSON file is returned as a dict."""
        data = _load_json_from_cwd(self._tmp)
        self.assertIsInstance(data, dict)
        self.assertEqual(data["key"], "value")
        self.assertEqual(data["num"], 42)

    def test_nonexistent_file_raises_file_not_found(self):
        """Missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            _load_json_from_cwd("__nonexistent_config_xyz__.json")

    def test_invalid_json_raises_value_error(self):
        """Malformed JSON raises ValueError."""
        bad = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=os.getcwd()
        )
        bad.write("{not valid json}")
        bad.close()
        try:
            with self.assertRaises(ValueError):
                _load_json_from_cwd(os.path.basename(bad.name))
        finally:
            os.unlink(bad.name)


class TestGetApplicationConfig(unittest.TestCase):
    """Unit tests for get_application_config."""

    def setUp(self):
        """Create a temp JSON file in the current directory."""
        self._tmp = None
        self._tmp = _write_temp_config({"app": "test", "version": 1})

    def tearDown(self):
        """Delete the temp JSON file created in setUp."""
        if self._tmp and os.path.exists(self._tmp):
            os.unlink(self._tmp)

    def test_returns_dict_from_file(self):
        """get_application_config reads and returns the JSON as a dict."""
        cfg = get_application_config(self._tmp)
        self.assertEqual(cfg["app"], "test")

    def test_missing_file_raises(self):
        """Missing config file propagates FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            get_application_config("__missing_app_config__.json")


class TestFormatNameComponent(unittest.TestCase):
    """Unit tests for the simulation-name formatter _format_name_component."""

    def test_scalar_returns_string(self):
        """A non-list value is converted to its string representation."""
        self.assertEqual(_format_name_component(42), "42")
        self.assertEqual(_format_name_component(3.14), "3.14")
        self.assertEqual(_format_name_component("hello"), "hello")

    def test_short_list_joined_by_underscores(self):
        """A list of ≤ 10 items is joined with underscores."""
        self.assertEqual(_format_name_component([1, 2, 3]), "1_2_3")
        self.assertEqual(_format_name_component([-1, 0, 1]), "-1_0_1")

    def test_long_list_shortened(self):
        """A list with > 10 items is shortened to _init_first_last_last."""
        result = _format_name_component(list(range(12)))
        # Should contain first (0) and last (11)
        self.assertIn("0", result)
        self.assertIn("11", result)
        self.assertIn("_init_", result)
        self.assertIn("_last_", result)

    def test_list_with_spaces_filtered(self):
        """Space strings are removed before formatting."""
        result = _format_name_component([1, " ", 2])
        self.assertNotIn(" ", result)
        self.assertIn("1", result)
        self.assertIn("2", result)

    def test_single_element_list(self):
        """A single-element list is returned as just that element string."""
        self.assertEqual(_format_name_component([7]), "7")


class TestEnsureSimulationDirectory(unittest.TestCase):
    """Unit tests for _ensure_simulation_directory."""

    def setUp(self):
        self._dir = f"__test_sim_dir_{os.getpid()}__"

    def tearDown(self):
        if os.path.isdir(self._dir):
            os.rmdir(self._dir)

    def test_creates_missing_directory(self):
        """Directory is created when it does not exist."""
        self.assertFalse(os.path.isdir(self._dir))
        _ensure_simulation_directory(self._dir)
        self.assertTrue(os.path.isdir(self._dir))

    def test_does_not_raise_if_already_exists(self):
        """No exception when the directory already exists."""
        os.mkdir(self._dir)
        _ensure_simulation_directory(self._dir)   # should not raise
        self.assertTrue(os.path.isdir(self._dir))


class TestSaveParametersToJson(unittest.TestCase):
    """Unit tests for save_parameters_to_json."""

    def setUp(self):
        self._file = f"__test_params_{os.getpid()}.json"

    def tearDown(self):
        if os.path.exists(self._file):
            os.remove(self._file)

    def test_saves_plain_dict(self):
        """Plain dict is written and can be read back identically."""
        params = {"Grid_resolution": [64, 64, 64], "dt": 0.01, "snapshots": 100}
        save_parameters_to_json(params, filepath=self._file)
        self.assertTrue(os.path.exists(self._file))
        with open(self._file) as f:
            loaded = json.load(f)
        self.assertEqual(params, loaded)

    def test_numpy_arrays_serialised_as_lists(self):
        """numpy arrays are transparently converted to JSON lists."""
        params = {"arr": np.array([1, 2, 3]), "val": 1.0}
        save_parameters_to_json(params, filepath=self._file)
        with open(self._file) as f:
            loaded = json.load(f)
        self.assertEqual(loaded["arr"], [1, 2, 3])

    def test_unsupported_type_raises_type_error(self):
        """A Python set (not JSON-serialisable) raises TypeError."""
        with self.assertRaises(TypeError):
            save_parameters_to_json({"bad": {1, 2, 3}}, filepath=self._file)


# =============================================================================
# Section 2 – Cartesian validators
# =============================================================================


class TestPerformGridChecks(unittest.TestCase):
    """Tests for _perform_grid_checks (Cartesian grid validation)."""

    def _params(self, **overrides):
        """Return a valid base param dict, optionally overriding keys."""
        p = {
            "x_min": [-10.0, -10.0, -10.0],
            "x_max": [10.0, 10.0, 10.0],
            "Grid_resolution": [64, 64, 64],
        }
        p.update(overrides)
        return p

    def test_valid_symmetric_grid_passes(self):
        """Symmetric, negative-x_min grid with positive resolution passes."""
        ok, msg = _perform_grid_checks(self._params())
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_positive_x_min_fails(self):
        """A positive x_min value is rejected (grid must be centred at 0)."""
        ok, msg = _perform_grid_checks(self._params(x_min=[5.0, -10.0, -10.0]))
        self.assertFalse(ok)
        self.assertIn("x_min for axis 1 is not negative", msg)

    def test_asymmetric_grid_fails(self):
        """Unequal |x_min| and |x_max| on any axis is rejected."""
        ok, msg = _perform_grid_checks(self._params(x_max=[10.0, 10.0, 5.0]))
        self.assertFalse(ok)
        self.assertIn("3 max and min are not symmetric", msg)

    def test_zero_resolution_fails(self):
        """A grid point count of zero is rejected."""
        ok, msg = _perform_grid_checks(
            self._params(Grid_resolution=[64, 0, 64])
        )
        self.assertFalse(ok)
        self.assertIn("grid resolution must be greater than zero", msg)

    def test_non_dict_raises_type_error(self):
        """Passing a non-dict raises TypeError."""
        with self.assertRaises(TypeError):
            _perform_grid_checks([(-10,), (10,), (64,)])


class TestPerformFrequencyChecks(unittest.TestCase):
    """Tests for _perform_frequency_checks (works for both Cartesian and cylindrical)."""

    def test_all_positive_frequencies_pass(self):
        """Three positive frequencies pass without an error message."""
        ok, msg = _perform_frequency_checks(
            {"Trapping_frequencies": [100.0, 100.0, 10.0]}
        )
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_two_positive_frequencies_pass(self):
        """Two positive frequencies (cylindrical case) also pass."""
        ok, msg = _perform_frequency_checks(
            {"Trapping_frequencies": [100.0, 10.0]}
        )
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_zero_frequency_fails(self):
        """A frequency of zero is rejected."""
        ok, msg = _perform_frequency_checks(
            {"Trapping_frequencies": [100.0, 0.0, 10.0]}
        )
        self.assertFalse(ok)
        self.assertIn("negative or zero", msg)

    def test_negative_frequency_fails(self):
        """A negative frequency is rejected."""
        ok, msg = _perform_frequency_checks(
            {"Trapping_frequencies": [-50.0, 100.0, 10.0]}
        )
        self.assertFalse(ok)
        self.assertIn("negative or zero", msg)

    def test_error_reports_correct_axis(self):
        """The error message identifies which frequency index is bad."""
        ok, msg = _perform_frequency_checks(
            {"Trapping_frequencies": [100.0, 100.0, -5.0]}
        )
        self.assertFalse(ok)
        self.assertIn("3", msg)  # third frequency


class TestPerformVortexChecks(unittest.TestCase):
    """Tests for _perform_vortex_checks (Cartesian vortex-parameter validation)."""

    def _params(self, **overrides):
        """Return a valid base vortex param dict."""
        p = {
            "Grid_resolution": [128, 128, 128],
            "vortex_charge": [[1, -1]],
            "vortex_position_x": [[10, -10]],
            "vortex_position_y": [[15, -15]],
            "repetitive": False,
            "imprinting_charge": [],
        }
        p.update(overrides)
        return p

    def test_valid_vortex_parameters_pass(self):
        """Well-formed vortex configuration passes all checks."""
        ok, msg = _perform_vortex_checks(self._params())
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_charge_x_count_mismatch_top_level(self):
        """Mismatch between the number of simulation groups in charges vs x positions."""
        ok, msg = _perform_vortex_checks(
            self._params(
                vortex_charge=[[1, -1], [2]],
                vortex_position_x=[[10, -10]],  # only one group
                vortex_position_y=[[15, -15]],
            )
        )
        self.assertFalse(ok)
        self.assertIn("list number of vortex charges", msg)

    def test_x_y_count_mismatch_top_level(self):
        """Mismatch between the number of x and y position groups."""
        ok, msg = _perform_vortex_checks(
            self._params(
                vortex_position_y=[[15, -15], [0]],  # extra group
            )
        )
        self.assertFalse(ok)
        self.assertIn("list number of x positions", msg)

    def test_charge_x_count_mismatch_per_sim(self):
        """Within a simulation, charge count must match x-position count."""
        ok, msg = _perform_vortex_checks(
            self._params(
                vortex_charge=[[1, -1]],
                vortex_position_x=[[10, -10, 30]],  # 3 positions for 2 charges
                vortex_position_y=[[15, -15]],
            )
        )
        self.assertFalse(ok)
        self.assertIn("number of charges doesn't agree with the number of x positions", msg)

    def test_position_exceeds_grid_x(self):
        """Radial position beyond n1 // 2 is rejected."""
        ok, msg = _perform_vortex_checks(
            self._params(
                vortex_position_x=[[80, -10]],  # 80 > 128//2=64
                vortex_position_y=[[15, -15]],
            )
        )
        self.assertFalse(ok)
        self.assertIn("greater than half the grid size", msg)

    def test_negative_position_below_grid_y(self):
        """Negative position below −n3 // 2 is rejected."""
        ok, _ = _perform_vortex_checks(
            self._params(
                vortex_position_y=[[15, -70]],  # -70 < -64
            )
        )
        self.assertFalse(ok)

    def test_repetitive_imprinting_charge_mismatch(self):
        """In repetitive mode, initial charge count must match imprinting charge count."""
        ok, msg = _perform_vortex_checks(
            self._params(
                repetitive=True,
                imprinting_charge=[[1, -1, 2], [1, 4]],  # 2 groups vs 1 group of charges
            )
        )
        self.assertFalse(ok)
        self.assertIn("number of initial vortex charges", msg)

    def test_multiple_simulations_all_valid(self):
        """Multiple simulation groups all within bounds pass."""
        ok, _ = _perform_vortex_checks(
            self._params(
                vortex_charge=[[1, -1], [2, -2]],
                vortex_position_x=[[10, -10], [20, -20]],
                vortex_position_y=[[15, -15], [25, -25]],
            )
        )
        self.assertTrue(ok)


class TestPerformReimprintChecks(unittest.TestCase):
    """Tests for _perform_reimprint_checks (re-imprinting parameter validation)."""

    def _params(self, **overrides):
        """Return a valid base reimprint param dict."""
        p = {
            "Total_simulation_time": 10.0,
            "shots": 100,
            "repetitive": True,
            "imprint_times": [[10, 20, 30], [40, 50, 60]],
            "imprinting_charge": [[1, -1], [2, -2]],
            "imprint_position_x": [[10, -10], [20, -20]],
            "imprint_position_y": [[15, -15], [25, -25]],
        }
        p.update(overrides)
        return p

    def test_valid_reimprint_configuration_passes(self):
        """Consistent imprint_times / charges / positions pass."""
        ok, msg = _perform_reimprint_checks(self._params())
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_non_repetitive_skips_all_checks(self):
        """When repetitive=False, all reimprint checks are skipped."""
        ok, msg = _perform_reimprint_checks(
            self._params(repetitive=False, imprint_times=[], imprinting_charge=[])
        )
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_imprint_times_charge_count_mismatch(self):
        """One imprint-times list per simulation is required."""
        ok, msg = _perform_reimprint_checks(
            self._params(imprint_times=[[10, 20, 30]])  # 1 list vs 2 sims
        )
        self.assertFalse(ok)
        self.assertIn("One list of imprinting times should be given for every simulation", msg)

    def test_imprint_time_exceeds_snapshot_count(self):
        """An imprint time larger than the snapshot count is rejected."""
        ok, msg = _perform_reimprint_checks(
            self._params(imprint_times=[[10, 20, 110], [40, 50, 60]])  # 110 > 100 shots
        )
        self.assertFalse(ok)
        self.assertIn("maximum imprint time is greater than the total simulation time", msg)

    def test_imprint_charge_x_position_mismatch(self):
        """Number of imprinting charges must match number of x positions."""
        ok, msg = _perform_reimprint_checks(
            self._params(imprint_position_x=[[10], [20, -20]])  # sim0: 1 pos vs 2 charges
        )
        self.assertFalse(ok)
        self.assertIn("doesn't agree with the number of x positions", msg)

    def test_imprint_charge_y_position_mismatch(self):
        """Number of imprinting charges must match number of y positions."""
        ok, msg = _perform_reimprint_checks(
            self._params(imprint_position_y=[[15], [25, -25]])  # sim0: 1 pos vs 2 charges
        )
        self.assertFalse(ok)
        self.assertIn("doesn't agree with the number of y positions", msg)


class TestCheckSimulationParameters(unittest.TestCase):
    """Tests for _check_simulation_parameters (Cartesian orchestrator)."""

    def test_all_valid_returns_true_empty_msg(self):
        """All sub-checks passing returns (True, '')."""
        ok, msg = _check_simulation_parameters(_CART_PARAMS)
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_bad_grid_returns_false(self):
        """A bad x_min propagates as failure with a non-empty message."""
        bad = dict(_CART_PARAMS, x_min=[5.0, -6.3, -6.3])  # positive x_min
        ok, msg = _check_simulation_parameters(bad)
        self.assertFalse(ok)
        self.assertNotEqual(msg, "")

    def test_bad_frequency_returns_false(self):
        """A zero frequency propagates as failure."""
        bad = dict(_CART_PARAMS, Trapping_frequencies=[0.0, 100.0, 10.0])
        ok, msg = _check_simulation_parameters(bad)
        self.assertFalse(ok)
        self.assertIn("negative or zero", msg)

    def test_messages_concatenated_on_multiple_failures(self):
        """When multiple checks fail, all messages are concatenated."""
        bad = dict(
            _CART_PARAMS,
            x_min=[5.0, -6.3, -6.3],                       # grid failure
            Trapping_frequencies=[0.0, 100.0, 10.0],        # frequency failure
        )
        ok, msg = _check_simulation_parameters(bad)
        self.assertFalse(ok)
        # Both errors should appear in the combined message
        self.assertIn("x_min", msg)
        self.assertIn("negative or zero", msg)


# =============================================================================
# Section 3 – Cartesian simulation builders
# =============================================================================


class TestSimulationsRepetitive(unittest.TestCase):
    """Tests for _simulations_repetitive (folder naming for repetitive imprinting)."""

    def _run(self, params_list):
        """Invoke _simulations_repetitive with filesystem operations mocked out."""
        with patch("src.utils.setup_simulations.os.path.isdir", return_value=False), \
             patch("src.utils.setup_simulations.os.mkdir"):
            return _simulations_repetitive(params_list)

    def _base_params(self, **overrides):
        p = {
            "vortex_charge": [1, -1],
            "imprinting_charge": [1, -1],
            "max_imprints": 10,
            "imprint_every": 10,
            "imprint_times": [10, 20],
        }
        p.update(overrides)
        return p

    def test_returns_list_of_name_params_pairs(self):
        """Each entry is a two-element list [simulation_name, params_dict]."""
        sims = self._run([self._base_params()])
        self.assertEqual(len(sims), 1)
        self.assertIsInstance(sims[0][0], str)
        self.assertIsInstance(sims[0][1], dict)

    def test_simulation_name_contains_charge_and_imprint(self):
        """Simulation name encodes charge, imprinting-charge, and snapshot info."""
        sims = self._run([self._base_params()])
        name = sims[0][0]
        self.assertIn("initCharge", name)
        self.assertIn("imprintCharge", name)
        self.assertIn("snapshots", name)

    def test_number_of_vortices_in_name(self):
        """The name starts with the count of vortices."""
        sims = self._run([self._base_params(vortex_charge=[1, -1, 2])])
        self.assertTrue(sims[0][0].startswith("3vort"))

    def test_non_list_charge_counted_as_one(self):
        """A scalar charge is treated as one vortex."""
        sims = self._run([self._base_params(vortex_charge=1)])
        self.assertTrue(sims[0][0].startswith("1vort"))

    def test_long_charge_list_shortened(self):
        """A charge list > 10 items is abbreviated with _init_…_last_… notation."""
        sims = self._run([self._base_params(vortex_charge=list(range(12)))])
        name = sims[0][0]
        self.assertIn("_init_", name)
        self.assertIn("_last_", name)

    def test_multiple_simulations_returned(self):
        """Multiple parameter dicts produce multiple simulation entries."""
        sims = self._run([self._base_params(), self._base_params(vortex_charge=[2])])
        self.assertEqual(len(sims), 2)

    def test_parameters_preserved_in_output(self):
        """The original params dict is stored as the second element of each pair."""
        p = self._base_params(max_imprints=999)
        sims = self._run([p])
        self.assertEqual(sims[0][1]["max_imprints"], 999)


class TestSimulationsMultiVortex(unittest.TestCase):
    """Tests for _simulations_multi_vortex (folder naming for multi-vortex scenarios)."""

    def _run(self, params_list):
        with patch("src.utils.setup_simulations.os.path.isdir", return_value=False), \
             patch("src.utils.setup_simulations.os.mkdir"):
            return _simulations_multi_vortex(params_list)

    def _base_params(self, **overrides):
        p = {
            "vortex_charge": [1, -1],
            "vortex_position_x": [5, -5],
            "vortex_position_y": [3, -3],
        }
        p.update(overrides)
        return p

    def test_returns_name_params_pairs(self):
        """Each entry is [simulation_name, params_dict]."""
        sims = self._run([self._base_params()])
        self.assertEqual(len(sims), 1)
        self.assertIsInstance(sims[0][0], str)

    def test_name_contains_vortex_count_and_positions(self):
        """Name encodes vortex count, charges, and x/y positions."""
        sims = self._run([self._base_params()])
        name = sims[0][0]
        self.assertIn("vortex_charges", name)
        self.assertIn("x-", name)
        self.assertIn("y-", name)

    def test_vortex_count_in_name(self):
        """The name starts with the number of vortices."""
        sims = self._run([self._base_params()])
        self.assertTrue(sims[0][0].startswith("2vortex"))

    def test_multiple_simulations(self):
        """Two distinct parameter sets produce two entries."""
        sims = self._run([
            self._base_params(vortex_charge=[1]),
            self._base_params(vortex_charge=[1, -1, 2]),
        ])
        self.assertEqual(len(sims), 2)


class TestGetSimulationCombinations(unittest.TestCase):
    """Integration tests for get_simulation_combinations (repetitive and non-repetitive)."""

    def _run(self, config):
        """Run get_simulation_combinations with os side-effects mocked."""
        with patch("src.utils.setup_simulations.os.path.isdir", return_value=False), \
             patch("src.utils.setup_simulations.os.mkdir"):
            return get_simulation_combinations(config)

    def _base_sims(self, repetitive=1):
        return {
            "vortex_excitation": True,
            "vortex_charge": [[1], [2]],
            "imprinting_charge": [[[1]], [[2]]],
            "vortex_position_x": [[0], [0]],
            "vortex_position_y": [[0], [0]],
            "imprint_position_x": [[[0]], [[0]]],
            "imprint_position_y": [[[0]], [[0]]],
            "imprint_times": [[10], [20]],
            "imprint_every": [10, 20],
            "max_imprints": [1, 1],
            "initial_imprint_time": [0, 0],
            "repetitive": repetitive,
        }

    def test_repetitive_returns_correct_count(self):
        """Repetitive mode creates one entry per simulation group."""
        sims = self._run(self._base_sims(repetitive=1))
        self.assertEqual(len(sims), 2)

    def test_non_repetitive_returns_correct_count(self):
        """Non-repetitive mode also creates one entry per charge group."""
        sims = self._run(self._base_sims(repetitive=0))
        self.assertEqual(len(sims), 2)

    def test_missing_required_key_raises(self):
        """Missing any REQUIRED_COMBINATION_KEY raises ValueError."""
        bad = self._base_sims()
        del bad["vortex_charge"]
        with self.assertRaises(ValueError):
            self._run(bad)

    def test_invalid_repetitive_value_raises(self):
        """A repetitive value other than 0 or 1 raises ValueError."""
        bad = self._base_sims(repetitive=2)
        with self.assertRaises(ValueError):
            self._run(bad)

    def test_finite_temp_defaults_passed_through(self):
        """Finite-temperature defaults appear in every simulation's params dict."""
        sims = self._run(self._base_sims(repetitive=0))
        for _, params in sims:
            self.assertIn("model_type", params)
            self.assertEqual(params["model_type"], "BEC")
            self.assertIn("temperature", params)


class TestDarkSolitonPerSimulation(unittest.TestCase):
    """get_simulation_combinations with per-simulation dark solitons."""

    def _run(self, config):
        with patch("src.utils.setup_simulations.os.path.isdir", return_value=False), \
             patch("src.utils.setup_simulations.os.mkdir"):
            return get_simulation_combinations(config)

    def _soliton_only_config(self, **overrides):
        # Two simulations: sim0 has one soliton at 0, sim1 has a pair at +-2.
        cfg = {
            "vortex_excitation": False,
            "dark_soliton": True,
            "soliton_positions": [[0.0], [-2.0, 2.0]],
            "soliton_widths": [[1.0], [1.0, 1.0]],
            "soliton_axes": [[3], [3, 3]],
            "soliton_greyness": [[0.0], [0.0, 0.0]],
            "soliton_imprint_time": [2, 5],
        }
        cfg.update(overrides)
        return cfg

    def test_soliton_only_builds_one_sim_per_entry(self):
        """Dark-soliton-only yields one simulation per soliton-list entry."""
        sims = self._run(self._soliton_only_config())
        self.assertEqual(len(sims), 2)

    def test_each_sim_gets_its_own_soliton_config(self):
        """Each simulation carries only its own (sliced) soliton parameters."""
        sims = self._run(self._soliton_only_config())
        (_, p0), (_, p1) = sims
        self.assertEqual(p0["soliton_positions"], [0.0])
        self.assertEqual(p0["soliton_axes"], [3])
        self.assertEqual(p0["soliton_imprint_time"], 2)
        self.assertEqual(p1["soliton_positions"], [-2.0, 2.0])
        self.assertEqual(p1["soliton_axes"], [3, 3])
        self.assertEqual(p1["soliton_imprint_time"], 5)
        for p in (p0, p1):
            self.assertEqual(p["vortex_excitation"], 0)
            self.assertTrue(p["dark_soliton"])
            self.assertEqual(p["model_type"], "BEC")  # finite-temp default flows through

    def test_no_excitation_returns_empty(self):
        """Neither vortices nor solitons -> no simulations."""
        self.assertEqual(self._run({"vortex_excitation": False}), [])

    def test_missing_soliton_keys_raises(self):
        """Enabling dark_soliton without the required lists raises ValueError."""
        cfg = self._soliton_only_config()
        del cfg["soliton_positions"]
        with self.assertRaises(ValueError):
            self._run(cfg)

    def test_mismatched_per_sim_lengths_raise(self):
        """Soliton lists must have one entry per simulation."""
        cfg = self._soliton_only_config(soliton_widths=[[1.0]])  # 1 vs 2 sims
        with self.assertRaises(ValueError):
            self._run(cfg)

    def test_vortex_plus_soliton_slices_per_sim(self):
        """With both on, each vortex simulation gets its own soliton slice."""
        cfg = {
            "vortex_excitation": True,
            "vortex_charge": [[1], [2]],
            "imprinting_charge": [[[1]], [[2]]],
            "vortex_position_x": [[0], [0]],
            "vortex_position_y": [[0], [0]],
            "imprint_position_x": [[[0]], [[0]]],
            "imprint_position_y": [[[0]], [[0]]],
            "imprint_times": [[10], [20]],
            "imprint_every": [10, 20],
            "max_imprints": [1, 1],
            "initial_imprint_time": [0, 0],
            "repetitive": 0,
            "dark_soliton": True,
            "soliton_positions": [[0.0], [-1.0, 1.0]],
            "soliton_widths": [[1.0], [1.0, 1.0]],
            "soliton_axes": [[3], [3, 3]],
        }
        sims = self._run(cfg)
        self.assertEqual(len(sims), 2)
        self.assertEqual(sims[0][1]["soliton_positions"], [0.0])
        self.assertEqual(sims[1][1]["soliton_positions"], [-1.0, 1.0])

    def test_vortex_plus_soliton_length_mismatch_raises(self):
        """Soliton lists must match the number of vortex simulations."""
        cfg = {
            "vortex_excitation": True,
            "vortex_charge": [[1], [2]],
            "imprinting_charge": [[[1]], [[2]]],
            "vortex_position_x": [[0], [0]],
            "vortex_position_y": [[0], [0]],
            "imprint_position_x": [[[0]], [[0]]],
            "imprint_position_y": [[[0]], [[0]]],
            "imprint_times": [[10], [20]],
            "imprint_every": [10, 20],
            "max_imprints": [1, 1],
            "initial_imprint_time": [0, 0],
            "repetitive": 0,
            "dark_soliton": True,
            "soliton_positions": [[0.0]],  # 1 entry for 2 sims
            "soliton_widths": [[1.0]],
            "soliton_axes": [[3]],
        }
        with self.assertRaises(ValueError):
            self._run(cfg)


# =============================================================================
# Section 4 – Cartesian end-to-end
# =============================================================================


class TestGetSimulationParameters(unittest.TestCase):
    """
    End-to-end tests for get_simulation_parameters.

    File I/O is bypassed by patching ``_load_json_from_cwd``
    (the function that actually opens the config file).
    """

    def _call(self, config):
        """Patch the JSON reader and invoke get_simulation_parameters_cartesian."""
        with patch(
            "src.sims_setup.cartesian_setup._load_json_from_cwd",
            return_value=config,
        ):
            return get_simulation_parameters_cartesian("dummy.json")

    def setUp(self):
        """Valid minimal Cartesian config used as the base for most tests."""
        self.cfg = {k: v for k, v in _CART_CONFIG.items()}

    # ── happy-path ─────────────────────────────────────────────────────────

    def test_valid_config_returns_params_dict(self):
        """A valid config returns a non-None params dict and an empty message."""
        params, msg = self._call(self.cfg)
        self.assertIsNotNone(params)
        self.assertEqual(msg, "")

    def test_derived_keys_present(self):
        """Derived quantities (omega_ho, a_ho, u, dx, dp, d_x) are in the output."""
        params, _ = self._call(self.cfg)
        for key in ("omega_ho", "a_ho", "u", "dx", "dp", "d_x", "dtau", "kmax"):
            self.assertIn(key, params, msg=f"Missing derived key: {key}")

    def test_grid_resolution_preserved(self):
        """Grid_resolution from config appears unchanged in the output."""
        params, _ = self._call(self.cfg)
        self.assertEqual(params["Grid_resolution"], [32, 32, 32])

    def test_normalised_x_min_is_negative(self):
        """After normalisation, x_min values must still be negative."""
        params, _ = self._call(self.cfg)
        self.assertTrue(all(v < 0 for v in params["x_min"]))

    def test_normalised_frequencies_positive(self):
        """Normalised trap frequencies w = ωi/ω_ho must be positive."""
        params, _ = self._call(self.cfg)
        self.assertTrue(all(w > 0 for w in params["w"]))

    def test_omega_ho_is_geometric_mean(self):
        """omega_ho = (ωx ωy ωz)^(1/3)."""
        params, _ = self._call(self.cfg)
        fx, fy, fz = self.cfg["Trapping_frequencies"]
        wx, wy, wz = 2 * math.pi * fx, 2 * math.pi * fy, 2 * math.pi * fz
        expected = (wx * wy * wz) ** (1 / 3)
        self.assertAlmostEqual(params["omega_ho"], expected, places=8)

    def test_kmax_computed_from_time_and_dt(self):
        """kmax = floor(Total_simulation_time / dt)."""
        params, _ = self._call(self.cfg)
        expected = int(self.cfg["Total_simulation_time"] // self.cfg["dt"])
        self.assertEqual(params["kmax"], expected)

    def test_three_body_losses_disabled_by_default(self):
        """k3 = 0 when the 'three-body-losses' key is absent."""
        params, _ = self._call(self.cfg)
        self.assertEqual(params["k3"], 0.0)

    def test_three_body_losses_enabled(self):
        """k3 is set to CONSTANTS.k3 when three-body-losses is True."""
        from src.library.parameters import CONSTANTS
        cfg = dict(self.cfg, **{"three-body-losses": True})
        params, _ = self._call(cfg)
        self.assertEqual(params["k3"], CONSTANTS.k3)

    def test_finite_temp_defaults(self):
        """model_type defaults to 'BEC' and temperature to 0.0."""
        params, _ = self._call(self.cfg)
        self.assertEqual(params["model_type"], "BEC")
        self.assertEqual(params["temperature"], 0.0)

    def test_absorber_defaults_to_disabled(self):
        """Absorber is disabled and strength is 0 when not specified."""
        params, _ = self._call(self.cfg)
        self.assertFalse(params["Absorber_enabled"])
        self.assertEqual(params["Absorber_strength"], 0.0)

    def test_chemical_potential_none_by_default(self):
        """chemical_potential is None when absent from config."""
        params, _ = self._call(self.cfg)
        self.assertIsNone(params["chemical_potential"])

    def test_chemical_potential_parsed_when_provided(self):
        """chemical_potential is parsed as a float when present."""
        cfg = dict(self.cfg, chemical_potential=3.5)
        params, _ = self._call(cfg)
        self.assertAlmostEqual(params["chemical_potential"], 3.5)

    # ── failure cases ───────────────────────────────────────────────────────

    def test_missing_required_key_returns_none(self):
        """A missing REQUIRED_SIMULATION_CONFIG_KEY returns (None, '[FATAL]...')."""
        cfg = {k: v for k, v in self.cfg.items() if k != "Grid_resolution"}
        params, msg = self._call(cfg)
        self.assertIsNone(params)
        self.assertIn("[FATAL]", msg)

    def test_invalid_grid_resolution_format_returns_none(self):
        """Only two values in Grid_resolution returns (None, '[FATAL]...')."""
        cfg = dict(self.cfg, Grid_resolution=[64, 64])
        params, msg = self._call(cfg)
        self.assertIsNone(params)
        self.assertIn("[FATAL]", msg)

    def test_invalid_trapping_frequencies_returns_none(self):
        """Only two values in Trapping_frequencies returns (None, '[FATAL]...')."""
        cfg = dict(self.cfg, Trapping_frequencies=[100, 10])
        params, msg = self._call(cfg)
        self.assertIsNone(params)
        self.assertIn("[FATAL]", msg)

    def test_repetitive_length_mismatch_returns_none(self):
        """imprint_every and imprint_times of different lengths returns (None, ...)."""
        cfg = dict(
            self.cfg,
            repetitive=1,
            imprint_every=[10],
            imprint_times=[[], []],  # 2 items vs 1
            max_imprints=[1],
            imprinting_charge=[[1]],
        )
        params, msg = self._call(cfg)
        self.assertIsNone(params)
        self.assertIn("imprint_every and imprint_times have different number", msg)

    def test_imprint_times_auto_generated_when_empty(self):
        """Empty imprint_times lists are auto-filled from imprint_every and max_imprints.

        With ``imprint_every = 5`` and ``max_imprints = 3`` the generated
        snapshots are 5, 10 and 15. The result is asserted unconditionally: an
        earlier version guarded the check with ``if params is not None``, so a
        config that started being rejected would have turned this into a test
        that silently checked nothing.
        """
        cfg = dict(
            self.cfg,
            repetitive=1,
            imprint_every=[5],
            imprint_times=[[]],   # empty → should be generated
            max_imprints=[3],
            imprinting_charge=[[1]],
        )
        params, msg = self._call(cfg)
        self.assertIsNotNone(params, msg=f"config was rejected: {msg}")
        self.assertEqual(params["imprint_times"][0], [5, 10, 15])


# =============================================================================
# Section 5 – Cylindrical validators
# =============================================================================


class TestPerformGridChecksCylindrical(unittest.TestCase):
    """Tests for _perform_grid_checks_cylindrical."""

    def _params(self, **overrides):
        p = {
            "r_max": 10.0,
            "z_min": -10.0,
            "z_max": 10.0,
            "Grid_resolution": [32, 8, 32],
        }
        p.update(overrides)
        return p

    def test_valid_cylindrical_grid_passes(self):
        """A well-formed cylindrical grid passes all checks."""
        ok, msg = _perform_grid_checks_cylindrical(self._params())
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_negative_r_max_fails(self):
        """r_max ≤ 0 is rejected."""
        ok, msg = _perform_grid_checks_cylindrical(self._params(r_max=-1.0))
        self.assertFalse(ok)
        self.assertIn("r_max", msg)

    def test_zero_r_max_fails(self):
        """r_max = 0 is rejected."""
        ok, _ = _perform_grid_checks_cylindrical(self._params(r_max=0.0))
        self.assertFalse(ok)

    def test_positive_z_min_fails(self):
        """z_min ≥ 0 is rejected (trap must be centred at z = 0)."""
        ok, msg = _perform_grid_checks_cylindrical(self._params(z_min=5.0))
        self.assertFalse(ok)
        self.assertIn("z_min", msg)

    def test_negative_z_max_fails(self):
        """z_max ≤ 0 is rejected."""
        ok, msg = _perform_grid_checks_cylindrical(self._params(z_max=-5.0))
        self.assertFalse(ok)
        self.assertIn("z_max", msg)

    def test_z_min_greater_than_z_max_fails(self):
        """z_min ≥ z_max is rejected regardless of individual signs."""
        ok, _ = _perform_grid_checks_cylindrical(
            self._params(z_min=-1.0, z_max=-5.0)
        )
        self.assertFalse(ok)

    def test_zero_grid_resolution_fails(self):
        """A grid point count of zero is rejected."""
        ok, _ = _perform_grid_checks_cylindrical(
            self._params(Grid_resolution=[32, 0, 32])
        )
        self.assertFalse(ok)

    def test_non_dict_raises_type_error(self):
        """Non-dict input raises TypeError."""
        with self.assertRaises(TypeError):
            _perform_grid_checks_cylindrical([1, 2, 3])


class TestPerformVortexChecksCylindrical(unittest.TestCase):
    """Tests for _perform_vortex_checks_cylindrical."""

    def _params(self, **overrides):
        p = {
            "Grid_resolution": [32, 8, 32],
            "r_max": 5.0,                   # radial boundary in dimensionless units
            "vortex_charge": [[1]],
            "vortex_position_x": [[0]],     # radial position r ≥ 0
            "vortex_position_y": [[0]],     # azimuthal angle φ ∈ [0, 2π]
            "imprinting_charge": [[1]],
            "repetitive": 0,
        }
        p.update(overrides)
        return p

    def test_on_axis_vortex_passes(self):
        """A vortex at r = 0 (on-axis) passes all checks."""
        ok, msg = _perform_vortex_checks_cylindrical(self._params())
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_negative_radial_position_fails(self):
        """r < 0 is invalid in cylindrical coordinates."""
        ok, msg = _perform_vortex_checks_cylindrical(
            self._params(vortex_position_x=[[-1]])
        )
        self.assertFalse(ok)
        self.assertIn("≥ 0", msg)

    def test_radial_position_exceeds_r_max_fails(self):
        """Radial position > r_max is rejected."""
        ok, msg = _perform_vortex_checks_cylindrical(
            self._params(vortex_position_x=[[999]])
        )
        self.assertFalse(ok)
        self.assertIn("r_max", msg)

    def test_angular_position_exceeds_2pi_fails(self):
        """Angular position > 2π is rejected."""
        ok, msg = _perform_vortex_checks_cylindrical(
            self._params(vortex_position_y=[[999]])
        )
        self.assertFalse(ok)
        self.assertIn("2π", msg)

    def test_angular_position_below_zero_fails(self):
        """Angular position φ < 0 is rejected."""
        ok, _ = _perform_vortex_checks_cylindrical(
            self._params(vortex_position_y=[[-1]])
        )
        self.assertFalse(ok)

    def test_charge_radial_count_mismatch_fails(self):
        """Number of charges must match number of radial positions."""
        ok, _ = _perform_vortex_checks_cylindrical(
            self._params(vortex_charge=[[1, -1]], vortex_position_x=[[0]])  # 2 vs 1
        )
        self.assertFalse(ok)

    def test_repetitive_charge_mismatch_fails(self):
        """In repetitive mode, initial and imprinting charge group counts must match."""
        ok, _ = _perform_vortex_checks_cylindrical(
            self._params(
                repetitive=1,
                vortex_charge=[[1]],
                imprinting_charge=[[1], [2]],   # 2 groups vs 1
            )
        )
        self.assertFalse(ok)

    def test_multiple_valid_simulations(self):
        """Multiple simulation groups all within bounds pass."""
        ok, _ = _perform_vortex_checks_cylindrical(
            self._params(
                Grid_resolution=[64, 8, 64],
                r_max=10.0,
                vortex_charge=[[1], [1, -1]],
                vortex_position_x=[[0], [0, 4.5]],
                vortex_position_y=[[0], [0, 1.0]],
                imprinting_charge=[[1], [1]],
            )
        )
        self.assertTrue(ok)


class TestCheckSimulationParametersCylindrical(unittest.TestCase):
    """Tests for _check_simulation_parameters_cylindrical (cylindrical orchestrator)."""

    def test_all_valid_returns_true_empty_msg(self):
        """All cylindrical sub-checks passing returns (True, '')."""
        ok, msg = _check_simulation_parameters_cylindrical(_CYL_PARAMS)
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_bad_r_max_returns_false(self):
        """Negative r_max propagates as a failure."""
        ok, msg = _check_simulation_parameters_cylindrical(
            dict(_CYL_PARAMS, r_max=-1.0)
        )
        self.assertFalse(ok)
        self.assertIn("r_max", msg)

    def test_bad_frequency_returns_false(self):
        """Zero frequency propagates as a failure."""
        ok, msg = _check_simulation_parameters_cylindrical(
            dict(_CYL_PARAMS, Trapping_frequencies=[0.0, 10.0])
        )
        self.assertFalse(ok)
        self.assertIn("negative or zero", msg)

    def test_negative_radial_vortex_position_fails(self):
        """Negative radial vortex position propagates as a failure."""
        ok, _ = _check_simulation_parameters_cylindrical(
            dict(_CYL_PARAMS, vortex_position_x=[[-1]])
        )
        self.assertFalse(ok)


# =============================================================================
# Section 6 – Cylindrical end-to-end
# =============================================================================


class TestGetSimulationParametersCylindrical(unittest.TestCase):
    """
    End-to-end tests for get_simulation_parameters_cylindrical.

    File I/O is bypassed by patching ``_load_json_from_cwd``.
    """

    def _call(self, config):
        with patch(
            "src.sims_setup.cylindrical_setup._load_json_from_cwd",
            return_value=config,
        ):
            return get_simulation_parameters_cylindrical("dummy.json")

    def setUp(self):
        self.cfg = {k: v for k, v in _CYL_CONFIG.items()}

    # ── happy-path ─────────────────────────────────────────────────────────

    def test_valid_2freq_config_returns_params(self):
        """[fr, fz] trapping frequencies produce a valid params dict."""
        params, msg = self._call(self.cfg)
        self.assertIsNotNone(params)
        self.assertEqual(msg, "")

    def test_valid_3freq_config_accepted(self):
        """[fr, fr, fz] (3-element compatibility form) also produces a valid params dict."""
        cfg = dict(self.cfg, Trapping_frequencies=[100.0, 100.0, 10.0])
        params, msg = self._call(cfg)
        self.assertIsNotNone(params)
        self.assertEqual(msg, "")

    def test_cylindrical_specific_keys_present(self):
        """Cylindrical-specific keys (r_max, z_min, z_max, dr, dphi, dz) are in output."""
        params, _ = self._call(self.cfg)
        for key in ("r_max", "z_min", "z_max", "dr", "dphi", "dz", "d_x"):
            self.assertIn(key, params, msg=f"Missing cylindrical key: {key}")

    def test_normalised_r_max_positive(self):
        """Normalised r_max must be > 0."""
        params, _ = self._call(self.cfg)
        self.assertGreater(params["r_max"], 0.0)

    def test_normalised_z_extent_symmetric(self):
        """z_min and z_max should be negatives of each other (symmetric input)."""
        params, _ = self._call(self.cfg)
        self.assertAlmostEqual(params["z_min"], -params["z_max"], places=10)

    def test_w_has_two_elements(self):
        """Normalised frequency list w = [wr_norm, wz_norm] has exactly two elements."""
        params, _ = self._call(self.cfg)
        self.assertEqual(len(params["w"]), 2)

    def test_w_elements_positive(self):
        """Both normalised frequencies must be positive."""
        params, _ = self._call(self.cfg)
        self.assertTrue(all(v > 0 for v in params["w"]))

    def test_omega_ho_is_geometric_mean_of_wr_wr_wz(self):
        """omega_ho = (wr² · wz)^(1/3) for an axially symmetric trap."""
        params, _ = self._call(self.cfg)
        fr, fz = self.cfg["Trapping_frequencies"]
        wr = 2 * math.pi * fr
        wz = 2 * math.pi * fz
        expected = (wr * wr * wz) ** (1 / 3)
        self.assertAlmostEqual(params["omega_ho"], expected, places=8)

    def test_dr_dphi_dz_positive(self):
        """All three cylindrical grid spacings are positive."""
        params, _ = self._call(self.cfg)
        self.assertGreater(params["dr"], 0.0)
        self.assertGreater(params["dphi"], 0.0)
        self.assertGreater(params["dz"], 0.0)

    def test_dphi_equals_2pi_over_n_phi(self):
        """dphi = 2π / n_phi."""
        params, _ = self._call(self.cfg)
        n_phi = self.cfg["Grid_resolution"][1]
        self.assertAlmostEqual(params["dphi"], 2 * math.pi / n_phi, places=12)

    def test_d_x_equals_dr_dphi_dz(self):
        """d_x = dr · dphi · dz (non-r-weighted part of the volume element)."""
        params, _ = self._call(self.cfg)
        expected = params["dr"] * params["dphi"] * params["dz"]
        self.assertAlmostEqual(params["d_x"], expected, delta=abs(expected) * 1e-10)

    def test_kmax_from_time_and_dt(self):
        """kmax = floor(Total_simulation_time / dt)."""
        params, _ = self._call(self.cfg)
        expected = int(self.cfg["Total_simulation_time"] // self.cfg["dt"])
        self.assertEqual(params["kmax"], expected)

    def test_finite_temp_defaults_present(self):
        """Finite-temperature fields default correctly when absent from config."""
        params, _ = self._call(self.cfg)
        self.assertEqual(params["model_type"], "BEC")
        self.assertEqual(params["temperature"], 0.0)
        self.assertIsNone(params["chemical_potential"])

    def test_absorber_defaults_to_disabled(self):
        """Absorber parameters default to disabled."""
        params, _ = self._call(self.cfg)
        self.assertFalse(params["Absorber_enabled"])
        self.assertEqual(params["Absorber_strength"], 0.0)

    # ── failure cases ───────────────────────────────────────────────────────

    def test_missing_r_max_returns_none(self):
        """Missing 'r_max' returns (None, '[FATAL]...')."""
        cfg = {k: v for k, v in self.cfg.items() if k != "r_max"}
        params, msg = self._call(cfg)
        self.assertIsNone(params)
        self.assertIn("[FATAL]", msg)

    def test_invalid_frequency_count_returns_none(self):
        """A single trapping frequency returns (None, '[FATAL]...')."""
        cfg = dict(self.cfg, Trapping_frequencies=[100.0])
        params, msg = self._call(cfg)
        self.assertIsNone(params)
        self.assertIn("[FATAL]", msg)

    def test_invalid_grid_resolution_returns_none(self):
        """Only two values in Grid_resolution returns (None, '[FATAL]...')."""
        cfg = dict(self.cfg, Grid_resolution=[32, 32])
        params, msg = self._call(cfg)
        self.assertIsNone(params)
        self.assertIn("[FATAL]", msg)

    def test_repetitive_mismatch_returns_none(self):
        """Mismatched imprint_every / imprint_times returns (None, ...)."""
        cfg = dict(
            self.cfg,
            vortex_excitation=True,
            repetitive=1,
            imprint_every=[5],
            imprint_times=[[], []],  # 2 lists vs 1 imprint_every entry
            max_imprints=[1],
            imprinting_charge=[[1]],
        )
        params, _ = self._call(cfg)
        self.assertIsNone(params)

    def test_bad_r_max_validation_gives_non_empty_msg(self):
        """Negative r_max triggers the cylindrical grid validator and returns a msg."""
        cfg = dict(self.cfg, r_max=-5.0)
        _, msg = self._call(cfg)
        # params may still be returned (validator returns False, not None)
        # but msg must be non-empty
        self.assertNotEqual(msg, "")

    def test_3freq_compat_matches_2freq(self):
        """[fr, fr, fz] and [fr, fz] produce identical omega_ho."""
        params2, _ = self._call(self.cfg)
        cfg3 = dict(self.cfg, Trapping_frequencies=[100.0, 100.0, 10.0])
        params3, _ = self._call(cfg3)
        self.assertAlmostEqual(params2["omega_ho"], params3["omega_ho"], places=8)
        self.assertAlmostEqual(params2["a_ho"], params3["a_ho"], places=12)


if __name__ == "__main__":
    unittest.main()
