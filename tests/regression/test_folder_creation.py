"""
Regression tests: simulation folder creation and per-simulation parameters
===========================================================================

These tests load a fixed configuration (fixtures/regression_config.json),
run the full two-step setup pipeline, and assert that:

1. The correct number of simulation folders is created.
2. Every folder name matches the golden value derived from the config.
3. Each folder physically exists on disk after the run.
4. A simulation_parameters.json is written inside every folder.
5. The shared simulation parameters (grid, timing, model defaults) are correct.
6. Per-simulation parameters (charges, imprint times, flags) are correct.

Simulation folders are created directly inside tests/regression/ so they
are easy to navigate to after a run.

Cleanup behaviour
-----------------
By default the simulation folders are deleted after the test run.  Set the
environment variable REGRESSION_KEEP_OUTPUT=1 to keep them::

    $env:REGRESSION_KEEP_OUTPUT = "1"
    python -m unittest tests.regression.test_folder_creation

The output directory is always printed at the start of the run.
"""

import json
import os
import shutil
import sys
import unittest

sys.path.append(".")

from src.sims_setup.cartesian_setup import get_simulation_parameters_cartesian
from src.sims_setup.cylindrical_setup import get_simulation_parameters_cylindrical
from src.utils.setup_simulations import get_simulation_combinations, save_parameters_to_json

# Simulation folders land directly in tests/regression/.
_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set REGRESSION_KEEP_OUTPUT=1 to prevent teardown from deleting the created folders.
_KEEP_OUTPUT = os.environ.get("REGRESSION_KEEP_OUTPUT", "0").strip() == "1"

# Path to the fixture config, relative to _OUTPUT_DIR (used after chdir).
_CONFIG_NAME = os.path.join("fixtures", "regression_config.json")

# Golden folder names produced by the fixture config.
# Derived by tracing _simulations_repetitive → _format_name_component for each
# simulation group defined in regression_config.json.
_EXPECTED_FOLDER_SIM0 = "1vort__initCharge1__imprintCharge[1]__snapshots3"
_EXPECTED_FOLDER_SIM1 = "1vort__initCharge2__imprintCharge[1, 2]__snapshots4_5"


class TestFolderCreationAndParameters(unittest.TestCase):
    """Full-pipeline regression: folder names, JSON files, and per-simulation parameters."""

    @classmethod
    def setUpClass(cls):
        cls._orig_cwd = os.getcwd()
        print(f"\n[regression] output directory: {_OUTPUT_DIR}")

        # chdir so that _ensure_simulation_directory creates folders relative here.
        os.chdir(_OUTPUT_DIR)

        params, msg = get_simulation_parameters_cartesian(_CONFIG_NAME)
        if params is None:
            raise RuntimeError(f"Fixture config rejected by pipeline: {msg}")

        cls._params = params
        cls._simulations = get_simulation_combinations(params)

        # Write simulation_parameters.json into every created folder.
        for name, p in cls._simulations:
            save_parameters_to_json(
                p, filepath=os.path.join(name, "simulation_parameters.json")
            )

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._orig_cwd)
        if _KEEP_OUTPUT:
            print(f"\n[regression] keeping folders in: {_OUTPUT_DIR}")
            for name, _ in cls._simulations:
                print(f"  {os.path.join(_OUTPUT_DIR, name)}{os.sep}")
        else:
            for name, _ in cls._simulations:
                shutil.rmtree(os.path.join(_OUTPUT_DIR, name), ignore_errors=True)

    # ── folder creation ────────────────────────────────────────────────────────

    def test_two_simulation_entries_returned(self):
        """The fixture config defines two charge groups, so two entries are returned."""
        self.assertEqual(len(self._simulations), 2)

    def test_sim0_folder_name(self):
        """Simulation 0 folder name encodes charge=1, imprint charge [1], snapshot=3."""
        names = [s[0] for s in self._simulations]
        self.assertIn(_EXPECTED_FOLDER_SIM0, names)

    def test_sim1_folder_name(self):
        """Simulation 1 folder name encodes charge=2, imprint charges [1,2], snapshots=4_5."""
        names = [s[0] for s in self._simulations]
        self.assertIn(_EXPECTED_FOLDER_SIM1, names)

    def test_all_folders_exist_on_disk(self):
        """Every returned folder name must correspond to a directory on disk."""
        for name, _ in self._simulations:
            path = os.path.join(_OUTPUT_DIR, name)
            self.assertTrue(os.path.isdir(path), f"Missing folder on disk: {name}")

    # ── simulation_parameters.json ─────────────────────────────────────────────

    def test_parameters_json_exists_in_each_folder(self):
        """save_parameters_to_json must write a file in every simulation folder."""
        for name, _ in self._simulations:
            path = os.path.join(_OUTPUT_DIR, name, "simulation_parameters.json")
            self.assertTrue(os.path.isfile(path), f"Missing JSON in folder: {name}")

    def test_parameters_json_is_valid_and_complete(self):
        """The written JSON must be parseable and contain the expected keys."""
        required = {"vortex_charge", "imprint_times", "repetitive", "model_type", "temperature"}
        for name, _ in self._simulations:
            path = os.path.join(_OUTPUT_DIR, name, "simulation_parameters.json")
            with open(path) as fh:
                data = json.load(fh)
            self.assertIsInstance(data, dict, f"JSON root is not a dict in: {name}")
            for key in required:
                self.assertIn(key, data, f"Key {key!r} missing from JSON in: {name}")

    def test_parameters_json_matches_returned_params(self):
        """JSON content must round-trip back to the same values returned by the pipeline."""
        for name, p in self._simulations:
            path = os.path.join(_OUTPUT_DIR, name, "simulation_parameters.json")
            with open(path) as fh:
                data = json.load(fh)
            self.assertEqual(data["vortex_charge"], p["vortex_charge"])
            self.assertEqual(data["imprint_times"], p["imprint_times"])
            self.assertEqual(data["repetitive"], p["repetitive"])

    # ── shared simulation parameters ──────────────────────────────────────────

    def test_grid_resolution_preserved(self):
        """Grid_resolution from the config passes through unchanged."""
        self.assertEqual(self._params["Grid_resolution"], [32, 32, 32])

    def test_kmax_derived_correctly(self):
        """kmax = int(Total_simulation_time // dt); 99 due to IEEE 754 rounding of 0.001/1e-5."""
        expected = int(0.001 // 1e-5)
        self.assertEqual(self._params["kmax"], expected)

    def test_shots_matches_snapshots(self):
        """'shots' mirrors the 'snapshots' key from the config."""
        self.assertEqual(self._params["shots"], 5)

    def test_model_type_defaults_to_bec(self):
        """model_type defaults to 'BEC' when absent from the config."""
        self.assertEqual(self._params["model_type"], "BEC")

    def test_temperature_defaults_to_zero(self):
        """temperature defaults to 0.0 when absent from the config."""
        self.assertAlmostEqual(self._params["temperature"], 0.0)

    def test_absorber_disabled_by_default(self):
        """Absorber is disabled and its strength is 0.0 when not configured."""
        self.assertFalse(self._params["Absorber_enabled"])
        self.assertAlmostEqual(self._params["Absorber_strength"], 0.0)

    def test_three_body_loss_disabled_by_default(self):
        """k3 = 0 when 'three-body-losses' is absent from the config."""
        self.assertAlmostEqual(self._params["k3"], 0.0)

    # ── per-simulation parameters ─────────────────────────────────────────────

    def _sim_by_name(self, fragment):
        """Return the params dict for the simulation whose folder name contains *fragment*."""
        for name, p in self._simulations:
            if fragment in name:
                return p
        self.fail(f"No simulation with name fragment {fragment!r}")

    def test_sim0_vortex_charge(self):
        """Simulation 0 starts with a single vortex of charge +1."""
        p = self._sim_by_name("initCharge1__imprintCharge[1]")
        self.assertEqual(p["vortex_charge"], [1])

    def test_sim0_imprint_times(self):
        """Simulation 0 re-imprints at snapshot 3."""
        p = self._sim_by_name("initCharge1__imprintCharge[1]")
        self.assertEqual(p["imprint_times"], [3])

    def test_sim0_initial_position(self):
        """Simulation 0 places the vortex at x=-2, y=0."""
        p = self._sim_by_name("initCharge1__imprintCharge[1]")
        self.assertEqual(p["vortex_position_x"], [-2])
        self.assertEqual(p["vortex_position_y"], [0])

    def test_sim1_vortex_charge(self):
        """Simulation 1 starts with a single vortex of charge +2."""
        p = self._sim_by_name("initCharge2")
        self.assertEqual(p["vortex_charge"], [2])

    def test_sim1_imprint_times(self):
        """Simulation 1 re-imprints at snapshots 4 and 5."""
        p = self._sim_by_name("initCharge2")
        self.assertEqual(p["imprint_times"], [4, 5])

    def test_sim1_initial_position(self):
        """Simulation 1 places the vortex on-axis at x=0, y=0."""
        p = self._sim_by_name("initCharge2")
        self.assertEqual(p["vortex_position_x"], [0])
        self.assertEqual(p["vortex_position_y"], [0])

    def test_repetitive_flag_set_in_all_sims(self):
        """Every simulation's params dict carries repetitive=1."""
        for _, p in self._simulations:
            self.assertEqual(p["repetitive"], 1)

    def test_finite_temp_defaults_injected_in_all_sims(self):
        """model_type and temperature defaults are injected into every simulation."""
        for _, p in self._simulations:
            self.assertEqual(p["model_type"], "BEC")
            self.assertAlmostEqual(p["temperature"], 0.0)

    def test_vortex_excitation_flag_set_in_all_sims(self):
        """vortex_excitation is truthy in every simulation."""
        for _, p in self._simulations:
            self.assertTrue(p["vortex_excitation"])


_CYL_CONFIG_NAME = os.path.join("fixtures", "regression_config_cylindrical.json")

_EXPECTED_FOLDER_CYL_SIM0 = "1vort__initCharge3__imprintCharge[3]__snapshots2"
_EXPECTED_FOLDER_CYL_SIM1 = "1vort__initCharge4__imprintCharge[3, 4]__snapshots3_4"


class TestCylindricalFolderCreationAndParameters(unittest.TestCase):
    """Full-pipeline regression for cylindrical coordinates: folder names, JSON files, and parameters."""

    @classmethod
    def setUpClass(cls):
        cls._orig_cwd = os.getcwd()
        print(f"\n[regression/cylindrical] output directory: {_OUTPUT_DIR}")

        os.chdir(_OUTPUT_DIR)

        params, msg = get_simulation_parameters_cylindrical(_CYL_CONFIG_NAME)
        if params is None:
            raise RuntimeError(f"Cylindrical fixture config rejected by pipeline: {msg}")

        cls._params = params
        cls._simulations = get_simulation_combinations(params)

        for name, p in cls._simulations:
            save_parameters_to_json(
                p, filepath=os.path.join(name, "simulation_parameters.json")
            )

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._orig_cwd)
        if _KEEP_OUTPUT:
            print(f"\n[regression/cylindrical] keeping folders in: {_OUTPUT_DIR}")
            for name, _ in cls._simulations:
                print(f"  {os.path.join(_OUTPUT_DIR, name)}{os.sep}")
        else:
            for name, _ in cls._simulations:
                shutil.rmtree(os.path.join(_OUTPUT_DIR, name), ignore_errors=True)

    # ── folder creation ────────────────────────────────────────────────────────

    def test_two_simulation_entries_returned(self):
        """The fixture config defines two charge groups, so two entries are returned."""
        self.assertEqual(len(self._simulations), 2)

    def test_sim0_folder_name(self):
        """Simulation 0 folder name encodes charge=3, imprint charge [3], snapshot=2."""
        names = [s[0] for s in self._simulations]
        self.assertIn(_EXPECTED_FOLDER_CYL_SIM0, names)

    def test_sim1_folder_name(self):
        """Simulation 1 folder name encodes charge=4, imprint charges [3,4], snapshots=3_4."""
        names = [s[0] for s in self._simulations]
        self.assertIn(_EXPECTED_FOLDER_CYL_SIM1, names)

    def test_all_folders_exist_on_disk(self):
        """Every returned folder name must correspond to a directory on disk."""
        for name, _ in self._simulations:
            path = os.path.join(_OUTPUT_DIR, name)
            self.assertTrue(os.path.isdir(path), f"Missing folder on disk: {name}")

    # ── simulation_parameters.json ─────────────────────────────────────────────

    def test_parameters_json_exists_in_each_folder(self):
        """save_parameters_to_json must write a file in every simulation folder."""
        for name, _ in self._simulations:
            path = os.path.join(_OUTPUT_DIR, name, "simulation_parameters.json")
            self.assertTrue(os.path.isfile(path), f"Missing JSON in folder: {name}")

    def test_parameters_json_is_valid_and_complete(self):
        """The written JSON must be parseable and contain the expected keys."""
        required = {"vortex_charge", "imprint_times", "repetitive", "model_type", "temperature"}
        for name, _ in self._simulations:
            path = os.path.join(_OUTPUT_DIR, name, "simulation_parameters.json")
            with open(path) as fh:
                data = json.load(fh)
            self.assertIsInstance(data, dict, f"JSON root is not a dict in: {name}")
            for key in required:
                self.assertIn(key, data, f"Key {key!r} missing from JSON in: {name}")

    def test_parameters_json_matches_returned_params(self):
        """JSON content must round-trip back to the same values returned by the pipeline."""
        for name, p in self._simulations:
            path = os.path.join(_OUTPUT_DIR, name, "simulation_parameters.json")
            with open(path) as fh:
                data = json.load(fh)
            self.assertEqual(data["vortex_charge"], p["vortex_charge"])
            self.assertEqual(data["imprint_times"], p["imprint_times"])
            self.assertEqual(data["repetitive"], p["repetitive"])

    # ── cylindrical-specific parameters ───────────────────────────────────────

    def test_cylindrical_keys_present(self):
        """All cylindrical-specific keys must be present in the shared params dict."""
        for key in ("r_max", "z_min", "z_max", "dr", "dphi", "dz", "d_x"):
            self.assertIn(key, self._params, f"Missing cylindrical key: {key}")

    def test_r_max_is_positive(self):
        """Normalised r_max must be positive."""
        self.assertGreater(self._params["r_max"], 0.0)

    def test_z_extent_is_symmetric(self):
        """z_min and z_max must be negatives of each other (symmetric input config)."""
        self.assertAlmostEqual(self._params["z_min"], -self._params["z_max"], places=10)

    def test_grid_spacings_are_positive(self):
        """dr, dphi, and dz must all be positive."""
        self.assertGreater(self._params["dr"], 0.0)
        self.assertGreater(self._params["dphi"], 0.0)
        self.assertGreater(self._params["dz"], 0.0)

    def test_dphi_equals_2pi_over_n_phi(self):
        """dphi = 2π / n_phi."""
        import math
        n_phi = 8  # from fixture Grid_resolution [32, 8, 32]
        self.assertAlmostEqual(self._params["dphi"], 2 * math.pi / n_phi, places=12)

    def test_d_x_equals_dr_times_dphi_times_dz(self):
        """d_x = dr · dphi · dz."""
        expected = self._params["dr"] * self._params["dphi"] * self._params["dz"]
        self.assertAlmostEqual(self._params["d_x"], expected, delta=abs(expected) * 1e-10)

    def test_normalised_frequencies_have_two_elements(self):
        """w = [wr_norm, wz_norm] must have exactly two elements for a cylindrical trap."""
        self.assertEqual(len(self._params["w"]), 2)

    def test_normalised_frequencies_are_positive(self):
        """Both normalised trapping frequencies must be positive."""
        self.assertTrue(all(v > 0 for v in self._params["w"]))

    # ── shared simulation parameters ──────────────────────────────────────────

    def test_grid_resolution_preserved(self):
        """Grid_resolution from the config passes through unchanged."""
        self.assertEqual(self._params["Grid_resolution"], [32, 8, 32])

    def test_kmax_derived_correctly(self):
        """kmax = int(Total_simulation_time // dt)."""
        expected = int(0.001 // 1e-5)
        self.assertEqual(self._params["kmax"], expected)

    def test_shots_matches_snapshots(self):
        """'shots' mirrors the 'snapshots' key from the config."""
        self.assertEqual(self._params["shots"], 5)

    def test_model_type_defaults_to_bec(self):
        """model_type defaults to 'BEC' when absent from the config."""
        self.assertEqual(self._params["model_type"], "BEC")

    def test_absorber_disabled_by_default(self):
        """Absorber is disabled and its strength is 0.0 when not configured."""
        self.assertFalse(self._params["Absorber_enabled"])
        self.assertAlmostEqual(self._params["Absorber_strength"], 0.0)

    def test_three_body_loss_disabled_by_default(self):
        """k3 = 0 when 'three-body-losses' is absent from the config."""
        self.assertAlmostEqual(self._params["k3"], 0.0)

    # ── per-simulation parameters ─────────────────────────────────────────────

    def _sim_by_name(self, fragment):
        """Return the params dict for the simulation whose folder name contains *fragment*."""
        for name, p in self._simulations:
            if fragment in name:
                return p
        self.fail(f"No simulation with name fragment {fragment!r}")

    def test_sim0_vortex_charge(self):
        """Simulation 0 starts with a single vortex of charge +3."""
        p = self._sim_by_name("initCharge3__imprintCharge[3]")
        self.assertEqual(p["vortex_charge"], [3])

    def test_sim0_imprint_times(self):
        """Simulation 0 re-imprints at snapshot 2."""
        p = self._sim_by_name("initCharge3__imprintCharge[3]")
        self.assertEqual(p["imprint_times"], [2])

    def test_sim0_radial_position(self):
        """Simulation 0 places the vortex on-axis (r=0, phi=0)."""
        p = self._sim_by_name("initCharge3__imprintCharge[3]")
        self.assertEqual(p["vortex_position_x"], [0.0])
        self.assertEqual(p["vortex_position_y"], [0.0])

    def test_sim1_vortex_charge(self):
        """Simulation 1 starts with a single vortex of charge +4."""
        p = self._sim_by_name("initCharge4")
        self.assertEqual(p["vortex_charge"], [4])

    def test_sim1_imprint_times(self):
        """Simulation 1 re-imprints at snapshots 3 and 4."""
        p = self._sim_by_name("initCharge4")
        self.assertEqual(p["imprint_times"], [3, 4])

    def test_repetitive_flag_set_in_all_sims(self):
        """Every simulation's params dict carries repetitive=1."""
        for _, p in self._simulations:
            self.assertEqual(p["repetitive"], 1)

    def test_finite_temp_defaults_injected_in_all_sims(self):
        """model_type and temperature defaults are injected into every simulation."""
        for _, p in self._simulations:
            self.assertEqual(p["model_type"], "BEC")
            self.assertAlmostEqual(p["temperature"], 0.0)

    def test_vortex_excitation_flag_set_in_all_sims(self):
        """vortex_excitation is truthy in every simulation."""
        for _, p in self._simulations:
            self.assertTrue(p["vortex_excitation"])


if __name__ == "__main__":
    unittest.main()
