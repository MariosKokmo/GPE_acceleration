import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import unittest
import os
import json
sys.path.append(".")
from src.utils.setup_simulations import _perform_vortex_checks
from src.utils.setup_simulations import _perform_reimprint_checks
from src.utils.setup_simulations import save_parameters_to_json
from src.utils.setup_simulations import _perform_grid_checks
from src.utils.setup_simulations import _simulations_repetitive
from src.utils.setup_simulations import get_simulation_parameters
from unittest.mock import patch
from src.utils.setup_simulations import get_simulation_combinations

class TestPerformVortexChecks(unittest.TestCase):
    def test_valid_vortex_parameters(self):
        """Test with valid vortex parameters."""
        simulation_params = {
            "Grid_resolution": [128, 128, 128],
            "vortex_charge": [[1, -1], [2, -2]],
            "vortex_position_x": [[10, -10], [20, -20]],
            "vortex_position_y": [[15, -15], [25, -25]],
            "repetitive": False,
            "imprinting_charge": []
        }
        result, msg = _perform_vortex_checks(simulation_params)
        self.assertTrue(result)
        self.assertEqual(msg, "")

    def test_mismatched_vortex_charge_and_position_x(self):
        """Test when the number of vortex charges doesn't match the number of x positions."""
        simulation_params = {
            "Grid_resolution": [128, 128, 128],
            "vortex_charge": [[1, -1]],
            "vortex_position_x": [[10, -10, 30]],
            "vortex_position_y": [[15, -15]],
            "repetitive": False,
            "imprinting_charge": []
        }
        result, msg = _perform_vortex_checks(simulation_params)
        self.assertFalse(result)
        self.assertIn("The number of charges doesn't agree with the number of x positions", msg)

    def test_mismatched_vortex_position_x_and_y(self):
        """Test when the number of x positions doesn't match the number of y positions."""
        simulation_params = {
            "Grid_resolution": [128, 128, 128],
            "vortex_charge": [[1, -1]],
            "vortex_position_x": [[10, -10]],
            "vortex_position_y": [[15]],
            "repetitive": False,
            "imprinting_charge": []
        }
        result, msg = _perform_vortex_checks(simulation_params)
        self.assertFalse(result)

    def test_vortex_position_out_of_bounds(self):
        """Test when vortex positions are out of bounds."""
        simulation_params = {
            "Grid_resolution": [128, 128, 128],
            "vortex_charge": [[1, -1]],
            "vortex_position_x": [[70, -70]],
            "vortex_position_y": [[70, -70]],
            "repetitive": False,
            "imprinting_charge": []
        }
        result, msg = _perform_vortex_checks(simulation_params)
        self.assertFalse(result)
        self.assertIn("is greater than half the grid size", msg)

    def test_negative_vortex_position_out_of_bounds(self):
        """Test when negative vortex positions are out of bounds."""
        simulation_params = {
            "Grid_resolution": [128, 128, 128],
            "vortex_charge": [[1, -1]],
            "vortex_position_x": [[-70, 70]],
            "vortex_position_y": [[-70, 70]],
            "repetitive": False,
            "imprinting_charge": []
        }
        result, msg = _perform_vortex_checks(simulation_params)
        self.assertFalse(result)

    def test_repetitive_with_mismatched_imprinting_charge(self):
        """Test when repetitive is True and imprinting charges mismatch."""
        simulation_params = {
            "Grid_resolution": [128, 128, 128],
            "vortex_charge": [[1, -1]],
            "vortex_position_x": [[10, -10]],
            "vortex_position_y": [[15, -15]],
            "repetitive": True,
            "imprinting_charge": [[1, -1, 2], [1, 4, 2]]
        }
        result, msg = _perform_vortex_checks(simulation_params)
        self.assertFalse(result)
        self.assertIn("The number of initial vortex charges doesn't agree with the number of imprinted charges", msg)

class TestPerformReimprintChecks(unittest.TestCase):
    def test_valid_reimprint_parameters(self):
        """Test with valid reimprint parameters."""
        simulation_params = {
            "Total_simulation_time": 10.0,
            "shots": 100,
            "repetitive": True,
            "imprint_times": [[10, 20, 30], [40, 50, 60]],
            "imprinting_charge": [[1, -1], [2, -2]],
            "vortex_charge": [[1, -1], [2, -2]],
            "imprint_position_x": [[10, -10], [20, -20]],
            "imprint_position_y": [[15, -15], [25, -25]]
        }
        result, msg = _perform_reimprint_checks(simulation_params)
        self.assertTrue(result)
        self.assertEqual(msg, "")

    def test_mismatched_imprint_times_and_charges(self):
        """Test when the number of imprint times doesn't match the number of imprinting charges."""
        simulation_params = {
            "Total_simulation_time": 10.0,
            "shots": 100,
            "repetitive": True,
            "imprint_times": [[10, 20, 30]],
            "imprinting_charge": [[1, -1], [2, -2]],
            "vortex_charge": [[1, -1], [2, -2]],
            "imprint_position_x": [[10, -10], [20, -20]],
            "imprint_position_y": [[15, -15], [25, -25]]
        }
        result, msg = _perform_reimprint_checks(simulation_params)
        self.assertFalse(result)
        self.assertIn("One list of imprinting times should be given for every simulation", msg)

    def test_mismatched_vortex_and_imprinting_charges(self):
        """Test when the number of vortex charges doesn't match the number of imprinting charges."""
        simulation_params = {
            "Total_simulation_time": 10.0,
            "shots": 100,
            "repetitive": True,
            "imprint_times": [[10, 20, 30],[6]],
            "imprinting_charge": [[1, -1],[2]],
            "vortex_charge": [[1, -1, 4]],
            "imprint_position_x": [[10, -10]],
            "imprint_position_y": [[15, -15]]
        }
        result, msg = _perform_reimprint_checks(simulation_params)
        self.assertFalse(result)
        self.assertIn("doesn't agree with the list number of imprinted", msg)
    
    def test_mismatched_individual_vortex_and_imprinting_charges(self):
        """Test when the number of vortex charges doesn't match the number of imprinting charges."""
        simulation_params = {
            "Total_simulation_time": 10.0,
            "shots": 100,
            "repetitive": True,
            "imprint_times": [[10, 20, 30]],
            "imprinting_charge": [[1, -1, 4]],
            "vortex_charge": [[1, -1]],
            "imprint_position_x": [[10, -10, 0]],
            "imprint_position_y": [[15, -15, 0]]
        }
        result, msg = _perform_reimprint_checks(simulation_params)
        self.assertTrue(result)
    
    def test_mismatched_individual_vortex_and_imprinting_charges_multi(self):
        """Test when the number of vortex charges doesn't match the number of imprinting charges."""
        simulation_params = {
            "Total_simulation_time": 10.0,
            "shots": 100,
            "repetitive": True,
            "imprint_times": [[10, 20, 30], [3, 4, 5]],
            "imprinting_charge": [[[1, -1], [4], [5]], [[3, 4], [5], [6]]],
            "vortex_charge": [[1, -1], [5, 7]],
            "imprint_position_x": [[[10, -10], [0], [0]], [[0, 0], [0], [0]]],
            "imprint_position_y": [[[10, -10], [0], [0]], [[0, 0], [0], [0]]]
        }
        result, msg = _perform_reimprint_checks(simulation_params)
        self.assertTrue(result)

    def test_mismatched_imprinting_charges_and_positions(self):
        """Test when the number of imprinting charges doesn't match the number of positions."""
        simulation_params = {
            "Total_simulation_time": 10.0,
            "shots": 100,
            "repetitive": True,
            "imprint_times": [[10, 20, 30], [40, 50, 60]],
            "imprinting_charge": [[1, -1], [2, -2]],
            "vortex_charge": [[1, -1], [2, -2]],
            "imprint_position_x": [[10, -10], [20]],
            "imprint_position_y": [[15, -15], [25, -25]]
        }
        result, msg = _perform_reimprint_checks(simulation_params)
        self.assertFalse(result)
        self.assertIn("doesn't agree with the number of x positions", msg)

    def test_imprint_time_exceeds_simulation_time(self):
        """Test when imprint times exceed the total simulation time."""
        simulation_params = {
            "Total_simulation_time": 10.0,
            "shots": 100,
            "repetitive": True,
            "imprint_times": [[10, 20, 110], [40, 50, 60]],
            "imprinting_charge": [[1, -1], [2, -2]],
            "vortex_charge": [[1, -1], [2, -2]],
            "imprint_position_x": [[10, -10], [20, -20]],
            "imprint_position_y": [[15, -15], [25, -25]]
        }
        result, msg = _perform_reimprint_checks(simulation_params)
        self.assertFalse(result)
        self.assertIn("The maximum imprint time is greater than the total simulation time", msg)

class TestSaveParametersToJson(unittest.TestCase):
    def setUp(self):
        """Set up a temporary file name for testing."""
        self.test_file = "test_simulation_parameters.json"

    def tearDown(self):
        """Clean up the test file after each test."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_save_parameters_to_json(self):
        """Test saving parameters to a JSON file."""
        parameters = {
            "Grid_resolution": [128, 128, 128],
            "Total_simulation_time": 10.0,
            "Trapping_frequencies": [1.0, 1.0, 1.0],
            "vortex_charge": [1, -1],
            "imprinting_charge": [2, -2],
            "vortex_position_x": [10, -10],
            "vortex_position_y": [15, -15]
        }

        # Save parameters to a JSON file
        save_parameters_to_json(parameters, filepath=self.test_file)

        # Check if the file was created
        self.assertTrue(os.path.exists(self.test_file))

        # Load the file and verify its contents
        with open(self.test_file, "r") as file:
            saved_parameters = json.load(file)

        self.assertEqual(parameters, saved_parameters)

    def test_save_numpy_array_to_json(self):
        """Test saving parameters with numpy arrays to a JSON file."""
        import numpy as np

        parameters = {
            "Grid_resolution": np.array([128, 128, 128]),
            "Total_simulation_time": 10.0,
            "Trapping_frequencies": np.array([1.0, 1.0, 1.0]),
        }

        # Save parameters to a JSON file
        save_parameters_to_json(parameters, filepath=self.test_file)

        # Check if the file was created
        self.assertTrue(os.path.exists(self.test_file))

        # Load the file and verify its contents
        with open(self.test_file, "r") as file:
            saved_parameters = json.load(file)

        # Convert numpy arrays to lists for comparison
        expected_parameters = {
            "Grid_resolution": [128, 128, 128],
            "Total_simulation_time": 10.0,
            "Trapping_frequencies": [1.0, 1.0, 1.0],
        }

        self.assertEqual(expected_parameters, saved_parameters)

    def test_invalid_type_raises_error(self):
        """Test that saving unsupported types raises a TypeError."""
        parameters = {
            "unsupported_type": set([1, 2, 3])  # Sets are not JSON serializable
        }

        with self.assertRaises(TypeError):
            save_parameters_to_json(parameters, filepath=self.test_file)

class TestPerformGridChecks(unittest.TestCase):

    def test_valid_grid(self):
        simulation_params = {
            "x_min": [-10, -10, -10],
            "x_max": [10, 10, 10],
            "Grid_resolution": [100, 100, 100]
        }
        ok, msg = _perform_grid_checks(simulation_params)
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_x_min_not_negative(self):
        simulation_params = {
            "x_min": [10, -10, -10],
            "x_max": [10, 10, 10],
            "Grid_resolution": [100, 100, 100]
        }
        ok, msg = _perform_grid_checks(simulation_params)
        self.assertFalse(ok)
        self.assertIn("x_min for axis 1 is not negative", msg)

    def test_grid_not_symmetric(self):
        simulation_params = {
            "x_min": [-10, -10, -5],
            "x_max": [10, 10, 10],
            "Grid_resolution": [100, 100, 100]
        }
        ok, msg = _perform_grid_checks(simulation_params)
        self.assertFalse(ok)
        self.assertIn("3 max and min are not symmetric", msg)

    def test_grid_resolution_zero(self):
        simulation_params = {
            "x_min": [-10, -10, -10],
            "x_max": [10, 10, 10],
            "Grid_resolution": [100, 0, 100]
        }
        ok, msg = _perform_grid_checks(simulation_params)
        self.assertFalse(ok)
        self.assertIn("The grid resolution must be greater than zero", msg)

    def test_invalid_simulation_params_type(self):
        simulation_params = [(-10, -10, -10), (10, 10, 10), (100, 100, 100)]  # Not a dictionary
        with self.assertRaises(TypeError) as context:
            _perform_grid_checks(simulation_params)
        self.assertIn("simulation_params must be a dictionary", str(context.exception))

class TestSimulationsRepetitive(unittest.TestCase):
    def setUp(self):
        # Define multiple base parameter sets for testing
        self.example_params = [
            {
                "params": {
                    "vortex_charge": [1, -1],
                    "imprinting_charge": [1, -1],
                    "max_imprints": 10,
                    "imprint_every": 10,
                    "imprint_times": [10, 20]
                },
                "expected_name_part": "2vort__initCharge[1, -1]__imprintCharge[1, -1]__snapshots10_20"
            },
            {
                "params": {
                    "vortex_charge": [2, 2, -2],
                    "imprinting_charge": [1],
                    "max_imprints": 5,
                    "imprint_every": 5,
                    "imprint_times": [5, 10, 15]
                },
                "expected_name_part": "3vort__initCharge[2, 2, -2]__imprintCharge[1]__snapshots5_10_15"
            },
            {
                "params": {
                    "vortex_charge": [1],
                    "imprinting_charge": [[1],[1],[1]],
                    "max_imprints": 3,
                    "imprint_every": 5,
                    "imprint_times": [5, 10, 15]
                },
                "expected_name_part": "1vort__initCharge[1]__imprintCharge[[1],[1],[1]]__snapshots5_10_15"
            },
            {
                "params": {
                    "vortex_charge": [1],
                    "imprinting_charge": [[1],[1],[1]],
                    "max_imprints": 3,
                    "imprint_every": 5,
                    "imprint_times": [5, 10, 15]
                },
                "expected_name_part": "1vort__initCharge[1]__imprintCharge[[1],[1],[1]]__snapshots5_10_15"
            }
        ]

    @patch("src.utils.setup_simulations.os.mkdir")
    @patch("src.utils.setup_simulations.os.path.isdir")
    def test_short_charges(self, mock_isdir, mock_mkdir):
        """Test shortening of charges list > 10."""
        mock_isdir.return_value = False
        
        # Pick the first example as base
        base = self.example_params[0]["params"]
        
        long_charges = [1] * 12
        params = base.copy()
        params["vortex_charge"] = long_charges
        
        sims = _simulations_repetitive([params])
        
        # charges > 10 -> f"{charges[0]}_{charges[-2]}_{charges[-1]}" -> "1_1_1"
        expected_part = "initCharge_init_1_last_1"
        
        self.assertIn(expected_part, sims[0][0])
        self.assertTrue(sims[0][0].startswith("12vort"))

    @patch("src.utils.setup_simulations.os.mkdir")
    @patch("src.utils.setup_simulations.os.path.isdir")
    def test_short_imprinting_charges(self, mock_isdir, mock_mkdir):
        """Test shortening of imprinting_charges list > 10."""
        mock_isdir.return_value = False
        
        base = self.example_params[0]["params"]
        long_imp_charges = [2] * 12
        params = base.copy()
        params["imprinting_charge"] = long_imp_charges
        
        sims = _simulations_repetitive([params])
        
        # imprinting_charge > 10 -> f"{imp[0]}_{imp[-2]}_{imp[-1]}" -> "2_2_2"
        expected_part = "imprintCharge_init_2_last_2"
        
        self.assertIn(expected_part, sims[0][0])

    @patch("src.utils.setup_simulations.os.mkdir")
    @patch("src.utils.setup_simulations.os.path.isdir")
    def test_short_imprint_times(self, mock_isdir, mock_mkdir):
        """Test shortening of imprint_times list > 10."""
        mock_isdir.return_value = False
        
        base = self.example_params[0]["params"]
        long_times = list(range(12))
        params = base.copy()
        params["imprint_times"] = long_times
        
        sims = _simulations_repetitive([params])
        
        # imprint_times > 10 -> f"{times[0]}_{times[-2]}_{times[-1]}" -> "0_10_11"
        expected_part = "snapshots_init_0_last_11"
        
        self.assertIn(expected_part, sims[0][0])

    @patch("src.utils.setup_simulations.os.mkdir")
    @patch("src.utils.setup_simulations.os.path.isdir")
    def test_non_list_charges(self, mock_isdir, mock_mkdir):
        """Test input where charges is not a list."""
        mock_isdir.return_value = False
        
        base = self.example_params[0]["params"]
        params = base.copy()
        params["vortex_charge"] = 1
        
        sims = _simulations_repetitive([params])
        
        expected_part = "1vort__initCharge1"
        
        self.assertIn(expected_part, sims[0][0])

class TestGetSimulationParameters(unittest.TestCase):
    def setUp(self):
        self.valid_config = {
            "Grid_resolution": [128, 128, 128],
            "Grid_negative_limits": [-10, -10, -10],
            "Grid_positive_limits": [10, 10, 10],
            "Trapping_frequencies": [10, 10, 10],
            "Total_simulation_time": 10.0,
            "dt": 0.01,
            "snapshots": 100,
            "vortex_excitation": 1,
            "vortex_charge": [1],
            "imprinting_charge": [[[1],[1],[1]]],
            "vortex_position_x": [0],
            "vortex_position_y": [0],
            "imprint_position_x": [[[0],[0],[0]]],
            "imprint_position_y": [[[0],[0],[0]]],
            "initial_imprint_time": [0],
            "repetitive": 1,
            "imprint_every": [0],
            "imprint_times": [[10,20,30]],
            "max_imprints": [3],
            "Potential_type": "Harmonic",
            "SwitchOff_time": 999
        }

    @patch("src.utils.setup_simulations._read_configuration_file")
    def test_valid_parameters(self, mock_read_config):
        mock_read_config.return_value = self.valid_config
        
        params, msg = get_simulation_parameters("dummy_path.json")
        
        self.assertIsNotNone(params)
        self.assertEqual(msg, "\n\n\n\n") # All checks pass, only newlines from the checks
        self.assertEqual(params["Grid_resolution"], [128, 128, 128])
        # Check derived parameters existence
        self.assertTrue("omega_ho" in params)
        self.assertTrue("u" in params)
        self.assertTrue("a_ho" in params)
        self.assertEqual(params["imprint_position_x"], [[[0],[0],[0]]])
        self.assertEqual(params["imprint_position_y"], [[[0],[0],[0]]])
        # Check normalization happened (x_max should be divided by something > 0)
        # Assuming a_ho is around 1e-6 or similar order of magnitude for typical atoms
        # Just check it ran without error and returned dict

    @patch("src.utils.setup_simulations._read_configuration_file")
    def test_repetitive_mismatch(self, mock_read_config):
        config = self.valid_config.copy()
        config["repetitive"] = True
        config["imprint_every"] = [10]
        config["imprint_times"] = [[], []] # Mismatch length
        config["max_imprints"] = [10]
        config["imprinting_charge"] = [1]
        
        mock_read_config.return_value = config
        
        params, msg = get_simulation_parameters("dummy_path.json")
        
        self.assertIsNone(params)
        self.assertIn("imprint_every and imprint_times have different number of simulations", msg)

    @patch("src.utils.setup_simulations._read_configuration_file")
    def test_imprint_times_calculation(self, mock_read_config):
        config = self.valid_config.copy()
        config["repetitive"] = True
        config["imprint_every"] = [2]
        config["imprint_times"] = [[10,20,30]]
        config["max_imprints"] = [3]
        config["imprinting_charge"] = [1]
        config["vortex_charge"] = [1]
        config["vortex_position_x"] = [0]
        config["vortex_position_y"] = [0]
        config["imprint_position_x"] = [[0,0,0]] # Need 3 positions for 3 imprints
        config["imprint_position_y"] = [[0,0,0]] # Need 3 positions
        
        mock_read_config.return_value = config
        
        with patch("src.utils.setup_simulations._check_simulation_parameters", return_value=(True, "")) as mock_check:
            params, msg = get_simulation_parameters("dummy_path.json")
            
            self.assertIsNotNone(params)
            self.assertEqual(params["imprint_times"][0], [10, 20, 30])

class TestFolderCreationIntegration(unittest.TestCase):
    def setUp(self):
        # Base configuration data (Repetitive = 1)
        self.config_data = {
            "Grid_resolution": [64, 64, 64],
            "Grid_negative_limits": [-10, -10, -10],
            "Grid_positive_limits": [10, 10, 10],
            "Trapping_frequencies": [10, 10, 10],
            "Total_simulation_time": 1.0,
            "dt": 0.01,
            "snapshots": 100,
            "vortex_excitation": True,
            "vortex_charge": [[1],[2],[3,3,3]],
            "imprinting_charge": [[[1],[1]],[[2],[2]],[[1,1,1]]],
            "vortex_position_x": [[0],[0],[0,0,0]],
            "vortex_position_y": [[0],[0],[0,0,0]],
            "imprint_position_x": [[[0],[0]], [[0],[0]],[[0,0,0]]],
            "imprint_position_y": [[[0],[0]], [[0],[0]],[[0,0,0]]],
            "initial_imprint_time": [0,0,0],
            "repetitive": 1,
            "imprint_every": [[],[],60],
            "imprint_times": [[10,20],[10,20],[]], 
            "max_imprints": [2,2,1],
            "Potential_type": "Harmonic",
            "SwitchOff_time": 999
        }

    @patch("src.utils.setup_simulations._read_configuration_file")
    @patch("src.utils.setup_simulations.os.mkdir")
    @patch("src.utils.setup_simulations.os.path.isdir")
    def test_all_simulation_folders_creation(self, mock_isdir, mock_mkdir, mock_read_config):
        mock_isdir.return_value = False
        mock_read_config.return_value = self.config_data.copy()
        
        # 1. Get parameters (mocked file read)
        params, msg = get_simulation_parameters("dummy_path.json")
        self.assertIsNotNone(params, f"Parameters validation failed: {msg}")
        
        # 2. Setup simulations
        sims = get_simulation_combinations(params)
        
        # 3. Verify folder creation
        expected_folders = [
            "1vort__initCharge1__imprintCharge[1]_[1]__snapshots10_20",
            "1vort__initCharge2__imprintCharge[2]_[2]__snapshots10_20",
            "3vort__initCharge3_3_3__imprintCharge[1, 1, 1]__snapshots60"
        ]
        
        self.assertEqual(len(sims), 3)
        
        # Check that mkdir was called for each expected folder
        for folder in expected_folders:
            mock_mkdir.assert_any_call(folder)
            
        # Check that the returned simulations match expected folders
        created_folders = [sim[0] for sim in sims]
        self.assertEqual(sorted(created_folders), sorted(expected_folders))

class TestFolderCreationIntegrationNonRepetitive(unittest.TestCase):
    def setUp(self):
        # Base configuration data (Repetitive = 0)
        self.config_data = {
            "Grid_resolution": [64, 64, 64],
            "Grid_negative_limits": [-10, -10, -10],
            "Grid_positive_limits": [10, 10, 10],
            "Trapping_frequencies": [10, 10, 10],
            "Total_simulation_time": 1.0,
            "dt": 0.01,
            "snapshots": 100,
            "vortex_excitation": True,
            "vortex_charge": [[1],[2],[3,3,3]],
            "imprinting_charge": [[[1],[1]],[[2],[2]],[[1,1,1]]],
            "vortex_position_x": [[0],[0],[0,0,0]],
            "vortex_position_y": [[0],[0],[0,0,0]],
            "imprint_position_x": [[[0],[0]], [[0],[0]],[[0,0,0]]],
            "imprint_position_y": [[[0],[0]], [[0],[0]],[[0,0,0]]],
            "initial_imprint_time": [0,0,0],
            "repetitive": 0,
            "imprint_every": [[],[],60],
            "imprint_times": [[10,20],[10,20],[]], 
            "max_imprints": [2,2,1],
            "Potential_type": "Harmonic",
            "SwitchOff_time": 999
        }

    @patch("src.utils.setup_simulations._read_configuration_file")
    @patch("src.utils.setup_simulations.os.mkdir")
    @patch("src.utils.setup_simulations.os.path.isdir")
    def test_all_simulation_folders_creation(self, mock_isdir, mock_mkdir, mock_read_config):
        mock_isdir.return_value = False
        mock_read_config.return_value = self.config_data.copy()
        
        # 1. Get parameters (mocked file read)
        params, msg = get_simulation_parameters("dummy_path.json")
        self.assertIsNotNone(params, f"Parameters validation failed: {msg}")
        
        # 2. Setup simulations
        sims = get_simulation_combinations(params)
        
        # 3. Verify folder creation
        expected_folders = [
            "1vortex_charges1__x-0__y-0",
            "1vortex_charges2__x-0__y-0",
            "3vortex_charges3_3_3__x-0_0_0__y-0_0_0"
        ]
        
        self.assertEqual(len(sims), 3)
        
        # Check that mkdir was called for each expected folder
        for folder in expected_folders:
            mock_mkdir.assert_any_call(folder)
            
        # Check that the returned simulations match expected folders
        created_folders = [sim[0] for sim in sims]
        self.assertEqual(sorted(created_folders), sorted(expected_folders))

if __name__ == "__main__":
    unittest.main()