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

if __name__ == "__main__":
    unittest.main()