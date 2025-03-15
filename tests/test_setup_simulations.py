import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import unittest
sys.path.append(".")
from src.utils import setup_simulations as ss
from src.utils.setup_simulations import _perform_vortex_checks
from src.utils.setup_simulations import _perform_reimprint_checks

# Test save_parameters_to_json
# params = {'x':np.array([1,2,3]), 'y':np.array([[1,2],[3,4]]), 'z':5, 't': 'potential'}
# ss.save_parameters_to_json(params)

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
        self.assertFalse(result)
        self.assertIn("The number of imprinting charges doesn't agree with the number of charges", msg)
    
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
        self.assertFalse(result)
        self.assertIn("The number of imprinting charges doesn't agree with the number of charges", msg)

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

if __name__ == "__main__":
    unittest.main()