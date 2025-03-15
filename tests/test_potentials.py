import unittest
import sys
sys.path.append('.')
from src.library.potentials import select_potential, HarmonicPot, ConstPot, RampPot, RampHarmonicPot, CustomPot

class MockApp:
    """Mock application class to simulate the app object."""
    def __init__(self, device="cpu"):
        self.device = device

class TestSelectPotential(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for the tests."""
        self.app = MockApp()
        self.simulation_parameters = {
            "Grid_resolution": [128, 128, 128],
            "x_min": [0.0, 0.0, 0.0],
            "dx": [0.1, 0.1, 0.1],
            "w": [1.0, 1.0, 1.0]
        }

    def test_select_harmonic_potential(self):
        """Test selecting the Harmonic potential."""
        potential = select_potential("harmonic", self.app, **self.simulation_parameters)
        self.assertIsInstance(potential, HarmonicPot)

    def test_select_constant_potential(self):
        """Test selecting the Constant potential."""
        potential = select_potential("constant", self.app, **self.simulation_parameters)
        self.assertIsInstance(potential, ConstPot)

    def test_select_ramp_potential(self):
        """Test selecting the Ramp potential."""
        potential = select_potential("ramp", self.app, **self.simulation_parameters)
        self.assertIsInstance(potential, RampPot)

    def test_select_ramp_harmonic_potential(self):
        """Test selecting the RampHarmonic potential."""
        potential = select_potential("rampharmonic", self.app, **self.simulation_parameters)
        self.assertIsInstance(potential, RampHarmonicPot)

    def test_select_custom_potential(self):
        """Test selecting the Custom potential."""
        potential = select_potential("custom", self.app, **self.simulation_parameters)
        self.assertIsNone(potential)  # CustomPot is not implemented in the function

    def test_invalid_potential_type(self):
        """Test selecting an invalid potential type."""
        with self.assertRaises(ValueError) as context:
            select_potential("invalid", self.app, **self.simulation_parameters)
        self.assertIn("Potential type invalid is not available", str(context.exception))

if __name__ == "__main__":
    unittest.main()