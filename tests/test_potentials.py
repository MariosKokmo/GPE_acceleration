import unittest
import sys
sys.path.append('.')
import torch
import numpy as np
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


class TestConstPot(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for the tests."""
        self.app = MockApp()
        self.simulation_parameters = {
            "Grid_resolution": [32, 32, 32],
            "x_min": [0.0, 0.0, 0.0],
            "dx": [0.1, 0.1, 0.1],
            "w": [1.0, 1.0, 1.0]
        }

    def test_constant_potential_value(self):
        """Test that ConstPot creates a constant potential with correct amplitude."""
        amplitude = 2.5
        potential = ConstPot(self.app, amplitude=amplitude, **self.simulation_parameters)
        
        # Check that all values are equal to the amplitude
        self.assertTrue(torch.allclose(potential.potential, torch.tensor(amplitude, dtype=torch.double)))
        
    def test_constant_potential_shape(self):
        """Test that ConstPot has the correct shape."""
        potential = ConstPot(self.app, amplitude=1.0, **self.simulation_parameters)
        expected_shape = tuple(self.simulation_parameters["Grid_resolution"])
        self.assertEqual(potential.potential.shape, expected_shape)
    
    def test_constant_potential_time_evolution(self):
        """Test that ConstPot doesn't change with time."""
        potential = ConstPot(self.app, amplitude=3.0, **self.simulation_parameters)
        
        # Evaluate at different times
        V_t0 = potential.evol(0.0)
        V_t1 = potential.evol(1.0)
        V_t10 = potential.evol(10.0)
        
        # All should be the same
        self.assertTrue(torch.allclose(V_t0, V_t1))
        self.assertTrue(torch.allclose(V_t0, V_t10))
        self.assertTrue(torch.allclose(V_t0, torch.full_like(V_t0, 3.0)))
    
    def test_constant_potential_zero(self):
        """Test the zero method sets potential to zero."""
        potential = ConstPot(self.app, amplitude=5.0, **self.simulation_parameters)
        potential.zero()
        
        self.assertTrue(torch.allclose(potential.potential, torch.zeros_like(potential.potential)))


class TestRampPot(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for the tests."""
        self.app = MockApp()
        self.simulation_parameters = {
            "Grid_resolution": [32, 32, 32],
            "x_min": [0.0, 0.0, 0.0],
            "dx": [0.1, 0.1, 0.1],
            "w": [1.0, 1.0, 1.0]
        }

    def test_ramp_potential_shape(self):
        """Test that RampPot has the correct shape."""
        potential = RampPot(self.app, initial=1.0, final=2.0, tfinal=1.0, **self.simulation_parameters)
        expected_shape = tuple(self.simulation_parameters["Grid_resolution"])
        self.assertEqual(potential.potential.shape, expected_shape)
    
    def test_ramp_time_evolution(self):
        """Test that RampPot evolves linearly in time."""
        initial = 1.0
        final = 3.0
        tfinal = 2.0
        potential = RampPot(self.app, initial=initial, final=final, tfinal=tfinal, **self.simulation_parameters)
        
        # Test at t=0 (should be initial)
        V_t0 = potential.evol(0.0)
        expected_t0 = torch.full_like(potential.potential, initial)
        self.assertTrue(torch.allclose(V_t0, expected_t0))
        
        # Test at t=tfinal (should be final)
        V_tfinal = potential.evol(tfinal)
        expected_tfinal = torch.full_like(potential.potential, final)
        self.assertTrue(torch.allclose(V_tfinal, expected_tfinal))
        
        # Test at t=tfinal/2 (should be midpoint)
        V_tmid = potential.evol(tfinal / 2)
        expected_tmid = torch.full_like(potential.potential, (initial + final) / 2)
        self.assertTrue(torch.allclose(V_tmid, expected_tmid))
    
    def test_ramp_linearity(self):
        """Test that the ramp evolution is linear."""
        initial = 0.5
        final = 2.5
        tfinal = 1.0
        potential = RampPot(self.app, initial=initial, final=final, tfinal=tfinal, **self.simulation_parameters)
        
        times = [0.0, 0.25, 0.5, 0.75, 1.0]
        expected_amplitudes = [0.5, 1.0, 1.5, 2.0, 2.5]
        
        for t, expected_amp in zip(times, expected_amplitudes):
            V_t = potential.evol(t)
            expected = torch.full_like(potential.potential, expected_amp)
            self.assertTrue(torch.allclose(V_t, expected, atol=1e-10))


class TestHarmonicPot(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for the tests."""
        self.app = MockApp()
        self.simulation_parameters = {
            "Grid_resolution": [16, 16, 16],
            "x_min": [-1.0, -1.0, -1.0],
            "dx": [0.125, 0.125, 0.125],
            "w": [1.0, 1.0, 1.0]
        }

    def test_harmonic_potential_shape(self):
        """Test that HarmonicPot has the correct shape."""
        potential = HarmonicPot(self.app, amplitude=1.0, **self.simulation_parameters)
        expected_shape = tuple(self.simulation_parameters["Grid_resolution"])
        self.assertEqual(potential.potential.shape, expected_shape)
    
    def test_harmonic_center_value(self):
        """Test that harmonic potential has minimum at center."""
        # Use symmetric grid centered at origin
        params = {
            "Grid_resolution": [17, 17, 17],  # Odd number to have center point
            "x_min": [-2.0, -2.0, -2.0],
            "dx": [0.25, 0.25, 0.25],
            "w": [1.0, 1.0, 1.0]
        }
        amplitude = 1.0
        potential = HarmonicPot(self.app, amplitude=amplitude, **params)
        
        # Center should be at index (8, 8, 8) and should have minimum value (close to 0)
        center_idx = 8
        center_value = potential.potential[center_idx, center_idx, center_idx].item()
        
        # The center value should be close to 0 (minimum of harmonic potential)
        self.assertLess(center_value, 0.01)  # Should be very small
    
    def test_harmonic_radial_symmetry(self):
        """Test that harmonic potential has radial symmetry for isotropic case."""
        # Use isotropic harmonic oscillator
        params = {
            "Grid_resolution": [17, 17, 17],
            "x_min": [-2.0, -2.0, -2.0],
            "dx": [0.25, 0.25, 0.25],
            "w": [1.0, 1.0, 1.0]
        }
        potential = HarmonicPot(self.app, amplitude=1.0, **params)
        
        # Points equidistant from center should have same potential value
        center = 8
        # Test a few points at same distance
        val1 = potential.potential[center + 2, center, center].item()
        val2 = potential.potential[center, center + 2, center].item()
        val3 = potential.potential[center, center, center + 2].item()
        
        self.assertAlmostEqual(val1, val2, places=10)
        self.assertAlmostEqual(val1, val3, places=10)
    
    def test_harmonic_anisotropic(self):
        """Test anisotropic harmonic potential."""
        params = {
            "Grid_resolution": [17, 17, 17],
            "x_min": [-2.0, -2.0, -2.0],
            "dx": [0.25, 0.25, 0.25],
            "w": [1.0, 2.0, 3.0]  # Different frequencies
        }
        potential = HarmonicPot(self.app, amplitude=1.0, **params)
        
        center = 8
        # Potential should be different along different axes
        val_x = potential.potential[center + 1, center, center].item()
        val_y = potential.potential[center, center + 1, center].item()
        val_z = potential.potential[center, center, center + 1].item()
        
        # Due to different w values, these should be different
        # val_x ~ 0.5 * w[0]^2 * dx^2 = 0.5 * 1^2 * 0.25^2
        # val_y ~ 0.5 * w[1]^2 * dx^2 = 0.5 * 2^2 * 0.25^2
        # val_z ~ 0.5 * w[2]^2 * dx^2 = 0.5 * 3^2 * 0.25^2
        self.assertAlmostEqual(val_x, 0.5 * (1.0 * 0.25)**2, places=10)
        self.assertAlmostEqual(val_y, 0.5 * (2.0 * 0.25)**2, places=10)
        self.assertAlmostEqual(val_z, 0.5 * (3.0 * 0.25)**2, places=10)
    
    def test_harmonic_time_evolution(self):
        """Test that static harmonic potential doesn't change with time."""
        potential = HarmonicPot(self.app, amplitude=1.0, **self.simulation_parameters)
        
        V_t0 = potential.evol(0.0)
        V_t1 = potential.evol(1.0)
        
        self.assertTrue(torch.allclose(V_t0, V_t1))
    
    def test_harmonic_amplitude_scaling(self):
        """Test that amplitude parameter scales the potential correctly."""
        amplitude1 = 1.0
        amplitude2 = 2.5
        
        pot1 = HarmonicPot(self.app, amplitude=amplitude1, **self.simulation_parameters)
        pot2 = HarmonicPot(self.app, amplitude=amplitude2, **self.simulation_parameters)
        
        # pot2 should be amplitude2/amplitude1 times pot1
        ratio = amplitude2 / amplitude1
        self.assertTrue(torch.allclose(pot2.potential, pot1.potential * ratio))
    
    def test_harmonic_zero_2D(self):
        """Test the zero_2D method."""
        potential = HarmonicPot(self.app, amplitude=1.0, **self.simulation_parameters)
        original_potential = potential.potential.clone()
        
        potential.zero_2D(amplitude=1.0)
        
        # The potential should not be the same as before
        self.assertFalse(torch.allclose(potential.potential, original_potential))
        
        # In the flattened dimensions (x and z), potential should be zero
        # Only y-direction should have harmonic confinement
        # Check that x=0, z=0 slices have the expected 1D harmonic form
        n1, n2, n3 = self.simulation_parameters["Grid_resolution"]
        center_x = n1 // 2
        center_z = n3 // 2
        
        # At the center of x and z, should have 1D harmonic potential in y
        slice_xz_center = potential.potential[center_x, :, center_z]
        # This should be a 1D harmonic along y-axis
        self.assertGreater(slice_xz_center.max().item(), 0)  # Should have non-zero values


class TestRampHarmonicPot(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for the tests."""
        self.app = MockApp()
        self.simulation_parameters = {
            "Grid_resolution": [16, 16, 16],
            "x_min": [-1.0, -1.0, -1.0],
            "dx": [0.125, 0.125, 0.125],
            "w": [1.0, 1.0, 1.0]
        }

    def test_ramp_harmonic_shape(self):
        """Test that RampHarmonicPot has the correct shape."""
        potential = RampHarmonicPot(self.app, initial=1.0, amplitude=2.0, 
                                    tinit=0.0, tfinal=1.0, **self.simulation_parameters)
        expected_shape = tuple(self.simulation_parameters["Grid_resolution"])
        self.assertEqual(potential.potential.shape, expected_shape)
    
    def test_ramp_harmonic_time_evolution(self):
        """Test that RampHarmonicPot evolves correctly in time."""
        initial = 0.5
        amplitude = 2.0
        tinit = 0.0
        tfinal = 1.0
        
        potential = RampHarmonicPot(self.app, initial=initial, amplitude=amplitude,
                                    tinit=tinit, tfinal=tfinal, **self.simulation_parameters)
        
        # Get the base harmonic potential shape
        base_pot = potential.potential.clone()
        
        # At t=tinit, should be initial * base_pot
        V_tinit = potential.evol(tinit)
        expected_tinit = initial * base_pot
        self.assertTrue(torch.allclose(V_tinit, expected_tinit))
        
        # At t=tfinal, should be amplitude * base_pot
        V_tfinal = potential.evol(tfinal)
        expected_tfinal = amplitude * base_pot
        self.assertTrue(torch.allclose(V_tfinal, expected_tfinal))
        
        # At midpoint, should be average
        V_tmid = potential.evol((tinit + tfinal) / 2)
        expected_tmid = ((initial + amplitude) / 2) * base_pot
        self.assertTrue(torch.allclose(V_tmid, expected_tmid))
    
    def test_ramp_harmonic_linearity(self):
        """Test linear ramping of the harmonic potential."""
        initial = 1.0
        amplitude = 3.0
        tinit = 0.5
        tfinal = 1.5
        
        potential = RampHarmonicPot(self.app, initial=initial, amplitude=amplitude,
                                    tinit=tinit, tfinal=tfinal, **self.simulation_parameters)
        
        base_pot = potential.potential.clone()
        
        # Test at several time points
        test_times = [0.5, 0.75, 1.0, 1.25, 1.5]
        expected_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        for t, factor in zip(test_times, expected_factors):
            V_t = potential.evol(t)
            expected = factor * base_pot
            self.assertTrue(torch.allclose(V_t, expected, atol=1e-10))
    
    def test_ramp_harmonic_spatial_structure(self):
        """Test that spatial structure remains harmonic during ramp."""
        potential = RampHarmonicPot(self.app, initial=1.0, amplitude=2.0,
                                    tinit=0.0, tfinal=1.0, **self.simulation_parameters)
        
        # The spatial structure should remain the same, only amplitude changes
        V_t0 = potential.evol(0.0)
        V_t1 = potential.evol(1.0)
        
        # Ratio should be constant everywhere (equal to amplitude/initial = 2.0)
        ratio = V_t1 / (V_t0 + 1e-12)  # Add small value to avoid division by zero
        
        # Ratio should be approximately constant (2.0) everywhere
        # (except where potential is very close to zero)
        mask = V_t0 > 0.01  # Only check where potential is significant
        self.assertTrue(torch.allclose(ratio[mask], torch.tensor(2.0, dtype=torch.double), atol=0.01))


class TestAbsorberPotential(unittest.TestCase):
    def setUp(self):
        self.app = MockApp()
        self.simulation_parameters = {
            "Grid_resolution": [32, 8, 32],
            "x_min": [-2.0, -0.5, -2.0],
            "dx": [0.125, 0.125, 0.125],
            "w": [1.0, 1.0, 1.0]
        }

    def test_absorber_optional_by_default(self):
        """Without the Absorber_* keys the potential stays purely real.

        A complex potential is what makes the propagator non-unitary, so a trap
        that silently acquired an imaginary part would leak atoms in every run.
        """
        potential = HarmonicPot(self.app, amplitude=1.0, **self.simulation_parameters)
        self.assertIsNone(potential.absorber_potential)

        evolved = potential.evol(0.5)
        self.assertFalse(torch.is_complex(evolved))

    def test_absorber_adds_imaginary_damping_near_edges(self):
        """With the absorber on, the potential gains a non-positive imaginary part
        that is strongest at the boundary.

        The trap amplitude is set to zero so the imaginary part is the absorber's
        doing alone. Damping means a negative imaginary part; the edge cell must
        be more strongly damped than the centre, which is where the profile is
        supposed to vanish.
        """
        params = {
            **self.simulation_parameters,
            "Absorber_enabled": True,
            "Absorber_strength": 2.0,
            "Absorber_start_ratio": 0.7,
            "Absorber_power": 2.0,
            "Absorber_tinit": 0.0
        }
        potential = HarmonicPot(self.app, amplitude=0.0, **params)
        evolved = potential.evol(1.0)

        self.assertTrue(torch.is_complex(evolved))

        center = evolved[evolved.shape[0] // 2, evolved.shape[1] // 2, evolved.shape[2] // 2].imag.item()
        edge = evolved[0, evolved.shape[1] // 2, 0].imag.item()

        # Imaginary component should be non-positive and stronger near boundaries.
        self.assertLessEqual(edge, 0.0)
        self.assertLess(edge, center)

    def test_absorber_time_ramp(self):
        """The absorber ramps in linearly between Absorber_tinit and Absorber_tfinal.

        Before tinit=0.5 it is exactly off, at t=1.0 it is partly on, and by t=2.0,
        past tfinal=1.5, it is at full strength. Sampling all three points is what
        distinguishes a ramp from a switch that is simply on.
        """
        params = {
            **self.simulation_parameters,
            "Absorber_enabled": True,
            "Absorber_strength": 1.0,
            "Absorber_start_ratio": 0.8,
            "Absorber_power": 2.0,
            "Absorber_tinit": 0.5,
            "Absorber_tfinal": 1.5
        }
        potential = ConstPot(self.app, amplitude=0.0, **params)

        before = potential.evol(0.25)
        mid = potential.evol(1.0)
        after = potential.evol(2.0)

        before_abs = torch.max(torch.abs(before.imag)).item()
        mid_abs = torch.max(torch.abs(mid.imag)).item()
        after_abs = torch.max(torch.abs(after.imag)).item()

        self.assertAlmostEqual(before_abs, 0.0, places=12)
        self.assertGreater(mid_abs, 0.0)
        self.assertGreater(after_abs, mid_abs)


if __name__ == "__main__":
    unittest.main()