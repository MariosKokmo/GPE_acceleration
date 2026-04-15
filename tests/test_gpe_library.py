import unittest
import numpy as np
import torch
import sys
sys.path.append('.')
from src.library.gpe_library import GPELibrary as gpe
from src.library.gpe_library import GPE2DLibrary as gpe2d
from src.library.gpe_library import GPE3DLibrary as gpe3d


class TestModGradPsi(unittest.TestCase):
    """Tests for the function that returns the modulus of the gradient"""
    @classmethod
    def setUpClass(self):
        """Creates the tensors"""
        self.N1 = 256
        self.N2 = 256
        self.N3 = 64
        self.N = np.array([self.N1, self.N2, self.N3])
        self.x_max = np.array([200, 200, 20])
        self.x_min = np.array([-200, -200, -20])
        self.dx = (self.x_max - self.x_min) / self.N
        self.dp = 2*np.pi/(self.x_max - self.x_min)
        # the error tolerance
        self.error_tol = 0.005

    def tearDown(self) -> None:
        return super().tearDown()
    
    def setup_grids(self, dim):
        """Return lists of maximum, minimum and length for every axis of the grid"""
        x1, x2, x3, p1, p2, p3, p_sq, space_grid, p_grid = gpe.init_grid(self.x_min, \
                        self.x_max, dx=self.dx, dp=self.dp, w=0, \
                        n1=self.N1, n2=self.N2, n3=self.N3, \
                            device='cpu')
        if dim==1:
            return [x1], [p1[0]], self.N1
        elif dim==2:
            return (x1, x2) , (p1[0], p2[0]), (self.N1, self.N2)
        elif dim==3:
            return (x1, x2, x3), (p1[0], p2[0], p3[0]), (self.N1, self.N2, self.N3)

    def test_tensor_flat_1D(self):
        space_grid, p_axes, n = self.setup_grids(dim=1)
        input = torch.ones_like(space_grid[0])
        grad = gpe.mod_grad_psi(input, p_axes)
        result = torch.zeros_like(space_grid[0])
        assert input.shape == result.shape
        assert grad.shape == result.shape, f"grad shape is {grad.shape} and expected result {result.shape}"
        error = np.linalg.norm(grad[n//3:-n//3] - result[n//3:-n//3])
        self.assertLessEqual(error, self.error_tol, msg=f"the error is not small enough")
    
    def test_tensor_1D(self):
        space_grid, p_axes, n = self.setup_grids(dim=1)
        input = torch.sin(space_grid[0])
        grad = gpe.mod_grad_psi(input, p_axes)
        result = torch.cos(space_grid[0])
        assert input.shape == result.shape
        assert grad.shape == result.shape, f"grad shape is {grad.shape} and expected result {result.shape}"
        error = np.linalg.norm(grad[n//3:-n//3] - result[n//3:-n//3])/np.linalg.norm(result[n//3:-n//3])
        self.assertLessEqual(error, self.error_tol, msg=f"the error is not small enough")

    def test_gaussian_1D(self):
        space_grid, p_axes, n = self.setup_grids(dim=1)
        input = torch.exp(-(space_grid[0]**2)/10)
        grad = gpe.mod_grad_psi(input, p_axes)
        result = -space_grid[0]/5 * input
        error = np.linalg.norm(grad[n//3:-n//3] - result[n//3:-n//3])/np.linalg.norm(result[n//3:-n//3])
        self.assertLessEqual(error, self.error_tol, msg=f"the error is not small enough")
    
    def test_sinExp_1D(self):
        space_grid, p_axes, n = self.setup_grids(dim=1)
        input = torch.sin(space_grid[0])*torch.exp(-space_grid[0]**2/100)
        grad = gpe.mod_grad_psi(input, p_axes)
        result = (torch.cos(space_grid[0])- (2 * space_grid[0] * torch.sin(space_grid[0])/100)) * torch.exp(-space_grid[0]**2/100)
        error = np.linalg.norm(grad[n//3:-n//3] - result[n//3:-n//3])/np.linalg.norm(result[n//3:-n//3])
        self.assertLessEqual(error, self.error_tol, msg=f"the error is not small enough")

    def test_tensor_flat_2D(self):
        grid, p_axes, n = self.setup_grids(dim=2)
        space_grid = torch.meshgrid(grid[0], grid[1])
        input = torch.ones_like(space_grid[0])
        result = torch.zeros_like(space_grid[0])
        assert input.shape == result.shape
        grad = gpe.mod_grad_psi(input, p_axes)
        assert grad.shape == result.shape, f"grad shape is {grad.shape} and expected result {result.shape}"
        error = np.linalg.norm(grad - result)
        self.assertLessEqual(error, self.error_tol, msg=f"the error is not small enough")

    def test_gaussian_2D(self):
        grid, p_axes, n = self.setup_grids(dim=2)
        space_grid = torch.meshgrid(grid[0], grid[1])
        input = torch.exp(-(space_grid[0]**2 + space_grid[1]**2)/200)
        assert len(p_axes) == 2
        grad = gpe.mod_grad_psi(input, p_axes)
        result = torch.sqrt(space_grid[0]**2 + space_grid[1]**2)/100 * input
        grad = grad[n[0]//5:-n[0]//5,n[1]//5:-n[1]//5]
        result = result[n[0]//5:-n[0]//5,n[1]//5:-n[1]//5]
        error = np.linalg.norm(grad - result)/np.linalg.norm(result)
        self.assertLessEqual(error, self.error_tol, msg=f"the error is not small enough")


    def test_tensor_flat_3D(self):
        grid, p_axes, n = self.setup_grids(dim=3)
        space_grid = torch.meshgrid(grid[0], grid[1], grid[2])
        input = torch.ones_like(space_grid[0])
        grad = gpe.mod_grad_psi(input, p_axes)
        result = torch.zeros_like(space_grid[0])
        error = np.linalg.norm(grad - result)
        self.assertLessEqual(error, self.error_tol, msg=f"the error is not small enough")

    def test_gaussian_3D(self):
        grid, p_axes, n = self.setup_grids(dim=3)
        space_grid = torch.meshgrid(grid[0], grid[1], grid[2])
        input = torch.exp(-(space_grid[0]**2 + space_grid[1]**2 + space_grid[2]**2)/200)
        grad = gpe.mod_grad_psi(input, p_axes)
        result = torch.sqrt(space_grid[0]**2 + space_grid[1]**2 + space_grid[2]**2)/100 * input
        # avoid the edges where we expect Gibbs oscillations
        grad = grad[n[0]//5:-n[0]//5,n[1]//5:-n[1]//5,n[2]//5:-n[2]//5]
        result = result[n[0]//5:-n[0]//5,n[1]//5:-n[1]//5,n[2]//5:-n[2]//5]
        error = np.linalg.norm(grad - result)/np.linalg.norm(result)
        self.assertLessEqual(error, self.error_tol, msg=f"the error is not small enough")

class TestInitGrid(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """Creates the tensors"""
        self.N1 = 256
        self.N3 = 32
        self.x_min = [-100, -100, -20]
        self.x_max = [100, 100, 20]
        self.dx = [200/self.N1, 200/self.N1, 40/self.N3]
        self.dp = [2*np.pi/200, 2*np.pi/200, 2*np.pi/40]
    
    def test_init_grid(self):
        x1, x2, x3, p1, p2, p3, p_sq, space_grid, p_grid = gpe.init_grid(self.x_min, \
                                                                         self.x_max, dx=self.dx, dp=self.dp, w=0, \
                                                                            n1=self.N1, n2=self.N1, n3=self.N3, \
                                                                                device='cpu')
        assert len(x1) == self.N1
        assert len(x2) == self.N1
        assert len(x3) == self.N3
        assert len(p1[0]) == self.N1, f"expected {self.N1} and got {len(p1)}"
        assert len(p2[0]) == self.N1
        assert len(p3[0]) == self.N3
        assert len(p_grid) == 3
        assert p_grid[0].shape == (self.N1, self.N1, self.N3), f"expected {(self.N1, self.N1, self.N3)} and got {p_grid[0].shape}"
        assert p_grid[1].shape == (self.N1, self.N1, self.N3), f"expected {(self.N1, self.N1, self.N3)} and got {p_grid[1].shape}"
        assert p_grid[2].shape == (self.N1, self.N1, self.N3), f"expected {(self.N1, self.N1, self.N3)} and got {p_grid[2].shape}"
        assert p_sq.shape == (self.N1, self.N1, self.N3), f"expected {(self.N1, self.N1, self.N3)} and got {p_sq.shape}"
        assert space_grid[0].shape == (self.N1, self.N1, self.N3), f"expected {(self.N1, self.N1, self.N3)} and got {space_grid[0].shape}"

class TestCreateVortices(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """Sets up the grid and parameters for testing"""
        self.N1 = 64
        self.N2 = 64
        self.N3 = 64
        self.x_min = [-10, -10, -10]
        self.x_max = [10, 10, 10]
        self.dx = [(self.x_max[i] - self.x_min[i]) / self.N1 for i in range(3)]
        self.dp = [2 * np.pi / (self.x_max[i] - self.x_min[i]) for i in range(3)]
        self.device = 'cpu'
        self.vortices = np.array([[0], [0], [1]])  # Single vortex at the origin with charge 1
        self.x1, self.x2, self.x3, _, _, _, _, _, _ = gpe.init_grid(
            self.x_min, self.x_max, self.dx, self.dp, 0, self.N1, self.N2, self.N3, self.device
        )

    def test_create_vortices_single(self):
        """Tests the creation of a single vortex"""
        phase = gpe2d.create_vortices(self.vortices, self.x1, self.x2, self.x3, self.N1, self.N2, self.N3, self.device)
        self.assertEqual(phase.shape, (self.N1, self.N2, self.N3), "Phase tensor shape mismatch")
        self.assertTrue(torch.is_tensor(phase), "Phase is not a tensor")
        self.assertTrue(torch.all(torch.isfinite(phase)), "Phase contains NaN or Inf values")

    def test_create_vortices_none(self):
        """Tests the behavior when no vortices are provided"""
        phase = gpe2d.create_vortices(None, self.x1, self.x2, self.x3, self.N1, self.N2, self.N3, self.device)
        self.assertIsNone(phase, "Phase should be None when no vortices are provided")

    def test_create_vortices_multiple(self):
        """Tests the creation of multiple vortices"""
        vortices = np.array([[0, 5], [0, -5], [1, -1]])  # Two vortices with different charges
        phase = gpe2d.create_vortices(vortices, self.x1, self.x2, self.x3, self.N1, self.N2, self.N3, self.device)
        self.assertEqual(phase.shape, (self.N1, self.N2, self.N3), "Phase tensor shape mismatch")
        self.assertTrue(torch.is_tensor(phase), "Phase is not a tensor")
        self.assertTrue(torch.all(torch.isfinite(phase)), "Phase contains NaN or Inf values")

class TestXEvolution(unittest.TestCase):
    def test_x_evolution(self):
        # Define test inputs
        psi1 = torch.tensor([[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]], dtype=torch.cdouble)
        utot1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        dtau = 0.1
        factor = 0.5

        # Expected output
        expected_output = torch.exp(-factor * dtau * 1j * utot1) * psi1

        # Run the function
        result = gpe.x_evolution(psi1, utot1, dtau, factor)

        # Assert the result is close to the expected output
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-6))

class TestCalculateDensityPeak(unittest.TestCase):
    def test_density_peak_simple(self):
        """Test with a simple 3D wavefunction where peak is known"""
        # Create a simple 3D wavefunction with known peak
        psi = torch.zeros((5, 5, 5), dtype=torch.cdouble)
        psi[2, 3, 1] = 5.0 + 0.0j  # Maximum at position (2, 3, 1)
        psi[1, 1, 1] = 2.0 + 0.0j
        psi[3, 3, 3] = 1.0 + 0.0j
        
        max_density, peak_indices = gpe2d.calculate_density_peak(psi)
        
        # Check that max_density is correct (|5.0|^2 = 25.0)
        self.assertAlmostEqual(max_density.item(), 25.0, places=6)
        
        # Check that peak_indices is correct
        self.assertEqual(peak_indices, (2, 3, 1))
    
    def test_density_peak_complex(self):
        """Test with complex wavefunction"""
        psi = torch.zeros((4, 4, 4), dtype=torch.cdouble)
        psi[1, 2, 3] = 3.0 + 4.0j  # |3+4j|^2 = 25
        psi[0, 0, 0] = 2.0 + 1.0j  # |2+1j|^2 = 5
        
        max_density, peak_indices = gpe2d.calculate_density_peak(psi)
        
        # Check that max_density is correct
        self.assertAlmostEqual(max_density.item(), 25.0, places=6)
        
        # Check that peak_indices is correct
        self.assertEqual(peak_indices, (1, 2, 3))
    
    def test_density_peak_gaussian(self):
        """Test with a Gaussian-like distribution"""
        # Create a 3D Gaussian centered at (5, 5, 5)
        x = torch.arange(11, dtype=torch.float64)
        y = torch.arange(11, dtype=torch.float64)
        z = torch.arange(11, dtype=torch.float64)
        X, Y, Z = torch.meshgrid(x, y, z)
        
        # Gaussian centered at (5, 5, 5)
        psi = torch.exp(-((X - 5)**2 + (Y - 5)**2 + (Z - 5)**2) / 2.0)
        psi = psi.type(torch.cdouble)
        
        max_density, peak_indices = gpe2d.calculate_density_peak(psi)
        
        # The peak should be at the center (5, 5, 5)
        self.assertEqual(peak_indices, (5, 5, 5))
        
        # The maximum density should be 1.0 (exp(0) = 1)
        self.assertAlmostEqual(max_density.item(), 1.0, places=6)
    
    def test_density_peak_corner(self):
        """Test when peak is at corner of grid"""
        psi = torch.zeros((10, 10, 10), dtype=torch.cdouble)
        psi[0, 0, 0] = 7.0 + 0.0j  # Peak at corner
        psi[5, 5, 5] = 3.0 + 0.0j
        
        max_density, peak_indices = gpe2d.calculate_density_peak(psi)
        
        self.assertAlmostEqual(max_density.item(), 49.0, places=6)
        self.assertEqual(peak_indices, (0, 0, 0))
    
    def test_density_peak_edge(self):
        """Test when peak is at edge but not corner"""
        psi = torch.zeros((8, 8, 8), dtype=torch.cdouble)
        psi[0, 4, 7] = 6.0 + 0.0j  # Peak at edge
        psi[4, 4, 4] = 2.0 + 0.0j
        
        max_density, peak_indices = gpe2d.calculate_density_peak(psi)
        
        self.assertAlmostEqual(max_density.item(), 36.0, places=6)
        self.assertEqual(peak_indices, (0, 4, 7))
    
    def test_density_peak_uniform(self):
        """Test when all values are equal"""
        psi = torch.ones((6, 6, 6), dtype=torch.cdouble) * 2.0
        
        max_density, peak_indices = gpe2d.calculate_density_peak(psi)
        
        # All densities are 4.0
        self.assertAlmostEqual(max_density.item(), 4.0, places=6)
        # Should return the first occurrence (0, 0, 0)
        self.assertEqual(peak_indices, (0, 0, 0))
    
    def test_density_peak_large_tensor(self):
        """Test with a larger tensor to verify unravel_index logic"""
        psi = torch.zeros((20, 30, 40), dtype=torch.cdouble)
        psi[15, 25, 35] = 10.0 + 0.0j  # Peak at specific position
        
        max_density, peak_indices = gpe2d.calculate_density_peak(psi)
        
        self.assertAlmostEqual(max_density.item(), 100.0, places=6)
        self.assertEqual(peak_indices, (15, 25, 35))
    
    def test_density_peak_negative_values(self):
        """Test that density is always positive (magnitude squared)"""
        psi = torch.zeros((5, 5, 5), dtype=torch.cdouble)
        psi[2, 2, 2] = -5.0 + 0.0j  # Negative real part
        psi[1, 1, 1] = 3.0 + 0.0j
        
        max_density, peak_indices = gpe2d.calculate_density_peak(psi)
        
        # |-5|^2 = 25, which is greater than |3|^2 = 9
        self.assertAlmostEqual(max_density.item(), 25.0, places=6)
        self.assertEqual(peak_indices, (2, 2, 2))
    
    def test_density_peak_multiple_complex(self):
        """Test with multiple complex values"""
        psi = torch.zeros((7, 7, 7), dtype=torch.cdouble)
        psi[1, 2, 3] = 2.0 + 2.0j   # |2+2j|^2 = 8
        psi[3, 3, 3] = 1.0 + 2.0j   # |1+2j|^2 = 5
        psi[5, 5, 5] = 2.0 + 1.0j   # |2+1j|^2 = 5
        psi[4, 4, 4] = 3.0 + 0.0j   # |3|^2 = 9
        
        max_density, peak_indices = gpe2d.calculate_density_peak(psi)
        
        self.assertAlmostEqual(max_density.item(), 9.0, places=6)
        self.assertEqual(peak_indices, (4, 4, 4))

class TestGPE3DLibrary(unittest.TestCase):
    """Tests for 3D-specific helper methods in GPE3DLibrary."""

    @classmethod
    def setUpClass(cls):
        cls.n1, cls.n2, cls.n3 = 24, 20, 16
        cls.x_min = [-6.0, -5.0, -4.0]
        cls.x_max = [6.0, 5.0, 4.0]
        cls.dx = [
            (cls.x_max[0] - cls.x_min[0]) / cls.n1,
            (cls.x_max[1] - cls.x_min[1]) / cls.n2,
            (cls.x_max[2] - cls.x_min[2]) / cls.n3,
        ]
        cls.dp = [
            2 * np.pi / (cls.x_max[0] - cls.x_min[0]),
            2 * np.pi / (cls.x_max[1] - cls.x_min[1]),
            2 * np.pi / (cls.x_max[2] - cls.x_min[2]),
        ]

        (
            cls.x1,
            cls.x2,
            cls.x3,
            cls.p1,
            cls.p2,
            cls.p3,
            cls.p_sq,
            cls.space_grid,
            cls.p_grid,
        ) = gpe.init_grid(
            cls.x_min,
            cls.x_max,
            dx=cls.dx,
            dp=cls.dp,
            w=0,
            n1=cls.n1,
            n2=cls.n2,
            n3=cls.n3,
            device="cpu",
        )

    def test_create_vortex_ring_shape_and_finite(self):
        phase = gpe3d.create_vortex_ring(
            self.x1,
            self.x2,
            self.x3,
            self.n1,
            self.n2,
            self.n3,
            ring_radius=1.2,
            center=(0.0, 0.0, 0.0),
            axis=3,
            charge=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(phase.shape, (self.n1, self.n2, self.n3))
        self.assertTrue(torch.all(torch.isfinite(phase)))

    def test_create_vortex_ring_charge_scaling_regression(self):
        """Phase should scale linearly with topological charge."""
        phase_q1 = gpe3d.create_vortex_ring(
            self.x1,
            self.x2,
            self.x3,
            self.n1,
            self.n2,
            self.n3,
            ring_radius=1.4,
            center=(0.0, 0.0, 0.0),
            axis=2,
            charge=1,
            device=torch.device("cpu"),
        )
        phase_q2 = gpe3d.create_vortex_ring(
            self.x1,
            self.x2,
            self.x3,
            self.n1,
            self.n2,
            self.n3,
            ring_radius=1.4,
            center=(0.0, 0.0, 0.0),
            axis=2,
            charge=2,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.allclose(phase_q2, 2.0 * phase_q1, atol=1e-12, rtol=0.0))

    def test_create_vortex_ring_invalid_axis_raises(self):
        with self.assertRaises(ValueError):
            gpe3d.create_vortex_ring(
                self.x1,
                self.x2,
                self.x3,
                self.n1,
                self.n2,
                self.n3,
                ring_radius=1.0,
                center=(0.0, 0.0, 0.0),
                axis=4,
            )

    def test_create_vortex_lines_shape_and_finite(self):
        phase = gpe3d.create_vortex_lines(
            self.x1,
            self.x2,
            self.x3,
            self.n1,
            self.n2,
            self.n3,
            positions=[(0.0, 0.0), (1.0, -1.0)],
            charges=[1, -1],
            axis=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(phase.shape, (self.n1, self.n2, self.n3))
        self.assertTrue(torch.all(torch.isfinite(phase)))

    def test_create_vortex_lines_invalid_axis_raises(self):
        with self.assertRaises(ValueError):
            gpe3d.create_vortex_lines(
                self.x1,
                self.x2,
                self.x3,
                self.n1,
                self.n2,
                self.n3,
                positions=[(0.0, 0.0)],
                charges=[1],
                axis=0,
            )

    def test_column_density_for_uniform_state(self):
        psi = torch.ones((self.n1, self.n2, self.n3), dtype=torch.cdouble)
        col1 = gpe3d.column_density(psi, axis=1)
        col2 = gpe3d.column_density(psi, axis=2)
        col3 = gpe3d.column_density(psi, axis=3)

        self.assertEqual(col1.shape, (self.n2, self.n3))
        self.assertEqual(col2.shape, (self.n1, self.n3))
        self.assertEqual(col3.shape, (self.n1, self.n2))
        self.assertTrue(torch.allclose(col1, torch.full_like(col1, self.n1, dtype=torch.float64)))
        self.assertTrue(torch.allclose(col2, torch.full_like(col2, self.n2, dtype=torch.float64)))
        self.assertTrue(torch.allclose(col3, torch.full_like(col3, self.n3, dtype=torch.float64)))

    def test_column_density_invalid_axis_raises(self):
        psi = torch.ones((self.n1, self.n2, self.n3), dtype=torch.cdouble)
        with self.assertRaises(ValueError):
            gpe3d.column_density(psi, axis=5)

    def test_cross_section_plane_default_and_index(self):
        density = torch.arange(self.n1 * self.n2 * self.n3, dtype=torch.float64).reshape(self.n1, self.n2, self.n3)
        psi = torch.sqrt(density).to(torch.cdouble)

        default_slice = gpe3d.cross_section_plane(psi, axis=2)
        explicit_slice = gpe3d.cross_section_plane(psi, axis=2, index=3)

        self.assertEqual(default_slice.shape, (self.n1, self.n3))
        self.assertEqual(explicit_slice.shape, (self.n1, self.n3))
        self.assertTrue(torch.allclose(default_slice, density[:, self.n2 // 2, :]))
        self.assertTrue(torch.allclose(explicit_slice, density[:, 3, :]))

    def test_cross_section_plane_invalid_axis_raises(self):
        psi = torch.ones((self.n1, self.n2, self.n3), dtype=torch.cdouble)
        with self.assertRaises(ValueError):
            gpe3d.cross_section_plane(psi, axis=-1)

    def test_calculate_velocity3d_zero_for_constant_phase(self):
        psi = torch.ones((self.n1, self.n2, self.n3), dtype=torch.cdouble)
        v1, v2, v3 = gpe3d.calculate_velocity3D(psi, self.p_grid)
        self.assertTrue(torch.allclose(v1, torch.zeros_like(v1), atol=1e-12))
        self.assertTrue(torch.allclose(v2, torch.zeros_like(v2), atol=1e-12))
        self.assertTrue(torch.allclose(v3, torch.zeros_like(v3), atol=1e-12))

    def test_calculate_velocity3d_plane_wave_x(self):
        gx, _, _ = self.space_grid
        kx = self.p1[0][1].item()
        psi = torch.exp(1j * kx * gx).to(torch.cdouble)

        v1, v2, v3 = gpe3d.calculate_velocity3D(psi, self.p_grid)
        self.assertAlmostEqual(torch.mean(v1).item(), kx, places=6)
        self.assertAlmostEqual(torch.mean(v2).item(), 0.0, places=6)
        self.assertAlmostEqual(torch.mean(v3).item(), 0.0, places=6)

    def test_angular_momentum_zero_for_real_symmetric_state(self):
        gx, gy, gz = self.space_grid
        psi_real = torch.exp(-(gx**2 + gy**2 + gz**2) / 3.0)
        psi = gpe.normalize(psi_real.to(torch.cdouble), d_x=np.prod(self.dx))

        l1 = gpe3d.angular_momentum(psi, self.space_grid, self.p_grid, component=1)
        l2 = gpe3d.angular_momentum(psi, self.space_grid, self.p_grid, component=2)
        l3 = gpe3d.angular_momentum(psi, self.space_grid, self.p_grid, component=3)

        self.assertAlmostEqual(l1.item(), 0.0, places=6)
        self.assertAlmostEqual(l2.item(), 0.0, places=6)
        self.assertAlmostEqual(l3.item(), 0.0, places=6)

    def test_angular_momentum_invalid_component_raises(self):
        psi = torch.ones((self.n1, self.n2, self.n3), dtype=torch.cdouble)
        with self.assertRaises(ValueError):
            gpe3d.angular_momentum(psi, self.space_grid, self.p_grid, component=0)

class TestDarkSoliton(unittest.TestCase):
    """Tests for dark soliton creation and imprinting."""

    def setUp(self):
        self.n1, self.n2, self.n3 = 64, 8, 64
        self.x1 = torch.linspace(-5.0, 5.0, self.n1, dtype=torch.float64)
        self.x3 = torch.linspace(-5.0, 5.0, self.n3, dtype=torch.float64)
        # uniform unit-amplitude wavefunction
        self.psi = torch.ones(self.n1, self.n2, self.n3, dtype=torch.cdouble)

    def test_black_soliton_density_dip(self):
        """A black soliton (greyness=0) should produce a zero-density line."""
        mask = gpe2d.create_dark_soliton(
            self.x1, self.x3,
            self.n1, self.n2, self.n3,
            positions=[0.0], widths=[0.5], axes=[3],
        )
        psi_s = gpe2d.imprint_dark_soliton(self.psi, mask)
        density = torch.abs(psi_s) ** 2

        # At z=0 (centre of axis 3) density should be ~0
        center_z = self.n3 // 2
        center_density = density[self.n1 // 2, self.n2 // 2, center_z].item()
        self.assertAlmostEqual(center_density, 0.0, places=5)

        # Far from soliton, density should approach 1
        edge_density = density[self.n1 // 2, self.n2 // 2, 0].item()
        self.assertAlmostEqual(edge_density, 1.0, delta=0.05)

    def test_black_soliton_phase_jump(self):
        """A black soliton should impart a pi phase jump across its centre."""
        mask = gpe2d.create_dark_soliton(
            self.x1, self.x3,
            self.n1, self.n2, self.n3,
            positions=[0.0], widths=[0.5], axes=[3],
        )
        psi_s = gpe2d.imprint_dark_soliton(self.psi, mask)

        mid = self.n2 // 2
        center_z = self.n3 // 2
        # Phase on opposite sides of the soliton
        phase_left = torch.angle(psi_s[self.n1 // 2, mid, center_z - 10]).item()
        phase_right = torch.angle(psi_s[self.n1 // 2, mid, center_z + 10]).item()
        phase_diff = abs(phase_left - phase_right)

        self.assertAlmostEqual(phase_diff, np.pi, delta=0.15)

    def test_grey_soliton_shallower_dip(self):
        """A grey soliton should have a shallower density dip than a black one."""
        mask_black = gpe2d.create_dark_soliton(
            self.x1, self.x3,
            self.n1, self.n2, self.n3,
            positions=[0.0], widths=[0.5], axes=[3], greyness=[0.0],
        )
        mask_grey = gpe2d.create_dark_soliton(
            self.x1, self.x3,
            self.n1, self.n2, self.n3,
            positions=[0.0], widths=[0.5], axes=[3], greyness=[0.5],
        )
        psi_black = gpe2d.imprint_dark_soliton(self.psi, mask_black)
        psi_grey = gpe2d.imprint_dark_soliton(self.psi, mask_grey)

        center = (self.n1 // 2, self.n2 // 2, self.n3 // 2)
        dens_black = torch.abs(psi_black[center]) ** 2
        dens_grey = torch.abs(psi_grey[center]) ** 2

        # Grey soliton should have higher density at the dip
        self.assertGreater(dens_grey.item(), dens_black.item())

    def test_soliton_along_axis_1(self):
        """Soliton perpendicular to x1 should leave axis 3 unaffected."""
        mask = gpe2d.create_dark_soliton(
            self.x1, self.x3,
            self.n1, self.n2, self.n3,
            positions=[0.0], widths=[0.5], axes=[1],
        )
        psi_s = gpe2d.imprint_dark_soliton(self.psi, mask)
        density = torch.abs(psi_s) ** 2

        # Density should be uniform along z at the x-centre
        center_x = self.n1 // 2
        line_along_z = density[center_x, self.n2 // 2, :]
        self.assertAlmostEqual(line_along_z.std().item(), 0.0, places=5)

        # But varying along x at a fixed z
        line_along_x = density[:, self.n2 // 2, self.n3 // 2]
        self.assertGreater(line_along_x.std().item(), 0.1)

    def test_multiple_solitons(self):
        """Multiple solitons should produce multiple density dips."""
        mask = gpe2d.create_dark_soliton(
            self.x1, self.x3,
            self.n1, self.n2, self.n3,
            positions=[-2.0, 2.0], widths=[0.5, 0.5], axes=[3, 3],
        )
        psi_s = gpe2d.imprint_dark_soliton(self.psi, mask)
        density = torch.abs(psi_s) ** 2

        mid = self.n2 // 2
        line = density[self.n1 // 2, mid, :]

        # Find positions of minimum density — should have two dips
        # At z ≈ -2 and z ≈ +2
        dz = (self.x3[-1] - self.x3[0]) / (self.n3 - 1)
        idx_neg2 = int((-2.0 - self.x3[0].item()) / dz.item())
        idx_pos2 = int((2.0 - self.x3[0].item()) / dz.item())

        self.assertLess(line[idx_neg2].item(), 0.05)
        self.assertLess(line[idx_pos2].item(), 0.05)
        # Between the two solitons density should recover
        self.assertGreater(line[self.n3 // 2].item(), 0.5)

    def test_mask_shape_and_dtype(self):
        """Mask should have correct shape and complex dtype."""
        mask = gpe2d.create_dark_soliton(
            self.x1, self.x3,
            self.n1, self.n2, self.n3,
            positions=[0.0], widths=[1.0], axes=[3],
        )
        self.assertEqual(mask.shape, (self.n1, self.n2, self.n3))
        self.assertEqual(mask.dtype, torch.cdouble)

    def test_invalid_axis_raises(self):
        """Axis other than 1 or 3 should raise ValueError."""
        with self.assertRaises(ValueError):
            gpe2d.create_dark_soliton(
                self.x1, self.x3,
                self.n1, self.n2, self.n3,
                positions=[0.0], widths=[1.0], axes=[2],
            )

if __name__ == '__main__':
    unittest.main(verbosity=3)