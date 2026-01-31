import unittest
import numpy as np
import torch
import sys
sys.path.append('.')
from src.library.gpe_library import GPELibrary as gpe
from src.library.gpe_library import GPE2DLibrary as gpe2d


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


if __name__ == '__main__':
    unittest.main(verbosity=3)