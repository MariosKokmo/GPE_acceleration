import unittest
import numpy as np
import torch
import sys
sys.path.append('.')
from src.library import gpe_library as gpe


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

if __name__ == "__main__":
    unittest.main(verbosity=3)