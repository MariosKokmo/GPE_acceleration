import unittest
import numpy as np
import torch

import src.library.gpe_library as gpe

class TestModGradPsi(unittest.TestCase):
    """Tests for the function that returns the modulus of the gradient"""
    @classmethod
    def setUpClass(self):
        """Creates the tensors"""
        N = 256
        x = torch.linspace(-200,200,N)
        y = torch.linspace(-200,200,N)
        z = torch.linspace(-200,200,N)
        self.grid_1D = x
        xv, yv = torch.meshgrid(x, y)
        self.grid_2D  = torch.stack((xv, yv)) # shape 2, N, N
        xxv, yyv, zzv = torch.meshgrid(x, y, z)
        self.grid_3D  = torch.stack((xxv, yyv, zzv)) # shape 3, N, N, N

        # for each testcase the first element is input and the second is the expected result
        self.testcases = {
            ##################### 1D test cases ############################
            'tensor_1d_flat' : (lambda : torch.ones_like(self.grid_1D),
                                lambda : torch.zeros_like(self.grid_1D)),
            # avoid Gibbs oscillations at the ends
            'tensor_1d' : (
                            lambda : torch.sin(self.grid_1D),
                            lambda : torch.cos(self.grid_1D),
                            ),

            'gaussian_1D' : (
                             lambda : torch.exp(-(self.grid_1D**2)/10),
                             lambda : -self.grid_1D/5 * torch.exp(-(self.grid_1D**2)/10)
                             ),

            'sinExp_1d' : (lambda : torch.sin(self.grid_1D)*torch.exp(-self.grid_1D**2/100),
                        lambda : (torch.cos(self.grid_1D)-2*self.grid_1D*torch.sin(self.grid_1D)/100)*torch.exp(-self.grid_1D**2/100)),

            ##################### 2D test cases ############################
            'tensor_2d_flat' : (lambda : torch.ones(self.grid_2D.shape[1:-1]),
                                lambda : torch.zeros(self.grid_2D.shape[1:-1])
                                ),

            'gaussian_2D' : (lambda : torch.exp(-(self.grid_2D[0]**2 + self.grid_2D[1]**2)/200),
                             lambda : torch.sqrt(self.grid_2D[0]**2 + self.grid_2D[1]**2)/100 * torch.exp(-(self.grid_2D[0]**2 + self.grid_2D[1]**2)/200)),
            
            ##################### 3D test cases ############################
            # for the flat we only want to keep the dimensions except the first
            'tensor_3d_flat' : (lambda : torch.ones(self.grid_3D.shape[1:-1]),
                                lambda : torch.zeros(self.grid_3D.shape[1:-1])
                                ),

            'gaussian_3D' : (lambda : torch.exp(-(self.grid_3D[0]**2 + self.grid_3D[1]**2 + self.grid_3D[2]**2)/200),
                             lambda : torch.sqrt(self.grid_3D[0]**2 + self.grid_3D[1]**2 + self.grid_3D[2]**2)/100 * torch.exp(-(self.grid_3D[0]**2 + self.grid_3D[1]**2 + self.grid_3D[2]**2)/200)
                             ),
        }

    @classmethod
    def tearDownClass(self):
        del self.testcases

    def setup_momentum_grid(self, dim, xmax, xmin, n):
        assert len(xmax) == len(xmin)
        assert len(n) == dim
        p = []
        for d in range(dim):
            dp = (2*np.pi/(xmax[d]-xmin[d]))
            p1 = torch.fft.fftfreq(n[d], d=1/n[d]) *dp
            #p1 = torch.arange(-n[d]//2, n[d]//2, dtype=torch.float64) * dp
            #p.append(torch.fft.fftshift(p1))
            p.append(p1)
        return p

    def setup_grid(self, dim):
        """Return lists of maximum, minimum and length for every axis of the grid"""
        grids = {1: self.grid_1D, # shape [512]
                 2: self.grid_2D, # shape [2, 512, 512]
                 3: self.grid_3D, # shaep [3, 512, 512, 512]
                 }
        grid = grids[dim]
        shape = grid.shape
        if dim == 1:
            xmax = [grid.max().item()]
            xmin = [grid.min().item()]
            n = [shape[0]]
        else:
            xmax = [grid[d].max().item() for d in range(dim)]
            xmin = [grid[d].min().item() for d in range(dim)]
            n = [shape[d] for d in range(1,dim+1)] # number of points in each axis
        return xmax, xmin, n
    
    def run_testcases(self):
        for name, testcase in self.testcases.items():
            self.setup_and_run_testcase(testcase, name)
    
    def setup_and_run_testcase(self, testcase, name):
        input = testcase[0]()
        dim = len(input.shape)
        xmax, xmin, n = self.setup_grid(dim=dim)
        p = self.setup_momentum_grid(dim=dim, xmax=xmax, xmin=xmin, n=n)
        grad = gpe.mod_grad_psi(input, p)
        result = testcase[1]()
        if  'flat' in name:
            error = np.linalg.norm(grad[n[0]//3:-n[0]//3] - result[n[0]//3:-n[0]//3])
            #self.assertLessEqual(error, 1, msg=f"the error is not small enough")
            if error < 0.01:
                print("{} ---- OK - error {}".format(name, error))
            else:
                print("{} ----- Failed - error {}".format(name, error))
        else:
            if dim == 1:
                # check far from Gibbs oscillations
                error = np.linalg.norm(grad[n[0]//3:-n[0]//3] - result[n[0]//3:-n[0]//3]) /np.linalg.norm(result[n[0]//3:-n[0]//3])
            else:    
                error = np.linalg.norm(grad - result) / np.linalg.norm(result)
            if error < 0.01:
                print("{} ---- OK - error {}".format(name, error))
            else:
                print("{} ----- Failed - error {}".format(name, error))
            #self.assertLessEqual(error, 1, msg=f"the error is not small enough")
        del grad
        del result

    def test_all(self):
        self.run_testcases()
    

if __name__ == "__main__":
    unittest.main(verbosity=2)