
"""Utilities for computing the condensate ground state.

The implementation uses imaginary-time propagation with a steepest-descent
update. All heavy tensor operations are performed with PyTorch so the solver
can run on either CPU or GPU depending on the supplied device and tensors.
"""
import numpy as np
import pandas as pd
import torch
from src.library.gpe_library import GPELibrary as gpe
from src.library.parameters import CONSTANTS
from src.utils.read_write_utils import write_psi

class GroundState:
    @staticmethod
    def find_ground_state(sim_params, system, file_name, device):
        """Compute and persist the stationary ground-state wavefunction.

        The initial state is seeded with a Thomas-Fermi profile and refined via
        imaginary-time steepest descent until the residual norm or relative
        energy change reaches the stopping criterion.

        Parameters
        ----------
        sim_params : dict
            Simulation configuration for the run. The current implementation
            reads the numerical grid and potential information from ``system``.
        system : object
            Simulation system carrying ``simulation_parameters`` and the
            external potential in ``system.uext.potential``.
        file_name : str
            Output path used by :func:`write_psi` to store the converged state.
        device : str or torch.device
            Device on which the tensors and FFTs are evaluated.

        Returns
        -------
        torch.Tensor
            Normalized complex tensor containing the converged ground state on
            ``device``.
        """
        n1, n2, n3 = system.simulation_parameters["Grid_resolution"]
        device = device
        d_x = system.simulation_parameters["d_x"]
        dx = system.simulation_parameters["dx"]
        dp = system.simulation_parameters["dp"]
        w = system.simulation_parameters["w"]
        x_min = system.simulation_parameters["x_min"]
        x_max = system.simulation_parameters["x_max"]
        dtau = 0.05*min(dx)**2
        a_ho = system.simulation_parameters["a_ho"]

        uext = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=device)
        psi1 = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=device)

        uext = system.uext.potential
        x1, x2, x3, p1, p2, p3, p_sq, _, _ = gpe.init_grid(x_min, x_max, dx, dp, w, n1, n2, n3, device)
        energy = 0
        energy_old = 0
        iter = 0
        mu = 0
        tol = 0
        done = False

        u = 4.*CONSTANTS.pi * CONSTANTS.nat * CONSTANTS.ascat/ a_ho
        mu_TF = 0.5 * (15/(4 * CONSTANTS.pi) * u)**(2/5)

        psi1 = torch.where(mu_TF > uext, torch.sqrt(mu_TF - uext) + 0j, psi1)
        psi1 = gpe.normalize(psi1, d_x)

        psi1, energy, tol, mu = GroundState.steepest_descent(psi1, dtau, p_sq, uext, d_x, u)
        print("{iter}\t{energy}\t{mu}\t{dtau:}\t{tol}".format(iter="iter",energy="energy",mu="mu",dtau="dtau",tol="tol"))

        energy_old = energy.real.item()

        while not done:
            iter = iter + 1
            psi1, energy, tol, mu = GroundState.steepest_descent(psi1, dtau, p_sq, uext, d_x, u)
            energy_value = energy.real.item()
            tol_value = tol.real.item()
            test_e = (energy_old - energy_value) / energy_value

            if test_e < 0:
                print(f"Changing dtau : {dtau}, {test_e}")
                dtau = dtau/2
            energy_old = energy_value

            if iter%50 == 0:
                mu_value = mu.real.item()
                print(f"{iter:10}\t{energy_value:10}\t{mu_value:10}\t{dtau:10}\t{tol_value:10}")
            if (tol_value < 1e-5) or (test_e == 0):
                done = True
        write_psi(file_name, psi1, n1, n2, n3)
        return psi1

    @staticmethod
    def steepest_descent(psi, dtau, p_sq, uext, d_x, u):
        """Advance the imaginary-time solver by one steepest-descent step.

        Parameters
        ----------
        psi : torch.Tensor
            Current condensate wavefunction. The tensor remains on its current
            device, so passing a CUDA tensor keeps the full update on the GPU.
        dtau : float
            Imaginary-time step size.
        p_sq : torch.Tensor
            Squared momentum grid used to apply the kinetic-energy operator in
            Fourier space.
        uext : torch.Tensor
            External trapping potential sampled on the spatial grid.
        d_x : float
            Volume element used for normalization and expectation values.
        u : float
            Contact-interaction strength.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Updated normalized wavefunction, total energy estimate, residual
            norm used as a convergence metric, and chemical potential. The
            scalar diagnostics are returned as zero-dimensional tensors on the
            same device as ``psi`` so callers can decide when to synchronize
            with the host.
        """
        dpsi = psi
        psiF = torch.fft.fftn(dpsi, norm='forward')
        psiF =  0.5 * p_sq * psiF
        dpsi = torch.fft.ifftn(psiF, norm='forward')

        utot = u*torch.abs(psi)**2 + uext
        dpsi = dpsi + utot*psi

        psi_conj = torch.conj(psi)
        mu = d_x * torch.sum(psi_conj*dpsi)
        mu = mu.abs()

        interaction = 0.5 * u * d_x * torch.sum(torch.abs(psi)**4)
        dpsi = dpsi - mu * psi
        tol = d_x * torch.sum(torch.abs(dpsi)**2)

        energy = mu - interaction
        psi = psi - dtau * dpsi
        psi = gpe.normalize(psi, d_x)

        return psi, energy, tol, mu

    @staticmethod
    def read_ground_state(data, n1, n2, n3):
        """Load a serialized ground-state wavefunction from disk.

        Parameters
        ----------
        data : str or path-like
            Text file containing the complex wavefunction values written as
            modulus/phase columns.
        n1, n2, n3 : int
            Grid resolution used to reshape the flattened data back into the
            simulation domain.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``(n1, n2, n3)`` on the CPU.
        """
        matrix = pd.read_csv(data, header=None, names=['modulus', 'phase'])
        matrix.modulus = matrix.modulus.str.strip(' (')
        matrix.phase = matrix.phase.str.strip(' )')
        matrix = matrix.astype(np.float64)

        psi1 = matrix.iloc[:,0] + matrix.iloc[:,1]*1j
        psi1 = psi1.values
        psi1 = psi1.reshape((n1,n2,n3))
        psi1 = torch.from_numpy(psi1)

        return psi1
