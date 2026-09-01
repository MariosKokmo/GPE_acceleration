
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
    def find_ground_state(sim_params, system, file_name, device, max_iterations=200000):
        """Compute and persist the stationary ground-state wavefunction.

        The initial state is seeded with a Thomas-Fermi profile and refined via
        imaginary-time steepest descent until the residual norm or relative
        energy change reaches the stopping criterion.

        Parameters
        ----------
        sim_params : dict
            Simulation configuration for the run. Only the interaction strength
            ``u`` is read from here (falling back to the value derived from
            :class:`CONSTANTS` when absent); the grid and potential come from
            ``system``.
        system : object
            Simulation system carrying ``simulation_parameters`` and the
            external potential in ``system.uext.potential``.
        file_name : str
            Output path used by :func:`write_psi` to store the converged state.
        device : str or torch.device
            Device on which the tensors and FFTs are evaluated.
        max_iterations : int
            Hard cap on descent iterations so a state that never meets the
            tolerance cannot spin forever.

        Returns
        -------
        torch.Tensor
            Normalized complex tensor containing the converged ground state on
            ``device``.
        """
        params = system.simulation_parameters
        n1, n2, n3 = params["Grid_resolution"]
        d_x = params["d_x"]
        dx = params["dx"]
        dp = params["dp"]
        x_min = params["x_min"]
        dtau = 0.05*min(dx)**2
        a_ho = params["a_ho"]

        uext = system.uext.potential
        x1, x2, x3, p1, p2, p3, p_sq, _, _ = gpe.init_grid(x_min, dx, dp, n1, n2, n3, device)

        # Prefer the interaction strength the configuration already derived so
        # there is a single source of truth for it.
        u = params.get("u")
        if u is None and isinstance(sim_params, dict):
            u = sim_params.get("u")
        if u is None:
            u = 4.*CONSTANTS.pi * CONSTANTS.nat * CONSTANTS.ascat / a_ho

        # Thomas-Fermi seed  psi = sqrt(max(mu_TF - V, 0)).  Only the real part
        # of the potential is a trapping energy; clamping keeps sqrt() off
        # negative arguments instead of relying on torch.where to discard NaNs.
        mu_TF = 0.5 * (15/(4 * CONSTANTS.pi) * u)**(2/5)
        v_real = uext.real if torch.is_complex(uext) else uext
        psi1 = torch.sqrt(torch.clamp(mu_TF - v_real, min=0.0)).to(torch.cdouble)

        # Weak or absent interactions put mu_TF below the trap minimum, leaving
        # the Thomas-Fermi profile identically zero — normalising that gives a
        # field of NaNs that the descent can never recover from. Fall back to a
        # Gaussian, which is the exact non-interacting ground state of a
        # harmonic trap and a reasonable guess for anything else.
        if not bool(torch.any(psi1.real > 0)):
            psi1 = torch.exp(-(v_real - v_real.min())).to(torch.cdouble)
        psi1 = gpe.normalize(psi1, d_x)

        psi1, energy, tol, mu = GroundState.steepest_descent(psi1, dtau, p_sq, uext, d_x, u)
        print("{iter}\t{energy}\t{mu}\t{dtau:}\t{tol}".format(iter="iter",energy="energy",mu="mu",dtau="dtau",tol="tol"))

        energy_old = energy.real.item()
        iteration = 0
        done = False

        while not done:
            iteration = iteration + 1
            psi1, energy, tol, mu = GroundState.steepest_descent(psi1, dtau, p_sq, uext, d_x, u)
            energy_value = energy.real.item()
            tol_value = tol.real.item()
            if not (np.isfinite(energy_value) and np.isfinite(tol_value)):
                # Every comparison against NaN is False, so without this the
                # loop would silently run to max_iterations.
                raise FloatingPointError(
                    f"Ground state diverged at iteration {iteration}: "
                    f"energy={energy_value}, residual={tol_value}"
                )
            # Relative change, guarded against a vanishing energy scale.
            scale = abs(energy_value) if energy_value != 0.0 else 1.0
            test_e = (energy_old - energy_value) / scale

            if test_e < 0:
                print(f"Changing dtau : {dtau}, {test_e}")
                dtau = dtau/2
            energy_old = energy_value

            if iteration % 50 == 0:
                mu_value = mu.real.item()
                print(f"{iteration:10}\t{energy_value:10}\t{mu_value:10}\t{dtau:10}\t{tol_value:10}")
            if (tol_value < 1e-5) or (test_e == 0):
                done = True
            elif iteration >= max_iterations:
                print(
                    f"Ground state did not converge in {max_iterations} iterations "
                    f"(residual {tol_value:.3e}, target 1e-5). Writing the current state."
                )
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
        # H is Hermitian so mu is real; take the real part rather than the
        # modulus, which would silently flip the sign of a negative chemical
        # potential and drive the descent the wrong way.
        mu = mu.real

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

        Each row of the file holds ``(real_part,imag_part)`` for one grid point,
        in row-major order over (n1, n2, n3) — the format written by
        :func:`write_psi`.

        Parameters
        ----------
        data : str or path-like
            Text file containing the complex wavefunction values.
        n1, n2, n3 : int
            Grid resolution used to reshape the flattened data back into the
            simulation domain.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``(n1, n2, n3)`` on the CPU.

        Raises
        ------
        ValueError
            If the file does not hold exactly ``n1*n2*n3`` points.
        """
        matrix = pd.read_csv(data, header=None, names=['real', 'imag'])
        matrix['real'] = matrix['real'].str.strip(' (')
        matrix['imag'] = matrix['imag'].str.strip(' )')
        matrix = matrix.astype(np.float64)

        expected = n1 * n2 * n3
        if len(matrix) != expected:
            raise ValueError(
                f"Ground state file '{data}' holds {len(matrix)} points but the "
                f"{n1}x{n2}x{n3} grid needs {expected}. It was probably written "
                f"for a different grid resolution."
            )

        psi1 = matrix.iloc[:,0].values + matrix.iloc[:,1].values*1j
        psi1 = psi1.reshape((n1,n2,n3))
        psi1 = torch.from_numpy(psi1)

        return psi1
