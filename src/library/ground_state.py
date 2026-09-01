r"""
Ground-state solver for the GPE on a Cartesian grid.

The stationary state is found by imaginary-time propagation with a
steepest-descent (gradient-flow) update. Writing the mean-field Hamiltonian in
dimensionless units (:math:`\hbar = m = \omega_\mathrm{ho} = 1`) as

.. math::

    H[\psi] = -\frac{\nabla^2}{2} + V_\mathrm{ext}(\mathbf{r})
              + u \lvert \psi \rvert^{2},

each iteration takes one step along the residual and renormalises,

.. math::

    \psi \leftarrow \mathcal{N}\Bigl[
        \psi - \Delta\tau\,\bigl(H[\psi] - \mu\bigr)\psi
    \Bigr],
    \qquad
    \mu = \langle \psi \lvert H \rvert \psi \rangle .

Projecting out :math:`\mu` at every step is what keeps the flow inside the
normalised manifold; the norm of the residual
:math:`\lVert (H - \mu)\psi \rVert^{2}` is what the loop watches for
convergence, since it vanishes exactly when :math:`\psi` is an eigenstate.

All heavy tensor operations go through PyTorch, so the solver runs on CPU or
GPU depending on the device of the supplied tensors.
"""
import numpy as np
import pandas as pd
import torch
from src.library.gpe_library import GPELibrary as gpe
from src.library.parameters import CONSTANTS
from src.utils.read_write_utils import write_psi

class GroundState:
    r"""
    Imaginary-time ground-state solver on a Cartesian grid.

    A namespace of static methods: :meth:`find_ground_state` drives the
    descent and writes the result to disk, :meth:`steepest_descent` performs a
    single iteration, and :meth:`read_ground_state` loads a state back from a
    file written by a previous run.
    """

    @staticmethod
    def find_ground_state(sim_params, system, file_name, device, max_iterations=200000):
        r"""
        Compute and persist the stationary ground-state wavefunction.

        The initial state is seeded with a Thomas-Fermi profile,

        .. math::

            \psi_\mathrm{TF} = \sqrt{\max(\mu_\mathrm{TF} - V_\mathrm{ext}, 0)},
            \qquad
            \mu_\mathrm{TF} = \frac{1}{2}
                \left(\frac{15\,u}{4\pi}\right)^{2/5},

        and refined by :meth:`steepest_descent` until the residual norm drops
        below ``1e-5`` or the relative energy change vanishes. The imaginary
        time step is set from the grid spacing as
        :math:`\Delta\tau = 0.05\,(\min \mathrm{d}x)^2`, and is halved whenever
        the energy goes *up*, which is the signature of a step that is too
        large for the current profile.

        Note:
            Weak or absent interactions put :math:`\mu_\mathrm{TF}` below the
            trap minimum, which makes the Thomas-Fermi profile identically
            zero and its normalisation a field of NaNs. The seed then falls
            back to a Gaussian, the exact non-interacting ground state of a
            harmonic trap.

        Args:
            sim_params (dict): Simulation configuration for the run. Only the
                interaction strength ``"u"`` is read from here, and only as a
                fallback; the grid and the potential come from ``system``.
            system: Simulation system carrying ``simulation_parameters`` and
                the external potential in ``system.uext.potential``.
            file_name (str): Output path, passed to
                :func:`~src.utils.read_write_utils.write_psi` to store the
                converged state.
            device (str or torch.device): Device on which the tensors and FFTs
                are evaluated.
            max_iterations (int, optional): Hard cap on descent iterations, so
                that a state which never meets the tolerance cannot spin
                forever (default ``200000``). On reaching it the current state
                is written anyway, with a message.

        Returns:
            torch.Tensor: Normalised complex wavefunction of shape
            ``Grid_resolution``, on ``device``.

        Raises:
            FloatingPointError: If the energy or the residual stops being
                finite, i.e. the descent has diverged. Without this check
                every comparison against NaN would be false and the loop would
                silently run to ``max_iterations``.
            KeyError: If ``system.simulation_parameters`` is missing a grid
                key.
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
        r"""
        Advance the imaginary-time solver by one steepest-descent step.

        The kinetic term is applied spectrally and the rest in real space,

        .. math::

            H\psi = \mathcal{F}^{-1}\!\left[\tfrac{1}{2} p^2\,
                        \mathcal{F}[\psi]\right]
                    + \left(u \lvert \psi \rvert^{2}
                        + V_\mathrm{ext}\right)\psi ,

        from which the step follows as

        .. math::

            \mu = \mathrm{d}V \sum \psi^{*} H\psi,
            \qquad
            \psi \leftarrow \mathcal{N}\!\left[
                \psi - \Delta\tau\,(H - \mu)\psi\right].

        The diagnostics returned alongside are the residual norm and the
        energy,

        .. math::

            \mathrm{tol} = \mathrm{d}V \sum \lvert (H - \mu)\psi \rvert^{2},
            \qquad
            E = \mu - \frac{u}{2}\,\mathrm{d}V \sum \lvert \psi \rvert^{4},

        the energy being the chemical potential minus the interaction energy,
        which is counted twice in :math:`\mu`.

        Note:
            :math:`H` is Hermitian, so :math:`\mu` is real and its real part is
            taken directly. Taking the modulus instead would silently flip the
            sign of a negative chemical potential and drive the descent the
            wrong way.

        Args:
            psi (torch.Tensor): Current condensate wavefunction. The tensor
                stays on its current device, so passing a CUDA tensor keeps the
                whole update on the GPU.
            dtau (float): Imaginary-time step size :math:`\Delta\tau`.
            p_sq (torch.Tensor): Squared momentum grid :math:`p^2`, used to
                apply the kinetic operator in Fourier space.
            uext (torch.Tensor): External trapping potential sampled on the
                spatial grid.
            d_x (float): Volume element, used for the normalisation and the
                expectation values.
            u (float): Contact-interaction strength :math:`u`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            The updated normalised wavefunction, the total energy estimate
            :math:`E`, the residual norm used as the convergence metric, and
            the chemical potential :math:`\mu`. The three scalar diagnostics
            are zero-dimensional tensors on the same device as ``psi``, so the
            caller decides when to synchronise with the host.
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
        r"""
        Load a serialised ground-state wavefunction from disk.

        Each row of the file holds ``(real_part,imag_part)`` for one grid
        point, in row-major order over ``(n1, n2, n3)`` — the format written
        by :func:`~src.utils.read_write_utils.write_psi`.

        Args:
            data (str or os.PathLike): Text file containing the complex
                wavefunction values.
            n1 (int): Number of grid points along the first axis.
            n2 (int): Number of grid points along the second axis.
            n3 (int): Number of grid points along the third axis.

        Returns:
            torch.Tensor: Complex tensor of shape ``(n1, n2, n3)``, on the CPU.
            Move it to the simulation device before use.

        Raises:
            ValueError: If the file does not hold exactly ``n1*n2*n3`` points,
                which normally means it was written for a different grid
                resolution.
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
