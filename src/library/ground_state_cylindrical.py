r"""
Ground-state solver for the GPE in cylindrical coordinates :math:`(r, \varphi, z)`.

Drop-in cylindrical counterpart to :mod:`src.library.ground_state`: same
imaginary-time steepest descent, same convergence criteria, same file format,
with two substitutions that follow from the geometry.

Every inner product and norm uses the cylindrical volume element

.. math::

    \mathrm{d}V = r\,\mathrm{d}r\,\mathrm{d}\varphi\,\mathrm{d}z ,

and the kinetic operator is applied spectrally in :math:`\varphi` and
:math:`z` but through a precomputed radial eigendecomposition in :math:`r`,
since the radial Laplacian is not diagonal in any Fourier basis (see
:class:`~src.library.gpe_cylindrical_library.GPECylindricalLibrary`). The
update rule itself is unchanged,

.. math::

    \psi \leftarrow \mathcal{N}\Bigl[
        \psi - \Delta\tau\,\bigl(H[\psi] - \mu\bigr)\psi
    \Bigr],
    \qquad
    \mu = \langle \psi \lvert H \rvert \psi \rangle .

Expected ``system.simulation_parameters`` keys
----------------------------------------------

``Grid_resolution``
    Tuple ``(n_r, n_phi, n_z)`` — number of grid points along each direction.
``r_max``
    Outer radial boundary, in the same units as ``a_ho``.
``z_min``, ``z_max``
    Axial extent of the box.
``a_ho``
    Harmonic oscillator length, in metres — used to derive :math:`u` when the
    configuration does not already supply it.

The cylindrical grid (:math:`r`, :math:`\varphi`, :math:`z`, :math:`k_z`, the
azimuthal modes and the spacings) and the radial eigendecomposition are both
built inside :meth:`CylindricalGroundState.find_ground_state` from those
parameters, so the caller has nothing to precompute.
"""

import math
import numpy as np
import pandas as pd
import torch
from src.library.gpe_cylindrical_library import GPECylindricalLibrary as cyl
from src.library.parameters import CONSTANTS
from src.utils.read_write_utils import write_psi


class CylindricalGroundState:
    r"""
    Imaginary-time ground-state solver on a cylindrical grid.

    A namespace of static methods mirroring
    :class:`~src.library.ground_state.GroundState`:
    :meth:`find_ground_state` drives the descent and writes the result to disk,
    :meth:`steepest_descent` performs a single iteration, :meth:`apply_kinetic`
    supplies the cylindrical kinetic operator that replaces the Cartesian FFT
    step, and :meth:`read_ground_state` loads a state back from a file.
    """

    # ------------------------------------------------------------------
    # Kinetic operator action
    # ------------------------------------------------------------------

    @staticmethod
    def apply_kinetic(
        psi: torch.Tensor,
        kz: torch.Tensor,
        m_modes: torch.Tensor,
        r: torch.Tensor,
        eigvecs_dict: dict,
        eigvals_dict: dict,
    ) -> torch.Tensor:
        r"""
        Apply the kinetic operator :math:`T` to a wavefunction in cylindrical
        coordinates.

        The azimuthal direction is periodic, so a DFT over :math:`\varphi`
        block-diagonalises the Laplacian: each azimuthal mode :math:`m` sees
        its own radial operator, and the kinetic term splits as

        .. math::

            T\psi = T_r^{m}\psi + T_z\psi .

        The radial part is applied through the eigendecomposition precomputed
        by
        :meth:`~src.library.gpe_cylindrical_library.GPECylindricalLibrary.build_radial_operators`,
        with the :math:`\sqrt{r}` similarity transform that makes the radial
        operator symmetric,

        .. math::

            T_r^{m}\psi_m = \frac{1}{\sqrt{r}}\,
                V_m \Lambda_m V_m^{\mathsf{T}}
                \left(\sqrt{r}\,\psi_m\right),

        while the axial part stays spectral,

        .. math::

            T_z \psi_m = \mathcal{F}^{-1}\!\left[
                \tfrac{1}{2} k_z^2\, \mathcal{F}[\psi_m]\right].

        Together these replace the single Cartesian step
        :math:`\mathcal{F}^{-1}[\tfrac{1}{2}p^2 \mathcal{F}[\psi]]`.

        Args:
            psi (torch.Tensor): Complex wavefunction of shape
                ``(n_r, n_phi, n_z)``.
            kz (torch.Tensor): Axial momentum grid, of shape ``(n_z,)``.
            m_modes (torch.Tensor): Azimuthal mode indices in DFT order, of
                shape ``(n_phi,)``.
            r (torch.Tensor): Radial grid, of shape ``(n_r,)``.
            eigvecs_dict (dict): Eigenvectors of the radial operator, keyed by
                the absolute azimuthal index and of shape ``(n_r, n_r)`` each,
                as returned by ``build_radial_operators``.
            eigvals_dict (dict): Matching eigenvalues, keyed the same way and
                of shape ``(n_r,)`` each.

        Returns:
            torch.Tensor: The kinetic term :math:`T\psi`, of shape
            ``(n_r, n_phi, n_z)``.
        """
        n_r, n_phi, n_z = psi.shape
        sqrt_r = torch.sqrt(r).to(device=psi.device, dtype=torch.float64)

        # 1. DFT over φ → azimuthal modes
        psi_m = torch.fft.fft(psi, dim=1, norm="ortho")

        # 2. Radial kinetic for every azimuthal mode in one batched matmul
        #    (see GPECylindricalLibrary._apply_radial_eigen); the diagonal
        #    factor is Λ itself because we apply T rather than exp(-i Δτ T).
        V_all, lam_all = cyl._stacked_radial(eigvecs_dict, eigvals_dict, m_modes)
        Tr_psi = cyl._apply_radial_eigen(
            psi_m, lam_all.to(torch.cdouble).unsqueeze(-1), V_all, sqrt_r
        )

        # 3. Axial kinetic: spectral derivative over the whole array at once
        psi_kz = torch.fft.fft(psi_m, dim=2, norm="forward")
        kz2 = (kz.reshape(1, 1, n_z) ** 2).to(torch.cdouble)
        Tz_psi = torch.fft.ifft(0.5 * kz2 * psi_kz, dim=2, norm="forward")

        # 4. IDFT over φ → physical space
        return torch.fft.ifft(Tr_psi + Tz_psi, dim=1, norm="ortho")

    # ------------------------------------------------------------------
    # Imaginary-time step
    # ------------------------------------------------------------------

    @staticmethod
    def steepest_descent(
        psi: torch.Tensor,
        dtau: float,
        kz: torch.Tensor,
        m_modes: torch.Tensor,
        r: torch.Tensor,
        eigvecs_dict: dict,
        eigvals_dict: dict,
        uext: torch.Tensor,
        dr: float,
        dphi: float,
        dz: float,
        u: float,
    ) -> tuple:
        r"""
        Advance the imaginary-time solver by one steepest-descent step.

        Mirrors :meth:`~src.library.ground_state.GroundState.steepest_descent`
        exactly, with two substitutions: the Cartesian kinetic step
        :math:`\mathcal{F}^{-1}[\tfrac{1}{2}p^2\mathcal{F}[\psi]]` becomes
        :meth:`apply_kinetic`, and the flat volume element becomes
        :math:`r\,\mathrm{d}r\,\mathrm{d}\varphi\,\mathrm{d}z` in every inner
        product. The step itself is

        .. math::

            \mu = \int \psi^{*} H\psi\; r\,\mathrm{d}r\,\mathrm{d}\varphi\,
                       \mathrm{d}z,
            \qquad
            \psi \leftarrow \mathcal{N}\!\left[
                \psi - \Delta\tau\,(H - \mu)\psi\right],

        with the diagnostics

        .. math::

            \mathrm{tol} = \int \lvert (H - \mu)\psi \rvert^{2}\,
                r\,\mathrm{d}r\,\mathrm{d}\varphi\,\mathrm{d}z,
            \qquad
            E = \mu - \frac{u}{2} \int \lvert \psi \rvert^{4}\,
                r\,\mathrm{d}r\,\mathrm{d}\varphi\,\mathrm{d}z .

        Note:
            :math:`H` is Hermitian, so :math:`\mu` is real and its real part is
            taken directly. Taking the modulus instead would silently flip the
            sign of a negative chemical potential and drive the descent the
            wrong way.

        Args:
            psi (torch.Tensor): Current wavefunction, of shape
                ``(n_r, n_phi, n_z)``.
            dtau (float): Imaginary-time step :math:`\Delta\tau`.
            kz (torch.Tensor): Axial momentum grid.
            m_modes (torch.Tensor): Azimuthal mode indices in DFT order.
            r (torch.Tensor): Radial grid.
            eigvecs_dict (dict): Eigenvectors of the radial operator, from
                ``build_radial_operators``.
            eigvals_dict (dict): Matching eigenvalues.
            uext (torch.Tensor): External trapping potential on the cylindrical
                grid.
            dr (float): Radial grid spacing.
            dphi (float): Azimuthal grid spacing.
            dz (float): Axial grid spacing.
            u (float): Contact-interaction strength :math:`u`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            The updated normalised wavefunction, the total energy estimate
            :math:`E`, the residual norm used as the convergence metric, and
            the chemical potential :math:`\mu`. The three scalar diagnostics
            are zero-dimensional tensors on the same device as ``psi``.
        """
        r_w = r.reshape(-1, 1, 1)
        dV = dr * dphi * dz

        # H|ψ⟩ = T|ψ⟩ + V_tot|ψ⟩
        dpsi = CylindricalGroundState.apply_kinetic(
            psi, kz, m_modes, r, eigvecs_dict, eigvals_dict
        )
        utot = u * torch.abs(psi) ** 2 + uext
        dpsi = dpsi + utot * psi

        # μ = ⟨ψ|H|ψ⟩  (cylindrical inner product).  H is Hermitian so μ is
        # real; take the real part rather than the modulus, which would
        # silently flip the sign of a negative μ and drive the descent the
        # wrong way.
        mu = torch.sum(psi.conj() * dpsi * r_w) * dV
        mu = mu.real

        # Interaction energy  e_int = (u/2) ∫|ψ|⁴ r dr dφ dz
        interaction = 0.5 * u * torch.sum(torch.abs(psi) ** 4 * r_w) * dV

        # Residual  (H − μ)|ψ⟩
        dpsi = dpsi - mu * psi
        tol = torch.sum(torch.abs(dpsi) ** 2 * r_w) * dV

        # Energy estimate: E = μ − e_int
        energy = mu - interaction

        # Gradient-flow update
        psi = psi - dtau * dpsi
        psi = cyl.normalize(psi, r, dr, dphi, dz)

        return psi, energy, tol, mu

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    @staticmethod
    def find_ground_state(sim_params, system, file_name, device, max_iterations=200000):
        r"""
        Compute and persist the cylindrical ground-state wavefunction.

        Builds the cylindrical grid and the radial operators, seeds the state
        with a Thomas-Fermi profile,

        .. math::

            \psi_\mathrm{TF} = \sqrt{\max(\mu_\mathrm{TF} - V_\mathrm{ext}, 0)},
            \qquad
            \mu_\mathrm{TF} = \frac{1}{2}
                \left(\frac{15\,u}{4\pi}\right)^{2/5},

        then refines it with :meth:`steepest_descent` until the residual norm
        drops below ``1e-5`` or the relative energy change vanishes. Stability
        of the explicit gradient flow requires a step well below the square of
        the smallest spacing, hence
        :math:`\Delta\tau = 0.05\,\min(\mathrm{d}r, \mathrm{d}z)^2`, halved
        whenever the energy goes *up*.

        The required ``system.simulation_parameters`` keys are listed in the
        module docstring.

        Note:
            Weak or absent interactions put :math:`\mu_\mathrm{TF}` below the
            trap minimum, which makes the Thomas-Fermi profile identically
            zero and its normalisation a field of NaNs. The seed then falls
            back to a Gaussian, the exact non-interacting ground state of a
            harmonic trap.

        Args:
            sim_params (dict): Simulation configuration. Only the interaction
                strength ``"u"`` is read from here, and only as a fallback
                behind ``system.simulation_parameters``; failing both, it is
                derived from :class:`~src.library.parameters.CONSTANTS`.
            system: System object with a ``simulation_parameters`` dict and the
                external potential in ``system.uext.potential``, sampled on the
                ``(n_r, n_phi, n_z)`` cylindrical grid.
            file_name (str): Output path for the serialised ground state.
            device (torch.device): Device on which the grid, the operators and
                the descent are evaluated.
            max_iterations (int, optional): Hard cap on descent iterations, so
                that a state which never meets the tolerance cannot spin
                forever (default ``200000``). On reaching it the current state
                is written anyway, with a message.

        Returns:
            torch.Tensor: Normalised complex wavefunction of shape
            ``(n_r, n_phi, n_z)``.

        Raises:
            FloatingPointError: If the energy or the residual stops being
                finite, i.e. the descent has diverged. Without this check
                every comparison against NaN would be false and the loop would
                silently run to ``max_iterations``.
            KeyError: If ``system.simulation_parameters`` is missing a grid
                key.
        """
        params = system.simulation_parameters
        n_r, n_phi, n_z = params["Grid_resolution"]
        r_max = params["r_max"]
        z_min = params["z_min"]
        z_max = params["z_max"]
        a_ho = params["a_ho"]

        # Build cylindrical grid and radial operators
        r, phi, z, kz, m_modes, dr, dphi, dz, _ = cyl.init_grid(
            r_max, z_min, z_max, n_r, n_phi, n_z, device
        )
        eigvecs_dict, eigvals_dict = cyl.build_radial_operators(r, dr, m_modes, device)

        uext = system.uext.potential

        # Dimensionless interaction strength — prefer the value the
        # configuration already derived so there is one source of truth.
        u = params.get("u")
        if u is None and isinstance(sim_params, dict):
            u = sim_params.get("u")
        if u is None:
            u = 4.0 * math.pi * CONSTANTS.nat * CONSTANTS.ascat / a_ho

        # Imaginary-time step: stability requires Δτ ≪ (min grid spacing)²
        dtau = 0.05 * min(dr, dz) ** 2

        # Thomas-Fermi seed: ψ_TF = sqrt(max(μ_TF − V_ext, 0))
        mu_TF = 0.5 * (15.0 / (4.0 * math.pi) * u) ** (2.0 / 5.0)
        v_real = uext.real if torch.is_complex(uext) else uext
        psi = torch.sqrt(torch.clamp(mu_TF - v_real, min=0.0)).to(torch.cdouble)

        # Weak or absent interactions put μ_TF below the trap minimum, leaving
        # the Thomas-Fermi profile identically zero — normalising that gives a
        # field of NaNs the descent can never recover from. Fall back to a
        # Gaussian, the exact non-interacting ground state of a harmonic trap.
        if not bool(torch.any(psi.real > 0)):
            psi = torch.exp(-(v_real - v_real.min())).to(torch.cdouble)
        psi = cyl.normalize(psi, r, dr, dphi, dz)

        # First step — initialise convergence state
        psi, energy, tol, mu = CylindricalGroundState.steepest_descent(
            psi, dtau, kz, m_modes, r, eigvecs_dict, eigvals_dict,
            uext, dr, dphi, dz, u
        )
        print(
            "{iter}\t{energy}\t{mu}\t{dtau}\t{tol}".format(
                iter="iter", energy="energy", mu="mu", dtau="dtau", tol="tol"
            )
        )
        energy_old = energy.real.item()

        done = False
        iteration = 0
        while not done:
            iteration += 1
            psi, energy, tol, mu = CylindricalGroundState.steepest_descent(
                psi, dtau, kz, m_modes, r, eigvecs_dict, eigvals_dict,
                uext, dr, dphi, dz, u
            )
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
                print(f"Changing dtau: {dtau}, relative change: {test_e:.3e}")
                dtau = dtau / 2.0
            energy_old = energy_value

            if iteration % 50 == 0:
                mu_value = mu.real.item()
                print(
                    f"{iteration:10}\t{energy_value:10.6f}\t{mu_value:10.6f}"
                    f"\t{dtau:10.3e}\t{tol_value:10.3e}"
                )

            if tol_value < 1e-5 or test_e == 0:
                done = True
            elif iteration >= max_iterations:
                print(
                    f"Ground state did not converge in {max_iterations} iterations "
                    f"(residual {tol_value:.3e}, target 1e-5). Writing the current state."
                )
                done = True

        write_psi(file_name, psi, n_r, n_phi, n_z)
        return psi

    # ------------------------------------------------------------------
    # Load from file
    # ------------------------------------------------------------------

    @staticmethod
    def read_ground_state(data, n_r: int, n_phi: int, n_z: int) -> torch.Tensor:
        r"""
        Load a serialised cylindrical ground-state wavefunction from disk.

        The file format is identical to the Cartesian
        :meth:`~src.library.ground_state.GroundState.read_ground_state`: each
        row contains ``(real_part, imag_part)`` for one grid point, written in
        row-major order over ``(n_r, n_phi, n_z)``.

        Args:
            data (str or os.PathLike): Text file produced by
                :func:`~src.utils.read_write_utils.write_psi`.
            n_r (int): Number of radial grid points.
            n_phi (int): Number of azimuthal grid points.
            n_z (int): Number of axial grid points.

        Returns:
            torch.Tensor: Complex tensor of shape ``(n_r, n_phi, n_z)``, on the
            CPU. Move it to the simulation device before use.

        Raises:
            ValueError: If the file does not hold exactly ``n_r*n_phi*n_z``
                points, which normally means it was written for a different
                grid resolution.
        """
        matrix = pd.read_csv(data, header=None, names=["real", "imag"])
        matrix["real"] = matrix["real"].str.strip(" (")
        matrix["imag"] = matrix["imag"].str.strip(" )")
        matrix = matrix.astype(np.float64)

        expected = n_r * n_phi * n_z
        if len(matrix) != expected:
            raise ValueError(
                f"Ground state file '{data}' holds {len(matrix)} points but the "
                f"{n_r}x{n_phi}x{n_z} grid needs {expected}. It was probably "
                f"written for a different grid resolution."
            )

        psi = matrix.iloc[:, 0].values + 1j * matrix.iloc[:, 1].values
        psi = psi.reshape((n_r, n_phi, n_z))
        return torch.from_numpy(psi)
