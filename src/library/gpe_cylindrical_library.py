r"""
GPE solver library in cylindrical coordinates :math:`(r, \varphi, z)`.

Cylindrical counterpart to :mod:`src.library.gpe_library`, in the same
dimensionless units (:math:`\hbar = m = \omega_\mathrm{ho} = 1`). The split-step
structure is unchanged; what differs is the kinetic operator, which is no
longer diagonal in a single Fourier basis, and the volume element
:math:`\mathrm{d}V = r\,\mathrm{d}r\,\mathrm{d}\varphi\,\mathrm{d}z` that every
integral carries.

Grid conventions
----------------

:math:`r`
    Half-point grid :math:`r_i = (i + 1/2)\,\mathrm{d}r`,
    :math:`i = 0 \ldots n_r - 1`. Keeping the points off :math:`r = 0` avoids
    the coordinate singularity and imposes a Neumann condition
    (:math:`\partial\psi/\partial r = 0`) at the axis for :math:`m = 0`.
:math:`\varphi`
    Uniform :math:`0 \ldots 2\pi` over ``n_phi`` points, in DFT order.
:math:`z`
    Uniform :math:`z_\mathrm{min} \ldots z_\mathrm{max}` over ``n_z`` points,
    in FFT order.

The kinetic operator
--------------------

The radial Laplacian for azimuthal mode :math:`m` is

.. math::

    L_r^{m}\psi = \frac{1}{r}\frac{\partial}{\partial r}
        \left(r \frac{\partial \psi}{\partial r}\right)
        - \frac{m^2}{r^2}\psi ,

discretised with a conservative flux scheme on the half-point grid. The
resulting tridiagonal matrix is *not* symmetric in the ordinary sense, but it
*is* self-adjoint in the weighted inner product
:math:`\langle f, g \rangle = \int f^{*} g\, r\,\mathrm{d}r`. The
:math:`\sqrt{r}` similarity transform

.. math::

    \tilde{T}_r^{m} = \sqrt{r}\; T_r^{m}\; \left(\sqrt{r}\right)^{-1}

is real-symmetric, so it can be diagonalised once with
:func:`torch.linalg.eigh` and reused for the whole run.

Call :meth:`GPECylindricalLibrary.build_radial_operators` at initialisation to
obtain the eigendecomposition dicts, then pass them to
:meth:`~GPECylindricalLibrary.p_evolution`,
:meth:`~GPECylindricalLibrary.split_step_step`,
:meth:`~GPECylindricalLibrary.sgpe_step` and the rest.
"""

import warnings
import numpy as np
import torch
from .common_utils import CommonUtils as cu

class GPECylindricalLibrary():
    r"""
    Core of the cylindrical-coordinate GPE solver.

    Reuses the coordinate-agnostic helpers from
    :class:`~src.library.common_utils.CommonUtils` (``x_evolution``,
    ``extract_phase``, ``add_phase``, ``update_phase``) and adds the operators
    that the geometry changes: :meth:`init_grid`,
    :meth:`build_radial_operators`, :meth:`normalize`, :meth:`p_evolution`,
    :meth:`split_step_step`, :meth:`mod_grad_psi`,
    :meth:`calculate_energy_allocation`, :meth:`calculate_chemical_potential`,
    :meth:`generate_thermal_noise` and :meth:`sgpe_step`. The one diagnostic
    defined here is :meth:`angular_momentum_z`.

    The remaining diagnostics (``rms_radius``, ``column_density_z``,
    ``column_density_radial``, ``radial_density_profile``) and the vortex
    imprinting (``create_vortices``, ``check_vortex_resolution``) live in the
    subclass :class:`GPE2DCylindricalLibrary`, mirroring the Cartesian split
    between :class:`~src.library.gpe_library.GPELibrary` and
    :class:`~src.library.gpe_library.GPE2DLibrary`. Because that subclass
    inherits from this one, it exposes everything in both.
    """

    #: Reserved key under which build_radial_operators caches the per-φ stacked
    #: eigenvectors/eigenvalues inside the returned dicts. Every other key is an
    #: int (|m|), so this cannot collide.
    _STACK_KEY = "stacked_by_phi"

    # ------------------------------------------------------------------
    # Grid initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _wrapped_axis(
        n: int,
        step: float,
        device: torch.device,
    ) -> torch.Tensor:
        r"""
        Build a spectral index axis in FFT (wrapped) order.

        Indices :math:`0 \ldots n/2 - 1` hold the positive values and
        :math:`n/2 \ldots n-1` the negative ones, matching the layout of
        :func:`torch.fft.fft`, so the axis can multiply a transformed field
        directly.

        Args:
            n (int): Number of grid points.
            step (float): Spacing between successive spectral values; ``1.0``
                gives the bare integer mode indices.
            device (torch.device): Device the axis is allocated on.

        Returns:
            torch.Tensor: The axis, of shape ``(n,)`` and dtype ``float64``.
        """
        k = torch.arange(n, dtype=torch.float64, device=device)
        k[n // 2:] -= n
        return float(step) * k

    @staticmethod
    def init_grid(
        r_max: float,
        z_min: float,
        z_max: float,
        n_r: int,
        n_phi: int,
        n_z: int,
        device: torch.device,
    ) -> tuple:
        r"""
        Initialise a cylindrical :math:`(r, \varphi, z)` grid.

        The radial direction uses a half-point layout, so the innermost point
        sits at :math:`\mathrm{d}r/2`. That avoids the coordinate singularity
        at :math:`r = 0` and naturally imposes a Neumann condition
        (:math:`\partial\psi/\partial r = 0`) at the axis for :math:`m = 0`.

        Args:
            r_max (float): Outer radial boundary.
            z_min (float): Lower axial bound.
            z_max (float): Upper axial bound.
            n_r (int): Number of radial grid points.
            n_phi (int): Number of azimuthal grid points.
            n_z (int): Number of axial grid points.
            device (torch.device): Device the tensors are allocated on.

        Returns:
            tuple: ``(r, phi, z, kz, m_modes, dr, dphi, dz, space_grid)`` —
            the three 1-D coordinate axes, the axial momentum axis in FFT
            order, the azimuthal mode indices in DFT order, the three grid
            spacings, and the meshgrids ``(gr, gphi, gz)`` of shape
            ``(n_r, n_phi, n_z)``.
        """
        dr = r_max / n_r
        r = (torch.arange(n_r, dtype=torch.float64, device=device) + 0.5) * dr

        dphi = 2.0 * np.pi / n_phi
        phi = torch.arange(n_phi, dtype=torch.float64, device=device) * dphi

        dz = (z_max - z_min) / n_z
        z = z_min + torch.arange(n_z, dtype=torch.float64, device=device) * dz

        # z-momentum (FFT natural order)
        kz = GPECylindricalLibrary._wrapped_axis(n_z, 2.0 * np.pi / (n_z * dz), device)

        # Azimuthal mode indices (DFT order)
        m_modes = GPECylindricalLibrary._wrapped_axis(n_phi, 1.0, device)

        gr, gphi, gz = torch.meshgrid(r, phi, z, indexing="ij")
        return r, phi, z, kz, m_modes, dr, dphi, dz, (gr, gphi, gz)

    # ------------------------------------------------------------------
    # Radial kinetic operator (precomputed once)
    # ------------------------------------------------------------------

    @staticmethod
    def build_radial_operators(
        r: torch.Tensor,
        dr: float,
        m_modes: torch.Tensor,
        device: torch.device,
    ) -> tuple:
        r"""
        Build and diagonalise the radial kinetic operator for every azimuthal
        mode.

        One operator is built per unique :math:`\lvert m \rvert` appearing in
        ``m_modes``. The conservative discretisation of
        :math:`(1/r)\,\partial_r(r\,\partial_r) - m^2/r^2` on the half-point
        grid gives a tridiagonal matrix, which is symmetrised by the
        :math:`\sqrt{r}` similarity transform so that
        :func:`torch.linalg.eigh` can be used,

        .. math::

            \tilde{T}_r^{m} =
                \mathrm{diag}\!\left(\sqrt{r}\right) T_r^{m}\,
                \mathrm{diag}\!\left(1/\sqrt{r}\right)
                = V \Lambda V^{\mathsf{T}} .

        The radial propagator for mode :math:`m` over a step
        :math:`\Delta\tau` then costs two matmuls instead of a matrix
        exponential,

        .. math::

            e^{-i \Delta\tau\, T_r^{m}}\psi =
                \frac{1}{\sqrt{r}}\, V\, e^{-i \Delta\tau \Lambda}\,
                V^{\mathsf{T}} \left(\sqrt{r}\,\psi\right).

        Doing this once at initialisation is what makes the cylindrical solver
        competitive with the Cartesian one, where the kinetic step is a plain
        FFT.

        Args:
            r (torch.Tensor): Radial grid (half-point), of shape ``(n_r,)``.
            dr (float): Radial grid spacing.
            m_modes (torch.Tensor): Azimuthal mode indices in DFT order.
            device (torch.device): Computation device the operators end up on.

        Returns:
            tuple[dict, dict]: ``(eigvecs_dict, eigvals_dict)``. Each maps the
            absolute azimuthal index to a tensor — eigenvectors of shape
            ``(n_r, n_r)`` and eigenvalues of shape ``(n_r,)``.
            ``eigvecs_dict`` additionally carries the reserved key
            ``_STACK_KEY``, holding ``(V_all, lam_all)``: the same operators
            re-indexed by :math:`\varphi` mode, so the whole azimuthal axis can
            be propagated with one batched matmul instead of a Python loop.
        """
        n_r = len(r)
        r_cpu = r.cpu().double()

        # Interface radii on the half-point grid
        r_plus = (torch.arange(n_r, dtype=torch.float64) + 1.0) * dr   # r_{i+1/2}
        r_minus = torch.arange(n_r, dtype=torch.float64) * dr           # r_{i-1/2}

        sqrt_r = torch.sqrt(r_cpu)   # shape (n_r,)

        unique_m_abs = torch.unique(torch.abs(m_modes).long()).tolist()

        eigvecs_dict: dict = {}
        eigvals_dict: dict = {}

        for m in unique_m_abs:
            m = int(m)

            # --- tridiagonal entries of L_r^m (not symmetric) ---
            diag = -(r_plus + r_minus) / (r_cpu * dr ** 2) - float(m) ** 2 / r_cpu ** 2
            sup = r_plus[:-1] / (r_cpu[:-1] * dr ** 2)   # upper off-diagonal
            sub = r_minus[1:] / (r_cpu[1:] * dr ** 2)    # lower off-diagonal

            T_diag = -0.5 * diag                          # (n_r,)
            T_sup = -0.5 * sqrt_r[:-1] * sup / sqrt_r[1:]  # (n_r-1,)
            T_sub = -0.5 * sqrt_r[1:] * sub / sqrt_r[:-1]  # (n_r-1,)

            T_tilde = (
                torch.diag(T_diag)
                + torch.diag(T_sup, 1)
                + torch.diag(T_sub, -1)
            )

            eigvals, eigvecs = torch.linalg.eigh(T_tilde.to(torch.float64))

            eigvecs_dict[m] = eigvecs.to(device=device, dtype=torch.float64)
            eigvals_dict[m] = eigvals.to(device=device, dtype=torch.float64)

        GPECylindricalLibrary._stacked_radial(eigvecs_dict, eigvals_dict, m_modes)
        return eigvecs_dict, eigvals_dict

    @staticmethod
    def _stacked_radial(
        eigvecs_dict: dict,
        eigvals_dict: dict,
        m_modes: torch.Tensor,
    ) -> tuple:
        r"""
        Re-index the per-mode eigendecomposition by azimuthal grid position.

        :meth:`p_evolution` and the ground-state kinetic operator need, for
        every :math:`\varphi` index, the eigenbasis of its own
        :math:`\lvert m \rvert`. Looking those up one at a time costs a Python
        iteration and a host synchronisation per :math:`\varphi` *per time
        step*; stacking them once lets the whole azimuthal axis go through a
        single batched matmul.

        Only ``V_all`` is materialised — the projection uses its transpose as a
        view, which benchmarks within a few percent of a contiguous copy while
        halving the memory. The result is cached inside ``eigvecs_dict`` under
        ``_STACK_KEY``, so it lives and dies with the operators it belongs to.

        Args:
            eigvecs_dict (dict): Eigenvectors from
                :meth:`build_radial_operators`.
            eigvals_dict (dict): Matching eigenvalues.
            m_modes (torch.Tensor): Azimuthal mode indices in DFT order, of
                shape ``(n_phi,)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(V_all, lam_all)``, of shapes
            ``(n_phi, n_r, n_r)`` and ``(n_phi, n_r)``.
        """
        key = GPECylindricalLibrary._STACK_KEY
        cached = eigvecs_dict.get(key)
        if cached is not None and cached[0].shape[0] == m_modes.shape[0]:
            return cached

        # One host transfer for the whole axis instead of one .item() per φ.
        abs_m = [int(m) for m in m_modes.abs().long().tolist()]
        V_all = torch.stack([eigvecs_dict[m] for m in abs_m])
        lam_all = torch.stack([eigvals_dict[m] for m in abs_m])
        eigvecs_dict[key] = (V_all, lam_all)
        return V_all, lam_all

    @staticmethod
    def _apply_radial_eigen(
        psi_m: torch.Tensor,
        factors: torch.Tensor,
        V_all: torch.Tensor,
        sqrt_r: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Apply the radial eigenbasis operator to every azimuthal mode at once.

        Evaluates

        .. math::

            \frac{1}{\sqrt{r}}\; V \,\mathrm{diag}(\text{factors})\,
                V^{\mathsf{T}} \left(\sqrt{r}\;\cdot\right),

        the :math:`\sqrt{r}`-symmetrised radial operator of the module
        docstring, in the precomputed eigenbasis.

        Note:
            ``V_all`` is real, so the complex field is pushed through the two
            matmuls as an interleaved real view: a real GEMM does half the work
            of the equivalent complex one, and the result is bit-identical.

        Args:
            psi_m (torch.Tensor): Complex field of shape
                ``(n_r, n_phi, n_z)``, already transformed over
                :math:`\varphi`.
            factors (torch.Tensor): Diagonal multiplier per radial eigenmode,
                broadcastable to ``(n_phi, n_r, n_z)``. Pass
                :math:`e^{-\text{damping}\,\Delta\tau\,\Lambda}` for a
                propagator, :math:`\Lambda` itself to apply the operator, or a
                0/1 mask to project.
            V_all (torch.Tensor): Stacked eigenvectors of shape
                ``(n_phi, n_r, n_r)``.
            sqrt_r (torch.Tensor): :math:`\sqrt{r}` on the radial grid, of
                shape ``(n_r,)``.

        Returns:
            torch.Tensor: The result, of shape ``(n_r, n_phi, n_z)``.
        """
        n_r, n_phi, n_z = psi_m.shape
        radial = sqrt_r.reshape(-1, 1, 1)

        x = (radial * psi_m).permute(1, 0, 2).contiguous()          # (n_phi, n_r, n_z)
        x_re = torch.view_as_real(x).reshape(n_phi, n_r, 2 * n_z)

        coeff = torch.bmm(V_all.transpose(1, 2), x_re).reshape(n_phi, n_r, n_z, 2)
        coeff = torch.view_as_complex(coeff.contiguous()) * factors

        c_re = torch.view_as_real(coeff.contiguous()).reshape(n_phi, n_r, 2 * n_z)
        out = torch.bmm(V_all, c_re).reshape(n_phi, n_r, n_z, 2)
        out = torch.view_as_complex(out.contiguous()).permute(1, 0, 2)
        return out / radial

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(
        psi: torch.Tensor,
        r: torch.Tensor,
        dr: float,
        dphi: float,
        dz: float,
    ) -> torch.Tensor:
        r"""
        Normalise the wavefunction with the cylindrical volume element.

        .. math::

            \int \lvert \psi \rvert^{2}\,
                r\,\mathrm{d}r\,\mathrm{d}\varphi\,\mathrm{d}z = 1

        Args:
            psi (torch.Tensor): Wavefunction of shape ``(n_r, n_phi, n_z)``.
            r (torch.Tensor): Radial grid, of shape ``(n_r,)``.
            dr (float): Radial grid spacing.
            dphi (float): Azimuthal grid spacing.
            dz (float): Axial grid spacing.

        Returns:
            torch.Tensor: The normalised wavefunction.
        """
        r_w = r.reshape(-1, 1, 1)
        norm_sq = torch.sum(torch.abs(psi) ** 2 * r_w) * (dr * dphi * dz)
        return psi / torch.sqrt(norm_sq)

    # ------------------------------------------------------------------
    # Kinetic (momentum-space) evolution
    # ------------------------------------------------------------------

    @staticmethod
    def p_evolution(
        psi: torch.Tensor,
        dtau: float,
        kz: torch.Tensor,
        m_modes: torch.Tensor,
        r: torch.Tensor,
        eigvecs_dict: dict,
        eigvals_dict: dict,
        damping: complex = 1j,
    ) -> torch.Tensor:
        r"""
        Apply the full kinetic evolution
        :math:`e^{-\text{damping}\,\Delta\tau\,T}` in cylindrical coordinates.

        The sequence is:

        1. DFT over :math:`\varphi` — to azimuthal modes :math:`m`;
        2. FFT over :math:`z` — to axial modes :math:`k_z`;
        3. the axial kinetic step
           :math:`e^{-\text{damping}\,\Delta\tau\,k_z^2/2}`, which is diagonal;
        4. the radial kinetic step for every :math:`m`, through the precomputed
           eigenbasis;
        5. the inverse transforms over :math:`z` and :math:`\varphi`.

        The radial propagator uses the :math:`\sqrt{r}` symmetrisation,

        .. math::

            e^{-\text{damping}\,\Delta\tau\, T_r^{m}}\psi =
                \frac{1}{\sqrt{r}}\, V\,
                e^{-\text{damping}\,\Delta\tau\,\Lambda}\,
                V^{\mathsf{T}} \left(\sqrt{r}\,\psi\right).

        Note:
            :math:`T_r` and :math:`T_z` act on different coordinates and
            therefore commute, so applying them one after the other is exact
            rather than a further splitting error.

        Args:
            psi (torch.Tensor): Complex wavefunction of shape
                ``(n_r, n_phi, n_z)``.
            dtau (float): Time step :math:`\Delta\tau`.
            kz (torch.Tensor): Axial momentum grid, of shape ``(n_z,)``.
            m_modes (torch.Tensor): Azimuthal mode indices, of shape
                ``(n_phi,)``.
            r (torch.Tensor): Radial grid, of shape ``(n_r,)``.
            eigvecs_dict (dict): Eigenvectors from
                :meth:`build_radial_operators`.
            eigvals_dict (dict): Matching eigenvalues.
            damping (complex, optional): Prefactor — ``1j`` for the ordinary
                GPE, :math:`i + \gamma` for the SGPE (default ``1j``).

        Returns:
            torch.Tensor: Updated wavefunction of shape
            ``(n_r, n_phi, n_z)``.
        """
        n_r, n_phi, n_z = psi.shape

        # 1-2: transform to (r, m, kz) space
        psi_m = torch.fft.fft(psi, dim=1, norm="ortho")
        psi_mk = torch.fft.fft(psi_m, dim=2, norm="forward")

        # 3: z kinetic step (diagonal)
        kz_sq = kz.reshape(1, 1, n_z) ** 2
        psi_mk = torch.exp(-damping * dtau * 0.5 * kz_sq) * psi_mk

        # √r weight for symmetrisation (shape n_r,)
        sqrt_r = torch.sqrt(r).to(dtype=torch.float64, device=psi.device)

        # 4: radial kinetic step, all azimuthal modes in one batched matmul.
        # T_r and T_z commute (they act on different coordinates), so applying
        # them in sequence is exact rather than a further splitting error.
        V_all, lam_all = GPECylindricalLibrary._stacked_radial(
            eigvecs_dict, eigvals_dict, m_modes
        )
        factors = torch.exp(-damping * dtau * lam_all).unsqueeze(-1)  # (n_phi, n_r, 1)
        result_mk = GPECylindricalLibrary._apply_radial_eigen(
            psi_mk, factors, V_all, sqrt_r
        )

        # 5: inverse transforms
        result_m = torch.fft.ifft(result_mk, dim=2, norm="forward")
        return torch.fft.ifft(result_m, dim=1, norm="ortho")

    # ------------------------------------------------------------------
    # Split-step step
    # ------------------------------------------------------------------

    @staticmethod
    def split_step_step(
        psi: torch.Tensor,
        utot: torch.Tensor,
        dtau: float,
        kz: torch.Tensor,
        m_modes: torch.Tensor,
        r: torch.Tensor,
        eigvecs_dict: dict,
        eigvals_dict: dict,
        dr: float,
        dphi: float,
        dz: float,
        renormalise: bool = False,
    ) -> torch.Tensor:
        r"""
        Advance the wavefunction by one split-step iteration.

        A Strang splitting — real-space half-step, full kinetic step,
        real-space half-step — matching
        :meth:`~src.library.gpe_library.GPELibrary.split_step_step`, with
        :meth:`p_evolution` supplying the cylindrical kinetic step.

        Note:
            For a real ``utot`` every factor applied here has unit modulus, so
            the norm is conserved to machine precision and no renormalisation
            is needed. When ``utot`` carries an imaginary part (three-body
            losses, a complex absorbing potential) the norm is *meant* to
            decay: renormalising would silently cancel the atom loss, which is
            why ``renormalise`` defaults to ``False``.

        Args:
            psi (torch.Tensor): Wavefunction of shape ``(n_r, n_phi, n_z)``.
            utot (torch.Tensor): Total real-space operator, frozen for the
                duration of the step.
            dtau (float): Time step :math:`\Delta\tau`.
            kz (torch.Tensor): Axial momentum grid.
            m_modes (torch.Tensor): Azimuthal mode indices.
            r (torch.Tensor): Radial grid.
            eigvecs_dict (dict): Eigenvectors from
                :meth:`build_radial_operators`.
            eigvals_dict (dict): Matching eigenvalues.
            dr (float): Radial grid spacing.
            dphi (float): Azimuthal grid spacing.
            dz (float): Axial grid spacing.
            renormalise (bool, optional): Force unit norm after the step. Only
                meaningful for a deliberately number-conserving run with a
                lossy potential (default ``False``).

        Returns:
            torch.Tensor: The wavefunction after one split-step.
        """
        psi = cu.x_evolution(psi, utot, dtau, factor=0.5)
        psi = GPECylindricalLibrary.p_evolution(
            psi, dtau, kz, m_modes, r, eigvecs_dict, eigvals_dict
        )
        psi = cu.x_evolution(psi, utot, dtau, factor=0.5)
        if renormalise:
            return GPECylindricalLibrary.normalize(psi, r, dr, dphi, dz)
        return psi

    # ------------------------------------------------------------------
    # Gradient / energy diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def mod_grad_psi(
        psi: torch.Tensor,
        r: torch.Tensor,
        dr: float,
        kz: torch.Tensor,
        m_modes: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute :math:`\lvert \nabla \psi \rvert` in cylindrical coordinates.

        .. math::

            \lvert \nabla \psi \rvert^{2} =
                \left\lvert \frac{\partial \psi}{\partial r} \right\rvert^{2}
                + \frac{1}{r^2}
                  \left\lvert \frac{\partial \psi}{\partial \varphi}
                  \right\rvert^{2}
                + \left\lvert \frac{\partial \psi}{\partial z} \right\rvert^{2}

        The two periodic directions are differentiated spectrally —
        :math:`\partial_\varphi \psi = \mathrm{IDFT}(i m \tilde\psi_m)` and
        :math:`\partial_z \psi = \mathrm{IFFT}(i k_z \tilde\psi_{k_z})` —
        while the radial derivative uses central finite differences, with
        second-order one-sided stencils at the two radial boundaries, since the
        grid is not periodic in :math:`r`.

        Note:
            The radial term is the only non-spectral piece in the library, so
            it is what sets the accuracy of the kinetic energy, at
            :math:`O(\mathrm{d}r^2)`. A first-order end point would dominate
            the error, which is why the boundary stencils are second-order too.

        Args:
            psi (torch.Tensor): Wavefunction of shape ``(n_r, n_phi, n_z)``.
            r (torch.Tensor): Radial grid, of shape ``(n_r,)``.
            dr (float): Radial grid spacing.
            kz (torch.Tensor): Axial momentum grid.
            m_modes (torch.Tensor): Azimuthal mode indices.

        Returns:
            torch.Tensor: Real-valued tensor of shape
            ``(n_r, n_phi, n_z)``.
        """
        n_r, n_phi, n_z = psi.shape
        r_w = r.reshape(-1, 1, 1)

        # z-gradient (spectral)
        psi_kz = torch.fft.fft(psi, dim=2, norm="forward")
        dpsi_dz = torch.fft.ifft(
            1j * kz.reshape(1, 1, n_z) * psi_kz, dim=2, norm="forward"
        )

        # φ-gradient (spectral)
        psi_m = torch.fft.fft(psi, dim=1, norm="ortho")
        dpsi_dphi = torch.fft.ifft(
            1j * m_modes.reshape(1, n_phi, 1) * psi_m, dim=1, norm="ortho"
        )

        # r-gradient (finite differences, central in the interior and
        # second-order one-sided at the boundaries so the whole profile is
        # O(dr²) — a first-order end point would dominate the kinetic energy)
        dpsi_dr = torch.empty_like(psi)
        dpsi_dr[1:-1] = (psi[2:] - psi[:-2]) / (2.0 * dr)
        if n_r >= 3:
            dpsi_dr[0] = (-3.0 * psi[0] + 4.0 * psi[1] - psi[2]) / (2.0 * dr)
            dpsi_dr[-1] = (3.0 * psi[-1] - 4.0 * psi[-2] + psi[-3]) / (2.0 * dr)
        else:
            dpsi_dr[0] = (psi[1] - psi[0]) / dr
            dpsi_dr[-1] = (psi[-1] - psi[-2]) / dr

        grad_sq = (
            torch.abs(dpsi_dr) ** 2
            + (torch.abs(dpsi_dphi) / r_w) ** 2
            + torch.abs(dpsi_dz) ** 2
        )
        return torch.sqrt(grad_sq)

    @staticmethod
    def calculate_energy_allocation(
        psi: torch.Tensor,
        Vext: torch.Tensor,
        r: torch.Tensor,
        dr: float,
        dphi: float,
        dz: float,
        kz: torch.Tensor,
        m_modes: torch.Tensor,
        **parameters,
    ) -> dict:
        r"""
        Split the condensate energy into its kinetic, potential and interaction
        parts, with the cylindrical volume element.

        .. math::

            e_\mathrm{kin} = \frac{1}{2}\int \lvert \nabla\psi \rvert^{2}\,
                r\,\mathrm{d}r\,\mathrm{d}\varphi\,\mathrm{d}z,

        .. math::

            e_\mathrm{pot} = \int V_\mathrm{ext}\lvert \psi \rvert^{2}\,
                r\,\mathrm{d}r\,\mathrm{d}\varphi\,\mathrm{d}z,
            \qquad
            e_\mathrm{int} = \frac{u}{2}\int \lvert \psi \rvert^{4}\,
                r\,\mathrm{d}r\,\mathrm{d}\varphi\,\mathrm{d}z

        Args:
            psi (torch.Tensor): Wavefunction of shape ``(n_r, n_phi, n_z)``.
            Vext (torch.Tensor): External potential on the same grid. If it is
                complex — an absorbing potential, say — only the real part
                contributes; the imaginary part is a loss rate, not an energy.
            r (torch.Tensor): Radial grid.
            dr (float): Radial grid spacing.
            dphi (float): Azimuthal grid spacing.
            dz (float): Axial grid spacing.
            kz (torch.Tensor): Axial momentum grid.
            m_modes (torch.Tensor): Azimuthal mode indices.
            **parameters: Must include the interaction strength ``"u"``.

        Returns:
            dict: The keys ``'e_kin'``, ``'e_pot'``, ``'e_int'`` and
            ``'E_total'``, in units of :math:`\hbar\omega_\mathrm{ho}`.

        Raises:
            ValueError: If ``"u"`` is not supplied.
        """
        if "u" not in parameters:
            raise ValueError("calculate_energy_allocation requires the interaction strength 'u'")
        u = parameters["u"]
        r_w = r.reshape(-1, 1, 1)
        dV = dr * dphi * dz

        potential = Vext.real if torch.is_complex(Vext) else Vext
        density = torch.abs(psi) ** 2
        grad_sq = (
            GPECylindricalLibrary.mod_grad_psi(psi, r, dr, kz, m_modes) ** 2
        )

        e_kin = 0.5 * torch.sum(grad_sq * r_w) * dV
        e_pot = torch.sum(potential * density * r_w) * dV
        e_int = 0.5 * u * torch.sum(density ** 2 * r_w) * dV
        E_total = e_kin + e_pot + e_int

        return {"e_kin": e_kin, "e_pot": e_pot, "e_int": e_int, "E_total": E_total}

    @staticmethod
    def calculate_chemical_potential(
        psi: torch.Tensor,
        uext: torch.Tensor,
        u: float,
        r: torch.Tensor,
        dr: float,
        dphi: float,
        dz: float,
        kz: torch.Tensor,
        m_modes: torch.Tensor,
    ) -> float:
        r"""
        Compute the mean-field chemical potential.

        .. math::

            \mu = e_\mathrm{kin} + e_\mathrm{pot} + 2\,e_\mathrm{int}

        Identical in form to
        :meth:`~src.library.gpe_library.GPELibrary.calculate_chemical_potential`
        — see there for why the interaction term is counted twice — but the
        energies come from the cylindrical
        :meth:`calculate_energy_allocation`.

        Args:
            psi (torch.Tensor): Normalised wavefunction of shape
                ``(n_r, n_phi, n_z)``.
            uext (torch.Tensor): External trapping potential on the grid.
            u (float): Dimensionless interaction strength :math:`u`.
            r (torch.Tensor): Radial grid.
            dr (float): Radial grid spacing.
            dphi (float): Azimuthal grid spacing.
            dz (float): Axial grid spacing.
            kz (torch.Tensor): Axial momentum grid.
            m_modes (torch.Tensor): Azimuthal mode indices.

        Returns:
            float: The chemical potential :math:`\mu`, in units of
            :math:`\hbar\omega_\mathrm{ho}`.
        """
        energy = GPECylindricalLibrary.calculate_energy_allocation(
            psi, uext, r, dr, dphi, dz, kz, m_modes, u=u
        )
        return float((energy["e_kin"] + energy["e_pot"] + 2.0 * energy["e_int"]).real)

    # ------------------------------------------------------------------
    # SGPE
    # ------------------------------------------------------------------

    @staticmethod
    def generate_thermal_noise(
        shape: tuple,
        gamma: float,
        kT: float,
        dtau: float,
        r: torch.Tensor,
        dr: float,
        dphi: float,
        dz: float,
        device: torch.device,
        kz: torch.Tensor = None,
        m_modes: torch.Tensor = None,
        eigvecs_dict: dict = None,
        eigvals_dict: dict = None,
        e_cut: float = None,
    ) -> torch.Tensor:
        r"""
        Generate a complex Gaussian noise field for one SGPE time step in
        cylindrical geometry.

        The local cell volume varies with radius,
        :math:`\mathrm{d}V(r_i) = r_i\,\mathrm{d}r\,\mathrm{d}\varphi\,
        \mathrm{d}z`, so the noise amplitude scales as
        :math:`1/\sqrt{\mathrm{d}V}` to satisfy the fluctuation-dissipation
        theorem,

        .. math::

            \text{amplitude}_i = \sqrt{\frac{\gamma\, k_B T\, \Delta\tau}
                {r_i\,\mathrm{d}r\,\mathrm{d}\varphi\,\mathrm{d}z}} .

        Note:
            **Projection — the "P" of the** *projected* **SGPE.**
            Delta-correlated noise is white across the whole grid, so it feeds
            energy into every mode the grid can represent. The c-field
            description is only valid below a cutoff energy, above which the
            modes belong to the thermal cloud rather than to the classical
            field. Supply the kinetic operators together with ``e_cut`` to keep
            only the modes with

            .. math::

                \Lambda_j^{\lvert m \rvert} + \frac{k_z^2}{2}
                    \le e_\mathrm{cut},

            where :math:`\Lambda` is the radial kinetic eigenvalue — the exact
            cylindrical analogue of the Cartesian
            :math:`p^2/2 \le e_\mathrm{cut}`. Without it the noise is
            unprojected and the run will heat artificially at short
            wavelengths.

        Args:
            shape (tuple): Grid shape ``(n_r, n_phi, n_z)``.
            gamma (float): Dimensionless damping coefficient :math:`\gamma`.
            kT (float): Dimensionless temperature
                :math:`k_B T / (\hbar\omega_\mathrm{ho})`.
            dtau (float): Time step :math:`\Delta\tau`.
            r (torch.Tensor): Radial grid, of shape ``(n_r,)``.
            dr (float): Radial grid spacing.
            dphi (float): Azimuthal grid spacing.
            dz (float): Axial grid spacing.
            device (torch.device): Computation device.
            kz (torch.Tensor, optional): Axial momentum grid; required for the
                projection.
            m_modes (torch.Tensor, optional): Azimuthal mode indices; required
                for the projection.
            eigvecs_dict (dict, optional): Eigenvectors from
                :meth:`build_radial_operators`; required for the projection.
            eigvals_dict (dict, optional): Matching eigenvalues; required for
                the projection.
            e_cut (float, optional): Cutoff energy in units of
                :math:`\hbar\omega_\mathrm{ho}`. The projection is applied only
                when it and all four operators above are supplied.

        Returns:
            torch.Tensor: Complex noise tensor of shape
            ``(n_r, n_phi, n_z)``.
        """
        n_r, n_phi, n_z = shape
        r_w = r.reshape(-1, 1, 1).expand(n_r, n_phi, n_z)
        dV_local = r_w * dr * dphi * dz
        amplitude = torch.sqrt(gamma * kT * dtau / dV_local)

        xi_r = torch.randn(shape, dtype=torch.float64, device=device)
        xi_i = torch.randn(shape, dtype=torch.float64, device=device)
        noise = (amplitude * (xi_r + 1j * xi_i)).to(torch.cdouble)

        operators = (kz, m_modes, eigvecs_dict, eigvals_dict)
        if e_cut is not None and all(o is not None for o in operators):
            V_all, lam_all = GPECylindricalLibrary._stacked_radial(
                eigvecs_dict, eigvals_dict, m_modes
            )
            # Mode energy Λ + kz²/2, indexed (n_phi, n_r, n_z)
            e_mode = lam_all.unsqueeze(-1) + 0.5 * kz.reshape(1, 1, n_z) ** 2
            keep = (e_mode <= e_cut).to(torch.cdouble)

            noise_m = torch.fft.fft(noise, dim=1, norm="ortho")
            noise_mk = torch.fft.fft(noise_m, dim=2, norm="forward")
            sqrt_r = torch.sqrt(r).to(dtype=torch.float64, device=noise.device)
            noise_mk = GPECylindricalLibrary._apply_radial_eigen(
                noise_mk, keep, V_all, sqrt_r
            )
            noise_m = torch.fft.ifft(noise_mk, dim=2, norm="forward")
            noise = torch.fft.ifft(noise_m, dim=1, norm="ortho")
        return noise

    @staticmethod
    def sgpe_step(
        psi: torch.Tensor,
        utot: torch.Tensor,
        mu: float,
        gamma: float,
        dtau: float,
        kz: torch.Tensor,
        m_modes: torch.Tensor,
        r: torch.Tensor,
        eigvecs_dict: dict,
        eigvals_dict: dict,
        dr: float,
        dphi: float,
        dz: float,
        renormalise: bool = False,
    ) -> torch.Tensor:
        r"""
        Perform one deterministic SGPE split-step with
        :math:`(1 - i\gamma)` dissipative damping.

        The SGPE replaces the unitary GPE operator with

        .. math::

            e^{-(i + \gamma)\,\Delta\tau\,(H_\mathrm{mf} - \mu)},

        applied as the same Strang splitting used everywhere else: a real-space
        half-step
        :math:`e^{-(i+\gamma)\Delta\tau (V_\mathrm{eff} - \mu)/2}`, a full
        kinetic step through :meth:`p_evolution` with the damped prefactor, and
        a second real-space half-step.

        Note:
            The norm is left free so that :math:`\mu` actually drives the
            dynamics; see
            :meth:`~src.library.gpe_library.GPELibrary.sgpe_step` for the full
            argument.

        Args:
            psi (torch.Tensor): Wavefunction of shape ``(n_r, n_phi, n_z)``.
            utot (torch.Tensor): Total potential
                :math:`V_\mathrm{ext} + u\lvert\psi\rvert^2`, frozen at the
                start of the step.
            mu (float): Reservoir chemical potential :math:`\mu`.
            gamma (float): Damping coefficient :math:`\gamma`.
            dtau (float): Time step :math:`\Delta\tau`.
            kz (torch.Tensor): Axial momentum grid.
            m_modes (torch.Tensor): Azimuthal mode indices.
            r (torch.Tensor): Radial grid.
            eigvecs_dict (dict): Eigenvectors from
                :meth:`build_radial_operators`.
            eigvals_dict (dict): Matching eigenvalues.
            dr (float): Radial grid spacing.
            dphi (float): Azimuthal grid spacing.
            dz (float): Axial grid spacing.
            renormalise (bool, optional): Force unit norm after the step, which
                disables the reservoir coupling (default ``False``).

        Returns:
            torch.Tensor: Updated wavefunction.
        """
        damping = 1j + gamma
        eff_pot = utot - mu

        psi = torch.exp(-damping * 0.5 * dtau * eff_pot) * psi
        psi = GPECylindricalLibrary.p_evolution(
            psi, dtau, kz, m_modes, r, eigvecs_dict, eigvals_dict, damping=damping
        )
        psi = torch.exp(-damping * 0.5 * dtau * eff_pot) * psi

        if renormalise:
            return GPECylindricalLibrary.normalize(psi, r, dr, dphi, dz)
        return psi

    # ------------------------------------------------------------------
    # Cylindrical diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def angular_momentum_z(
        psi: torch.Tensor,
        m_modes: torch.Tensor,
        r: torch.Tensor,
        dr: float,
        dphi: float,
        dz: float,
    ) -> torch.Tensor:
        r"""
        Compute the expectation value of
        :math:`L_z = -i\,\partial/\partial\varphi`, in units of :math:`\hbar`.

        Cylindrical coordinates make this diagnostic almost free: in the DFT
        basis :math:`L_z` is diagonal with eigenvalue :math:`m`, so no
        derivative has to be evaluated at all,

        .. math::

            \langle L_z \rangle = \sum_m m
                \int \lvert \psi_m(r, z) \rvert^{2}\,
                r\,\mathrm{d}r\,\mathrm{d}z\;\mathrm{d}\varphi .

        The ``norm='ortho'`` DFT ensures that
        :math:`\sum_m \lvert \psi_m \rvert^{2}` reproduces
        :math:`\lvert \psi \rvert^{2}` in the azimuthal integral.

        Args:
            psi (torch.Tensor): Normalised wavefunction of shape
                ``(n_r, n_phi, n_z)``.
            m_modes (torch.Tensor): Azimuthal mode indices, of shape
                ``(n_phi,)``.
            r (torch.Tensor): Radial grid.
            dr (float): Radial grid spacing.
            dphi (float): Azimuthal grid spacing.
            dz (float): Axial grid spacing.

        Returns:
            torch.Tensor: Scalar expectation value
            :math:`\langle L_z \rangle`.
        """
        r_w = r.reshape(-1, 1, 1)
        psi_m = torch.fft.fft(psi, dim=1, norm="ortho")
        m_grid = m_modes.reshape(1, -1, 1)
        Lz = torch.sum(m_grid * torch.abs(psi_m) ** 2 * r_w) * (dr * dphi * dz)
        return torch.real(Lz)

class GPE2DCylindricalLibrary(GPECylindricalLibrary):
    r"""
    Vortex imprinting and density diagnostics on a cylindrical grid.

    Mirrors the Cartesian
    :class:`~src.library.gpe_library.GPE2DLibrary` relationship to
    :class:`~src.library.gpe_library.GPELibrary`: it inherits every core
    operator from :class:`GPECylindricalLibrary` and adds the diagnostics the
    simulation loop reports (:meth:`rms_radius`, the column densities,
    :meth:`radial_density_profile`) plus vortex-line imprinting
    (:meth:`create_vortices`, :meth:`check_vortex_resolution`).
    """

    @staticmethod
    def create_vortices(
        r: torch.Tensor,
        phi: torch.Tensor,
        n_r: int,
        n_phi: int,
        n_z: int,
        positions: list,
        charges: list,
        r_core: float,
        device: torch.device,
        dr: float = None,
        dphi: float = None,
    ) -> torch.Tensor:
        r"""
        Build a phase-and-amplitude mask for one or more vortex lines at
        arbitrary positions.

        Each vortex runs along :math:`z`, its core intersecting the
        :math:`(r, \varphi)` plane at the Cartesian position
        :math:`(x_0, y_0)`. Unlike the pure-phase Cartesian imprint, this mask
        also suppresses the density at the core, so the state starts closer to
        a relaxed vortex. Vortex :math:`n` contributes

        .. math::

            d_n(r, \varphi) = \sqrt{(r\cos\varphi - x_0)^2
                                  + (r\sin\varphi - y_0)^2},

        .. math::

            A_n = \tanh\!\left(\frac{d_n}{r_\mathrm{core}}\right),
            \qquad
            \theta_n = q_n\, \mathrm{atan2}\left(
                r\sin\varphi - y_0,\; r\cos\varphi - x_0\right),

        and all contributions are multiplied together,

        .. math::

            \text{mask} = \prod_n A_n\, e^{i\theta_n}.

        A vortex at the origin reduces to the on-axis case
        :math:`\tanh(r/r_\mathrm{core})\,e^{i q \varphi}`.

        If ``dr`` and ``dphi`` are given, a resolution check is run and a
        warning is issued for any vortex whose core is under-resolved on the
        cylindrical grid; see :meth:`check_vortex_resolution`.

        Args:
            r (torch.Tensor): Radial axis, of shape ``(n_r,)``.
            phi (torch.Tensor): Azimuthal axis, of shape ``(n_phi,)``.
            n_r (int): Number of radial grid points.
            n_phi (int): Number of azimuthal grid points.
            n_z (int): Number of axial grid points.
            positions (list): Cartesian vortex-core positions
                :math:`(x_0, y_0)`.
            charges (list): Topological charge :math:`q` of each vortex.
            r_core (float): Healing-length core radius, the same for all
                vortices.
            device (torch.device): Computation device.
            dr (float, optional): Radial grid spacing; supplying it enables the
                resolution check.
            dphi (float, optional): Azimuthal grid spacing
                :math:`2\pi / n_\varphi`; supplying it enables the resolution
                check.

        Returns:
            torch.Tensor: Complex tensor of shape ``(n_r, n_phi, n_z)`` to
            multiply onto :math:`\psi`.

        Raises:
            ValueError: If ``positions`` and ``charges`` have different
                lengths.

        Warns:
            UserWarning: For each vortex whose core is under-resolved, when the
                grid spacings are supplied.
        """
        if len(charges) != len(positions):
            # zip() would otherwise silently drop the surplus vortices.
            raise ValueError(
                f"charges has {len(charges)} entries but there are "
                f"{len(positions)} vortex positions"
            )

        if dr is not None and dphi is not None:
            report = GPE2DCylindricalLibrary.check_vortex_resolution(
                positions, r_core, dr, dphi
            )
            for info in report:
                if not info["resolved"]:
                    warnings.warn(
                        f"Vortex at {info['position']} is under-resolved: "
                        f"r_core / dr_eff = {info['ratio']:.2f} "
                        f"(bottleneck: {info['bottleneck']}, "
                        f"r0={info['r0']:.3g}, dr_eff={info['dr_eff']:.3g}). "
                        f"Increase n_phi or use a larger r_core.",
                        stacklevel=2,
                    )

        # The vortex lines run along z, so the mask is z-independent: build it
        # on the (r, φ) plane and broadcast, instead of materialising n_z
        # identical copies of every intermediate.
        gr = r.reshape(n_r, 1).to(device=device, dtype=torch.float64)
        gphi = phi.reshape(1, n_phi).to(device=device, dtype=torch.float64)

        gx = gr * torch.cos(gphi)
        gy = gr * torch.sin(gphi)

        mask = torch.ones(n_r, n_phi, dtype=torch.cdouble, device=device)
        for (x0, y0), q in zip(positions, charges):
            dx = gx - x0
            dy = gy - y0
            d = torch.sqrt(dx ** 2 + dy ** 2)
            amplitude = torch.tanh(d / r_core)
            phase = q * torch.atan2(dy, dx)
            mask = mask * amplitude.to(torch.cdouble) * torch.exp(1j * phase.to(torch.cdouble))

        return mask.reshape(n_r, n_phi, 1).expand(n_r, n_phi, n_z)

    @staticmethod
    def check_vortex_resolution(
        positions: list,
        r_core: float,
        dr: float,
        dphi: float,
        min_points_per_core: float = 2.0,
    ) -> list:
        r"""
        Check whether the cylindrical grid resolves each vortex core.

        A polar grid gets coarser in arc length as the radius grows, so a
        vortex that is well resolved near the axis may not be near the edge. At
        radial position :math:`r_0` the effective local Cartesian spacing is
        the worse of the two directions,

        .. math::

            \mathrm{d}r_\mathrm{eff} =
                \max\left(\mathrm{d}r,\; r_0\,\mathrm{d}\varphi\right),

        and the core counts as resolved when at least
        ``min_points_per_core`` cells span one healing length,

        .. math::

            \frac{r_\mathrm{core}}{\mathrm{d}r_\mathrm{eff}}
                \ge n_\mathrm{min} .

        Args:
            positions (list): Cartesian vortex positions :math:`(x_0, y_0)`.
            r_core (float): Healing-length core radius.
            dr (float): Radial grid spacing.
            dphi (float): Azimuthal grid spacing :math:`2\pi / n_\varphi`.
            min_points_per_core (float, optional): Minimum required resolution
                ratio (default ``2.0``).

        Returns:
            list[dict]: One dict per vortex, with the keys ``'position'``
            :math:`(x_0, y_0)`; ``'r0'``, the distance of the core from the
            axis; ``'dr_eff'``, the effective local spacing; ``'ratio'``,
            :math:`r_\mathrm{core}/\mathrm{d}r_\mathrm{eff}`; ``'resolved'``, a
            bool; and ``'bottleneck'``, either ``'azimuthal'`` or ``'radial'``
            depending on which direction limits the resolution.
        """
        report = []
        for (x0, y0) in positions:
            r0 = float(np.sqrt(x0 ** 2 + y0 ** 2))
            arc = r0 * dphi
            dr_eff = max(dr, arc)
            ratio = r_core / dr_eff
            report.append(
                {
                    "position": (x0, y0),
                    "r0": r0,
                    "dr_eff": dr_eff,
                    "ratio": ratio,
                    "resolved": ratio >= min_points_per_core,
                    "bottleneck": "azimuthal" if arc >= dr else "radial",
                }
            )
        return report

    @staticmethod
    def column_density_z(
        psi: torch.Tensor,
        dz: float,
    ) -> torch.Tensor:
        r"""
        Compute the column density integrated along the axial direction.

        .. math::

            n(r, \varphi) = \int \lvert \psi \rvert^{2}\,\mathrm{d}z

        Args:
            psi (torch.Tensor): Wavefunction of shape ``(n_r, n_phi, n_z)``.
            dz (float): Axial grid spacing.

        Returns:
            torch.Tensor: Column density of shape ``(n_r, n_phi)``.
        """
        return torch.sum(torch.abs(psi) ** 2, dim=2) * dz

    @staticmethod
    def column_density_radial(
        psi: torch.Tensor,
        r: torch.Tensor,
        dr: float,
    ) -> torch.Tensor:
        r"""
        Compute the density integrated over the radial direction.

        .. math::

            n(\varphi, z) = \int \lvert \psi \rvert^{2}\, r\,\mathrm{d}r

        The :math:`r` weight is the cylindrical volume element, without which
        the inner radii would be over-counted.

        Args:
            psi (torch.Tensor): Wavefunction of shape ``(n_r, n_phi, n_z)``.
            r (torch.Tensor): Radial grid.
            dr (float): Radial grid spacing.

        Returns:
            torch.Tensor: Density of shape ``(n_phi, n_z)``.
        """
        r_w = r.reshape(-1, 1, 1)
        return torch.sum(torch.abs(psi) ** 2 * r_w, dim=0) * dr

    @staticmethod
    def radial_density_profile(
        psi: torch.Tensor,
        dphi: float,
        dz: float,
    ) -> torch.Tensor:
        r"""
        Compute the azimuthally and axially integrated radial profile.

        .. math::

            n(r) = \int \lvert \psi \rvert^{2}\,
                \mathrm{d}\varphi\,\mathrm{d}z

        Args:
            psi (torch.Tensor): Wavefunction of shape ``(n_r, n_phi, n_z)``.
            dphi (float): Azimuthal grid spacing.
            dz (float): Axial grid spacing.

        Returns:
            torch.Tensor: Radial profile of shape ``(n_r,)``.
        """
        return torch.sum(torch.abs(psi) ** 2, dim=(1, 2)) * (dphi * dz)

    @staticmethod
    def rms_radius(
        psi: torch.Tensor,
        r: torch.Tensor,
        dr: float,
        dphi: float,
        dz: float,
    ) -> torch.Tensor:
        r"""
        Compute the RMS radial extent of the condensate.

        .. math::

            \langle r^2 \rangle =
                \frac{\int r^2 \lvert \psi \rvert^{2}\,
                      r\,\mathrm{d}r\,\mathrm{d}\varphi\,\mathrm{d}z}
                     {\int \lvert \psi \rvert^{2}\,
                      r\,\mathrm{d}r\,\mathrm{d}\varphi\,\mathrm{d}z},
            \qquad
            r_\mathrm{rms} = \sqrt{\langle r^2 \rangle}

        The volume element appears in both integrals, so the numerator carries
        a factor :math:`r^3` — :math:`r^2` from the observable and one more
        from :math:`\mathrm{d}V` — and the spacings cancel.

        Args:
            psi (torch.Tensor): Wavefunction of shape ``(n_r, n_phi, n_z)``.
            r (torch.Tensor): Radial grid.
            dr (float): Radial grid spacing.
            dphi (float): Azimuthal grid spacing.
            dz (float): Axial grid spacing.

        Returns:
            torch.Tensor: Scalar RMS radius.
        """
        r_w = r.reshape(-1, 1, 1)
        density = torch.abs(psi) ** 2
        dV = dr * dphi * dz
        # both numerator and denominator share dV → it cancels
        total = torch.sum(density * r_w)
        r2_mean = torch.sum(density * r_w ** 3) / total   # r³ = r² × r (volume element)
        return torch.sqrt(r2_mean)
