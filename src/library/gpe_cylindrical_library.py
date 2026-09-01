"""GPE library in cylindrical coordinates (r, φ, z).

Grid conventions
----------------
r  : half-point grid r_i = (i + 1/2) dr  (i = 0 … n_r-1), avoids r = 0 singularity.
φ  : uniform  0 … 2π  (n_phi points, DFT-ordered).
z  : uniform  z_min … z_max  (n_z points, FFT-ordered).

Kinetic operator
----------------
The radial Laplacian for azimuthal mode m is

    L_r^m ψ = (1/r) d/dr(r dψ/dr) − m²/r² ψ

discretised with the conservative flux scheme on the half-point grid.  The
resulting tridiagonal matrix is NOT symmetric in the ordinary sense, but IS
self-adjoint in the weighted inner product ⟨f,g⟩ = ∫ f* g r dr.  The
symmetrised matrix T̃_r^m = √r · T_r^m · (√r)^{-1} IS real-symmetric and can
be diagonalised once via `torch.linalg.eigh`.

Call `build_radial_operators` at initialisation to obtain the eigendecomposition
dicts; pass them to `p_evolution`, `split_step_step`, `sgpe_step`, etc.

Dimensionless units: ħ = m = ω_ho = 1  (same as GPELibrary).
"""

import warnings
import numpy as np
import torch
from .common_utils import CommonUtils as cu

class GPECylindricalLibrary():
    """
    Cylindrical-coordinate GPE library.

    Reuses shared, coordinate-agnostic helpers from CommonUtils:
        x_evolution, extract_phase, add_phase, update_phase.

    Core operators for cylindrical geometry:
        init_grid, build_radial_operators, normalize, p_evolution,
        split_step_step, mod_grad_psi, calculate_energy_allocation,
        calculate_chemical_potential, generate_thermal_noise, sgpe_step.

    Diagnostics defined here:
        angular_momentum_z.

    The remaining diagnostics (rms_radius, column_density_z,
    column_density_radial, radial_density_profile) and vortex imprinting
    (create_vortices, check_vortex_resolution) live in the subclass
    :class:`GPE2DCylindricalLibrary`, mirroring the Cartesian split between
    ``GPELibrary`` and ``GPE2DLibrary``. Because that subclass inherits from
    this one, it exposes everything in both.
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
        """
        Spectral index axis in FFT (wrapped) order, built directly on ``device``.

        Indices 0 … n/2-1 hold the positive values and n/2 … n-1 the negative
        ones, matching the layout of :func:`torch.fft.fft`.
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
        """
        Initialise a cylindrical (r, φ, z) grid.

        r uses a half-point layout so that the innermost point is at dr/2,
        avoiding the coordinate singularity at r = 0 and naturally imposing
        a Neumann condition (dψ/dr = 0) for m = 0 at the axis.

        Args:
            r_max  : outer radial boundary.
            z_min, z_max : axial extent.
            n_r, n_phi, n_z : grid point counts.
            device : torch device.

        Returns:
            r, phi, z       – 1-D coordinate arrays.
            kz              – z-momentum array (FFT order).
            m_modes         – azimuthal mode indices (DFT order, integer values).
            dr, dphi, dz    – grid spacings.
            space_grid      – (gr, gphi, gz) 3-D meshgrids of shape (n_r, n_phi, n_z).
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
        """
        Build and diagonalise the radial kinetic operator T_r^m for every
        unique |m| appearing in m_modes.

        The conservative discretisation of (1/r)(d/dr)(r d/dr) − m²/r² on the
        half-point grid gives a tridiagonal matrix.  It is symmetrised by the
        √r similarity transform so that torch.linalg.eigh can be used.

        T̃_r^m = diag(√r) · T_r^m · diag(1/√r)

        The radial propagator for mode m over time step Δτ is then

            exp(−i Δτ T_r^m) ψ = (1/√r) · V · exp(−i Δτ Λ) · V^T · (√r ψ)

        where V, Λ come from T̃_r^m = V Λ V^T.

        Args:
            r       : 1-D radial grid (half-point, length n_r).
            dr      : radial spacing.
            m_modes : azimuthal mode indices (DFT order).
            device  : computation device.

        Returns:
            tuple: ``(eigvecs_dict, eigvals_dict)``. Each maps ``|m|`` to a
            tensor — eigenvectors of shape (n_r, n_r) and eigenvalues of
            shape (n_r,). ``eigvecs_dict`` additionally carries the reserved
            key ``_STACK_KEY``, holding ``(V_all, lam_all)``: the same
            operators re-indexed by φ mode, so the whole azimuthal axis can
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
        """
        Re-index the per-|m| eigendecomposition by azimuthal grid position.

        ``p_evolution`` and the ground-state kinetic operator need, for every φ
        index, the eigenbasis of its own |m|. Looking those up one at a time
        costs a Python iteration and a host synchronisation per φ *per time
        step*; stacking them once lets the whole azimuthal axis go through a
        single batched matmul.

        Only ``V_all`` is materialised — the projection uses ``V_allᵀ`` as a
        transposed view, which benchmarks within a few percent of a contiguous
        copy while halving the memory.

        The result is cached inside ``eigvecs_dict`` (under ``_STACK_KEY``), so
        it lives and dies with the operators it belongs to.

        Args:
            eigvecs_dict, eigvals_dict : from :meth:`build_radial_operators`.
            m_modes : azimuthal mode indices (n_phi,), DFT order.

        Returns:
            (V_all, lam_all) of shapes (n_phi, n_r, n_r) and (n_phi, n_r).
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
        """
        Apply ``(1/√r) · V · diag(factors) · Vᵀ · (√r ·)`` to every φ mode at once.

        This is the √r-symmetrised radial operator of the module docstring,
        evaluated in the precomputed eigenbasis. ``V_all`` is real, so the
        complex field is pushed through the two matmuls as an interleaved
        real view: a real GEMM does half the work of the equivalent complex
        one, and the result is bit-identical.

        Args:
            psi_m   : (n_r, n_phi, n_z) complex field, already DFT'd over φ.
            factors : diagonal multiplier per radial eigenmode, broadcastable
                to (n_phi, n_r, n_z). ``exp(−damping·Δτ·Λ)`` for a propagator,
                ``Λ`` itself to apply the operator, a 0/1 mask to project.
            V_all   : (n_phi, n_r, n_r) stacked eigenvectors.
            sqrt_r  : √r on the radial grid (n_r,).

        Returns:
            Tensor of shape (n_r, n_phi, n_z).
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
        """
        Normalise ψ so that ∫|ψ|² r dr dφ dz = 1.

        Args:
            psi           : wavefunction of shape (n_r, n_phi, n_z).
            r             : 1-D radial grid (n_r,).
            dr, dphi, dz  : grid spacings.

        Returns:
            Normalised wavefunction.
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
        """
        Full kinetic evolution exp(−damping · Δτ · T_kin) in cylindrical coordinates.

        Sequence:
            1. DFT over φ  →  azimuthal modes m.
            2. FFT over z  →  axial modes kz.
            3. Apply z kinetic: exp(−damping · Δτ · kz²/2)  [diagonal].
            4. For each m, apply radial kinetic via precomputed eigenbasis.
            5. Inverse FFT over z, inverse DFT over φ.

        The radial propagator uses the √r symmetrisation:
            exp(−damping · Δτ · T_r^m) ψ = (1/√r) V exp(−damping·Δτ·Λ) V^T (√r ψ)

        ``damping`` is 1j for ordinary GPE, (1j + γ) for SGPE.

        Args:
            psi            : (n_r, n_phi, n_z) complex wavefunction.
            dtau           : time step.
            kz             : z-momentum grid (n_z,).
            m_modes        : azimuthal mode indices (n_phi,).
            r              : radial grid (n_r,).
            eigvecs_dict   : {|m|: (n_r, n_r)} from build_radial_operators.
            eigvals_dict   : {|m|: (n_r,)}   from build_radial_operators.
            damping        : complex prefactor (1j for GPE, 1j+γ for SGPE).

        Returns:
            Updated wavefunction of shape (n_r, n_phi, n_z).
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
        """
        Strang split-step: x-half, p-full, x-half.

        For a real ``utot`` every factor applied here has unit modulus, so the
        norm is conserved to machine precision and no renormalisation is needed.
        When ``utot`` carries an imaginary part (three-body losses, complex
        absorbing potential) the norm is *meant* to decay: renormalising would
        silently cancel the atom loss, which is why ``renormalise`` defaults to
        ``False``. Matches :meth:`GPELibrary.split_step_step`.

        Args match those of p_evolution / x_evolution / normalize.

        Returns:
            Wavefunction after one split-step.
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
        """
        Compute |∇ψ| in cylindrical coordinates.

        |∇ψ|² = |∂ψ/∂r|² + (1/r²)|∂ψ/∂φ|² + |∂ψ/∂z|²

        ::

            r-derivative : central finite differences, second-order one-sided
                           at the two radial boundaries (the grid is not
                           periodic in r, so no spectral derivative there).
            φ-derivative : spectral via DFT  →  ∂ψ/∂φ = IDFT(im ψ̃_m).
            z-derivative : spectral via FFT  →  ∂ψ/∂z = IFFT(ikz ψ̃_kz).

        The radial term is the only non-spectral piece in the library, so it
        sets the accuracy of the kinetic energy at O(dr²).

        Returns:
            Real-valued tensor of shape (n_r, n_phi, n_z).
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
        """
        Energy components with cylindrical volume element r dr dφ dz.

        Args:
            psi, Vext : wavefunction and external potential (n_r, n_phi, n_z).
                If Vext is complex (e.g. an absorbing potential), only the real
                part contributes — the imaginary part is a loss rate, not an
                energy.
            r         : radial grid.
            dr, dphi, dz : spacings.
            kz        : z-momentum grid.
            m_modes   : azimuthal mode indices.
            **parameters : must include 'u' (interaction strength).

        Returns:
            dict with keys 'e_kin', 'e_pot', 'e_int', 'E_total'.

        Raises:
            ValueError: If 'u' is not supplied.
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
        """
        Mean-field chemical potential μ = e_kin + e_pot + 2 e_int.

        Identical formula to GPELibrary but uses cylindrical energy calculation.
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
        """
        Complex Gaussian noise for one SGPE time step in cylindrical geometry.

        The local cell volume is dV(r_i) = r_i dr dφ dz, so the noise amplitude
        scales with 1/√dV to satisfy the fluctuation-dissipation theorem:

            amplitude_i = √(γ kT Δτ / (r_i dr dφ dz))

        Projection (the "P" of the *projected* SGPE)
        --------------------------------------------
        Delta-correlated noise is white across the whole grid, so it feeds
        energy into every mode the grid can represent. The c-field description
        is only valid below a cutoff energy, above which the modes belong to
        the thermal cloud rather than the classical field. Supply the kinetic
        operators together with ``e_cut`` to keep only the modes with

            Λ_j^{|m|} + kz²/2  ≤  e_cut

        where Λ is the radial kinetic eigenvalue — the exact cylindrical
        analogue of the Cartesian ``p²/2 ≤ e_cut``. Without it the noise is
        unprojected and the run will heat artificially at short wavelengths.

        Args:
            shape         : (n_r, n_phi, n_z).
            gamma, kT     : damping and temperature.
            dtau          : time step.
            r             : radial grid (n_r,).
            dr, dphi, dz  : spacings.
            device        : computation device.
            kz, m_modes, eigvecs_dict, eigvals_dict : kinetic operators, all
                required for projection (from init_grid / build_radial_operators).
            e_cut         : cutoff energy in units of ħ·ω_ho. Projection is
                applied only when it and the operators above are all supplied.

        Returns:
            Complex noise tensor of shape (n_r, n_phi, n_z).
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
        """
        Deterministic SGPE split-step with (1 − iγ) dissipative damping.

        The SGPE replaces GPE's unitary operator with:

            exp(−(i + γ) Δτ (H_mf − μ))

        Strang splitting:
            1. Real-space half-step  exp(−(i+γ) Δτ/2 (V_eff − μ))
            2. Kinetic full-step     (cylindrical p_evolution with damping i+γ)
            3. Real-space half-step

        The norm is left free so that μ actually drives the dynamics; see
        :meth:`GPELibrary.sgpe_step` for the full argument.

        Args:
            psi             : (n_r, n_phi, n_z) wavefunction.
            utot            : total potential V_ext + u|ψ|² (frozen at step start).
            mu              : chemical potential (reservoir).
            gamma           : damping coefficient.
            dtau            : time step.
            kz              : z-momentum grid.
            m_modes         : azimuthal mode indices.
            r               : radial grid.
            eigvecs_dict, eigvals_dict : from build_radial_operators.
            dr, dphi, dz    : spacings.
            renormalise     : force ∫|ψ|² dV = 1 after the step, which disables
                              the reservoir coupling. Default False.

        Returns:
            Updated wavefunction.
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
        """
        Expectation value of L_z = −i ∂/∂φ in units of ħ.

        In the DFT basis L_z is diagonal with eigenvalue m, so

            ⟨L_z⟩ = Σ_m  m · ∫ |ψ_m(r,z)|² r dr dz · dphi

        The DFT norm='ortho' ensures |ψ_m|² sums to |ψ|² in the phi integral.

        Args:
            psi          : normalised wavefunction (n_r, n_phi, n_z).
            m_modes      : azimuthal mode indices (n_phi,).
            r            : radial grid.
            dr, dphi, dz : spacings.

        Returns:
            Scalar ⟨L_z⟩.
        """
        r_w = r.reshape(-1, 1, 1)
        psi_m = torch.fft.fft(psi, dim=1, norm="ortho")
        m_grid = m_modes.reshape(1, -1, 1)
        Lz = torch.sum(m_grid * torch.abs(psi_m) ** 2 * r_w) * (dr * dphi * dz)
        return torch.real(Lz)

class GPE2DCylindricalLibrary(GPECylindricalLibrary):
    """
    Vortex imprinting and density diagnostics on a cylindrical grid.

    Mirrors the Cartesian ``GPE2DLibrary(GPELibrary)`` relationship: it
    inherits every core operator from :class:`GPECylindricalLibrary` and adds
    the diagnostics the simulation loop reports (rms_radius, column densities,
    radial profile) plus vortex-line imprinting.
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
        """
        Phase and amplitude mask for one or more vortex lines at arbitrary positions.

        Each vortex runs along z.  Its core intersects the (r, φ) plane at the
        Cartesian position (x0, y0) given in ``positions``.  The contribution of
        vortex n to the wavefunction is:

            d_n(r, φ)   = √[(r cos φ − x0)² + (r sin φ − y0)²]
            amplitude_n = tanh(d_n / r_core)
            phase_n     = q_n · atan2(r sin φ − y0, r cos φ − x0)

        All contributions are multiplied together:

            mask = ∏_n  amplitude_n · exp(i · phase_n)

        A vortex at (x0, y0) = (0, 0) reduces to the on-axis case
        tanh(r / r_core) · exp(i q φ).

        If ``dr`` and ``dphi`` are provided, a resolution check is run and a
        warning is issued for any vortex whose core is under-resolved on the
        cylindrical grid.  See ``check_vortex_resolution`` for details.

        Args:
            r, phi      : 1-D cylindrical coordinate axes (lengths n_r, n_phi).
            n_r, n_phi, n_z : grid point counts.
            positions   : list of (x0, y0) Cartesian vortex-core positions.
            charges     : topological charge (winding number) for each vortex.
            r_core      : healing-length core radius (same for all vortices).
            device      : computation device.
            dr          : radial grid spacing (optional, enables resolution check).
            dphi        : azimuthal grid spacing 2π/n_phi (optional, enables check).

        Returns:
            Complex tensor of shape (n_r, n_phi, n_z) to multiply onto ψ.

        Raises:
            ValueError: If ``positions`` and ``charges`` have different lengths.
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
        """
        Check whether the cylindrical grid resolves each vortex core.

        At radial position r0 the effective Cartesian grid spacing is

            dr_eff = max(dr, r0 · dφ)

        where ``dr`` is the radial step and ``r0 · dφ`` is the local arc-length
        step.  The core is considered resolved when

            r_core / dr_eff  >=  min_points_per_core

        i.e. at least ``min_points_per_core`` grid cells span one healing length.

        Args:
            positions          : list of (x0, y0) Cartesian vortex positions.
            r_core             : healing-length core radius.
            dr                 : radial grid spacing.
            dphi               : azimuthal spacing 2π / n_phi.
            min_points_per_core: minimum required resolution ratio (default 2).

        Returns:
            List of dicts, one per vortex:
                'position'   – (x0, y0)
                'r0'         – radial distance of the core from the axis
                'dr_eff'     – effective local Cartesian spacing max(dr, r0·dφ)
                'ratio'      – r_core / dr_eff  (> min_points_per_core = resolved)
                'resolved'   – bool
                'bottleneck' – 'azimuthal' | 'radial'
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
        """
        Column density integrated along z: n(r, φ) = ∫ |ψ|² dz.

        Returns:
            Tensor of shape (n_r, n_phi).
        """
        return torch.sum(torch.abs(psi) ** 2, dim=2) * dz

    @staticmethod
    def column_density_radial(
        psi: torch.Tensor,
        r: torch.Tensor,
        dr: float,
    ) -> torch.Tensor:
        """
        Density integrated over r (with volume weight): n(φ, z) = ∫ |ψ|² r dr.

        Returns:
            Tensor of shape (n_phi, n_z).
        """
        r_w = r.reshape(-1, 1, 1)
        return torch.sum(torch.abs(psi) ** 2 * r_w, dim=0) * dr

    @staticmethod
    def radial_density_profile(
        psi: torch.Tensor,
        dphi: float,
        dz: float,
    ) -> torch.Tensor:
        """
        Azimuthally and axially integrated radial profile n(r) = ∫ |ψ|² dφ dz.

        Returns:
            Tensor of shape (n_r,).
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
        """
        RMS radial extent √⟨r²⟩ weighted by the cylindrical density.

            ⟨r²⟩ = ∫ r² |ψ|² r dr dφ dz / ∫ |ψ|² r dr dφ dz

        Returns:
            Scalar RMS radius.
        """
        r_w = r.reshape(-1, 1, 1)
        density = torch.abs(psi) ** 2
        dV = dr * dphi * dz
        # both numerator and denominator share dV → it cancels
        total = torch.sum(density * r_w)
        r2_mean = torch.sum(density * r_w ** 3) / total   # r³ = r² × r (volume element)
        return torch.sqrt(r2_mean)
