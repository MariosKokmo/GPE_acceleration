import numpy as np
import torch
from .parameters import CONSTANTS
from .common_utils import CommonUtils as cu

class GPELibrary:
    @staticmethod
    def _real_axis(
        n: int,
        x_min: float,
        dx: float,
        device: torch.device
    ) -> torch.Tensor:
        """Real-space axis x_i = x_min + i·dx, i = 0 … n-1, built on ``device``."""
        return float(x_min) + torch.arange(n, dtype=torch.float64, device=device) * float(dx)

    @staticmethod
    def _momentum_axis(
        n: int,
        dp: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Momentum axis in FFT (wrapped) ordering, built on ``device``.

        Indices 0 … n/2-1 carry positive momenta, indices n/2 … n-1 carry the
        negative ones, matching the layout of :func:`torch.fft.fftn`.
        """
        k = torch.arange(n, dtype=torch.float64, device=device)
        k[n // 2:] -= n
        return float(dp) * k

    @staticmethod
    def _broadcast_momentum_axes(
        p_axes,
        dim: int,
        zero_nyquist: bool = False
    ) -> list:
        """
        Normalise a momentum-grid argument to ``dim`` broadcastable tensors.

        Both conventions used across this library are accepted:

        - a tuple of **1-D axes** ``(p1, p2, p3)`` as returned by
          :meth:`init_grid` — each axis is reshaped so it broadcasts along its
          own dimension (no meshgrid materialised);
        - a tuple of **N-D meshgrids** ``(px, py, pz)`` (``p_grid`` from
          :meth:`init_grid`) — passed through unchanged.

        The Nyquist mode
        ----------------
        On an even-sized grid the highest mode is its own alias (+π/dx and
        −π/dx are the same sampled wave), so its *first* derivative is
        undefined: ``i·p`` there breaks the Hermitian symmetry that keeps the
        derivative of a real field real. The leak is tiny in absolute terms but
        gets divided by the density in the velocity field, where it can
        dominate the dilute tail. Odd-order derivatives should therefore set
        that coefficient to zero — ``p²`` in the kinetic operator is an *even*
        order and must keep it, which is why this is opt-in.

        Args:
            p_axes: Sequence of momentum tensors, at least ``dim`` of them.
            dim (int): Number of spatial dimensions of the field.
            zero_nyquist (bool): Drop the Nyquist coefficient of each axis.
                Set this for any first derivative.

        Returns:
            list[torch.Tensor]: ``dim`` tensors that broadcast against the field.

        Raises:
            ValueError: If too few axes are supplied or an axis has a rank that
                is neither 1 nor ``dim``.
        """
        if len(p_axes) < dim:
            raise ValueError(
                f"Expected at least {dim} momentum axes for a {dim}-D field, got {len(p_axes)}"
            )
        broadcast = []
        for k, p in enumerate(p_axes[:dim]):
            if p.dim() == 1:
                if zero_nyquist and p.numel() % 2 == 0:
                    p = p.clone()
                    p[p.numel() // 2] = 0.0
                shape = [1] * dim
                shape[k] = p.numel()
                broadcast.append(p.reshape(shape))
            elif p.dim() == dim:
                if zero_nyquist and p.shape[k] % 2 == 0:
                    p = p.clone()
                    p.index_fill_(k, torch.tensor([p.shape[k] // 2], device=p.device), 0.0)
                broadcast.append(p)
            else:
                raise ValueError(
                    f"Momentum axis {k} has {p.dim()} dimensions; expected 1 (axis) or {dim} (meshgrid)"
                )
        return broadcast

    @staticmethod
    def superfluid_velocity(
        psi: torch.Tensor,
        p_axes,
        components=None,
        density_floor: float = 1e-12
    ) -> list:
        """
        Superfluid velocity components, in dimensionless units (ℏ/m = 1):

            v_i = Im( ψ* ∂_i ψ ) / |ψ|²

        Take the derivative of ψ, never of its phase
        --------------------------------------------
        This expression equals ∂_i θ for the *unwrapped* phase θ, but it is
        computed from ψ, which is single-valued and smooth. Differentiating
        ``angle(ψ)`` instead looks equivalent and is not: the wrapped phase
        jumps by 2π across a branch cut running out of every vortex core, and a
        spectral derivative of a discontinuous field rings across the *whole*
        domain, not just near the cut. Measured against an analytic
        vortex-antivortex field, the phase route is wrong by more than 100%
        everywhere while this one is limited only by grid resolution.

        The velocity is undefined where there is no condensate. In floating
        point the density is essentially never exactly zero, so a bare
        ``density > 0`` test lets the dilute tail (n ~ 1e-30) produce enormous
        meaningless values. The cut is therefore taken *relative* to the peak
        density, and the division itself is guarded so no inf/NaN is generated
        in the discarded region.

        Args:
            psi (torch.Tensor): Wavefunction.
            p_axes: Momentum axes or meshgrids, one per dimension of ``psi``.
            components (iterable[int], optional): Which components to compute.
                Defaults to all of them.
            density_floor (float, optional): Threshold relative to the peak
                density below which the velocity is reported as zero.

        Returns:
            list[torch.Tensor]: One velocity field per requested component.
        """
        axes = GPELibrary._broadcast_momentum_axes(p_axes, psi.dim(), zero_nyquist=True)
        if components is None:
            components = range(len(axes))

        psi_f = torch.fft.fftn(psi, norm='forward')
        density = torch.abs(psi) ** 2

        valid = density > density_floor * torch.max(density)
        # Divide by a safe denominator so the masked-out region never evaluates
        # 0/0; torch.where would otherwise still compute (and propagate) NaNs.
        safe_density = torch.where(valid, density, torch.ones_like(density))

        velocities = []
        for k in components:
            dpsi = torch.fft.ifftn(1j * axes[k] * psi_f, norm='forward')
            numerator = torch.imag(psi.conj() * dpsi)
            velocities.append(
                torch.where(valid, numerator / safe_density, torch.zeros_like(numerator))
            )
        return velocities

    @staticmethod
    def init_grid(
        x_min: list,
        dx: list,
        dp: list,
        n1: int,
        n2: int,
        n3: int,
        device: torch.device
    ) -> tuple:
        """
        Initializes the grid for real and momentum space.

        All tensors are allocated directly on ``device`` so that no host/device
        transfer is needed later in the evolution loop.

        Args:
            x_min (list): Minimum values for the x-axis in each dimension.
            dx (list): Grid spacing in real space for each dimension.
            dp (list): Grid spacing in momentum space for each dimension.
            n1, n2, n3 (int): Number of grid points in each dimension.
            device (torch.device): Device to allocate tensors (CPU or GPU).

        Returns:
            tuple: (x1, x2, x3, p1, p2, p3, p_sq, space_grid, p_grid) where
            x1..x3 and p1..p3 are 1-D axes, p_sq is |p|² on the full grid, and
            space_grid / p_grid are the corresponding 3-D meshgrids.
        """
        x1 = GPELibrary._real_axis(n1, x_min[0], dx[0], device)
        x2 = GPELibrary._real_axis(n2, x_min[1], dx[1], device)
        x3 = GPELibrary._real_axis(n3, x_min[2], dx[2], device)

        p1 = GPELibrary._momentum_axis(n1, dp[0], device)
        p2 = GPELibrary._momentum_axis(n2, dp[1], device)
        p3 = GPELibrary._momentum_axis(n3, dp[2], device)

        # indexing='ij' is required: the default flips to 'xy' in future PyTorch
        # releases, which would silently transpose the first two axes.
        g_px, g_py, g_pz = torch.meshgrid(p1, p2, p3, indexing='ij')
        p_sq = g_px**2 + g_py**2 + g_pz**2
        p_grid = (g_px, g_py, g_pz)

        g_x, g_y, g_z = torch.meshgrid(x1, x2, x3, indexing='ij')
        space_grid = (g_x, g_y, g_z)
        return x1, x2, x3, p1, p2, p3, p_sq, space_grid, p_grid

    @staticmethod
    def p_evolution(
        psi1: torch.Tensor,
        dtau: float,
        p_sq: torch.Tensor
    ) -> torch.Tensor:
        """
        Momentum-space evolution step for the wavefunction.

        Args:
            psi1 (torch.Tensor): Wavefunction of the system.
            dtau (float): Time evolution step.
            p_sq (torch.Tensor): Squared momentum grid.

        Returns:
            torch.Tensor: Updated wavefunction.
        """
        psiF = torch.fft.fftn(psi1, norm='forward')
        psiF = torch.exp(-1j * dtau * 0.5 * p_sq) * psiF
        return torch.fft.ifftn(psiF, norm='forward')

    @staticmethod
    def normalize(
        phi: torch.Tensor,
        d_x: float
    ) -> torch.Tensor:
        """
        Normalize the wavefunction.

        Args:
            phi (torch.Tensor): Wavefunction to be normalized.
            d_x (float): Grid cell volume.

        Returns:
            torch.Tensor: Normalized wavefunction, ∫|φ|² dV = 1.

        Note:
            No guard is placed on a vanishing norm; feeding in an identically
            zero field returns NaN. Checking for it here would force a
            host/device synchronisation on every time step.
        """
        return phi / torch.sqrt(d_x * torch.sum(torch.abs(phi) ** 2))

    @staticmethod
    def split_step_step(
        psi1: torch.Tensor,
        utot1: torch.Tensor,
        dtau: float,
        p_sq: torch.Tensor,
        d_x: float,
        renormalise: bool = False
    ) -> torch.Tensor:
        """
        Perform a step of the split-step Fourier transform.

        For a real ``utot1`` every factor applied here has unit modulus, so the
        norm is conserved to machine precision and no renormalisation is needed.
        When ``utot1`` carries an imaginary part (three-body losses, complex
        absorbing potential) the norm is *meant* to decay: renormalising would
        silently cancel the atom loss, which is why ``renormalise`` defaults to
        ``False``.

        Args:
            psi1 (torch.Tensor): Wavefunction of the system.
            utot1 (torch.Tensor): Total potential.
            dtau (float): Time evolution step.
            p_sq (torch.Tensor): Squared momentum grid.
            d_x (float): Grid cell volume (only used when ``renormalise``).
            renormalise (bool, optional): Force ∫|ψ|² dV = 1 after the step.
                Only meaningful for a number-conserving run with a lossy
                potential. Default ``False``.

        Returns:
            torch.Tensor: Updated wavefunction.
        """
        psi1 = cu.x_evolution(psi1, utot1, dtau)
        psi1 = GPELibrary.p_evolution(psi1, dtau, p_sq)
        psi1 = cu.x_evolution(psi1, utot1, dtau)
        if renormalise:
            return GPELibrary.normalize(psi1, d_x)
        return psi1

    @staticmethod
    def mod_grad_psi(
        psi: torch.Tensor,
        p_axes: list
    ) -> torch.Tensor:
        """
        Calculate the modulus of the gradient of the wavefunction, |∇ψ|.

        The derivative of a complex field is itself complex, so each component
        contributes ``|∂_i ψ|² = Re(∂_i ψ)² + Im(∂_i ψ)²``. Keeping only the
        real part (as an earlier version did) understates the kinetic energy of
        any state carrying phase structure — vortices, solitons, or simply a
        state undergoing real-time evolution.

        Works for 1-, 2- and 3-D fields. ``p_axes`` may be either 1-D momentum
        axes or full meshgrids (see :meth:`_broadcast_momentum_axes`).

        Args:
            psi (torch.Tensor): Condensate wavefunction.
            p_axes (list): Momentum space axes (or meshgrids), one per dimension.

        Returns:
            torch.Tensor: Modulus of the gradient of the wavefunction, same
            shape as ``psi``.
        """
        dim = psi.dim()
        if dim not in (1, 2, 3):
            raise ValueError(f"psi must be 1-, 2- or 3-dimensional; got {dim} dimensions")

        # First derivative: drop the Nyquist coefficient (see the helper).
        axes = GPELibrary._broadcast_momentum_axes(p_axes, dim, zero_nyquist=True)
        # One forward transform serves every component.
        psi_f = torch.fft.fftn(psi, norm='forward')

        grad_sq = None
        for p in axes:
            grad = torch.fft.ifftn(1j * p * psi_f, norm='forward')
            term = torch.abs(grad) ** 2
            grad_sq = term if grad_sq is None else grad_sq + term
        return torch.sqrt(grad_sq)

    @staticmethod
    def calculate_energy_allocation(
        psi: torch.Tensor,
        Vext: torch.Tensor,
        p_grid: tuple,
        d_x: float,
        **parameters
    ) -> dict:
        """
        Calculate energy allocation for the condensate.

        Every term is an integral over the grid, so each sum carries the cell
        volume ``d_x``; ψ is normalised as ``d_x·Σ|ψ|² = 1``. Omitting it makes
        the energies scale with the grid spacing instead of being physical.

            e_kin = ½ ∫|∇ψ|² dV
            e_pot = ∫ V_ext |ψ|² dV
            e_int = ½ u ∫|ψ|⁴ dV

        Args:
            psi (torch.Tensor): BEC wavefunction.
            Vext (torch.Tensor): External potential. If complex (e.g. an
                absorbing potential), only the real part contributes to the
                energy — the imaginary part is a loss rate, not an energy.
            p_grid (tuple): Momentum space axes (1-D) or meshgrids.
            d_x (float): Grid cell volume (product of dx in each dimension).
            **parameters: Must include the interaction strength ``u``.

        Returns:
            dict: Energy terms (kinetic, potential, interaction, total), in
            units of ħ·ω_ho.

        Raises:
            ValueError: If ``u`` is not supplied.
        """
        if 'u' not in parameters:
            raise ValueError("calculate_energy_allocation requires the interaction strength 'u'")
        u = parameters['u']

        potential = Vext.real if torch.is_complex(Vext) else Vext
        density = torch.abs(psi) ** 2
        grad_sq = GPELibrary.mod_grad_psi(psi, p_grid) ** 2

        e_kin = 0.5 * torch.sum(grad_sq) * d_x
        e_pot = torch.sum(potential * density) * d_x
        e_int = 0.5 * u * torch.sum(density ** 2) * d_x
        E_total = e_kin + e_pot + e_int
        return {
            'e_kin': e_kin,
            'e_pot': e_pot,
            'e_int': e_int,
            'E_total': E_total
        }

    @staticmethod
    def calculate_chemical_potential(
        psi: torch.Tensor,
        uext: torch.Tensor,
        u: float,
        p_grid: tuple,
        d_x: float
    ) -> float:
        """
        Compute the mean-field chemical potential μ = ⟨ψ|H_mf|ψ⟩.

        This is used by the SGPE as the grand-canonical reservoir potential
        that drives condensate growth (modes below μ) and decay (modes above μ).

        The mean-field Hamiltonian in dimensionless units (ħ = m = ω_ho = 1) is:

            H_mf = -∇²/2 + V_ext + u|ψ|²

        The chemical potential is then:

            μ = ⟨ψ|H_mf|ψ⟩ = e_kin + e_pot + 2·e_int

        where e_int = (u/2)∫|ψ|⁴ dV.  The interaction term is counted *twice*
        because μ = ∂E/∂N and differentiating the (u/2)N² term yields uN.

        Args:
            psi (torch.Tensor): Normalised BEC wavefunction (n1, n2, n3).
            uext (torch.Tensor): External trapping potential on the grid.
            u (float): Dimensionless interaction strength.
            p_grid (tuple): (p1, p2, p3) momentum axes, either as 1-D axes or
                as 3-D meshgrids.
            d_x (float): Grid cell volume (product of dx in each dimension).

        Returns:
            float: Chemical potential μ in units of ħ·ω_ho.
        """
        energy = GPELibrary.calculate_energy_allocation(psi, uext, p_grid, d_x, u=u)
        mu = energy['e_kin'] + energy['e_pot'] + 2.0 * energy['e_int']
        return float(mu.real)

    @staticmethod
    def generate_thermal_noise(
        shape: tuple,
        gamma: float,
        kT: float,
        dtau: float,
        d_x: float,
        device: torch.device,
        p_sq: torch.Tensor = None,
        e_cut: float = None
    ) -> torch.Tensor:
        """
        Generate a complex Gaussian noise field for one SGPE time step.

        The SGPE noise must satisfy the fluctuation-dissipation theorem:

            ⟨η*(r,t) η(r',t')⟩ = 2γ·k_BT · δ(r−r') · δ(t−t')

        Discretising on a grid with cell volume δV = d_x and time step δt = dtau:

            noise amplitude = √(γ · kT · dtau / d_x)

        so that ⟨|Δψ_noise|²⟩ = 2·γ·kT·dtau/d_x per grid point, matching the
        continuous fluctuation-dissipation relation.

        Projection (the "P" of the *projected* SGPE)
        --------------------------------------------
        Delta-correlated noise is white across the whole grid, so it feeds
        energy into every mode up to the Nyquist momentum. The c-field
        description is only valid up to a cutoff energy, above which the modes
        belong to the thermal cloud rather than the classical field. Pass
        ``p_sq`` together with ``e_cut`` to zero the noise in every mode with
        p²/2 > e_cut; without it the noise is unprojected and the run will heat
        artificially at large momenta.

        Args:
            shape (tuple): Grid shape (n1, n2, n3).
            gamma (float): Dimensionless damping coefficient γ.
            kT (float): Dimensionless temperature k_B·T / (ħ·ω_ho).
            dtau (float): Dimensionless time step ω_ho·dt.
            d_x (float): Grid cell volume (product of dx in each dimension).
            device (torch.device): Computation device.
            p_sq (torch.Tensor, optional): Squared momentum grid |p|², required
                for projection.
            e_cut (float, optional): Cutoff energy p²/2 in units of ħ·ω_ho.
                Ignored unless ``p_sq`` is also given.

        Returns:
            torch.Tensor: Complex noise tensor of shape (n1, n2, n3).
        """
        amplitude = (gamma * kT * dtau / d_x) ** 0.5
        xi_real = torch.randn(shape, dtype=torch.float64, device=device)
        xi_imag = torch.randn(shape, dtype=torch.float64, device=device)
        noise = amplitude * (xi_real + 1j * xi_imag).to(torch.cdouble)

        if p_sq is not None and e_cut is not None:
            noise_f = torch.fft.fftn(noise, norm='forward')
            noise_f = torch.where(0.5 * p_sq <= e_cut, noise_f, torch.zeros_like(noise_f))
            noise = torch.fft.ifftn(noise_f, norm='forward')
        return noise

    @staticmethod
    def sgpe_step(
        psi: torch.Tensor,
        utot: torch.Tensor,
        mu: float,
        gamma: float,
        dtau: float,
        p_sq: torch.Tensor,
        d_x: float,
        renormalise: bool = False
    ) -> torch.Tensor:
        """
        Perform one deterministic SGPE split-step with (1 − iγ) damping.

        The SGPE modifies the GPE by replacing the purely unitary evolution
        operator with a dissipative one:

            GPE:   exp(−i · dt · H_mf)
            SGPE:  exp(−(i + γ) · dt · (H_mf − μ))

        The (i + γ) factor arises from the (1 − iγ) prefactor in the SGPE:

            ∂ψ/∂t = −(i + γ)(H_mf − μ)ψ + noise

        Modes with H_mf > μ are exponentially damped (energy removed to reservoir).
        Modes with H_mf < μ are amplified (energy drawn from reservoir).
        This drives the system toward the thermal equilibrium state at temperature T.

        The split-step sequence (Strang splitting) is:

            1. Real-space half-step:  ψ ← exp(−(i+γ)·Δτ/2·(V_eff − μ)) · ψ
            2. Momentum full-step:    ψ̃ ← exp(−(i+γ)·Δτ·p²/2) · ψ̃
            3. Real-space half-step:  ψ ← exp(−(i+γ)·Δτ/2·(V_eff − μ)) · ψ

        where V_eff = u|ψ|² + V_ext is frozen at the start of the step.

        The norm is deliberately left free
        ----------------------------------
        μ enters only as the constant shift ``utot − mu``, so the whole step
        multiplies ψ by the global factor ``exp((i+γ)·Δτ·μ)``. Renormalising
        afterwards divides that factor straight back out and leaves nothing but
        an unobservable global phase: with a forced norm, μ has *no effect
        whatsoever* on the dynamics and the grand-canonical growth/decay the
        SGPE is built around simply does not happen.

        Letting the norm evolve is also self-consistent with the units used
        here. With ``d_x·Σ|ψ|² = 1`` at t=0 and ``u = 4πN₀a/a_ho``, the
        mean-field term ``u|ψ|²`` equals ``(4πa/a_ho)·N₀|ψ|²``, i.e. the
        interaction energy of the *current* atom number N(t) = N₀·‖ψ‖². The
        reservoir therefore sets N through μ, exactly as intended.

        Args:
            psi (torch.Tensor): Wavefunction (n1, n2, n3), complex double.
            utot (torch.Tensor): Total mean-field potential V_ext + u|ψ|².
            mu (float): Reservoir chemical potential μ in units of ħ·ω_ho.
            gamma (float): Dimensionless damping coefficient γ.
            dtau (float): Dimensionless time step ω_ho·dt.
            p_sq (torch.Tensor): Squared momentum grid |p|².
            d_x (float): Grid cell volume (only used when ``renormalise``).
            renormalise (bool, optional): Force ∫|ψ|² dV = 1 after the step.
                This disables the reservoir coupling described above; only use
                it for a deliberately number-conserving damped GPE. Default
                ``False``.

        Returns:
            torch.Tensor: Updated wavefunction.
        """
        damping = 1j + gamma
        eff_pot = utot - mu

        # Real-space half-step
        psi = torch.exp(-damping * 0.5 * dtau * eff_pot) * psi

        # Momentum full-step
        psiF = torch.fft.fftn(psi, norm='forward')
        psiF = torch.exp(-damping * dtau * 0.5 * p_sq) * psiF
        psi = torch.fft.ifftn(psiF, norm='forward')

        # Real-space half-step
        psi = torch.exp(-damping * 0.5 * dtau * eff_pot) * psi

        if renormalise:
            return GPELibrary.normalize(psi, d_x)
        return psi

    @staticmethod
    def calculate_density_peak(
        psi: torch.Tensor
    ) -> tuple:
        """
        Calculate the maximum density and its position in the wavefunction.

        Args:
            psi (torch.Tensor): BEC normalised wavefunction.

        Returns:
            tuple: (max_density, peak_indices) where max_density is a scalar tensor
                   and peak_indices is a tuple of integers (i, j, k) representing
                   the grid position of the maximum density.
        """
        density = torch.abs(psi) ** 2
        density_flat = density.flatten()
        peak_position = torch.argmax(density_flat).item()
        max_density = density_flat[peak_position]
        
        # Manual unravel_index for 3D tensor
        shape = density.shape
        k = peak_position % shape[2]
        j = (peak_position // shape[2]) % shape[1]
        i = peak_position // (shape[2] * shape[1])
        
        return max_density, (i, j, k)

class GPE2DLibrary(GPELibrary):
    @staticmethod
    def create_vortices(
        vortices: np.ndarray,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        n1: int,
        n2: int,
        n3: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Creates vortices on the condensate by calculating a new phase to be added (2D BEC).

        The winding is built with the half-angle identity
        ``atan2(y, r + t) = θ/2`` (r = √(t²+y²)), which is free of the branch
        cut that a plain ``atan2(y, t)`` would introduce. The identity breaks
        down only on the negative-t half-axis (y = 0, t < 0), where r + t = 0;
        those points are assigned the limiting value q·π explicitly.

        The phase is identical for every y index, so it is computed once on the
        (n1, n3) plane and broadcast along the second dimension. The previous
        implementation looped over n1·n3 grid points in Python and indexed the
        tensor element by element, which dominated the setup cost of a run.

        Args:
            vortices (np.ndarray): Shape (3, number_of_vortices), rows are x positions,
                z positions, vortex charges. Positions are grid offsets relative
                to the centre of the grid. A flat (3,) array is accepted for a
                single vortex.
            x1, x2, x3 (torch.Tensor): Real space axes for each dimension.
            n1, n2, n3 (int): Number of grid points in each dimension.
            device (torch.device): Device to allocate tensors (CPU or GPU).

        Returns:
            torch.Tensor: Real phase (n1, n2, n3) to be added to the condensate
            phase, or None if ``vortices`` is None.

        Raises:
            ValueError: If ``vortices`` is not of shape (3, N) or a vortex sits
                outside the grid.
        """
        if vortices is None:
            return None

        v = np.asarray(vortices)
        if v.ndim == 1:
            v = v.reshape(3, 1)
        if v.ndim != 2 or v.shape[0] != 3:
            raise ValueError(
                f"vortices must have shape (3, number_of_vortices); got {v.shape}"
            )

        ax1 = x1.to(device=device, dtype=torch.float64).reshape(n1, 1)
        ax3 = x3.to(device=device, dtype=torch.float64).reshape(1, n3)
        k_idx = torch.arange(n1, device=device).reshape(n1, 1)
        i_idx = torch.arange(n3, device=device).reshape(1, n3)

        phase_xz = torch.zeros((n1, n3), dtype=torch.float64, device=device)
        for n in range(v.shape[1]):
            core_x = int(v[0][n]) + n1 // 2
            core_z = int(v[1][n]) + n3 // 2
            q = float(v[2][n])
            if not (0 <= core_x < n1 and 0 <= core_z < n3):
                raise ValueError(
                    f"Vortex {n} at grid index ({core_x}, {core_z}) lies outside "
                    f"the {n1}x{n3} grid"
                )

            t = ax1 - ax1[core_x, 0]          # (n1, 1)
            y = ax3 - ax3[0, core_z]          # (1, n3)
            r = torch.sqrt(t ** 2 + y ** 2)   # (n1, n3) by broadcasting
            winding = 2.0 * q * torch.atan2(y, r + t)

            # Negative-t half-axis: the half-angle formula degenerates to
            # atan2(0, 0); the correct limit of the winding there is q·π.
            on_cut = (i_idx == core_z) & (k_idx < core_x)
            phase_xz = phase_xz + torch.where(
                on_cut, torch.full_like(winding, q * CONSTANTS.pi), winding
            )

        return phase_xz.reshape(n1, 1, n3).expand(n1, n2, n3)

    @staticmethod
    def repetitive_imprint(
        psi1: torch.Tensor,
        repetitive_phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Re-imprint the wavefunction by adding the repetitive phase (2D BEC).

        Args:
            psi1 (torch.Tensor): Wavefunction of the system.
            repetitive_phase (torch.Tensor): Repetitive phase to be added.

        Returns:
            torch.Tensor: Updated wavefunction.
        """
        return cu.update_phase(psi1, repetitive_phase)

    @staticmethod
    def calculate_velocity2D(
        psi: torch.Tensor,
        p_grid: tuple,
        density_floor: float = 1e-12
    ) -> tuple:
        """
        In-plane superfluid velocity of the condensate, as magnitude and direction.

        The simulation plane of the 2-D model is x–z (n1–n3), so the in-plane
        components are v_x and v_z; for a genuinely 2-D field they are the two
        axes it has. Multiply by ℏ/m for a dimensional velocity.

        Takes the wavefunction, not the phase
        -------------------------------------
        This used to differentiate ``angle(ψ)``. That is wrong wherever the
        condensate carries circulation: the wrapped phase has a 2π branch cut
        running out of every vortex core, and because a spectral derivative is
        global, the discontinuity corrupts the velocity across the entire
        domain rather than only near the cut. Against an analytic
        vortex-antivortex field the phase route was off by 146% at the median
        and 5383% at worst, versus ~5% (grid resolution) for this one. See
        :meth:`GPELibrary.superfluid_velocity`.

        Args:
            psi (torch.Tensor): Condensate wavefunction.
            p_grid (tuple): Momentum axes or meshgrids, one per dimension of
                ``psi``.
            density_floor (float, optional): Threshold relative to the peak
                density below which the velocity is reported as zero.

        Returns:
            tuple: (|v| in the plane, direction of v as atan2(v_z, v_x)).
        """
        # x–z for a 3-D array; the only two axes of a genuinely 2-D one.
        in_plane = (0, 2) if psi.dim() == 3 else (0, 1)
        v_a, v_b = GPELibrary.superfluid_velocity(
            psi, p_grid, components=in_plane, density_floor=density_floor)
        return torch.sqrt(v_a ** 2 + v_b ** 2), torch.atan2(v_b, v_a)

    @staticmethod
    def rms_radius(
        psi: torch.Tensor,
        center: list,
        space_grid: tuple
    ) -> torch.Tensor:
        """
        Calculate the RMS radius of the condensate.

        Args:
            psi (torch.Tensor): Normalized wavefunction.
            center (list): Centers of the space axes (x1, x2, x3).
            space_grid (tuple): Meshgrid of the space.

        Returns:
            torch.Tensor: RMS radius of the condensate.
        """
        center_x, center_y, center_z = center
        # Use sum of probability densities as the normalization factor for weighted
        # average. The cell volume cancels between numerator and denominator, so it
        # is deliberately omitted here.
        total_density = torch.sum(torch.abs(psi) ** 2)
        
        g_x, g_y, g_z = space_grid
        d_sq = (g_x - center_x) ** 2 + (g_y - center_y) ** 2 + (g_z - center_z) ** 2
        
        # RMS = sqrt( sum(r^2 * density) / sum(density) )
        rms = (torch.sum(d_sq * (torch.abs(psi) ** 2)) / total_density) ** 0.5
        return rms

    @staticmethod
    def create_dark_soliton(
        x1: torch.Tensor,
        x3: torch.Tensor,
        n1: int,
        n2: int,
        n3: int,
        positions: list,
        widths: list,
        axes: list,
        greyness: list = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Create a multiplicative dark-soliton profile for the wavefunction.

        Each soliton is a stripe of suppressed density across the condensate.
        The profile for a single dark soliton along coordinate *r* is

            f(r) = cos(alpha) * tanh( cos(alpha) * (r - r0) / width ) + i * sin(alpha)

        where ``alpha = 0`` gives a stationary black soliton (full density dip
        plus a pi phase jump) and ``0 < alpha < pi/2`` gives a grey (moving)
        soliton with a shallower dip.

        When multiple solitons are requested the individual profiles are
        multiplied together.

        Args:
            x1 (torch.Tensor): 1-D real-space axis along the first grid dimension.
            x3 (torch.Tensor): 1-D real-space axis along the third grid dimension.
            n1, n2, n3 (int): Number of grid points in each dimension.
            positions (list[float]): Centre positions of each soliton (in the
                same units as the corresponding axis).
            widths (list[float]): Characteristic width of each soliton.  For a
                BEC this is typically the healing length xi.
            axes (list[int]): Axis for each soliton: ``1`` for x (stripe
                perpendicular to x1) or ``3`` for z (stripe perpendicular to x3).
            greyness (list[float], optional): Grey-soliton angle alpha for
                each soliton in radians.  ``0`` = black (default), values up
                to ``pi/2`` make the soliton progressively greyer / faster.
            device (torch.device): Device for the output tensor.

        Returns:
            torch.Tensor: Complex-valued tensor of shape ``(n1, n2, n3)`` to
            be multiplied element-wise with the wavefunction.

        Raises:
            ValueError: If the per-soliton lists have inconsistent lengths, or
                an axis is neither 1 nor 3.
        """
        n_solitons = len(positions)
        if greyness is None:
            greyness = [0.0] * n_solitons

        # zip() would silently drop solitons on a length mismatch.
        for name, values in (("widths", widths), ("axes", axes), ("greyness", greyness)):
            if len(values) != n_solitons:
                raise ValueError(
                    f"soliton '{name}' has {len(values)} entries but there are "
                    f"{n_solitons} positions"
                )

        # Build 3-D coordinate grids (only the two relevant axes matter)
        grid_x1 = x1.to(device=device, dtype=torch.float64)
        grid_x3 = x3.to(device=device, dtype=torch.float64)
        # shape: (n1, 1, 1) and (1, 1, n3)
        gx = grid_x1.reshape(n1, 1, 1).expand(n1, n2, n3)
        gz = grid_x3.reshape(1, 1, n3).expand(n1, n2, n3)

        mask = torch.ones(n1, n2, n3, dtype=torch.cdouble, device=device)

        for pos, w, ax, alpha in zip(positions, widths, axes, greyness):
            if ax == 1:
                r = gx
            elif ax == 3:
                r = gz
            else:
                raise ValueError(f"Soliton axis must be 1 (x) or 3 (z), got {ax}")

            cos_a = float(np.cos(alpha))
            sin_a = float(np.sin(alpha))
            profile = cos_a * torch.tanh(cos_a * (r - pos) / w) + 1j * sin_a
            mask = mask * profile

        return mask

    @staticmethod
    def imprint_dark_soliton(
        psi: torch.Tensor,
        soliton_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply a dark-soliton mask to the wavefunction.

        Unlike vortex imprinting (which adds pure phase), this multiplies
        the wavefunction by a complex profile that modifies both amplitude
        and phase simultaneously.

        Args:
            psi (torch.Tensor): Current wavefunction (n1, n2, n3).
            soliton_mask (torch.Tensor): Profile returned by
                :meth:`create_dark_soliton`.

        Returns:
            torch.Tensor: Updated wavefunction.
        """
        return psi * soliton_mask

    @staticmethod
    def calculate_cross_section_line(
        psi: torch.Tensor,
        axis: int = 1
    ) -> torch.Tensor:
        """
        Calculate the column density on a line that crosses the condensate.

        The line runs *along* the requested axis through the centre of the
        grid, with the trivial y direction (n2) integrated out:

            axis=1 → n(x) = Σ_y |ψ[:, :, n3//2]|²   (length n1)
            axis=2 → n(z) = Σ_y |ψ[n1//2, :, :]|²   (length n3)

        Args:
            psi (torch.Tensor): BEC wavefunction.
            axis (int, optional): Axis the line runs along (1 for x, 2 for z).
                Default is 1.

        Returns:
            torch.Tensor: Cross-section line through the center of the BEC.

        Raises:
            ValueError: If ``axis`` is neither 1 nor 2.
        """
        n1, n2, n3 = psi.shape
        if axis == 1:
            return torch.sum(torch.abs(psi[:, :, n3 // 2]) ** 2, dim=1)
        elif axis == 2:
            return torch.sum(torch.abs(psi[n1 // 2, :, :]) ** 2, dim=0)
        raise ValueError(f"axis must be 1 (x) or 2 (z); got {axis}")

class GPE3DLibrary(GPELibrary):
    """
    Extensions of the GPE library for fully three-dimensional BEC simulations.

    Provides tools for 3D-specific topological structures (vortex rings,
    vortex lines), density diagnostics (column densities, 2D slices), and
    observables (superfluid velocity field, angular momentum components).
    """

    @staticmethod
    def create_vortex_ring(
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        n1: int,
        n2: int,
        n3: int,
        ring_radius: float,
        center: tuple,
        axis: int,
        charge: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Create the phase texture for a quantized vortex ring.

        The ring is a circle of radius ``ring_radius`` centred at ``center``
        lying in the plane perpendicular to ``axis``.  The phase winds by
        2π * charge around the vortex core (the ring itself) using toroidal
        coordinates:

            phi = atan2(x_axis - c_axis, rho - ring_radius)

        where rho is the cylindrical radius in the plane of the ring.

        Args:
            x1, x2, x3 (torch.Tensor): 1-D real-space axes.
            n1, n2, n3 (int): Grid point counts per dimension.
            ring_radius (float): Radius of the vortex ring, in the same units
                as the real-space axes.
            center (tuple): (c1, c2, c3) – 3-D centre of the ring, in the same
                real-space units as x1, x2, x3 (not grid indices).
            axis (int): 1, 2, or 3 – axis the ring encircles (ring lies in the
                perpendicular plane).
            charge (int): Topological charge (winding number). Default 1.
            device (torch.device): Computation device.

        Returns:
            torch.Tensor: Real phase tensor of shape (n1, n2, n3).
        """
        if axis not in (1, 2, 3):
            raise ValueError(f"axis must be 1, 2, or 3; got {axis}")

        c1, c2, c3 = center
        gx = x1.to(device=device, dtype=torch.float64).reshape(n1, 1, 1).expand(n1, n2, n3)
        gy = x2.to(device=device, dtype=torch.float64).reshape(1, n2, 1).expand(n1, n2, n3)
        gz = x3.to(device=device, dtype=torch.float64).reshape(1, 1, n3).expand(n1, n2, n3)

        if axis == 1:
            rho = torch.sqrt((gy - c2) ** 2 + (gz - c3) ** 2)
            phi = torch.atan2(gx - c1, rho - ring_radius)
        elif axis == 2:
            rho = torch.sqrt((gx - c1) ** 2 + (gz - c3) ** 2)
            phi = torch.atan2(gy - c2, rho - ring_radius)
        else:  # axis == 3
            rho = torch.sqrt((gx - c1) ** 2 + (gy - c2) ** 2)
            phi = torch.atan2(gz - c3, rho - ring_radius)

        # atan2 is defined everywhere (atan2(0, 0) = 0), so no NaN filtering is needed.
        return charge * phi

    @staticmethod
    def create_vortex_lines(
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        n1: int,
        n2: int,
        n3: int,
        positions: list,
        charges: list,
        axis: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Create the phase texture for one or more straight vortex lines.

        Each vortex line runs parallel to ``axis``.  Its core intersects the
        perpendicular plane at the coordinate pair given in ``positions``:

        - axis=1: each position is (c2, c3)
        - axis=2: each position is (c1, c3)
        - axis=3: each position is (c1, c2)

        Args:
            x1, x2, x3 (torch.Tensor): 1-D real-space axes.
            n1, n2, n3 (int): Grid point counts.
            positions (list[tuple]): Core positions in the perpendicular plane.
            charges (list[int]): Topological charges for each line.
            axis (int): 1, 2, or 3 – axis the vortex lines run along.
            device (torch.device): Computation device.

        Returns:
            torch.Tensor: Real phase tensor of shape (n1, n2, n3).
        """
        if axis not in (1, 2, 3):
            raise ValueError(f"axis must be 1, 2, or 3; got {axis}")

        gx = x1.to(device=device, dtype=torch.float64).reshape(n1, 1, 1).expand(n1, n2, n3)
        gy = x2.to(device=device, dtype=torch.float64).reshape(1, n2, 1).expand(n1, n2, n3)
        gz = x3.to(device=device, dtype=torch.float64).reshape(1, 1, n3).expand(n1, n2, n3)

        phase = torch.zeros(n1, n2, n3, dtype=torch.float64, device=device)
        for (ca, cb), q in zip(positions, charges):
            if axis == 1:
                da, db = gy - ca, gz - cb
            elif axis == 2:
                da, db = gx - ca, gz - cb
            else:  # axis == 3
                da, db = gx - ca, gy - cb
            phase = phase + q * torch.atan2(db, da)

        # atan2 is defined everywhere (atan2(0, 0) = 0), so no NaN filtering is needed.
        return phase

    @staticmethod
    def column_density(
        psi: torch.Tensor,
        axis: int,
        d_axis: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the column density by integrating |psi|² along the given axis.

        Args:
            psi (torch.Tensor): BEC wavefunction (n1, n2, n3).
            axis (int): 1, 2, or 3 – axis to integrate along.
            d_axis (float, optional): Grid spacing along ``axis``. Pass it to
                get a true line integral ∫|ψ|² dx_axis; the default of 1.0
                returns the bare sum over grid points.

        Returns:
            torch.Tensor: 2-D column density tensor in the remaining plane.
        """
        if axis not in (1, 2, 3):
            raise ValueError(f"axis must be 1, 2, or 3; got {axis}")
        return torch.sum(torch.abs(psi) ** 2, dim=axis - 1) * d_axis

    @staticmethod
    def cross_section_plane(
        psi: torch.Tensor,
        axis: int,
        index: int = None,
    ) -> torch.Tensor:
        """
        Extract a 2-D density slice orthogonal to the given axis.

        Args:
            psi (torch.Tensor): BEC wavefunction (n1, n2, n3).
            axis (int): 1, 2, or 3 – normal axis of the slice.
            index (int, optional): Grid index along ``axis``. Defaults to the
                centre of that axis.

        Returns:
            torch.Tensor: 2-D density slice.
        """
        if axis not in (1, 2, 3):
            raise ValueError(f"axis must be 1, 2, or 3; got {axis}")
        density = torch.abs(psi) ** 2
        idx = index if index is not None else psi.shape[axis - 1] // 2
        if axis == 1:
            return density[idx, :, :]
        elif axis == 2:
            return density[:, idx, :]
        else:
            return density[:, :, idx]

    @staticmethod
    def calculate_velocity3D(
        psi: torch.Tensor,
        p_grid: tuple,
        density_floor: float = 1e-12,
    ) -> tuple:
        """
        Compute the 3-D superfluid velocity field using spectral derivatives.

        In dimensionless units (ℏ/m = 1) the superfluid velocity is:

            v_i = Im( ψ* ∂_i ψ ) / |ψ|²

        where the spatial derivative is evaluated spectrally.

        The velocity is undefined where there is no condensate. In floating
        point the density is essentially never exactly zero, so a bare
        ``density > 0`` test lets the dilute tail (n ~ 1e-30) produce enormous
        meaningless velocities. The cut is therefore taken *relative* to the
        peak density, and the division itself is guarded so that no inf/NaN is
        generated in the discarded region.

        Args:
            psi (torch.Tensor): BEC wavefunction (n1, n2, n3).
            p_grid (tuple): (px, py, pz) – 3-D momentum meshgrids (1-D axes are
                also accepted).
            density_floor (float, optional): Density threshold relative to the
                peak density below which the velocity is reported as zero.
                Default 1e-12.

        Returns:
            tuple: (v1, v2, v3) – velocity component tensors, each (n1, n2, n3).
        """
        return tuple(GPELibrary.superfluid_velocity(
            psi, p_grid, density_floor=density_floor))

    @staticmethod
    def angular_momentum(
        psi: torch.Tensor,
        space_grid: tuple,
        p_grid: tuple,
        component: int,
        d_x: float,
    ) -> torch.Tensor:
        """
        Calculate the expectation value of one angular momentum component.

        In units of ℏ:

            ⟨L_1⟩ = ⟨ψ | -i(x2 ∂_3 - x3 ∂_2) | ψ⟩
            ⟨L_2⟩ = ⟨ψ | -i(x3 ∂_1 - x1 ∂_3) | ψ⟩
            ⟨L_3⟩ = ⟨ψ | -i(x1 ∂_2 - x2 ∂_1) | ψ⟩

        Spatial derivatives are evaluated spectrally. As with the energies, the
        expectation value is an integral, so the sum carries the cell volume
        ``d_x`` — ψ is normalised as ``d_x·Σ|ψ|² = 1``.

        Args:
            psi (torch.Tensor): Normalised BEC wavefunction (n1, n2, n3).
            space_grid (tuple): (g_x, g_y, g_z) – 3-D real-space meshgrids.
            p_grid (tuple): (px, py, pz) – 3-D momentum meshgrids (1-D axes are
                also accepted).
            component (int): 1, 2, or 3.
            d_x (float): Grid cell volume (product of dx in each dimension).

        Returns:
            torch.Tensor: Scalar expectation value ⟨L_component⟩ (in ℏ).
        """
        if component not in (1, 2, 3):
            raise ValueError(f"component must be 1, 2, or 3; got {component}")

        gx, gy, gz = space_grid
        px, py, pz = GPELibrary._broadcast_momentum_axes(p_grid, psi.dim(), zero_nyquist=True)
        psi_f = torch.fft.fftn(psi, norm='forward')

        def _d(p_comp):
            return torch.fft.ifftn(1j * p_comp * psi_f, norm='forward')

        if component == 1:
            Lpsi = -1j * (gy * _d(pz) - gz * _d(py))
        elif component == 2:
            Lpsi = -1j * (gz * _d(px) - gx * _d(pz))
        else:  # component == 3
            Lpsi = -1j * (gx * _d(py) - gy * _d(px))

        return torch.real(torch.sum(psi.conj() * Lpsi)) * d_x

