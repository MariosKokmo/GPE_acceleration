r"""
Core GPE solver library on a Cartesian grid.

Everything needed to propagate the Gross-Pitaevskii equation

.. math::

    i\,\frac{\partial \psi}{\partial t} =
        \left[-\frac{\nabla^2}{2} + V_\mathrm{ext}(\mathbf{r}, t)
              + u \lvert \psi \rvert^{2}\right]\psi

with the split-step Fourier method, in dimensionless units
(:math:`\hbar = m = \omega_\mathrm{ho} = 1`). One iteration is a Strang
splitting — a real-space half-step, a full kinetic step evaluated in Fourier
space, then a second real-space half-step,

.. math::

    \psi(t + \Delta\tau) =
        e^{-i U_\mathrm{tot} \Delta\tau / 2}\,
        \mathcal{F}^{-1}\!\left[e^{-i p^2 \Delta\tau / 2}\,
            \mathcal{F}\left[
                e^{-i U_\mathrm{tot} \Delta\tau / 2}\, \psi(t)\right]\right],

which is second-order accurate in :math:`\Delta\tau` because the two
non-commuting operators are applied symmetrically.

The module is split in three:

:class:`GPELibrary`
    The dimension-agnostic core: grids, the propagation steps, the energy and
    chemical-potential diagnostics, and the stochastic (SGPE) machinery.
:class:`GPE2DLibrary`
    Additions for the effectively two-dimensional model, whose simulation
    plane is x–z: point vortices, dark solitons, in-plane velocity.
:class:`GPE3DLibrary`
    Additions for fully three-dimensional runs: vortex rings and lines,
    column densities and slices, the full velocity field and angular momentum.

Both subclasses inherit from :class:`GPELibrary`, so either one exposes the
whole core as well.

Grid conventions
----------------

Real-space axes are :math:`x_i = x_\mathrm{min} + i\,\mathrm{d}x`. Momentum
axes are in FFT (wrapped) order — the first half positive, the second half
negative — matching :func:`torch.fft.fftn`, and are held as 1-D axes that
broadcast rather than as materialised meshgrids wherever possible. Every
tensor is allocated on the requested device, so the evolution loop needs no
host/device transfer.

Wavefunctions are normalised as :math:`\mathrm{d}V \sum \lvert\psi\rvert^2 = 1`,
so every integral in this module carries the cell volume ``d_x`` explicitly.
"""
import numpy as np
import torch
from .parameters import CONSTANTS
from .common_utils import CommonUtils as cu

class GPELibrary:
    r"""
    Coordinate-agnostic core of the Cartesian GPE solver.

    A namespace of static methods covering the whole simulation loop: grid
    construction (:meth:`init_grid`), the split-step propagator
    (:meth:`split_step_step` and its kinetic half :meth:`p_evolution`), the
    normalisation, the diagnostics (:meth:`calculate_energy_allocation`,
    :meth:`calculate_chemical_potential`, :meth:`superfluid_velocity`,
    :meth:`mod_grad_psi`) and the stochastic extension
    (:meth:`generate_thermal_noise`, :meth:`sgpe_step`).

    Nothing here is specific to a number of dimensions; the 2-D and 3-D
    specialisations live in :class:`GPE2DLibrary` and :class:`GPE3DLibrary`.
    """

    @staticmethod
    def _real_axis(
        n: int,
        x_min: float,
        dx: float,
        device: torch.device
    ) -> torch.Tensor:
        r"""
        Build a real-space axis :math:`x_i = x_\mathrm{min} + i\,\mathrm{d}x`.

        Args:
            n (int): Number of grid points, so that :math:`i = 0 \ldots n-1`.
            x_min (float): Coordinate of the first point.
            dx (float): Grid spacing.
            device (torch.device): Device the axis is allocated on, so that no
                host/device transfer is needed later.

        Returns:
            torch.Tensor: The axis, of shape ``(n,)`` and dtype ``float64``.
        """
        return float(x_min) + torch.arange(n, dtype=torch.float64, device=device) * float(dx)

    @staticmethod
    def _momentum_axis(
        n: int,
        dp: float,
        device: torch.device
    ) -> torch.Tensor:
        r"""
        Build a momentum axis in FFT (wrapped) ordering.

        Indices :math:`0 \ldots n/2 - 1` carry the positive momenta and
        :math:`n/2 \ldots n-1` the negative ones, matching the layout of
        :func:`torch.fft.fftn`, so the axis can multiply a transformed field
        directly.

        Args:
            n (int): Number of grid points.
            dp (float): Momentum-space grid spacing.
            device (torch.device): Device the axis is allocated on.

        Returns:
            torch.Tensor: The axis, of shape ``(n,)`` and dtype ``float64``.
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
        r"""
        Normalise a momentum-grid argument to ``dim`` broadcastable tensors.

        Both conventions used across this library are accepted:

        - a tuple of **1-D axes** ``(p1, p2, p3)`` as returned by
          :meth:`init_grid` — each axis is reshaped so it broadcasts along its
          own dimension (no meshgrid materialised);
        - a tuple of **N-D meshgrids** ``(px, py, pz)`` (``p_grid`` from
          :meth:`init_grid`) — passed through unchanged.

        Note:
            **The Nyquist mode.** On an even-sized grid the highest mode is its
            own alias (:math:`+\pi/\mathrm{d}x` and :math:`-\pi/\mathrm{d}x`
            are the same sampled wave), so its *first* derivative is undefined:
            multiplying by :math:`i p` there breaks the Hermitian symmetry that
            keeps the derivative of a real field real. The leak is tiny in
            absolute terms but gets divided by the density in the velocity
            field, where it can dominate the dilute tail. Odd-order derivatives
            should therefore set that coefficient to zero — :math:`p^2` in the
            kinetic operator is an *even* order and must keep it, which is why
            this is opt-in.

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
        r"""
        Compute the superfluid velocity components.

        In dimensionless units (:math:`\hbar/m = 1`),

        .. math::

            v_i = \frac{\mathrm{Im}\left(\psi^{*}\,
                \partial_i \psi\right)}{\lvert \psi \rvert^{2}} ,

        with the derivative evaluated spectrally.

        Note:
            **Take the derivative of** :math:`\psi`, **never of its phase.**
            The expression above equals :math:`\partial_i \theta` for the
            *unwrapped* phase :math:`\theta`, but it is computed from
            :math:`\psi`, which is single-valued and smooth. Differentiating
            ``angle(psi)`` instead looks equivalent and is not: the wrapped
            phase jumps by :math:`2\pi` across a branch cut running out of
            every vortex core, and a spectral derivative of a discontinuous
            field rings across the *whole* domain, not just near the cut.
            Measured against an analytic vortex-antivortex field, the phase
            route is wrong by more than 100% everywhere while this one is
            limited only by grid resolution.

        The velocity is undefined where there is no condensate. In floating
        point the density is essentially never exactly zero, so a bare
        ``density > 0`` test lets the dilute tail (:math:`n \sim 10^{-30}`)
        produce enormous meaningless values. The cut is therefore taken
        *relative* to the peak density, and the division itself is guarded so
        no inf/NaN is generated in the discarded region.

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
        r"""
        Initialise the real-space and momentum-space grids.

        All tensors are allocated directly on ``device``, so that no
        host/device transfer is needed later in the evolution loop.

        Args:
            x_min (list): Coordinate of the first real-space point in each
                dimension.
            dx (list): Real-space grid spacing in each dimension.
            dp (list): Momentum-space grid spacing in each dimension.
            n1 (int): Number of grid points along the first dimension.
            n2 (int): Number of grid points along the second dimension.
            n3 (int): Number of grid points along the third dimension.
            device (torch.device): Device to allocate the tensors on (CPU or
                GPU).

        Returns:
            tuple: ``(x1, x2, x3, p1, p2, p3, p_sq, space_grid, p_grid)`` —
            ``x1``–``x3`` and ``p1``–``p3`` are the 1-D axes, ``p_sq`` is
            :math:`\lvert p \rvert^2` on the full grid, and ``space_grid`` /
            ``p_grid`` are the corresponding 3-D meshgrids.

        Note:
            The meshgrids are built with ``indexing='ij'``. The PyTorch default
            flips to ``'xy'`` in future releases, which would silently
            transpose the first two axes.
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
        r"""
        Apply the kinetic evolution operator in momentum space.

        The kinetic term is diagonal in Fourier space, so the full step is a
        transform, a pointwise multiplication and a transform back,

        .. math::

            \psi \leftarrow \mathcal{F}^{-1}\!\left[
                e^{-i\,\Delta\tau\, p^2 / 2}\, \mathcal{F}[\psi]\right].

        Args:
            psi1 (torch.Tensor): Wavefunction of the system.
            dtau (float): Time step :math:`\Delta\tau`.
            p_sq (torch.Tensor): Squared momentum grid :math:`\lvert p\rvert^2`.

        Returns:
            torch.Tensor: Updated wavefunction, of the same shape as ``psi1``.
        """
        psiF = torch.fft.fftn(psi1, norm='forward')
        psiF = torch.exp(-1j * dtau * 0.5 * p_sq) * psiF
        return torch.fft.ifftn(psiF, norm='forward')

    @staticmethod
    def normalize(
        phi: torch.Tensor,
        d_x: float
    ) -> torch.Tensor:
        r"""
        Normalise the wavefunction to unit norm.

        .. math::

            \phi \leftarrow \frac{\phi}
                {\sqrt{\mathrm{d}V \sum \lvert \phi \rvert^{2}}}

        Args:
            phi (torch.Tensor): Wavefunction to be normalised.
            d_x (float): Grid cell volume :math:`\mathrm{d}V`.

        Returns:
            torch.Tensor: Normalised wavefunction, satisfying
            :math:`\int \lvert \phi \rvert^{2}\,\mathrm{d}V = 1`.

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
        r"""
        Advance the wavefunction by one split-step Fourier iteration.

        A Strang splitting — real-space half-step, full kinetic step, real-space
        half-step,

        .. math::

            \psi(t + \Delta\tau) =
                e^{-i U_\mathrm{tot} \Delta\tau / 2}\,
                \mathcal{F}^{-1}\!\left[e^{-i p^2 \Delta\tau / 2}\,
                    \mathcal{F}\left[
                        e^{-i U_\mathrm{tot} \Delta\tau / 2}\,
                        \psi(t)\right]\right],

        which is second-order accurate in :math:`\Delta\tau` because the two
        non-commuting operators are applied symmetrically.

        Note:
            For a real ``utot1`` every factor applied here has unit modulus, so
            the norm is conserved to machine precision and no renormalisation
            is needed. When ``utot1`` carries an imaginary part (three-body
            losses, a complex absorbing potential) the norm is *meant* to
            decay: renormalising would silently cancel the atom loss, which is
            why ``renormalise`` defaults to ``False``.

        Args:
            psi1 (torch.Tensor): Wavefunction of the system.
            utot1 (torch.Tensor): Total real-space operator
                :math:`U_\mathrm{tot}`, frozen for the duration of the step.
            dtau (float): Time step :math:`\Delta\tau`.
            p_sq (torch.Tensor): Squared momentum grid.
            d_x (float): Grid cell volume; only used when ``renormalise`` is
                set.
            renormalise (bool, optional): Force
                :math:`\int \lvert \psi \rvert^{2}\,\mathrm{d}V = 1` after the
                step. Only meaningful for a deliberately number-conserving run
                with a lossy potential (default ``False``).

        Returns:
            torch.Tensor: Updated wavefunction, of the same shape as ``psi1``.
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
        r"""
        Compute the modulus of the gradient of the wavefunction,
        :math:`\lvert \nabla \psi \rvert`.

        Each derivative is taken spectrally, and the components are combined as

        .. math::

            \lvert \nabla \psi \rvert =
                \sqrt{\sum_i \lvert \partial_i \psi \rvert^{2}},
            \qquad
            \lvert \partial_i \psi \rvert^{2} =
                \mathrm{Re}(\partial_i \psi)^{2}
                + \mathrm{Im}(\partial_i \psi)^{2}.

        The derivative of a complex field is itself complex, so both parts
        count. Keeping only the real part (as an earlier version did)
        understates the kinetic energy of any state carrying phase structure —
        vortices, solitons, or simply a state undergoing real-time evolution.

        Works for 1-, 2- and 3-D fields. ``p_axes`` may be either 1-D momentum
        axes or full meshgrids (see :meth:`_broadcast_momentum_axes`), and the
        Nyquist coefficient is dropped since this is a first derivative.

        Args:
            psi (torch.Tensor): Condensate wavefunction.
            p_axes (list): Momentum-space axes (or meshgrids), one per
                dimension.

        Returns:
            torch.Tensor: Modulus of the gradient, of the same shape as
            ``psi``.

        Raises:
            ValueError: If ``psi`` is not 1-, 2- or 3-dimensional.
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
        r"""
        Split the condensate energy into its kinetic, potential and interaction
        parts.

        .. math::

            e_\mathrm{kin} = \frac{1}{2}\int \lvert \nabla\psi \rvert^{2}\,
                \mathrm{d}V,
            \qquad
            e_\mathrm{pot} = \int V_\mathrm{ext}\lvert \psi \rvert^{2}\,
                \mathrm{d}V,
            \qquad
            e_\mathrm{int} = \frac{u}{2}\int \lvert \psi \rvert^{4}\,
                \mathrm{d}V

        Every term is an integral over the grid, so each sum carries the cell
        volume ``d_x``; :math:`\psi` is normalised as
        :math:`\mathrm{d}V \sum \lvert\psi\rvert^2 = 1`. Omitting it would make
        the energies scale with the grid spacing instead of being physical.

        Args:
            psi (torch.Tensor): BEC wavefunction.
            Vext (torch.Tensor): External potential. If complex (e.g. an
                absorbing potential), only the real part contributes to the
                energy — the imaginary part is a loss rate, not an energy.
            p_grid (tuple): Momentum space axes (1-D) or meshgrids.
            d_x (float): Grid cell volume (product of dx in each dimension).
            **parameters: Must include the interaction strength ``"u"``.

        Returns:
            dict: The keys ``'e_kin'``, ``'e_pot'``, ``'e_int'`` and
            ``'E_total'``, in units of :math:`\hbar\omega_\mathrm{ho}`.

        Raises:
            ValueError: If ``"u"`` is not supplied.
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
        r"""
        Compute the mean-field chemical potential
        :math:`\mu = \langle \psi \lvert H_\mathrm{mf} \rvert \psi \rangle`.

        With the mean-field Hamiltonian in dimensionless units
        (:math:`\hbar = m = \omega_\mathrm{ho} = 1`),

        .. math::

            H_\mathrm{mf} = -\frac{\nabla^2}{2} + V_\mathrm{ext}
                            + u \lvert \psi \rvert^{2},

        the chemical potential follows from the energy terms of
        :meth:`calculate_energy_allocation` as

        .. math::

            \mu = e_\mathrm{kin} + e_\mathrm{pot} + 2\,e_\mathrm{int}.

        The interaction term is counted *twice* because
        :math:`\mu = \partial E / \partial N`, and differentiating the
        :math:`(u/2)N^2` term yields :math:`uN`.

        This is what the SGPE uses as the grand-canonical reservoir potential:
        modes below :math:`\mu` grow, modes above it decay.

        Args:
            psi (torch.Tensor): Normalised BEC wavefunction of shape
                ``(n1, n2, n3)``.
            uext (torch.Tensor): External trapping potential on the grid.
            u (float): Dimensionless interaction strength :math:`u`.
            p_grid (tuple): Momentum axes ``(p1, p2, p3)``, either as 1-D axes
                or as 3-D meshgrids.
            d_x (float): Grid cell volume (the product of ``dx`` over the
                dimensions).

        Returns:
            float: The chemical potential :math:`\mu`, in units of
            :math:`\hbar\omega_\mathrm{ho}`.
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
        r"""
        Generate a complex Gaussian noise field for one SGPE time step.

        The noise has to satisfy the fluctuation-dissipation theorem,

        .. math::

            \langle \eta^{*}(\mathbf{r}, t)\, \eta(\mathbf{r}', t') \rangle =
                2\gamma\, k_B T\, \delta(\mathbf{r} - \mathbf{r}')\,
                \delta(t - t'),

        which on a grid with cell volume :math:`\delta V = \mathrm{d}V` and time
        step :math:`\delta t = \Delta\tau` fixes the per-point amplitude at

        .. math::

            \sqrt{\frac{\gamma\, k_B T\, \Delta\tau}{\mathrm{d}V}},

        so that
        :math:`\langle \lvert \Delta\psi_\mathrm{noise} \rvert^{2}\rangle =
        2\gamma k_B T \Delta\tau / \mathrm{d}V` per grid point, matching the
        continuous relation.

        Note:
            **Projection — the "P" of the** *projected* **SGPE.**
            Delta-correlated noise is white across the whole grid, so it feeds
            energy into every mode up to the Nyquist momentum. The c-field
            description is only valid up to a cutoff energy, above which the
            modes belong to the thermal cloud rather than to the classical
            field. Pass ``p_sq`` together with ``e_cut`` to zero the noise in
            every mode with :math:`p^2/2 > e_\mathrm{cut}`; without it the
            noise is unprojected and the run will heat artificially at large
            momenta.

        Args:
            shape (tuple): Grid shape ``(n1, n2, n3)``.
            gamma (float): Dimensionless damping coefficient :math:`\gamma`.
            kT (float): Dimensionless temperature
                :math:`k_B T / (\hbar\omega_\mathrm{ho})`.
            dtau (float): Dimensionless time step
                :math:`\omega_\mathrm{ho}\,\mathrm{d}t`.
            d_x (float): Grid cell volume (the product of ``dx`` over the
                dimensions).
            device (torch.device): Computation device.
            p_sq (torch.Tensor, optional): Squared momentum grid
                :math:`\lvert p \rvert^{2}`, required for the projection.
            e_cut (float, optional): Cutoff energy :math:`p^2/2` in units of
                :math:`\hbar\omega_\mathrm{ho}`. Ignored unless ``p_sq`` is
                also given.

        Returns:
            torch.Tensor: Complex noise tensor of shape ``(n1, n2, n3)``.
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
        r"""
        Perform one deterministic SGPE split-step with :math:`(1 - i\gamma)`
        damping.

        The SGPE replaces the unitary GPE evolution operator with a dissipative
        one,

        .. math::

            \text{GPE:}\quad e^{-i\,\mathrm{d}t\, H_\mathrm{mf}}
            \qquad\longrightarrow\qquad
            \text{SGPE:}\quad
                e^{-(i + \gamma)\,\mathrm{d}t\,(H_\mathrm{mf} - \mu)},

        the :math:`(i + \gamma)` factor coming from the :math:`(1 - i\gamma)`
        prefactor of the equation of motion,

        .. math::

            \frac{\partial \psi}{\partial t} =
                -(i + \gamma)\,(H_\mathrm{mf} - \mu)\,\psi + \eta .

        Modes with :math:`H_\mathrm{mf} > \mu` are exponentially damped (energy
        goes to the reservoir) and modes with :math:`H_\mathrm{mf} < \mu` are
        amplified (energy is drawn from it), which drives the system towards
        thermal equilibrium at temperature :math:`T`.

        The step is the same Strang splitting as :meth:`split_step_step`, with
        :math:`V_\mathrm{eff} = u\lvert\psi\rvert^2 + V_\mathrm{ext}` frozen at
        the start of the step:

        1. real-space half-step
           :math:`\psi \leftarrow e^{-(i+\gamma)\Delta\tau (V_\mathrm{eff} - \mu)/2}\psi`;
        2. momentum full-step
           :math:`\tilde\psi \leftarrow e^{-(i+\gamma)\Delta\tau p^2/2}\tilde\psi`;
        3. real-space half-step, as in 1.

        Note:
            **The norm is deliberately left free.** :math:`\mu` enters only as
            the constant shift ``utot - mu``, so the whole step multiplies
            :math:`\psi` by the global factor
            :math:`e^{(i+\gamma)\Delta\tau\mu}`. Renormalising afterwards
            divides that factor straight back out and leaves nothing but an
            unobservable global phase: with a forced norm, :math:`\mu` has *no
            effect whatsoever* on the dynamics and the grand-canonical
            growth/decay the SGPE is built around simply does not happen.

            Letting the norm evolve is also self-consistent with the units used
            here. With :math:`\mathrm{d}V \sum \lvert\psi\rvert^2 = 1` at
            :math:`t = 0` and :math:`u = 4\pi N_0 a / a_\mathrm{ho}`, the
            mean-field term :math:`u\lvert\psi\rvert^2` equals
            :math:`(4\pi a / a_\mathrm{ho})\,N_0\lvert\psi\rvert^2`, i.e. the
            interaction energy of the *current* atom number
            :math:`N(t) = N_0 \lVert \psi \rVert^2`. The reservoir therefore
            sets :math:`N` through :math:`\mu`, exactly as intended.

        Args:
            psi (torch.Tensor): Wavefunction of shape ``(n1, n2, n3)``, complex
                double.
            utot (torch.Tensor): Total mean-field potential
                :math:`V_\mathrm{ext} + u\lvert\psi\rvert^2`.
            mu (float): Reservoir chemical potential :math:`\mu`, in units of
                :math:`\hbar\omega_\mathrm{ho}`.
            gamma (float): Dimensionless damping coefficient :math:`\gamma`.
            dtau (float): Dimensionless time step
                :math:`\omega_\mathrm{ho}\,\mathrm{d}t`.
            p_sq (torch.Tensor): Squared momentum grid
                :math:`\lvert p \rvert^{2}`.
            d_x (float): Grid cell volume; only used when ``renormalise`` is
                set.
            renormalise (bool, optional): Force
                :math:`\int \lvert \psi \rvert^{2}\,\mathrm{d}V = 1` after the
                step. This disables the reservoir coupling described above;
                only use it for a deliberately number-conserving damped GPE
                (default ``False``).

        Returns:
            torch.Tensor: Updated wavefunction, of the same shape as ``psi``.
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
        r"""
        Find the peak density and where on the grid it sits.

        Args:
            psi (torch.Tensor): Normalised BEC wavefunction of shape
                ``(n1, n2, n3)``.

        Returns:
            tuple: ``(max_density, peak_indices)`` — the peak of
            :math:`\lvert \psi \rvert^{2}` as a scalar tensor, and the grid
            position ``(i, j, k)`` at which it occurs.
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
    r"""
    Extensions of the GPE library for effectively two-dimensional simulations.

    The simulation plane is x–z (grid dimensions ``n1``–``n3``), with the ``y``
    direction held flat: the tight axis is frozen in its transverse ground
    state, so the field is stored on the full 3-D grid but is constant along
    ``n2``. Provides point-vortex and dark-soliton imprinting, the in-plane
    velocity field, and the density diagnostics the 2-D runs report.

    Inherits every core operator from :class:`GPELibrary`.
    """

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
        r"""
        Build the phase texture that imprints point vortices on a 2-D
        condensate.

        Each vortex of charge :math:`q` contributes a winding of
        :math:`2\pi q` around its core. It is built with the half-angle
        identity

        .. math::

            2\,\mathrm{atan2}\left(y,\; \rho + t\right) = \theta,
            \qquad \rho = \sqrt{t^2 + y^2},

        with :math:`t` and :math:`y` measured from the core. Written this way
        the winding is free of the branch cut that a plain
        :math:`\mathrm{atan2}(y, t)` would introduce. The identity degenerates
        only on the negative-:math:`t` half-axis (:math:`y = 0`,
        :math:`t < 0`), where :math:`\rho + t = 0`; those points are assigned
        the limiting value :math:`q\pi` explicitly.

        The phase is identical for every ``y`` index, so it is computed once on
        the ``(n1, n3)`` plane and broadcast along the second dimension. The
        previous implementation looped over ``n1·n3`` grid points in Python and
        indexed the tensor element by element, which dominated the setup cost
        of a run.

        Args:
            vortices (numpy.ndarray): Array of shape
                ``(3, number_of_vortices)`` whose rows are the x positions, the
                z positions and the charges. Positions are grid offsets
                relative to the centre of the grid. A flat ``(3,)`` array is
                accepted for a single vortex.
            x1 (torch.Tensor): Real-space axis along the first dimension.
            x2 (torch.Tensor): Real-space axis along the second dimension.
            x3 (torch.Tensor): Real-space axis along the third dimension.
            n1 (int): Number of grid points along the first dimension.
            n2 (int): Number of grid points along the second dimension.
            n3 (int): Number of grid points along the third dimension.
            device (torch.device): Device to allocate the tensors on.

        Returns:
            torch.Tensor: Real phase of shape ``(n1, n2, n3)`` to be added to
            the condensate phase, or ``None`` if ``vortices`` is ``None``.

        Raises:
            ValueError: If ``vortices`` is not of shape ``(3, N)``, or if a
                vortex sits outside the grid.
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
        r"""
        Re-imprint the wavefunction by adding a phase again.

        Used to re-apply a phase texture periodically during a run, e.g. to
        hold a vortex pattern in place against the dynamics.

        Args:
            psi1 (torch.Tensor): Wavefunction of the system.
            repetitive_phase (torch.Tensor): Real phase to add.

        Returns:
            torch.Tensor: Updated wavefunction,
            :math:`\psi e^{i\theta}`.
        """
        return cu.update_phase(psi1, repetitive_phase)

    @staticmethod
    def calculate_velocity2D(
        psi: torch.Tensor,
        p_grid: tuple,
        density_floor: float = 1e-12
    ) -> tuple:
        r"""
        Compute the in-plane superfluid velocity, as magnitude and direction.

        The simulation plane of the 2-D model is x–z (``n1``–``n3``), so the
        in-plane components are :math:`v_x` and :math:`v_z`; for a genuinely
        2-D field they are the two axes it has. The result is

        .. math::

            \lvert v \rvert = \sqrt{v_x^2 + v_z^2},
            \qquad
            \arg v = \mathrm{atan2}(v_z, v_x),

        in dimensionless units — multiply by :math:`\hbar/m` for a dimensional
        velocity.

        Note:
            **Takes the wavefunction, not the phase.** This used to
            differentiate ``angle(psi)``, which is wrong wherever the
            condensate carries circulation: the wrapped phase has a
            :math:`2\pi` branch cut running out of every vortex core, and
            because a spectral derivative is global, the discontinuity corrupts
            the velocity across the entire domain rather than only near the
            cut. Against an analytic vortex-antivortex field the phase route
            was off by 146% at the median and 5383% at worst, versus about 5%
            (grid resolution) for this one. See
            :meth:`GPELibrary.superfluid_velocity`.

        Args:
            psi (torch.Tensor): Condensate wavefunction.
            p_grid (tuple): Momentum axes or meshgrids, one per dimension of
                ``psi``.
            density_floor (float, optional): Threshold relative to the peak
                density below which the velocity is reported as zero (default
                ``1e-12``).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The in-plane speed
            :math:`\lvert v \rvert` and its direction, each of the same shape
            as ``psi``.
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
        r"""
        Compute the RMS radius of the condensate about a given centre.

        .. math::

            r_\mathrm{rms} = \sqrt{
                \frac{\int \lvert \mathbf{r} - \mathbf{r}_0 \rvert^{2}
                      \lvert \psi \rvert^{2}\,\mathrm{d}V}
                     {\int \lvert \psi \rvert^{2}\,\mathrm{d}V}}

        The cell volume cancels between numerator and denominator and is
        therefore omitted.

        Args:
            psi (torch.Tensor): Normalised wavefunction.
            center (list): Centre :math:`\mathbf{r}_0` as
                ``(center_x, center_y, center_z)``, in real-space units.
            space_grid (tuple): Real-space meshgrids ``(g_x, g_y, g_z)``.

        Returns:
            torch.Tensor: Scalar RMS radius of the condensate.
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
        r"""
        Build a multiplicative dark-soliton profile for the wavefunction.

        Each soliton is a stripe of suppressed density across the condensate.
        The profile for a single soliton along a coordinate :math:`s` is

        .. math::

            f(s) = \cos\alpha\,
                   \tanh\!\left(\frac{\cos\alpha\,(s - s_0)}{w}\right)
                   + i \sin\alpha ,

        where :math:`\alpha = 0` gives a stationary black soliton (a full
        density dip plus a :math:`\pi` phase jump) and
        :math:`0 < \alpha < \pi/2` gives a grey — that is, moving — soliton
        with a shallower dip and a smaller phase step. When several solitons
        are requested their profiles are multiplied together.

        Args:
            x1 (torch.Tensor): Real-space axis along the first grid dimension.
            x3 (torch.Tensor): Real-space axis along the third grid dimension.
            n1 (int): Number of grid points along the first dimension.
            n2 (int): Number of grid points along the second dimension.
            n3 (int): Number of grid points along the third dimension.
            positions (list[float]): Centre :math:`s_0` of each soliton, in the
                same units as the corresponding axis.
            widths (list[float]): Characteristic width :math:`w` of each
                soliton. For a BEC this is typically the healing length
                :math:`\xi`.
            axes (list[int]): Axis for each soliton: ``1`` for x (a stripe
                perpendicular to ``x1``) or ``3`` for z (perpendicular to
                ``x3``).
            greyness (list[float], optional): Grey-soliton angle
                :math:`\alpha` for each soliton, in radians. ``0`` is black
                (the default); values up to :math:`\pi/2` make the soliton
                progressively greyer and faster.
            device (torch.device, optional): Device for the output tensor.

        Returns:
            torch.Tensor: Complex tensor of shape ``(n1, n2, n3)``, to be
            multiplied element-wise with the wavefunction.

        Raises:
            ValueError: If the per-soliton lists have inconsistent lengths, or
                if an axis is neither ``1`` nor ``3``.
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
        r"""
        Apply a dark-soliton mask to the wavefunction.

        Unlike vortex imprinting, which adds pure phase, this multiplies the
        wavefunction by a complex profile and so modifies the amplitude and the
        phase at once.

        Args:
            psi (torch.Tensor): Current wavefunction of shape
                ``(n1, n2, n3)``.
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
        r"""
        Compute the column density along a line crossing the condensate.

        The line runs *along* the requested axis through the centre of the
        grid, with the trivial ``y`` direction (``n2``) integrated out:

        .. math::

            \text{axis} = 1:\quad
                n(x) = \sum_y \lvert \psi[:, :, n_3/2] \rvert^{2}
                \quad(\text{length } n_1),

        .. math::

            \text{axis} = 2:\quad
                n(z) = \sum_y \lvert \psi[n_1/2, :, :] \rvert^{2}
                \quad(\text{length } n_3).

        Args:
            psi (torch.Tensor): BEC wavefunction of shape ``(n1, n2, n3)``.
            axis (int, optional): Axis the line runs along — ``1`` for x,
                ``2`` for z (default ``1``).

        Returns:
            torch.Tensor: The 1-D cross-section line through the centre of the
            condensate.

        Raises:
            ValueError: If ``axis`` is neither ``1`` nor ``2``.
        """
        n1, n2, n3 = psi.shape
        if axis == 1:
            return torch.sum(torch.abs(psi[:, :, n3 // 2]) ** 2, dim=1)
        elif axis == 2:
            return torch.sum(torch.abs(psi[n1 // 2, :, :]) ** 2, dim=0)
        raise ValueError(f"axis must be 1 (x) or 2 (z); got {axis}")

class GPE3DLibrary(GPELibrary):
    r"""
    Extensions of the GPE library for fully three-dimensional simulations.

    Provides the topological structures that only exist in 3-D (vortex rings
    and vortex lines), the density diagnostics that reduce a 3-D field to
    something viewable (column densities, 2-D slices), and the observables
    (the full superfluid velocity field, the angular momentum components).

    Inherits every core operator from :class:`GPELibrary`.
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
        r"""
        Build the phase texture for a quantised vortex ring.

        The ring is a circle of radius :math:`R` centred at ``center``, lying
        in the plane perpendicular to ``axis``. The phase winds by
        :math:`2\pi q` around the core — the ring itself — which in toroidal
        coordinates is

        .. math::

            \theta = q\,\mathrm{atan2}\left(
                x_\parallel - c_\parallel,\; \rho - R\right),

        where :math:`x_\parallel` is the coordinate along ``axis`` and
        :math:`\rho` is the cylindrical radius in the plane of the ring.

        Args:
            x1 (torch.Tensor): Real-space axis along the first dimension.
            x2 (torch.Tensor): Real-space axis along the second dimension.
            x3 (torch.Tensor): Real-space axis along the third dimension.
            n1 (int): Number of grid points along the first dimension.
            n2 (int): Number of grid points along the second dimension.
            n3 (int): Number of grid points along the third dimension.
            ring_radius (float): Radius :math:`R` of the ring, in the same
                units as the real-space axes.
            center (tuple): Centre ``(c1, c2, c3)`` of the ring, in real-space
                units rather than grid indices.
            axis (int): ``1``, ``2`` or ``3`` — the axis the ring encircles;
                the ring lies in the perpendicular plane.
            charge (int, optional): Topological charge :math:`q`, i.e. the
                winding number (default ``1``).
            device (torch.device, optional): Computation device.

        Returns:
            torch.Tensor: Real phase tensor of shape ``(n1, n2, n3)``.

        Raises:
            ValueError: If ``axis`` is not ``1``, ``2`` or ``3``.

        Note:
            ``atan2`` is defined everywhere, including at
            ``atan2(0, 0) = 0``, so no NaN filtering is needed at the core.
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
        r"""
        Build the phase texture for one or more straight vortex lines.

        Each line runs parallel to ``axis`` and contributes a winding
        :math:`q\,\mathrm{atan2}(\delta_b, \delta_a)` about its core, where
        :math:`(\delta_a, \delta_b)` is the offset from the core in the
        perpendicular plane. The core intersects that plane at the coordinate
        pair given in ``positions``:

        - ``axis=1``: each position is ``(c2, c3)``;
        - ``axis=2``: each position is ``(c1, c3)``;
        - ``axis=3``: each position is ``(c1, c2)``.

        Args:
            x1 (torch.Tensor): Real-space axis along the first dimension.
            x2 (torch.Tensor): Real-space axis along the second dimension.
            x3 (torch.Tensor): Real-space axis along the third dimension.
            n1 (int): Number of grid points along the first dimension.
            n2 (int): Number of grid points along the second dimension.
            n3 (int): Number of grid points along the third dimension.
            positions (list[tuple]): Core positions in the perpendicular plane.
            charges (list[int]): Topological charge of each line.
            axis (int): ``1``, ``2`` or ``3`` — the axis the lines run along.
            device (torch.device, optional): Computation device.

        Returns:
            torch.Tensor: Real phase tensor of shape ``(n1, n2, n3)``.

        Raises:
            ValueError: If ``axis`` is not ``1``, ``2`` or ``3``.

        Note:
            ``atan2`` is defined everywhere, including at
            ``atan2(0, 0) = 0``, so no NaN filtering is needed at the core.
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
        r"""
        Compute the column density by integrating the density along one axis.

        .. math::

            n(\mathbf{r}_\perp) = \int \lvert \psi \rvert^{2}\,
                \mathrm{d}x_\mathrm{axis}

        Args:
            psi (torch.Tensor): BEC wavefunction of shape ``(n1, n2, n3)``.
            axis (int): ``1``, ``2`` or ``3`` — the axis to integrate along.
            d_axis (float, optional): Grid spacing along ``axis``. Pass it for
                a true line integral; the default of ``1.0`` returns the bare
                sum over grid points.

        Returns:
            torch.Tensor: The 2-D column density in the remaining plane.

        Raises:
            ValueError: If ``axis`` is not ``1``, ``2`` or ``3``.
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
        r"""
        Extract a 2-D density slice orthogonal to the given axis.

        Args:
            psi (torch.Tensor): BEC wavefunction of shape ``(n1, n2, n3)``.
            axis (int): ``1``, ``2`` or ``3`` — the normal axis of the slice.
            index (int, optional): Grid index along ``axis``. Defaults to the
                centre of that axis.

        Returns:
            torch.Tensor: The 2-D density slice
            :math:`\lvert \psi \rvert^{2}` at that index.

        Raises:
            ValueError: If ``axis`` is not ``1``, ``2`` or ``3``.
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
        r"""
        Compute the full 3-D superfluid velocity field.

        In dimensionless units (:math:`\hbar/m = 1`),

        .. math::

            v_i = \frac{\mathrm{Im}\left(\psi^{*}\,
                \partial_i \psi\right)}{\lvert \psi \rvert^{2}} ,

        with the spatial derivatives evaluated spectrally. See
        :meth:`GPELibrary.superfluid_velocity`, which this delegates to, for
        why the derivative is taken of :math:`\psi` rather than of its phase.

        The velocity is undefined where there is no condensate. In floating
        point the density is essentially never exactly zero, so a bare
        ``density > 0`` test lets the dilute tail (:math:`n \sim 10^{-30}`)
        produce enormous meaningless velocities. The cut is therefore taken
        *relative* to the peak density, and the division itself is guarded so
        that no inf/NaN is generated in the discarded region.

        Args:
            psi (torch.Tensor): BEC wavefunction of shape ``(n1, n2, n3)``.
            p_grid (tuple): Momentum meshgrids ``(px, py, pz)``; 1-D axes are
                also accepted.
            density_floor (float, optional): Threshold relative to the peak
                density below which the velocity is reported as zero (default
                ``1e-12``).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The velocity
            components ``(v1, v2, v3)``, each of shape ``(n1, n2, n3)``.
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
        r"""
        Compute the expectation value of one angular momentum component.

        In units of :math:`\hbar`,

        .. math::

            \langle L_1 \rangle = \langle \psi \lvert
                -i\left(x_2 \partial_3 - x_3 \partial_2\right)
                \rvert \psi \rangle,

        .. math::

            \langle L_2 \rangle = \langle \psi \lvert
                -i\left(x_3 \partial_1 - x_1 \partial_3\right)
                \rvert \psi \rangle,

        .. math::

            \langle L_3 \rangle = \langle \psi \lvert
                -i\left(x_1 \partial_2 - x_2 \partial_1\right)
                \rvert \psi \rangle .

        The spatial derivatives are evaluated spectrally. As with the energies,
        the expectation value is an integral, so the sum carries the cell
        volume ``d_x`` — :math:`\psi` is normalised as
        :math:`\mathrm{d}V \sum \lvert\psi\rvert^2 = 1`.

        Args:
            psi (torch.Tensor): Normalised BEC wavefunction of shape
                ``(n1, n2, n3)``.
            space_grid (tuple): Real-space meshgrids ``(g_x, g_y, g_z)``.
            p_grid (tuple): Momentum meshgrids ``(px, py, pz)``; 1-D axes are
                also accepted.
            component (int): ``1``, ``2`` or ``3`` — which component to
                compute.
            d_x (float): Grid cell volume (the product of ``dx`` over the
                dimensions).

        Returns:
            torch.Tensor: Scalar expectation value
            :math:`\langle L_\mathrm{component} \rangle`, in units of
            :math:`\hbar`.

        Raises:
            ValueError: If ``component`` is not ``1``, ``2`` or ``3``.
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

