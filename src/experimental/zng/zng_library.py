r"""
Pure physics functions for the ZNG framework.

All functions here are stateless: they take tensors and scalars and return
tensors. No simulation objects are imported, so this module can be tested in
isolation.

Dimensionless units are used throughout
(:math:`\hbar = m = \omega_\mathrm{ho} = 1`):

length
    :math:`a_\mathrm{ho} = \sqrt{\hbar / m \omega_\mathrm{ho}}`
energy
    :math:`\hbar \omega_\mathrm{ho}`
momentum
    :math:`\hbar / a_\mathrm{ho}`
time
    :math:`1 / \omega_\mathrm{ho}`
temperature
    :math:`k_B T / (\hbar \omega_\mathrm{ho})`, referred to as ``kT`` below
"""

import torch
import math
from typing import Tuple


# ---------------------------------------------------------------------------
# Grid <-> particle interpolation
# ---------------------------------------------------------------------------

def thermal_density_from_particles(
    positions: torch.Tensor,
    n1: int,
    n2: int,
    n3: int,
    x_min: torch.Tensor,
    dx: torch.Tensor,
    d_x: float,
    device: torch.device,
    particle_weight: float = 1.0,
) -> torch.Tensor:
    r"""
    Deposit test-particle positions onto the 3-D grid to obtain the thermal
    number density :math:`\tilde{n}(\mathbf{r})`.

    Cloud-In-Cell (CIC) trilinear weighting assigns each particle a fractional
    weight to the eight surrounding lattice sites, proportional to the overlap
    of a unit cell centred on the particle with each grid cell. For a particle
    at fractional offset :math:`(f_x, f_y, f_z)` within its base cell
    :math:`(i_0, j_0, k_0)`,

    .. math::

        w_{i_0 + \delta_x,\, j_0 + \delta_y,\, k_0 + \delta_z}
            = \prod_{\alpha = x, y, z}
              \bigl[\, \delta_\alpha f_\alpha
                     + (1 - \delta_\alpha)(1 - f_\alpha) \,\bigr],
        \qquad \delta_\alpha \in \{0, 1\}.

    The deposited count is divided by the cell volume ``d_x``, so the result
    carries units of number density (in dimensionless :math:`a_\mathrm{ho}`
    units).

    Args:
        positions (torch.Tensor): Particle positions in dimensionless units,
            shape ``(N_test, 3)``. Particles outside the box are clamped to the
            nearest boundary cell.
        n1 (int): Number of grid points along the first axis.
        n2 (int): Number of grid points along the second axis.
        n3 (int): Number of grid points along the third axis.
        x_min (torch.Tensor): Lower box boundaries, shape ``(3,)``.
        dx (torch.Tensor): Grid spacings per axis, shape ``(3,)``.
        d_x (float): Grid cell volume, ``dx[0] * dx[1] * dx[2]``.
        device (torch.device): Computation device.
        particle_weight (float): How many atoms one test particle stands for,
            in the same normalisation as the condensate (where
            :math:`\int \lvert \psi \rvert^{2}\, \mathrm{d}V = 1` is the whole
            cloud). With the default of ``1.0`` the result integrates to the
            raw test-particle count, which puts :math:`\tilde{n}` on a
            different scale from :math:`n_c` and makes every mean-field term
            that mixes them depend on :math:`N_\mathrm{test}`. Pass
            ``thermal_fraction / n_test`` so that
            :math:`\int \tilde{n}\, \mathrm{d}V` is the thermal fraction and
            the physics is independent of the particle count.

    Returns:
        torch.Tensor: Thermal number density :math:`\tilde{n}(\mathbf{r})`, of
        shape ``(n1, n2, n3)``.
    """
    x_min = x_min.to(device=device, dtype=torch.float64)
    dx = dx.to(device=device, dtype=torch.float64)

    # Base cell indices (clamped so we never go out of bounds on the +1 cell)
    i0 = ((positions[:, 0] - x_min[0]) / dx[0]).long().clamp(0, n1 - 2)
    j0 = ((positions[:, 1] - x_min[1]) / dx[1]).long().clamp(0, n2 - 2)
    k0 = ((positions[:, 2] - x_min[2]) / dx[2]).long().clamp(0, n3 - 2)

    # Fractional offsets within the base cell [0, 1)
    fx = ((positions[:, 0] - x_min[0]) / dx[0] - i0.double()).clamp(0.0, 1.0)
    fy = ((positions[:, 1] - x_min[1]) / dx[1] - j0.double()).clamp(0.0, 1.0)
    fz = ((positions[:, 2] - x_min[2]) / dx[2] - k0.double()).clamp(0.0, 1.0)

    n_tilde_flat = torch.zeros(n1 * n2 * n3, dtype=torch.float64, device=device)

    # Deposit weight to all 8 surrounding cells
    for di in (0, 1):
        wx = fx if di == 1 else (1.0 - fx)
        for dj in (0, 1):
            wy = fy if dj == 1 else (1.0 - fy)
            for dk in (0, 1):
                wz = fz if dk == 1 else (1.0 - fz)
                weight = wx * wy * wz
                idx = (i0 + di) * (n2 * n3) + (j0 + dj) * n3 + (k0 + dk)
                n_tilde_flat.scatter_add_(0, idx, weight)

    # Scale by the atoms-per-test-particle weight and divide by the cell volume
    # to convert the deposited count into a number density.
    return (n_tilde_flat * particle_weight / d_x).reshape(n1, n2, n3)


def interpolate_to_particles(
    field: torch.Tensor,
    positions: torch.Tensor,
    n1: int,
    n2: int,
    n3: int,
    x_min: torch.Tensor,
    dx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    r"""
    Interpolate a 3-D grid field trilinearly to the particle positions.

    The CIC weights are the same as in :func:`thermal_density_from_particles`,
    but here they read from the grid instead of writing to it. This is how the
    mean-field potential :math:`U(\mathbf{r}_i)` and the forces are evaluated
    at the particle positions.

    Args:
        field (torch.Tensor): Grid-based field, shape ``(n1, n2, n3)``.
        positions (torch.Tensor): Particle positions, shape ``(N_test, 3)``.
        n1 (int): Number of grid points along the first axis.
        n2 (int): Number of grid points along the second axis.
        n3 (int): Number of grid points along the third axis.
        x_min (torch.Tensor): Lower box boundaries, shape ``(3,)``.
        dx (torch.Tensor): Grid spacings per axis, shape ``(3,)``.
        device (torch.device): Computation device.

    Returns:
        torch.Tensor: Field values at the particle positions, shape
        ``(N_test,)``.
    """
    x_min = x_min.to(device=device, dtype=torch.float64)
    dx = dx.to(device=device, dtype=torch.float64)

    i0 = ((positions[:, 0] - x_min[0]) / dx[0]).long().clamp(0, n1 - 2)
    j0 = ((positions[:, 1] - x_min[1]) / dx[1]).long().clamp(0, n2 - 2)
    k0 = ((positions[:, 2] - x_min[2]) / dx[2]).long().clamp(0, n3 - 2)

    fx = ((positions[:, 0] - x_min[0]) / dx[0] - i0.double()).clamp(0.0, 1.0)
    fy = ((positions[:, 1] - x_min[1]) / dx[1] - j0.double()).clamp(0.0, 1.0)
    fz = ((positions[:, 2] - x_min[2]) / dx[2] - k0.double()).clamp(0.0, 1.0)

    field_flat = field.reshape(-1).double()
    values = torch.zeros(positions.shape[0], dtype=torch.float64, device=device)

    for di in (0, 1):
        wx = fx if di == 1 else (1.0 - fx)
        for dj in (0, 1):
            wy = fy if dj == 1 else (1.0 - fy)
            for dk in (0, 1):
                wz = fz if dk == 1 else (1.0 - fz)
                idx = (i0 + di) * (n2 * n3) + (j0 + dj) * n3 + (k0 + dk)
                values += wx * wy * wz * field_flat[idx]

    return values


def spectral_gradient_3d(
    field: torch.Tensor,
    p_grid: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Compute the 3-D gradient of a real-valued grid field by spectral
    (FFT-based) differentiation.

    In momentum space the derivative is a multiplication,

    .. math::

        \frac{\partial f}{\partial x_\alpha}
            = \mathcal{F}^{-1}\bigl[\, i\, p_\alpha\, \mathcal{F}[f] \,\bigr],

    which is the approach used in :meth:`~src.library.gpe_library.GPELibrary.mod_grad_psi` and gives
    spectral accuracy for smooth (periodic) fields.

    Args:
        field (torch.Tensor): Real-valued grid field of shape ``(n1, n2, n3)``,
            e.g. the mean-field potential :math:`U`.
        p_grid (tuple): Momentum meshgrids ``(px, py, pz)``, each of shape
            ``(n1, n2, n3)``.

    Returns:
        tuple: The real-valued gradient components
        ``(grad_x, grad_y, grad_z)``, each of shape ``(n1, n2, n3)``.
    """
    px, py, pz = p_grid
    field_f = torch.fft.fftn(field.to(torch.cdouble), norm='forward')
    grad_x = torch.fft.ifftn(1j * px * field_f, norm='forward').real
    grad_y = torch.fft.ifftn(1j * py * field_f, norm='forward').real
    grad_z = torch.fft.ifftn(1j * pz * field_f, norm='forward').real
    return grad_x, grad_y, grad_z


# ---------------------------------------------------------------------------
# Mean-field potentials
# ---------------------------------------------------------------------------

def mean_field_potential_for_thermal(
    n_c: torch.Tensor,
    n_tilde: torch.Tensor,
    uext: torch.Tensor,
    u: float,
) -> torch.Tensor:
    r"""
    Evaluate the mean-field potential felt by thermal (non-condensed) atoms.

    Thermal atoms interact with the condensate through the exchange-symmetric
    Hartree-Fock potential,

    .. math::

        U(\mathbf{r}) = V_\mathrm{ext}(\mathbf{r})
            + 2u \bigl[\, n_c(\mathbf{r}) + \tilde{n}(\mathbf{r}) \,\bigr].

    The factor of 2 in front of the densities arises because there are two
    distinct s-wave scattering channels (direct and exchange) between a thermal
    atom and a condensate atom, whereas two condensate atoms interact through
    only one. This potential drives the classical equations of motion for the
    test particles in the Monte Carlo step.

    Args:
        n_c (torch.Tensor): Condensate density
            :math:`n_c = \lvert \psi \rvert^{2}`, shape ``(n1, n2, n3)``.
        n_tilde (torch.Tensor): Thermal density :math:`\tilde{n}`, shape
            ``(n1, n2, n3)``.
        uext (torch.Tensor): External trapping potential, shape
            ``(n1, n2, n3)``.
        u (float): Dimensionless interaction strength
            :math:`g / (\hbar \omega_\mathrm{ho} a_\mathrm{ho}^{3})`.

    Returns:
        torch.Tensor: The potential :math:`U(\mathbf{r})`, shape
        ``(n1, n2, n3)``.

    References:
        E. Zaremba, T. Nikuni and A. Griffin, *J. Low Temp. Phys.* **116**, 277
        (1999), Eq. (2.13).
    """
    return uext + 2.0 * u * (n_c + n_tilde)


def condensate_gpe_potential(
    n_c: torch.Tensor,
    n_tilde: torch.Tensor,
    uext: torch.Tensor,
    u: float,
) -> torch.Tensor:
    r"""
    Evaluate the mean-field potential that enters the condensate GPE.

    The condensate feels a different mean field from the thermal atoms than
    thermal atoms feel from each other (cf.
    :func:`mean_field_potential_for_thermal`),

    .. math::

        V_\mathrm{GP}(\mathbf{r}) = V_\mathrm{ext}(\mathbf{r})
            + u \bigl[\, n_c(\mathbf{r}) + 2 \tilde{n}(\mathbf{r}) \,\bigr].

    The factor of 2 on :math:`\tilde{n}` (and not on :math:`n_c`) reflects the
    Bose enhancement of scattering between a condensate atom and a thermal
    atom.

    Args:
        n_c (torch.Tensor): Condensate density, shape ``(n1, n2, n3)``.
        n_tilde (torch.Tensor): Thermal density :math:`\tilde{n}`, shape
            ``(n1, n2, n3)``.
        uext (torch.Tensor): External trapping potential, shape
            ``(n1, n2, n3)``.
        u (float): Dimensionless interaction strength.

    Returns:
        torch.Tensor: The potential :math:`V_\mathrm{GP}(\mathbf{r})`, shape
        ``(n1, n2, n3)``.

    References:
        E. Zaremba, T. Nikuni and A. Griffin, *J. Low Temp. Phys.* **116**, 277
        (1999), Eq. (2.9).
    """
    return uext + u * (n_c + 2.0 * n_tilde)


# ---------------------------------------------------------------------------
# Source term (C_12 in the mean-field / local equilibrium approximation)
# ---------------------------------------------------------------------------

def condensate_source_term(
    n_c: torch.Tensor,
    n_tilde: torch.Tensor,
    uext: torch.Tensor,
    u: float,
    mu: float,
    gamma_12: float,
) -> torch.Tensor:
    r"""
    Evaluate the local-equilibrium approximation to the :math:`C_{12}`
    condensate source/sink rate :math:`R(\mathbf{r})`.

    :math:`C_{12}` is the collision integral that transfers atoms between the
    condensate and the thermal cloud. In the semiclassical local approximation,
    valid when the thermal cloud varies slowly on the scale of the healing
    length,

    .. math::

        R(\mathbf{r}) = 2 \gamma_{12}
            \bigl[\, \mu - V_\mathrm{eff}(\mathbf{r}) \,\bigr],
        \qquad
        V_\mathrm{eff} = V_\mathrm{ext}
            + 2u \bigl[\, n_c + \tilde{n} \,\bigr],

    where :math:`V_\mathrm{eff}` is the full mean-field potential that drives
    the thermal particle dynamics. The sign convention is that :math:`R > 0`
    grows the condensate (trap centre, where :math:`V_\mathrm{eff} < \mu`)
    and :math:`R < 0` shrinks it (trap wings, where
    :math:`V_\mathrm{eff} > \mu`).

    :math:`R` enters the modified condensate GPE as a complex source term,

    .. math::

        i \frac{\partial \psi}{\partial t}
            = \Bigl[\, H_\mathrm{GP}
                     + \tfrac{i}{2} R(\mathbf{r}) \,\Bigr] \psi,

    where the :math:`+iR/2` (rather than :math:`-iR/2`) gives
    :math:`\partial \lvert \psi \rvert^{2} / \partial t
    = R \lvert \psi \rvert^{2}`, i.e. growth where :math:`R > 0`.

    Args:
        n_c (torch.Tensor): Condensate density, shape ``(n1, n2, n3)``.
        n_tilde (torch.Tensor): Thermal density :math:`\tilde{n}`, shape
            ``(n1, n2, n3)``.
        uext (torch.Tensor): External trapping potential, shape
            ``(n1, n2, n3)``.
        u (float): Dimensionless interaction strength.
        mu (float): Chemical potential :math:`\mu` of the condensate, in units
            of :math:`\hbar \omega_\mathrm{ho}`.
        gamma_12 (float): Dimensionless phenomenological rate
            :math:`\gamma_{12}` for the :math:`C_{12}` condensate-thermal
            exchange. Typical cold-atom values are 0.01-0.3.

    Returns:
        torch.Tensor: The real-valued rate :math:`R(\mathbf{r})`, shape
        ``(n1, n2, n3)``.

    References:
        T. Nikuni, E. Zaremba and A. Griffin, *Phys. Rev. Lett.* **83**, 10
        (1999), Eq. (6).
    """
    v_eff = uext + 2.0 * u * (n_c + n_tilde)
    return 2.0 * gamma_12 * (mu - v_eff)


# ---------------------------------------------------------------------------
# Initial thermal cloud sampling
# ---------------------------------------------------------------------------

def sample_initial_thermal_cloud(
    n_test: int,
    w: torch.Tensor,
    kT: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Sample initial test-particle positions and momenta for the thermal cloud.

    The semiclassical Boltzmann distribution is used in the harmonic-oscillator
    approximation. For a trap with dimensionless frequencies
    :math:`(\omega_x, \omega_y, \omega_z) / \omega_\mathrm{ho}` (the ``w``
    argument below), the thermal equilibrium distribution is

    .. math::

        f(\mathbf{r}, \mathbf{p}) \propto
            \exp\Bigl[ -\beta \bigl(
                \tfrac{1}{2} p^{2} + V_\mathrm{trap}(\mathbf{r})
            \bigr) \Bigr],
        \qquad
        V_\mathrm{trap}
            = \tfrac{1}{2} \sum_\alpha \omega_\alpha^{2} x_\alpha^{2},

    with :math:`\beta = 1 / kT` dimensionless. This factorises, so positions
    and momenta are sampled independently:

    .. math::

        x_\alpha \sim \mathcal{N}\bigl(0, \sigma_\alpha^{2}\bigr),
        \quad \sigma_\alpha = \frac{\sqrt{kT}}{\omega_\alpha},
        \qquad
        p_\alpha \sim \mathcal{N}\bigl(0, kT\bigr).

    The position sampling is exact for a harmonic trap and the
    Maxwell-Boltzmann momenta are valid for any trap. For an anharmonic trap
    the position distribution is approximate, but running the simulation for a
    few trap periods thermalises the cloud to the correct shape.

    Args:
        n_test (int): Number of test particles :math:`N_\mathrm{test}`.
        w (torch.Tensor): Dimensionless trap frequencies
            :math:`(\omega_x, \omega_y, \omega_z) / \omega_\mathrm{ho}`, shape
            ``(3,)``. Obtain from ``system.simulation_parameters["w"]``.
        kT (float): Dimensionless temperature
            :math:`k_B T / (\hbar \omega_\mathrm{ho})`.
        device (torch.device): Computation device.

    Returns:
        tuple: The pair ``(positions, momenta)``, each a ``torch.Tensor`` of
        shape ``(N_test, 3)``.

    Raises:
        ValueError: If ``kT`` is not strictly positive.
    """
    if kT <= 0.0:
        raise ValueError("kT must be > 0 to sample a thermal cloud.")

    sigma_x = (kT ** 0.5) / w[0].item()
    sigma_y = (kT ** 0.5) / w[1].item()
    sigma_z = (kT ** 0.5) / w[2].item()
    sigma_p = kT ** 0.5

    positions = torch.zeros(n_test, 3, dtype=torch.float64, device=device)
    positions[:, 0] = torch.randn(n_test, dtype=torch.float64, device=device) * sigma_x
    positions[:, 1] = torch.randn(n_test, dtype=torch.float64, device=device) * sigma_y
    positions[:, 2] = torch.randn(n_test, dtype=torch.float64, device=device) * sigma_z

    momenta = torch.randn(n_test, 3, dtype=torch.float64, device=device) * sigma_p

    return positions, momenta
