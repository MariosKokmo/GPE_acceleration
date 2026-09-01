"""
Pure physics functions for the ZNG framework.

All functions are stateless — they take tensors and scalars, return tensors.
No simulation objects are imported here so this file can be tested in isolation.

Dimensionless units throughout (ħ = m = ω_ho = 1):
  length   → a_ho = sqrt(ħ / m ω_ho)
  energy   → ħ ω_ho
  momentum → ħ / a_ho
  time     → 1 / ω_ho
  temperature → k_B T / (ħ ω_ho)   (referred to as kT below)
"""

import torch
import math
from typing import Tuple


# ---------------------------------------------------------------------------
# Grid ↔ particle interpolation
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
    """
    Deposit test-particle positions onto the 3-D grid using Cloud-In-Cell (CIC)
    trilinear weighting to obtain the thermal number density ñ(r).

    CIC assigns each particle fractional weight to the 8 surrounding lattice
    sites proportional to the overlap of a unit cell centred on the particle
    with each grid cell.  For a particle at fractional offset (fx, fy, fz)
    within its base cell (i0, j0, k0):

        w[i0+di, j0+dj, k0+dk] = (fx if di==1 else 1−fx)
                                × (fy if dj==1 else 1−fy)
                                × (fz if dk==1 else 1−fz)

    The deposited count is then divided by the cell volume d_x so the result
    has units of number density (in dimensionless a_ho units).

    Args:
        positions (torch.Tensor): Shape (N_test, 3), particle positions in
            dimensionless units.  Particles outside the box are clamped to the
            nearest boundary cell.
        n1, n2, n3 (int): Grid point counts per dimension.
        x_min (torch.Tensor): 1-D tensor of length 3 — lower box boundaries.
        dx (torch.Tensor): 1-D tensor of length 3 — grid spacings per axis.
        d_x (float): Grid cell volume = dx[0]*dx[1]*dx[2].
        device (torch.device): Computation device.
        particle_weight (float): How many atoms one test particle stands for,
            in the same normalisation as the condensate (where
            ``∫|psi|^2 dV = 1`` is the whole cloud). With the default of 1.0
            the result integrates to the raw test-particle count, which puts
            ñ on a different scale from n_c and makes every mean-field term
            that mixes them depend on N_test. Pass
            ``thermal_fraction / n_test`` so that ``∫ñ dV`` is the thermal
            fraction and the physics is independent of the particle count.

    Returns:
        torch.Tensor: Shape (n1, n2, n3), thermal number density ñ(r).
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
    """
    Trilinear interpolation of a 3-D grid field to particle positions.

    Uses the same CIC weights as :func:`thermal_density_from_particles` but
    reads from the grid instead of writing to it.  This is used to evaluate
    the mean-field potential U(r_i) and forces at particle positions.

    Args:
        field (torch.Tensor): Shape (n1, n2, n3) — the grid-based field.
        positions (torch.Tensor): Shape (N_test, 3) — particle positions.
        n1, n2, n3 (int): Grid point counts.
        x_min (torch.Tensor): Lower box boundaries, length 3.
        dx (torch.Tensor): Grid spacings, length 3.
        device (torch.device): Computation device.

    Returns:
        torch.Tensor: Shape (N_test,) — field values at particle positions.
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
    """
    Compute the 3-D gradient of a real-valued grid field using spectral
    (FFT-based) differentiation.

    In momentum space: ∂f/∂x_i = IFFT(i·p_i · FFT(f))

    This is the same approach used in GPELibrary.mod_grad_psi and gives
    spectral accuracy for smooth (periodic) fields.

    Args:
        field (torch.Tensor): Shape (n1, n2, n3) — real-valued grid field
            (e.g. the mean-field potential U).
        p_grid (tuple): (px, py, pz) — 3-D momentum meshgrids, each (n1,n2,n3).

    Returns:
        tuple: (grad_x, grad_y, grad_z) — gradient components, each (n1,n2,n3),
            real-valued.
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
    """
    Mean-field potential felt by thermal (non-condensed) atoms.

    Thermal atoms interact with the condensate via the exchange-symmetric
    Hartree-Fock potential.  The factor of 2 in front of the condensate density
    arises because there are two distinct s-wave scattering channels (direct
    and exchange) between a thermal atom and a condensate atom, whereas two
    condensate atoms interact through only one:

        U(r) = V_ext(r) + 2u [n_c(r) + ñ(r)]

    This potential drives the classical equations of motion for the test
    particles in the Monte Carlo step.

    Reference: E. Zaremba, T. Nikuni, A. Griffin, J. Low Temp. Phys. 116,
               277 (1999), Eq. (2.13).

    Args:
        n_c (torch.Tensor): Condensate density |ψ|², shape (n1, n2, n3).
        n_tilde (torch.Tensor): Thermal density ñ, shape (n1, n2, n3).
        uext (torch.Tensor): External trapping potential, shape (n1, n2, n3).
        u (float): Dimensionless interaction strength g/(ħ ω_ho a_ho³).

    Returns:
        torch.Tensor: U(r), shape (n1, n2, n3).
    """
    return uext + 2.0 * u * (n_c + n_tilde)


def condensate_gpe_potential(
    n_c: torch.Tensor,
    n_tilde: torch.Tensor,
    uext: torch.Tensor,
    u: float,
) -> torch.Tensor:
    """
    Mean-field potential that enters the condensate GPE.

    The condensate feels a different mean-field from thermal atoms than
    thermal atoms feel from each other (cf. :func:`mean_field_potential_for_thermal`):

        V_GP(r) = V_ext(r) + u [n_c(r) + 2ñ(r)]

    The factor of 2 on ñ (not on n_c) reflects the Bose enhancement of
    scattering between a condensate atom and a thermal atom.

    Reference: ZNG 1999, Eq. (2.9).

    Args:
        n_c (torch.Tensor): Condensate density, shape (n1, n2, n3).
        n_tilde (torch.Tensor): Thermal density ñ, shape (n1, n2, n3).
        uext (torch.Tensor): External trapping potential, shape (n1, n2, n3).
        u (float): Dimensionless interaction strength.

    Returns:
        torch.Tensor: V_GP(r), shape (n1, n2, n3).
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
    """
    Local-equilibrium approximation to the C_12 condensate source/sink rate R(r).

    C_12 is the collision integral that transfers atoms between the condensate
    and the thermal cloud.  In the semiclassical local approximation (valid when
    the thermal cloud varies slowly on the scale of the healing length):

        R(r) = 2 γ_12 [μ − V_eff(r)]

    where V_eff(r) = V_ext(r) + 2u [n_c(r) + ñ(r)] is the full mean-field
    potential that drives the thermal particle dynamics.

    Sign convention:
        R(r) > 0  →  condensate grows   (trap centre, where V_eff < μ)
        R(r) < 0  →  condensate shrinks (trap wings, where V_eff > μ)

    R enters the modified condensate GPE as a complex source term:

        i ∂ψ/∂t = [H_GP + iR(r)/2] ψ

    The +iR/2 (not −iR/2) gives ∂|ψ|²/∂t = R|ψ², i.e. growth where R > 0.

    Reference: T. Nikuni, E. Zaremba, A. Griffin, PRL 83, 10 (1999), Eq. (6).

    Args:
        n_c (torch.Tensor): Condensate density, shape (n1, n2, n3).
        n_tilde (torch.Tensor): Thermal density ñ, shape (n1, n2, n3).
        uext (torch.Tensor): External trapping potential, shape (n1, n2, n3).
        u (float): Dimensionless interaction strength.
        mu (float): Chemical potential μ of the condensate (ħ ω_ho units).
        gamma_12 (float): Dimensionless phenomenological rate for C_12
            condensate↔thermal exchange.  Typical cold-atom values: 0.01–0.3.

    Returns:
        torch.Tensor: R(r), shape (n1, n2, n3), real-valued.
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
    """
    Sample initial test-particle positions and momenta for the thermal cloud.

    Uses the semiclassical Boltzmann distribution in the harmonic-oscillator
    approximation.  For a trap with dimensionless frequencies
    ω = (ω_x, ω_y, ω_z) / ω_ho (the ``w`` argument below), the thermal
    equilibrium distribution is:

        f(r, p) ∝ exp[−β (p²/2 + V_trap(r))]

    where β = 1/kT (dimensionless) and V_trap = ½ Σ ω_i² x_i².

    This factorises: positions and momenta are sampled independently.

    Position sampling (exact for a harmonic trap):
        x_i ~ Normal(0, σ_x)   with  σ_x = √(kT) / ω_x

    Momentum sampling (Maxwell-Boltzmann, valid for any trap):
        p_i ~ Normal(0, √kT)

    For an anharmonic trap the position distribution is approximate; running
    the simulation for a few trap periods thermalises the cloud to the correct
    shape.

    Args:
        n_test (int): Number of test particles N_test.
        w (torch.Tensor): Dimensionless trap frequencies (ω_x, ω_y, ω_z) / ω_ho,
            shape (3,).  Obtain from system.simulation_parameters["w"].
        kT (float): Dimensionless temperature k_B T / (ħ ω_ho).
        device (torch.device): Computation device.

    Returns:
        positions (torch.Tensor): Shape (N_test, 3), sampled positions.
        momenta  (torch.Tensor): Shape (N_test, 3), sampled momenta.
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
