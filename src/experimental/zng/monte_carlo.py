"""
Test-particle Monte Carlo dynamics for the thermal cloud in the ZNG framework.

This module evolves N_test classical particles under the mean-field potential
U(r) = V_ext + 2u(n_c + ñ) and handles stochastic condensate↔thermal
collisions (C_12) and the thermal–thermal scattering stub (C_22).

All functions are stateless — they take tensors, return tensors.
No simulation objects are imported.

Dimensionless units: ħ = m = ω_ho = 1.
"""

import math
import torch
from typing import Tuple

from src.experimental.zng.zng_library import (
    spectral_gradient_3d,
    interpolate_to_particles,
)


# ---------------------------------------------------------------------------
# Particle equations of motion
# ---------------------------------------------------------------------------

def compute_particle_forces(
    U: torch.Tensor,
    positions: torch.Tensor,
    n1: int,
    n2: int,
    n3: int,
    x_min: torch.Tensor,
    dx: torch.Tensor,
    p_grid: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the classical force F = −∇U on each test particle.

    Step 1 — spectral gradient:  compute (∂U/∂x, ∂U/∂y, ∂U/∂z) on the grid
    using FFT-based differentiation (spectral accuracy for smooth, periodic fields).

    Step 2 — grid→particle interpolation:  trilinearly interpolate each gradient
    component from the grid to the particle position.

    In dimensionless units (m = 1): F_i = −(∇U)_i.

    Args:
        U (torch.Tensor): Mean-field potential on the grid, shape (n1, n2, n3).
            Should be real-valued; any imaginary part is discarded.
        positions (torch.Tensor): Particle positions, shape (N_test, 3).
        n1, n2, n3 (int): Grid point counts.
        x_min (torch.Tensor): Lower box boundaries, length 3.
        dx (torch.Tensor): Grid spacings, length 3.
        p_grid (tuple): (px, py, pz) 3-D momentum meshgrids for FFT gradients.
        device (torch.device): Computation device.

    Returns:
        torch.Tensor: Forces, shape (N_test, 3).
    """
    grad_x, grad_y, grad_z = spectral_gradient_3d(U.real, p_grid)

    forces = torch.zeros_like(positions)
    forces[:, 0] = -interpolate_to_particles(grad_x, positions, n1, n2, n3, x_min, dx, device)
    forces[:, 1] = -interpolate_to_particles(grad_y, positions, n1, n2, n3, x_min, dx, device)
    forces[:, 2] = -interpolate_to_particles(grad_z, positions, n1, n2, n3, x_min, dx, device)
    return forces


def advance_particles_leapfrog(
    positions: torch.Tensor,
    momenta: torch.Tensor,
    U_current: torch.Tensor,
    U_next: torch.Tensor,
    n1: int,
    n2: int,
    n3: int,
    x_min: torch.Tensor,
    dx: torch.Tensor,
    p_grid: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dtau: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Advance test particles by one time step using the velocity-Verlet
    (leapfrog) integrator.

    Leapfrog is symplectic — it conserves a shadow Hamiltonian exactly, giving
    no secular drift in particle energy even over long integrations.  This is
    important for correctly representing the equilibrium Boltzmann distribution.

    Algorithm (velocity Verlet form):
        1.  F_n    = −∇U_n(r_n)           (force at current position)
        2.  p_half = p_n + (dt/2) F_n      (half momentum step)
        3.  r_{n+1} = r_n + dt p_half      (full position step, m=1)
        4.  F_{n+1} = −∇U_{n+1}(r_{n+1})  (force at new position and new potential)
        5.  p_{n+1} = p_half + (dt/2) F_{n+1}  (second half momentum step)

    U_current and U_next are provided separately because U depends on n_c and ñ,
    which change during the condensate evolution step.

    Args:
        positions (torch.Tensor): Current positions, shape (N_test, 3).
        momenta (torch.Tensor): Current momenta, shape (N_test, 3).
        U_current (torch.Tensor): Mean-field potential at time t, (n1,n2,n3).
        U_next (torch.Tensor): Mean-field potential at time t+dt, (n1,n2,n3).
            Pass U_current here on the first call (predictor step); refine if
            a second-order accurate scheme is needed.
        n1, n2, n3 (int): Grid dimensions.
        x_min (torch.Tensor): Lower box boundaries.
        dx (torch.Tensor): Grid spacings.
        p_grid (tuple): Momentum meshgrids for spectral gradients.
        dtau (float): Dimensionless time step ω_ho · dt.
        device (torch.device): Computation device.

    Returns:
        new_positions (torch.Tensor): Shape (N_test, 3).
        new_momenta  (torch.Tensor): Shape (N_test, 3).
    """
    # Step 1 & 2: half momentum step with current forces
    F_current = compute_particle_forces(
        U_current, positions, n1, n2, n3, x_min, dx, p_grid, device
    )
    momenta_half = momenta + 0.5 * dtau * F_current

    # Step 3: full position step (m = 1 in dimensionless units)
    new_positions = positions + dtau * momenta_half

    # Step 4 & 5: half momentum step with forces at new positions and new potential
    F_next = compute_particle_forces(
        U_next, new_positions, n1, n2, n3, x_min, dx, p_grid, device
    )
    new_momenta = momenta_half + 0.5 * dtau * F_next

    return new_positions, new_momenta


# ---------------------------------------------------------------------------
# C_12: stochastic condensate ↔ thermal exchange
# ---------------------------------------------------------------------------

def _stochastic_round(value: float) -> int:
    """
    Round to a whole number of test particles without bias.

    A transfer of, say, 2.3 particles becomes 2 particles 70% of the time and 3
    the other 30%, so the *expected* count is exactly 2.3. Always rounding the
    same way would bias every step in one direction, which is precisely the
    systematic drift this module has to avoid.
    """
    base = int(math.floor(value))
    frac = value - base
    if frac > 0.0 and float(torch.rand(())) < frac:
        base += 1
    return base


def apply_c12_collisions(
    positions: torch.Tensor,
    momenta: torch.Tensor,
    U: torch.Tensor,
    n_c: torch.Tensor,
    mu: float,
    gamma_12: float,
    dtau: float,
    n1: int,
    n2: int,
    n3: int,
    x_min: torch.Tensor,
    dx: torch.Tensor,
    kT: float,
    device: torch.device,
    particle_weight: float,
    atoms_gained_by_condensate: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Move atoms between the thermal cloud and the condensate, conserving the total.

    The transfer size is whatever the condensate actually did
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The condensate's own equation already carries the mean C_12 exchange, as
    the source term R(r) = 2 gamma_12 [mu - V_eff] in the modified GPE. The
    number of atoms leaving the cloud is therefore not a free rate to be
    modelled a second time here: it is fixed by the norm the condensate just
    gained or lost, which the caller measures across the split step and passes
    in as ``atoms_gained_by_condensate``.

    Deriving the two sides independently is what let them drift apart. The
    emission rate ``gamma_12 n_c (V_eff - mu)`` is half the ``|R| n_c`` the
    condensate sheds, and absorption was keyed on the particle energy ``eps``
    rather than on ``V_eff``, so the books never balanced. Total atom number
    now closes by construction, to within the single test particle that
    discreteness allows in any one step -- and that residual is unbiased, see
    :func:`_stochastic_round`.

    The physical rates still choose *which* atoms move
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    They set the sampling distribution rather than the magnitude:

    - Absorption (cloud -> condensate) favours particles energetically ready to
      join it, with weight ``gamma_12 n_c(r) max(0, mu - eps)`` where
      ``eps = p^2/2 + U(r)``.
    - Emission (condensate -> cloud) favours cells where the condensate is
      unbound, with weight ``gamma_12 n_c(r) max(0, V_eff(r) - mu)``.

    Emitted particles get Maxwell-Boltzmann momenta at ``kT``, resampled so
    ``eps > mu`` and they genuinely belong to the thermal cloud.

    Args:
        positions (torch.Tensor): Particle positions, shape (N_test, 3).
        momenta (torch.Tensor): Particle momenta, shape (N_test, 3).
        U (torch.Tensor): Mean-field potential U(r), shape (n1, n2, n3).
        n_c (torch.Tensor): Condensate density |psi|^2, shape (n1, n2, n3).
        mu (float): Condensate chemical potential (hbar omega_ho units).
        gamma_12 (float): Dimensionless C_12 coupling rate.
        dtau (float): Dimensionless time step. Retained for signature stability;
            the transfer size now comes from the measured norm change.
        n1, n2, n3 (int): Grid dimensions.
        x_min (torch.Tensor): Lower box boundaries.
        dx (torch.Tensor): Grid spacings.
        kT (float): Dimensionless temperature k_B T / (hbar omega_ho).
        device (torch.device): Computation device.
        particle_weight (float): Atoms represented by one test particle, in the
            condensate normalisation.
        atoms_gained_by_condensate (float): Net atoms the condensate gained over
            the step. Positive removes that many atoms from the cloud, negative
            adds them.

    Returns:
        tuple: ``(new_positions, new_momenta, atoms_transferred)`` — the
        updated particle arrays, and the atoms actually handed to the
        condensate (negative if the cloud received them). The transfer can
        fall short of the request when the cloud does not hold enough atoms
        to give; the caller reconciles the condensate against this figure, so
        the total is conserved even when the exchange saturates.
    """
    if particle_weight <= 0.0 or atoms_gained_by_condensate == 0.0:
        return positions, momenta, 0.0

    count = _stochastic_round(abs(atoms_gained_by_condensate) / particle_weight)
    if count == 0:
        return positions, momenta, 0.0

    x_min_d = x_min.to(device=device, dtype=torch.float64)
    dx_d = dx.to(device=device, dtype=torch.float64)

    # ------------------------------------------------------------------
    # Condensate grew: take the atoms out of the thermal cloud
    # ------------------------------------------------------------------
    if atoms_gained_by_condensate > 0.0:
        available = positions.shape[0]
        if available == 0:
            return positions, momenta, 0.0

        U_at_particles = interpolate_to_particles(
            U, positions, n1, n2, n3, x_min, dx, device
        )
        n_c_at_particles = interpolate_to_particles(
            n_c, positions, n1, n2, n3, x_min, dx, device
        )
        epsilon = 0.5 * (momenta ** 2).sum(dim=1) + U_at_particles
        weights = (gamma_12 * n_c_at_particles
                   * (mu - epsilon).clamp(min=0.0)).clamp(min=0.0).double()

        # Nothing is energetically ready to be absorbed: pick uniformly instead,
        # so the atom count still balances.
        eligible = int((weights > 0).sum())
        if eligible == 0:
            weights = torch.ones(available, dtype=torch.float64, device=device)
            eligible = available

        count = min(count, eligible, available)
        if count == 0:
            return positions, momenta, 0.0

        chosen = torch.multinomial(weights, count, replacement=False)
        keep = torch.ones(available, dtype=torch.bool, device=device)
        keep[chosen] = False
        return positions[keep], momenta[keep], count * particle_weight

    # ------------------------------------------------------------------
    # Condensate shrank: put the atoms back into the thermal cloud
    # ------------------------------------------------------------------
    emission = (gamma_12 * n_c * (U.real - mu).clamp(min=0.0)).reshape(-1)
    emission = emission.clamp(min=0.0).double()
    if float(emission.sum()) <= 0.0:
        # No unbound region: emit from wherever the condensate actually is.
        emission = n_c.reshape(-1).clamp(min=0.0).double()
        if float(emission.sum()) <= 0.0:
            return positions, momenta, 0.0

    cells = torch.multinomial(emission, count, replacement=True)
    k_idx = cells % n3
    j_idx = (cells // n3) % n2
    i_idx = cells // (n2 * n3)

    # Cell centre plus uniform jitter within the cell
    jitter = torch.rand(count, 3, dtype=torch.float64, device=device) - 0.5
    new_pos = torch.stack([
        x_min_d[0] + (i_idx.double() + 0.5) * dx_d[0] + jitter[:, 0] * dx_d[0],
        x_min_d[1] + (j_idx.double() + 0.5) * dx_d[1] + jitter[:, 1] * dx_d[1],
        x_min_d[2] + (k_idx.double() + 0.5) * dx_d[2] + jitter[:, 2] * dx_d[2],
    ], dim=1)

    # Maxwell-Boltzmann momenta, resampled until the particle sits above mu so
    # it belongs to the thermal cloud rather than being reabsorbed immediately.
    U_new = interpolate_to_particles(U, new_pos, n1, n2, n3, x_min, dx, device)
    new_mom = torch.randn(count, 3, dtype=torch.float64, device=device) * (kT ** 0.5)
    for _ in range(8):
        invalid = (0.5 * (new_mom ** 2).sum(dim=1) + U_new) <= mu
        if not bool(invalid.any()):
            break
        resampled = torch.randn(count, 3, dtype=torch.float64, device=device) * (kT ** 0.5)
        new_mom = torch.where(invalid.unsqueeze(1), resampled, new_mom)

    # Stragglers get just enough kinetic energy to clear mu, so the emitted
    # count is exact rather than silently short (which would leak atoms).
    invalid = (0.5 * (new_mom ** 2).sum(dim=1) + U_new) <= mu
    if bool(invalid.any()):
        needed = (mu - U_new).clamp(min=0.0) + max(kT, 1e-12)
        direction = torch.nn.functional.normalize(new_mom + 1e-12, dim=1, eps=1e-30)
        boosted = direction * torch.sqrt(2.0 * needed).unsqueeze(1)
        new_mom = torch.where(invalid.unsqueeze(1), boosted, new_mom)

    positions = torch.cat([positions, new_pos], dim=0)
    momenta = torch.cat([momenta, new_mom], dim=0)
    return positions, momenta, -count * particle_weight


# ---------------------------------------------------------------------------
# C_22: thermal–thermal scattering (stub)
# ---------------------------------------------------------------------------

def apply_c22_collisions(
    positions: torch.Tensor,
    momenta: torch.Tensor,
    kT: float,
    dtau: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply C_22 thermal–thermal scattering collisions (stub).

    C_22 is the Boltzmann collision integral for two thermal atoms scattering
    off each other.  It conserves total particle number and total momentum
    but redistributes energy between particles, driving the thermal cloud toward
    a local Maxwell-Boltzmann distribution.

    The full C_22 requires evaluating a 6-dimensional collision integral for
    every pair of particles — computationally expensive and typically implemented
    via:

      - Bird's direct simulation Monte Carlo (DSMC) algorithm, or
      - a mean-free-path relaxation-time approximation (BGK model).

    This stub is a placeholder.  It returns the inputs unchanged and logs a
    warning.  Implement one of the above schemes here when needed.

    Physical significance of C_22:
        Without C_22, the thermal cloud thermalises only through C_12 (coupling
        to the condensate).  For most BEC dynamics near equilibrium this is
        sufficient.  C_22 matters when the thermal cloud is driven far from
        equilibrium, e.g. during rapid evaporative cooling or quenches.

    Args:
        positions (torch.Tensor): Particle positions, shape (N_test, 3).
        momenta (torch.Tensor): Particle momenta, shape (N_test, 3).
        kT (float): Dimensionless temperature (for future relaxation-time use).
        dtau (float): Dimensionless time step (for future DSMC use).
        device (torch.device): Computation device.

    Returns:
        positions (torch.Tensor): Unchanged.
        momenta  (torch.Tensor): Unchanged.
    """
    # C_22 not yet implemented — thermal cloud thermalises via C_12 only.
    return positions, momenta
