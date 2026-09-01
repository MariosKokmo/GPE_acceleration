r"""
Test-particle Monte Carlo dynamics for the thermal cloud in the ZNG framework.

This module evolves :math:`N_\mathrm{test}` classical particles under the
mean-field potential
:math:`U = V_\mathrm{ext} + 2u (n_c + \tilde{n})`, and handles the stochastic
condensate-thermal collisions (:math:`C_{12}`) together with the
thermal-thermal scattering stub (:math:`C_{22}`).

All functions here are stateless: they take tensors and return tensors. No
simulation objects are imported.

Dimensionless units are used throughout
(:math:`\hbar = m = \omega_\mathrm{ho} = 1`).
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
    r"""
    Compute the classical force :math:`\mathbf{F} = -\nabla U` on each test
    particle.

    The force is obtained in two steps. First the gradient
    :math:`(\partial_x U, \partial_y U, \partial_z U)` is evaluated on the grid
    by FFT-based differentiation, which is spectrally accurate for smooth,
    periodic fields. Each gradient component is then interpolated trilinearly
    from the grid to the particle position. In dimensionless units
    (:math:`m = 1`) the force is simply

    .. math::

        F_\alpha = -\bigl(\nabla U\bigr)_\alpha .

    Args:
        U (torch.Tensor): Mean-field potential on the grid, shape
            ``(n1, n2, n3)``. It should be real-valued; any imaginary part is
            discarded.
        positions (torch.Tensor): Particle positions, shape ``(N_test, 3)``.
        n1 (int): Number of grid points along the first axis.
        n2 (int): Number of grid points along the second axis.
        n3 (int): Number of grid points along the third axis.
        x_min (torch.Tensor): Lower box boundaries, shape ``(3,)``.
        dx (torch.Tensor): Grid spacings per axis, shape ``(3,)``.
        p_grid (tuple): Momentum meshgrids ``(px, py, pz)`` used for the FFT
            gradients.
        device (torch.device): Computation device.

    Returns:
        torch.Tensor: Forces on the particles, shape ``(N_test, 3)``.
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
    r"""
    Advance the test particles by one time step with the velocity-Verlet
    (leapfrog) integrator.

    Leapfrog is symplectic: it conserves a shadow Hamiltonian exactly, so there
    is no secular drift in particle energy even over long integrations. That is
    what keeps the equilibrium Boltzmann distribution correctly represented.
    One step, in velocity-Verlet form, is

    .. math::

        \mathbf{F}_n &= -\nabla U_n(\mathbf{r}_n), \\
        \mathbf{p}_{n + 1/2} &= \mathbf{p}_n
            + \tfrac{1}{2} \Delta t\, \mathbf{F}_n, \\
        \mathbf{r}_{n+1} &= \mathbf{r}_n
            + \Delta t\, \mathbf{p}_{n + 1/2}, \\
        \mathbf{F}_{n+1} &= -\nabla U_{n+1}(\mathbf{r}_{n+1}), \\
        \mathbf{p}_{n+1} &= \mathbf{p}_{n + 1/2}
            + \tfrac{1}{2} \Delta t\, \mathbf{F}_{n+1},

    where the position step uses :math:`m = 1`. ``U_current`` and ``U_next``
    are supplied separately because :math:`U` depends on :math:`n_c` and
    :math:`\tilde{n}`, which change during the condensate evolution step.

    Args:
        positions (torch.Tensor): Current positions, shape ``(N_test, 3)``.
        momenta (torch.Tensor): Current momenta, shape ``(N_test, 3)``.
        U_current (torch.Tensor): Mean-field potential at time :math:`t`, shape
            ``(n1, n2, n3)``.
        U_next (torch.Tensor): Mean-field potential at time
            :math:`t + \Delta t`, shape ``(n1, n2, n3)``. Pass ``U_current``
            here on the first call (predictor step); refine it if a
            second-order accurate scheme is needed.
        n1 (int): Number of grid points along the first axis.
        n2 (int): Number of grid points along the second axis.
        n3 (int): Number of grid points along the third axis.
        x_min (torch.Tensor): Lower box boundaries, shape ``(3,)``.
        dx (torch.Tensor): Grid spacings per axis, shape ``(3,)``.
        p_grid (tuple): Momentum meshgrids ``(px, py, pz)`` for the spectral
            gradients.
        dtau (float): Dimensionless time step
            :math:`\omega_\mathrm{ho} \Delta t`.
        device (torch.device): Computation device.

    Returns:
        tuple: The pair ``(new_positions, new_momenta)``, each a
        ``torch.Tensor`` of shape ``(N_test, 3)``.
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
# C_12: stochastic condensate <-> thermal exchange
# ---------------------------------------------------------------------------

def _stochastic_round(value: float) -> int:
    r"""
    Round to a whole number of test particles without bias.

    A transfer of, say, 2.3 particles becomes 2 particles 70% of the time and 3
    the other 30%, so the *expected* count is exactly 2.3. Always rounding the
    same way would bias every step in one direction, which is precisely the
    systematic drift this module has to avoid.

    Args:
        value (float): Non-negative real number of test particles to round.

    Returns:
        int: A whole number of particles whose expectation value is ``value``.
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
    r"""
    Move atoms between the thermal cloud and the condensate, conserving the
    total.

    **The transfer size is whatever the condensate actually did.** The
    condensate's own equation already carries the mean :math:`C_{12}` exchange,
    as the source term
    :math:`R(\mathbf{r}) = 2 \gamma_{12} [\mu - V_\mathrm{eff}]` in the
    modified GPE. The number of atoms leaving the cloud is therefore not a free
    rate to be modelled a second time here: it is fixed by the norm the
    condensate just gained or lost, which the caller measures across the split
    step and passes in as ``atoms_gained_by_condensate``.

    Deriving the two sides independently is what let them drift apart. The
    emission rate :math:`\gamma_{12} n_c (V_\mathrm{eff} - \mu)` is half the
    :math:`\lvert R \rvert n_c` the condensate sheds, and absorption was keyed
    on the particle energy :math:`\epsilon` rather than on
    :math:`V_\mathrm{eff}`, so the books never balanced. Total atom number now
    closes by construction, to within the single test particle that
    discreteness allows in any one step, and that residual is unbiased (see
    :func:`_stochastic_round`).

    **The physical rates still choose which atoms move.** They set the sampling
    distribution rather than the magnitude:

    - Absorption (cloud to condensate) favours particles energetically ready to
      join it, with weight

      .. math::

          w_\mathrm{abs} = \gamma_{12}\, n_c(\mathbf{r})\,
              \max\bigl(0,\, \mu - \epsilon\bigr),
          \qquad \epsilon = \tfrac{1}{2} p^{2} + U(\mathbf{r}).

    - Emission (condensate to cloud) favours cells where the condensate is
      unbound, with weight

      .. math::

          w_\mathrm{em} = \gamma_{12}\, n_c(\mathbf{r})\,
              \max\bigl(0,\, V_\mathrm{eff}(\mathbf{r}) - \mu\bigr).

    Emitted particles are given Maxwell-Boltzmann momenta at ``kT``, resampled
    until :math:`\epsilon > \mu` so that they genuinely belong to the thermal
    cloud.

    Args:
        positions (torch.Tensor): Particle positions, shape ``(N_test, 3)``.
        momenta (torch.Tensor): Particle momenta, shape ``(N_test, 3)``.
        U (torch.Tensor): Mean-field potential :math:`U(\mathbf{r})`, shape
            ``(n1, n2, n3)``.
        n_c (torch.Tensor): Condensate density
            :math:`\lvert \psi \rvert^{2}`, shape ``(n1, n2, n3)``.
        mu (float): Condensate chemical potential :math:`\mu`, in units of
            :math:`\hbar \omega_\mathrm{ho}`.
        gamma_12 (float): Dimensionless :math:`C_{12}` coupling rate
            :math:`\gamma_{12}`.
        dtau (float): Dimensionless time step. Retained for signature
            stability; the transfer size now comes from the measured norm
            change.
        n1 (int): Number of grid points along the first axis.
        n2 (int): Number of grid points along the second axis.
        n3 (int): Number of grid points along the third axis.
        x_min (torch.Tensor): Lower box boundaries, shape ``(3,)``.
        dx (torch.Tensor): Grid spacings per axis, shape ``(3,)``.
        kT (float): Dimensionless temperature
            :math:`k_B T / (\hbar \omega_\mathrm{ho})`.
        device (torch.device): Computation device.
        particle_weight (float): Atoms represented by one test particle, in the
            condensate normalisation.
        atoms_gained_by_condensate (float): Net atoms the condensate gained
            over the step. A positive value removes that many atoms from the
            cloud, a negative value adds them.

    Returns:
        tuple: ``(new_positions, new_momenta, atoms_transferred)`` — the
        updated particle arrays, and the atoms actually handed to the
        condensate (negative if the cloud received them). The transfer can fall
        short of the request when the cloud does not hold enough atoms to give;
        the caller reconciles the condensate against this figure, so the total
        is conserved even when the exchange saturates.
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
# C_22: thermal-thermal scattering (stub)
# ---------------------------------------------------------------------------

def apply_c22_collisions(
    positions: torch.Tensor,
    momenta: torch.Tensor,
    kT: float,
    dtau: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Apply :math:`C_{22}` thermal-thermal scattering collisions (stub).

    :math:`C_{22}` is the Boltzmann collision integral for two thermal atoms
    scattering off each other. It conserves total particle number and total
    momentum but redistributes energy between particles, driving the thermal
    cloud towards a local Maxwell-Boltzmann distribution.

    The full :math:`C_{22}` requires evaluating a six-dimensional collision
    integral for every pair of particles, which is computationally expensive
    and typically implemented either with Bird's direct simulation Monte Carlo
    (DSMC) algorithm, or with a mean-free-path relaxation-time approximation
    (BGK model).

    This stub is a placeholder: it returns the inputs unchanged. Implement one
    of the above schemes here when needed.

    Note:
        Without :math:`C_{22}`, the thermal cloud thermalises only through
        :math:`C_{12}`, i.e. through its coupling to the condensate. For most
        BEC dynamics near equilibrium that is sufficient. :math:`C_{22}`
        matters when the thermal cloud is driven far from equilibrium, for
        example during rapid evaporative cooling or quenches.

    Args:
        positions (torch.Tensor): Particle positions, shape ``(N_test, 3)``.
        momenta (torch.Tensor): Particle momenta, shape ``(N_test, 3)``.
        kT (float): Dimensionless temperature, for future relaxation-time use.
        dtau (float): Dimensionless time step, for future DSMC use.
        device (torch.device): Computation device.

    Returns:
        tuple: The pair ``(positions, momenta)``, unchanged.
    """
    # C_22 not yet implemented — thermal cloud thermalises via C_12 only.
    return positions, momenta
