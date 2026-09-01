r"""
Zaremba-Nikuni-Griffin (ZNG) finite-temperature BEC model.

:class:`ZNGBEC` inherits all ground-state finding, I/O, snapshotting and
split-step machinery from :class:`BaseBEC`. It overrides only
:meth:`ZNGBEC._initialize_custom_parameters` and
:meth:`ZNGBEC._main_simulation_loop` to implement the two-component ZNG
dynamics.

In dimensionless units (:math:`\hbar = m = \omega_\mathrm{ho} = 1`) the
condensate obeys the modified GPE

.. math::

    i \frac{\partial \psi}{\partial t}
        &= \Bigl[\, H_\mathrm{GP}
                 + \tfrac{i}{2} R(\mathbf{r}, t) \,\Bigr] \psi, \\
    H_\mathrm{GP}
        &= -\tfrac{1}{2} \nabla^{2} + V_\mathrm{ext}
           + u \bigl(n_c + 2 \tilde{n}\bigr), \\
    R(\mathbf{r})
        &= 2 \gamma_{12} \bigl[\, \mu - V_\mathrm{eff}(\mathbf{r}) \,\bigr],
        \qquad
    V_\mathrm{eff} = V_\mathrm{ext} + 2u \bigl(n_c + \tilde{n}\bigr),

where :math:`R` is the condensate source/sink rate and
:math:`V_\mathrm{eff}` the full mean field seen by thermal atoms. The thermal
cloud is represented by classical test particles obeying

.. math::

    \frac{\mathrm{d}\mathbf{r}_i}{\mathrm{d}t} = \mathbf{p}_i,
    \qquad
    \frac{\mathrm{d}\mathbf{p}_i}{\mathrm{d}t} = -\nabla U(\mathbf{r}_i),
    \qquad
    U = V_\mathrm{ext} + 2u \bigl(n_c + \tilde{n}\bigr) = V_\mathrm{eff}.

The two equations are coupled through the densities
:math:`n_c = \lvert \psi \rvert^{2}` and :math:`\tilde{n}(\mathbf{r})`, the
latter deposited from the test-particle positions onto the grid.

References:
    E. Zaremba, T. Nikuni and A. Griffin, *J. Low Temp. Phys.* **116**, 277
    (1999).

    T. Nikuni, E. Zaremba and A. Griffin, *Phys. Rev. Lett.* **83**, 10 (1999).

    S. Gardiner, D. Jaksch, R. Dum, J. Cirac and P. Zoller, *Phys. Rev. A*
    **62**, 023612 (2000).
"""

import torch
import numpy as np
from typing import Optional

from src.models.base_BEC import BaseBEC
from src.library.gpe_library import GPELibrary as gpe
from src.library.parameters import CONSTANTS

from src.experimental.zng.zng_library import (
    thermal_density_from_particles,
    mean_field_potential_for_thermal,
    condensate_gpe_potential,
    condensate_source_term,
    sample_initial_thermal_cloud,
)
from src.experimental.zng.monte_carlo import (
    advance_particles_leapfrog,
    apply_c12_collisions,
    apply_c22_collisions,
)


class ZNGBEC(BaseBEC):
    r"""
    Finite-temperature BEC simulation using the full Zaremba-Nikuni-Griffin
    (ZNG) two-component framework.

    The following keys are read from the simulation parameters (dict or JSON):

    ``temperature`` (float)
        Dimensionless temperature :math:`k_B T / (\hbar \omega_\mathrm{ho})`.
        Must be positive; use ``FiniteTempBEC`` / SGPE for the
        :math:`T \to 0` limit.
    ``n_test_particles`` (int, default 10000)
        Number of Monte Carlo test particles representing the thermal cloud.
        A larger :math:`N_\mathrm{test}` gives a smoother
        :math:`\tilde{n}(\mathbf{r})` with less shot noise, at more compute.
    ``gamma_12`` (float, default 0.1)
        Dimensionless :math:`C_{12}` coupling rate :math:`\gamma_{12}` between
        condensate and thermal cloud. It controls how fast atoms are exchanged
        between the two components; :math:`\gamma_{12} = 0` decouples them
        entirely.
    ``chemical_potential`` (float or None, default None)
        Reservoir chemical potential :math:`\mu`, in units of
        :math:`\hbar \omega_\mathrm{ho}`. If ``None``, it is computed once from
        the initial ground-state wavefunction.
    ``enable_c22`` (bool, default False)
        Whether to apply :math:`C_{22}` (thermal-thermal) collisions.
        Currently a no-op stub; set it to ``True`` once a real :math:`C_{22}`
        is implemented.

    See :meth:`_initialize_custom_parameters` for the remaining ZNG-specific
    keys.

    Attributes:
        psi (torch.Tensor): Condensate wavefunction :math:`\psi(\mathbf{r})`,
            shape ``(n1, n2, n3)``, ``cdouble``.
        particle_positions (torch.Tensor): Thermal test-particle positions,
            shape ``(N_test, 3)``.
        particle_momenta (torch.Tensor): Thermal test-particle momenta, shape
            ``(N_test, 3)``.
        n_tilde (torch.Tensor): Thermal density :math:`\tilde{n}(\mathbf{r})`
            on the grid, shape ``(n1, n2, n3)``.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_custom_parameters(self) -> None:
        r"""
        Read the ZNG-specific parameters and allocate the thermal cloud arrays.

        ``temperature`` (float)
            :math:`k_B T / (\hbar \omega_\mathrm{ho})`, which controls the
            thermal cloud width and its noise.
        ``n_test_particles`` (int)
            :math:`N_\mathrm{test}`, the number of test particles for the Monte
            Carlo thermal cloud. More particles give a smoother
            :math:`\tilde{n}` but a slower step.
        ``gamma_12`` (float)
            Phenomenological :math:`C_{12}` rate. It drives the atom exchange
            between condensate and thermal cloud on the timescale
            :math:`1 / \gamma_{12}`.
        ``chemical_potential`` (float or None)
            If not provided, :math:`\mu` is computed from the :math:`T = 0`
            ground state as
            :math:`\mu = E_\mathrm{kin} + E_\mathrm{pot} + 2 E_\mathrm{int}`
            and held fixed throughout.
        ``enable_c22`` (bool)
            Placeholder flag. :math:`C_{22}` is the thermal-thermal Boltzmann
            collision integral; it thermalises the distribution further but is
            expensive.
        ``zng_condensate_exchange`` (bool, default False)
            Whether the condensate is allowed to exchange atoms with the
            thermal cloud.

            ``False`` — the condensate is renormalised after every split step,
            so its atom number is pinned. The source term :math:`R` still
            reshapes the condensate (:math:`R` varies in space, so rescaling
            does not cancel it), but no atoms actually leave or enter. This is
            the behaviour the model was originally written with, and it is the
            default so that existing runs reproduce.

            ``True`` — the norm is left free. :math:`R` then does what it is
            meant to do: where :math:`R > 0` the condensate grows, where
            :math:`R < 0` it decays, and the :math:`C_{12}` step moves the
            corresponding test particles the other way. This is the physically
            consistent choice, since the :math:`C_{12}` collision step already
            creates and destroys test particles; with the norm pinned those
            transfers are one-sided and total atom number is not conserved.

        See :meth:`_resolve_thermal_fraction` for
        ``zng_thermal_fraction_mode`` and ``zng_thermal_fraction``.
        """
        self.temperature: float = self.parameters.get("temperature", 1.0)
        self.n_test: int = int(self.parameters.get("n_test_particles", 10_000))
        self.gamma_12: float = self.parameters.get("gamma_12", 0.1)
        self.mu: Optional[float] = self.parameters.get("chemical_potential", None)
        self.enable_c22: bool = bool(self.parameters.get("enable_c22", False))
        self.condensate_exchange: bool = bool(
            self.parameters.get("zng_condensate_exchange", False)
        )
        self.thermal_fraction_mode: str = str(
            self.parameters.get("zng_thermal_fraction_mode", "temperature")
        ).lower()
        self.thermal_fraction: float = self._resolve_thermal_fraction()
        # Atoms represented by one test particle, in the condensate's own
        # normalisation. This is what puts ñ and n_c on a common scale.
        self.particle_weight: float = (
            self.thermal_fraction / self.n_test if self.n_test > 0 else 0.0
        )

        # These are populated in _initialise() after the ground state is loaded.
        self.particle_positions: Optional[torch.Tensor] = None
        self.particle_momenta: Optional[torch.Tensor] = None
        self.n_tilde: Optional[torch.Tensor] = None

        self.logger.info(
            f"ZNGBEC: T = {self.temperature} [ħω_ho/k_B], "
            f"N_test = {self.n_test}, γ_12 = {self.gamma_12}, "
            f"μ = {'(computed from ground state)' if self.mu is None else self.mu}, "
            f"C_22 = {'enabled (stub)' if self.enable_c22 else 'disabled'}, "
            f"condensate = {'exchanging with the cloud (free norm)' if self.condensate_exchange else 'number-pinned (renormalised each step)'}"
        )

    def _initialise(self) -> None:
        r"""
        Load the condensate ground state and initialise the thermal cloud.

        The parent :meth:`BaseBEC._initialise` is called first to find or read
        :math:`\psi_0` from disk. The sample is then split between the two
        components, :math:`\mu` is computed (unless it was supplied), and the
        initial test-particle distribution is drawn from the semiclassical
        Boltzmann distribution.
        """
        # Load condensate ground state (ψ₀) via BaseBEC
        super()._initialise()

        # The ground state arrives normalised to 1, i.e. holding every atom.
        # Split the sample between the two components so that
        #   ∫n_c dV = 1 − f   and   ∫ñ dV = f
        # and the total is 1 in the condensate's own normalisation.
        if self.thermal_fraction > 0.0:
            self.psi = self.psi * ((1.0 - self.thermal_fraction) ** 0.5)
            self.logger.info(
                f"ZNG: thermal fraction f = {self.thermal_fraction:.4f} "
                f"({self.thermal_fraction_mode} convention); condensate scaled to "
                f"{1.0 - self.thermal_fraction:.4f} of the sample, "
                f"{self.particle_weight:.3e} atoms per test particle."
            )

        # Compute μ from the ground-state wavefunction if not externally fixed.
        # μ is the reservoir chemical potential held constant throughout the run.
        if self.mu is None:
            energy = gpe.calculate_energy_allocation(
                self.psi, self.uext,
                (self.p1, self.p2, self.p3), self.d_x, u=self.u
            )
            self.mu = float((energy['e_kin'] + energy['e_pot'] + 2.0 * energy['e_int']).real)
            self.logger.info(f"ZNG: computed chemical potential μ = {self.mu:.6f} [ħω_ho]")

        # Sample initial thermal cloud from the harmonic Boltzmann distribution.
        # For an anharmonic trap, the cloud thermalises to the correct shape
        # after a few trap periods of free evolution.
        w = torch.tensor(
            self.system.simulation_parameters["w"],
            dtype=torch.float64, device=self.device
        )
        self.particle_positions, self.particle_momenta = sample_initial_thermal_cloud(
            self.n_test, w, self.temperature, self.device
        )
        self.logger.info(
            f"ZNG: initialised {self.n_test} thermal test particles."
        )

        # Compute initial ñ from the particle positions
        self._update_thermal_density()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_thermal_density(self) -> None:
        r"""
        Deposit the test-particle positions onto the grid to get
        :math:`\tilde{n}(\mathbf{r})`.

        This is called at the start of each time step, so that the condensate
        GPE and the particle equations of motion see a consistent
        :math:`\tilde{n}`.
        """
        params = self.system.simulation_parameters
        n1, n2, n3 = params["Grid_resolution"]
        x_min = torch.tensor(params["x_min"], dtype=torch.float64, device=self.device)
        dx = torch.tensor(params["dx"], dtype=torch.float64, device=self.device)

        self.n_tilde = thermal_density_from_particles(
            self.particle_positions, n1, n2, n3,
            x_min, dx, self.d_x, self.device,
            particle_weight=self.particle_weight,
        )

    #: Accepted values for ``zng_thermal_fraction_mode``.
    THERMAL_FRACTION_MODES = ("temperature", "explicit")

    def _resolve_thermal_fraction(self) -> float:
        r"""
        Work out what fraction of the atoms starts in the thermal cloud.

        The condensate is normalised so that
        :math:`\int \lvert \psi \rvert^{2}\, \mathrm{d}V = 1` stands for the
        whole sample, so the thermal cloud has to be measured on the same
        scale: the fraction returned here is what
        :math:`\int \tilde{n}\, \mathrm{d}V` integrates to, and the condensate
        is scaled to hold the remaining :math:`1 - f`. Without this the two
        components are in different units, and every mean-field term that mixes
        them — and so the :math:`C_{12}` exchange rate itself — scales with the
        test-particle count, which is a convergence knob rather than physics.

        There are two conventions, selected by
        ``zng_thermal_fraction_mode``:

        ``"temperature"`` (default)
            Derive :math:`f` from the ideal-Bose result for a 3-D harmonic
            trap. With
            :math:`k_B T_c = \hbar \omega_\mathrm{ho} (N / \zeta(3))^{1/3}`,
            the condensed fraction below :math:`T_c` is
            :math:`1 - (T / T_c)^{3}`, so

            .. math::

                f = \Bigl(\frac{T}{T_c}\Bigr)^{3}
                  = \frac{T^{3} \zeta(3)}{N},

            capped at 1, using the dimensionless
            :math:`T = k_B T / (\hbar \omega_\mathrm{ho})` already configured
            as ``temperature`` and the atom number :math:`N` from
            ``CONSTANTS``. Above :math:`T_c` the whole sample is thermal and
            :math:`f` saturates at 1.
        ``"explicit"``
            Take :math:`f` straight from ``zng_thermal_fraction``. Use this
            when the trap is not harmonic, when you are matching a measured
            condensate fraction, or when you want to sweep :math:`f`
            independently of :math:`T`.

        Returns:
            float: The thermal fraction :math:`f \in [0, 1]`.

        Raises:
            ValueError: If the mode is unknown, if the explicit fraction is
                missing or out of range, or if ``CONSTANTS.nat`` is not
                positive.
        """
        mode = self.thermal_fraction_mode
        if mode not in self.THERMAL_FRACTION_MODES:
            raise ValueError(
                f"zng_thermal_fraction_mode must be one of "
                f"{self.THERMAL_FRACTION_MODES}; got {mode!r}"
            )

        if mode == "explicit":
            raw = self.parameters.get("zng_thermal_fraction", None)
            if raw is None:
                raise ValueError(
                    "zng_thermal_fraction_mode='explicit' requires "
                    "zng_thermal_fraction to be set"
                )
            fraction = float(raw)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"zng_thermal_fraction must lie in [0, 1]; got {fraction}"
                )
            return fraction

        # mode == "temperature"
        zeta_3 = 1.2020569031595943
        n_atoms = float(CONSTANTS.nat)
        if n_atoms <= 0:
            raise ValueError("CONSTANTS.nat must be positive to derive a thermal fraction")
        return min(1.0, float(self.temperature) ** 3 * zeta_3 / n_atoms)

    def _reconcile_condensate(self, target_number: float) -> None:
        r"""
        Rescale :math:`\psi` so the condensate holds exactly ``target_number``
        atoms.

        The factor is uniform, so whatever shape the source term :math:`R`
        imposed during the step is untouched and only the total is set. This is
        called with the number :math:`C_{12}` actually transferred, which is
        what closes the particle budget between the two components.

        Args:
            target_number (float): Condensate atom number to rescale to, in the
                condensate's own normalisation. Non-positive values are
                ignored.
        """
        if self.psi is None or target_number <= 0.0:
            return
        current = self.condensate_number()
        if current > 0.0:
            self.psi = self.psi * ((target_number / current) ** 0.5)

    def condensate_number(self) -> float:
        r"""
        Return the condensate atom number as a fraction of its initial value,
        :math:`\int \lvert \psi \rvert^{2}\, \mathrm{d}V`.

        This is exactly 1 while ``zng_condensate_exchange`` is off, since the
        state is renormalised every step. With exchange on it moves, and
        watching it against the test-particle count is how you check that what
        the condensate loses the cloud gains.

        Returns:
            float: The condensate norm, or ``0.0`` if :math:`\psi` has not been
            initialised.
        """
        if self.psi is None:
            return 0.0
        return float(self.d_x * torch.sum(torch.abs(self.psi) ** 2))

    def _get_grid_arrays(self):
        r"""
        Return ``x_min`` and ``dx`` as device tensors.

        This avoids repeating the conversion at every use site.

        Returns:
            tuple: The pair ``(x_min, dx)``, each a ``torch.Tensor`` of shape
            ``(3,)`` on the simulation device.
        """
        params = self.system.simulation_parameters
        x_min = torch.tensor(params["x_min"], dtype=torch.float64, device=self.device)
        dx = torch.tensor(params["dx"], dtype=torch.float64, device=self.device)
        return x_min, dx

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def _main_simulation_loop(self) -> None:
        r"""
        Run the ZNG coupled time-evolution loop.

        Each iteration couples the condensate and the thermal cloud through
        their shared densities :math:`(n_c, \tilde{n})`. The sequence is:

        1. Compute the condensate density
           :math:`n_c = \lvert \psi \rvert^{2}`.
        2. Deposit the thermal cloud onto the grid,
           :math:`\tilde{n} = \mathrm{CIC}(\text{particle positions})`.
        3. Build the condensate GPE potential
           :math:`V_\mathrm{GP} = V_\mathrm{ext} + u (n_c + 2 \tilde{n})`.
        4. Build the condensate source term (the :math:`C_{12}` mean field)
           :math:`R = 2 \gamma_{12} [\mu - V_\mathrm{eff}]`, with
           :math:`V_\mathrm{eff} = V_\mathrm{ext} + 2u (n_c + \tilde{n})`.
        5. Evolve the condensate one step with the modified GPE
           :math:`i\, \partial_t \psi = [H_\mathrm{GP} + iR/2]\, \psi`, using
           the split-step method with the complex potential
           :math:`V_\mathrm{GP} + iR/2`.
        6. Compute the thermal particle potential
           :math:`U = V_\mathrm{ext} + 2u (n_c^\mathrm{new} + \tilde{n})`.
        7. Advance the test particles by leapfrog under :math:`U`.
        8. Apply the :math:`C_{12}` stochastic collisions (absorption and
           emission), then reconcile the condensate against the transfer that
           actually happened.
        9. Apply :math:`C_{22}` (a no-op stub unless ``enable_c22`` is set).
        10. Update :math:`\tilde{n}` from the new particle positions.

        Note:
            **Why the source term enters as** :math:`+iR/2`. The modified GPE
            is written as
            :math:`i\, \partial_t \psi = H_\mathrm{GP} \psi + i (R/2) \psi`;
            multiplying through by :math:`-i` gives

            .. math::

                \frac{\partial \psi}{\partial t}
                    = -i H_\mathrm{GP} \psi + \frac{R}{2} \psi .

            The :math:`+R/2` (rather than :math:`-R/2`) gives
            :math:`\partial_t \lvert \psi \rvert^{2} = R \lvert \psi \rvert^{2}`,
            so the condensate density grows where :math:`R > 0`. In the
            split-step this is implemented by passing the complex potential
            :math:`V_\mathrm{GP} + iR/2` to the real-space evolution operator,

            .. math::

                e^{-i \Delta t (V_\mathrm{GP} + iR/2)}
                    = e^{\Delta t R / 2}\, e^{-i \Delta t V_\mathrm{GP}},

            where the real exponential :math:`e^{\Delta t R / 2}` provides the
            growth or decay.

        Raises:
            RuntimeError: If the condensate wavefunction has not been
                initialised.
        """
        if self.psi is None:
            raise RuntimeError("BEC wavefunction (psi) is not initialized.")

        count = 0
        snapshot_interval = self._get_snapshot_interval()
        n1, n2, n3 = self.system.simulation_parameters["Grid_resolution"]
        x_min, dx = self._get_grid_arrays()

        self.logger.info("Starting ZNG simulation loop...")

        try:
            for iteration in range(self.kmax):
                t = self.dt * iteration * self.omega_ho

                # --- 1. Condensate density ---
                n_c = torch.abs(self.psi) ** 2

                # --- 2. Thermal density (already updated at end of previous step) ---
                # n_tilde is self.n_tilde

                # --- 3. Condensate GPE potential: V_GP = V_ext + u(n_c + 2ñ) ---
                V_gpe = condensate_gpe_potential(n_c, self.n_tilde, self.uext, self.u)

                # --- 4. Source term R(r) = 2 γ_12 [μ − V_eff] ---
                R = condensate_source_term(
                    n_c, self.n_tilde, self.uext, self.u, self.mu, self.gamma_12
                )

                # Snapshot before evolving (first frame is t=0 state)
                if (
                    snapshot_interval is not None
                    and iteration % snapshot_interval == 0
                    and count < self.shots
                ):
                    self._write_iteration_data(count, t)
                    count += 1

                # Imprint dark solitons at the configured snapshot (no-op if disabled)
                self._maybe_imprint_solitons(iteration)

                # --- 5. Evolve condensate: modified GPE split-step ---
                # Complex potential = V_GP + i·R/2
                # The +i·R/2 source term drives growth (R>0) / decay (R<0).
                utot_complex = V_gpe.to(torch.cdouble) + 1j * R.to(torch.cdouble) * 0.5
                # Renormalising pins the condensate atom number. Unlike the
                # SGPE — where mu is a constant shift that renormalisation
                # cancels outright — R varies in space, so rescaling still
                # leaves the shape R imposes; it only removes the net transfer.
                # zng_condensate_exchange=True lets that transfer happen.
                # Measure the condensate number either side of the step: the
                # difference is exactly the number of atoms C_12 has to move
                # the other way, which is what keeps the two components' books
                # balanced instead of letting independently-modelled rates
                # drift apart.
                n_condensate_before = self.condensate_number()
                self.psi = gpe.split_step_step(
                    self.psi, utot_complex, self.dtau, self.p_sq, self.d_x,
                    renormalise=False,
                )
                # How many atoms R is asking the cloud for. With exchange off
                # the answer is none: the condensate is pinned, and step 8b
                # below scales it straight back to where it started.
                atoms_gained = (
                    self.condensate_number() - n_condensate_before
                    if self.condensate_exchange else 0.0
                )

                # --- 6. Thermal particle potential after condensate step ---
                n_c_new = torch.abs(self.psi) ** 2
                U_thermal = mean_field_potential_for_thermal(
                    n_c_new, self.n_tilde, self.uext, self.u
                )
                # U_next approximation: use U_thermal for the end-of-step force
                # (a predictor; a corrector pass would use the updated ñ)
                U_next = U_thermal

                # --- 7. Advance test particles (leapfrog) ---
                U_current = mean_field_potential_for_thermal(
                    n_c, self.n_tilde, self.uext, self.u
                )
                self.particle_positions, self.particle_momenta = (
                    advance_particles_leapfrog(
                        self.particle_positions, self.particle_momenta,
                        U_current, U_next,
                        n1, n2, n3, x_min, dx,
                        self.system.p_grid, self.dtau, self.device
                    )
                )

                # --- 8. C_12: stochastic condensate ↔ thermal exchange ---
                (self.particle_positions, self.particle_momenta,
                 atoms_transferred) = apply_c12_collisions(
                    self.particle_positions, self.particle_momenta,
                    U_thermal, n_c_new,
                    self.mu, self.gamma_12, self.dtau,
                    n1, n2, n3, x_min, dx,
                    self.temperature, self.device,
                    particle_weight=self.particle_weight,
                    atoms_gained_by_condensate=atoms_gained,
                )

                # --- 8b. Reconcile the condensate against what actually moved ---
                # C_12 can fall short of what R asked for — the cloud may simply
                # not hold that many atoms — so the condensate is scaled to the
                # transfer that really happened. The factor is uniform, so the
                # shape R imposed survives; only the total is corrected. This is
                # what makes the two components conserve atoms exactly, and it
                # doubles as the pin when exchange is off (transfer = 0, so the
                # condensate returns to the number it started the step with).
                self._reconcile_condensate(n_condensate_before + atoms_transferred)

                # --- 9. C_22: thermal–thermal scattering (stub) ---
                if self.enable_c22:
                    self.particle_positions, self.particle_momenta = apply_c22_collisions(
                        self.particle_positions, self.particle_momenta,
                        self.temperature, self.dtau, self.device
                    )

                # --- 10. Update ñ from new particle positions ---
                self._update_thermal_density()

                # Update time-dependent external potential if supported
                if hasattr(self.system.uext, 'evol'):
                    self.uext = self.system.uext.evol(t)

            self.logger.info(
                f"ZNG loop completed after {self.kmax} iterations. "
                f"Final N_test = {self.particle_positions.shape[0]} "
                f"(started with {self.n_test}). "
                f"Condensate number {self.condensate_number():.6f} of its initial value "
                f"({'free to exchange' if self.condensate_exchange else 'pinned by renormalisation'})."
            )
        except Exception as e:
            self.logger.exception(
                f"Error in ZNG loop at iteration "
                f"{iteration if 'iteration' in locals() else 'unknown'}."
            )
            raise
