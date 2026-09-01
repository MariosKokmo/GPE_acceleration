r"""
Finite-temperature BEC model using the Stochastic Projected Gross-Pitaevskii
Equation (SGPE).

At zero temperature the condensate evolves under the Gross-Pitaevskii equation
(GPE). At finite temperature it is coupled to a thermal reservoir of
non-condensed atoms, and the SGPE captures that coupling with two extra terms,
a damping term and a noise term:

.. math::

    \frac{\partial \psi}{\partial t}
        = \underbrace{-(i + \gamma)\bigl(H_\mathrm{mf} - \mu\bigr) \psi}
                     _{\text{damped GPE}}
        + \underbrace{\eta(\mathbf{r}, t)}_{\text{thermal noise}},

where the mean-field Hamiltonian, the reservoir chemical potential :math:`\mu`
and the dimensionless damping rate :math:`\gamma` are

.. math::

    H_\mathrm{mf} = -\tfrac{1}{2} \nabla^{2} + V_\mathrm{ext}
        + u \lvert \psi \rvert^{2},

and the noise obeys the fluctuation-dissipation relation

.. math::

    \bigl\langle \eta^{*}(\mathbf{r}, t)\, \eta(\mathbf{r}', t') \bigr\rangle
        = 2 \gamma k_B T\,
          \delta(\mathbf{r} - \mathbf{r}')\, \delta(t - t').

All quantities are in dimensionless units
(:math:`\hbar = m = \omega_\mathrm{ho} = 1`).

References:
    C. W. Gardiner, J. R. Anglin and T. I. A. Fudge, *J. Phys. B* **35**, 1555
    (2002).

    S. J. Rooney, P. B. Blakie and A. S. Bradley, *Phys. Rev. A* **86**, 053634
    (2012).
"""
import torch
from src.models.base_BEC import BaseBEC


class FiniteTempBEC(BaseBEC):
    r"""
    Finite-temperature BEC simulation using the SGPE.

    All grid setup, ground-state initialisation, I/O and split-step machinery
    is inherited from :class:`~src.models.base_BEC.BaseBEC`; only the simulation
    loop and the parameter initialisation are overridden.

    The following keys are read in addition to the standard
    :class:`~src.models.base_BEC.BaseBEC` ones:

    ``temperature`` (float)
        Dimensionless temperature :math:`k_B T / (\hbar \omega_\mathrm{ho})`.
        Set it to 0 to recover the damped GPE (no noise, pure dissipation).
        Typical values for a Rb BEC near :math:`T_c` are 0.1-5.
    ``damping_coefficient`` (float, default 0.03)
        Dimensionless damping rate :math:`\gamma`, which controls the strength
        of the energy exchange with the thermal reservoir. :math:`\gamma = 0`
        gives the standard GPE with no thermal effects at all, while
        :math:`\gamma \approx 0.01` to :math:`0.1` is the physically motivated
        range for cold-atom BECs.
    ``chemical_potential`` (float or None, default None)
        Reservoir chemical potential :math:`\mu`, in units of
        :math:`\hbar \omega_\mathrm{ho}`. If ``None``, it is computed from the
        initial ground-state wavefunction as
        :math:`\mu = E_\mathrm{kin} + E_\mathrm{pot} + 2 E_\mathrm{int}` and
        kept fixed for the entire run. The factor of 2 on
        :math:`E_\mathrm{int}` arises because :math:`\mu = \partial E /
        \partial N` while the interaction energy scales as :math:`N^{2}`.

    Example:
        Set ``"model_type": "FiniteTempBEC"`` and add these keys to the
        simulation parameters dictionary or configuration JSON::

            "temperature": 1.5,
            "damping_coefficient": 0.03
    """

    def _initialize_custom_parameters(self) -> None:
        r"""
        Read the SGPE-specific parameters from the per-simulation parameters
        dict.

        ``temperature`` (float)
            :math:`k_B T / (\hbar \omega_\mathrm{ho})`, the dimensionless
            thermal energy of the reservoir. It feeds directly into the noise
            amplitude

            .. math::

                \sigma = \sqrt{\frac{\gamma\, kT\, \Delta\tau}{\delta V}},

            so a larger :math:`T` gives stronger fluctuations and a more
            strongly depleted condensate.
        ``damping_coefficient`` (float)
            :math:`\gamma`, the dimensionless coupling to the reservoir. The
            :math:`(1 - i\gamma)` prefactor in the SGPE damps modes above
            :math:`\mu` and amplifies modes below it, thermalising the system
            to temperature :math:`T`.
        ``chemical_potential`` (float or None)
            :math:`\mu`. If it is not provided in the parameters, it is
            computed once from the ground-state wavefunction at the start of
            :meth:`_main_simulation_loop`.
        """
        self.temperature: float = self.parameters.get("temperature", 0.0)
        self.gamma: float = self.parameters.get("damping_coefficient", 0.03)
        self.mu: float | None = self.parameters.get("chemical_potential", None)

        self.logger.info(
            f"FiniteTempBEC: T = {self.temperature} [ħω_ho/k_B], "
            f"γ = {self.gamma}, "
            f"μ = {'(computed from ground state)' if self.mu is None else self.mu}"
        )

    def _main_simulation_loop(self) -> None:
        r"""
        Run the SGPE time-evolution loop.

        Each iteration applies the following operations in sequence:

        1. **SGPE deterministic step** (see
           :meth:`GPELibrary.sgpe_step <src.library.gpe_library.GPELibrary.sgpe_step>`),
           the Strang splitting of the damped Hamiltonian

           .. math::

               \exp\bigl[-(i + \gamma) \Delta\tau
                         (H_\mathrm{mf} - \mu)\bigr].

           Modes with :math:`H_\mathrm{mf} > \mu` lose energy to the reservoir
           and modes with :math:`H_\mathrm{mf} < \mu` gain energy from it,
           which is the mechanism that drives condensate growth at finite
           :math:`T`.
        2. **Three-body loss** (see
           :meth:`~src.models.base_BEC.BaseBEC._apply_three_body_loss`),
           Strang-split as its own operator, half a step either side of the
           propagator, when :math:`K_3` is non-zero. It is a *separate* channel
           from the reservoir coupling above, not a duplicate of it:
           :math:`\gamma` exchanges atoms between the condensate and the
           thermal cloud, while three-body recombination ejects them from the
           trap entirely, so a finite-temperature run generally wants both.
        3. **Stochastic noise injection** (see
           :meth:`GPELibrary.generate_thermal_noise <src.library.gpe_library.GPELibrary.generate_thermal_noise>`),
           which adds a complex Gaussian noise field :math:`\eta` satisfying

           .. math::

               \bigl\langle \eta^{*}(\mathbf{r}, t)\,
                            \eta(\mathbf{r}', t') \bigr\rangle
                   = 2 \gamma k_B T\,
                     \delta(\mathbf{r} - \mathbf{r}')\, \delta(t - t').

           This is applied only when :math:`T > 0`; setting :math:`T = 0` with
           :math:`\gamma > 0` gives a purely dissipative, imaginary-time-like
           damped GPE.

        Note:
            **The norm is deliberately not reset.**
            :math:`\int \lvert \psi \rvert^{2}\, \mathrm{d}V` is left free to
            evolve, which is what makes the ensemble grand-canonical:
            :math:`\mu` enters the propagator only as the constant shift
            :math:`(H_\mathrm{mf} - \mu)`, so forcing the norm back to 1 after
            every step divides that factor straight back out and reduces the
            run to a number-conserving damped GPE in which :math:`\mu` has no
            effect at all. With the norm free,
            :math:`N(t) = N_0 \lVert \psi \rVert^{2}` and the reservoir sets
            the atom number through :math:`\mu`. The initial ground state is
            very nearly a fixed point of the damped propagator, so
            :math:`\lVert \psi \rVert` stays close to 1 unless the state is
            genuinely out of equilibrium with the reservoir.

        Note:
            **Chemical potential.** :math:`\mu` is computed once from the
            ground-state wavefunction at the first iteration if it was not
            given in the parameters, and is then held fixed throughout the run,
            representing the static thermal reservoir.

        Raises:
            RuntimeError: If the condensate wavefunction has not been
                initialised.
        """
        if self.psi is None:
            raise RuntimeError("BEC wavefunction (psi) is not initialized.")

        count = 0
        snapshot_interval = self._get_snapshot_interval()

        # Compute μ from the initial ground state when not externally specified.
        # μ fixes the reservoir chemical potential for the entire simulation.
        if self.mu is None:
            if self._coord == "cylindrical":
                self.mu = self._lib.calculate_chemical_potential(
                    self.psi, self.uext, self.u,
                    self.r, self.dr, self.dphi, self.dz,
                    self.kz, self.m_modes,
                )
            else:
                self.mu = self._lib.calculate_chemical_potential(
                    self.psi, self.uext, self.u,
                    (self.p1, self.p2, self.p3), self.d_x,
                )
            self.logger.info(f"Computed chemical potential μ = {self.mu:.6f} [ħω_ho]")

        self.logger.info("Starting SGPE simulation loop...")

        try:
            for iteration in range(self.kmax):
                t = self.dt * iteration * self.omega_ho

                # Total mean-field potential at this step. It stays real: the
                # three-body loss is applied as a separate operator below
                # rather than as an imaginary part of utot, because everything
                # placed in utot is multiplied by the (i + γ) prefactor.
                utot = self.u * torch.abs(self.psi) ** 2 + self.uext

                # Write snapshot before evolving so the first frame is t=0.
                if (
                    snapshot_interval is not None
                    and iteration % snapshot_interval == 0
                    and count < self.shots
                ):
                    self._write_iteration_data(count, t)
                    count += 1

                # Imprint dark solitons at the configured snapshot (no-op if disabled)
                self._maybe_imprint_solitons(iteration)

                # --- Step 1: three-body loss, first half-step ---
                # Strang-split around the propagator and applied as its own
                # operator, never folded into utot: see _apply_three_body_loss.
                self._apply_three_body_loss(0.5 * self.dtau)

                # --- Step 2: SGPE deterministic split-step with (1−iγ) damping ---
                # exp(−(i+γ)·Δτ·(H_mf − μ)) applied via Strang splitting.
                # This is the finite-temperature generalisation of split_step_step.
                if self._coord == "cylindrical":
                    self.psi = self._lib.sgpe_step(
                        self.psi, utot, self.mu, self.gamma, self.dtau,
                        self.kz, self.m_modes, self.r,
                        self.eigvecs_dict, self.eigvals_dict,
                        self.dr, self.dphi, self.dz,
                    )
                else:
                    self.psi = self._lib.sgpe_step(
                        self.psi, utot, self.mu, self.gamma,
                        self.dtau, self.p_sq, self.d_x,
                    )

                # --- Step 3: three-body loss, second half-step ---
                self._apply_three_body_loss(0.5 * self.dtau)

                # --- Step 4: Stochastic noise (fluctuation-dissipation theorem) ---
                # Only active at T > 0.  At T = 0 the loop reduces to a
                # damped (dissipative) GPE, useful for ground-state search.
                if self.temperature > 0.0:
                    if self._coord == "cylindrical":
                        noise = self._lib.generate_thermal_noise(
                            self.psi.shape, self.gamma, self.temperature,
                            self.dtau, self.r, self.dr, self.dphi, self.dz, self.device,
                        )
                    else:
                        noise = self._lib.generate_thermal_noise(
                            self.psi.shape, self.gamma, self.temperature,
                            self.dtau, self.d_x, self.device,
                        )
                    # The norm is *not* reset here: see the note in the method
                    # docstring — renormalising cancels the reservoir coupling.
                    self.psi = self.psi + noise

                # Update time-dependent external potential if the potential
                # object supports it (e.g. a rotating trap or a stirring beam).
                if hasattr(self.system.uext, 'evol'):
                    self.uext = self.system.uext.evol(t)

            self.logger.info(
                f"SGPE loop completed after {self.kmax} iterations "
                f"(T = {self.temperature:.4f} [ħω_ho/k_B], γ = {self.gamma})."
            )
        except Exception:
            self.logger.exception(
                f"Error in SGPE loop at iteration "
                f"{iteration if 'iteration' in locals() else 'unknown'}."
            )
            raise
