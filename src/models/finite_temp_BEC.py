"""
Finite-temperature BEC model using the Stochastic Projected
Gross-Pitaevskii Equation (SGPE).

Physics background
------------------
At zero temperature the condensate evolves under the Gross-Pitaevskii equation
(GPE).  At finite temperature the condensate is coupled to a thermal reservoir
of non-condensed atoms.  The SGPE captures this coupling through two extra terms:

    ∂ψ/∂t = −(i + γ)(H_mf − μ)ψ  +  η(r, t)
              ↑ damped GPE            ↑ thermal noise

    H_mf = −∇²/2 + V_ext + u|ψ|²       (mean-field Hamiltonian, dimensionless)
    μ                                     (reservoir chemical potential)
    γ                                     (dimensionless damping rate)
    ⟨η*(r,t) η(r',t')⟩ = 2γ k_BT δ(r−r') δ(t−t')   (fluctuation-dissipation)

All quantities are in dimensionless units:  ħ = m = ω_ho = 1.

Reference:  C. W. Gardiner, J. R. Anglin, T. I. A. Fudge,
            J. Phys. B 35, 1555 (2002).
            S. J. Rooney, P. B. Blakie, A. S. Bradley,
            Phys. Rev. A 86, 053634 (2012).
"""
import torch
from src.models.base_BEC import BaseBEC


class FiniteTempBEC(BaseBEC):
    """
    Finite-temperature BEC simulation using the SGPE.

    Inherits all grid setup, ground-state initialisation, I/O, and split-step
    machinery from BaseBEC.  Only the simulation loop and the parameter
    initialisation are overridden.

    Configuration parameters (in addition to the standard BaseBEC ones)
    -------------------------------------------------------------------
    temperature : float
        Dimensionless temperature k_B T / (ħ ω_ho).
        Set to 0 to recover the damped GPE (no noise, pure dissipation).
        Typical values for a Rb BEC near T_c: 0.1–5.

    damping_coefficient : float  (default 0.03)
        Dimensionless damping rate γ.  Controls the strength of energy exchange
        with the thermal reservoir.

        - γ = 0   → standard GPE (no thermal effects at all)
        - γ ≈ 0.01–0.1 is the physically motivated range for cold-atom BECs.

    chemical_potential : float or None  (default None)
        Reservoir chemical potential μ in units of ħ ω_ho.
        If None, μ is computed from the initial ground-state wavefunction
        as μ = e_kin + e_pot + 2·e_int, and kept fixed for the entire run.  The factor of 2 on e_int arises
        because μ = ∂E/∂N and the interaction energy scales as N².

    Usage example
    -------------
    Replace BEC with FiniteTempBEC in simulation.py, and add these keys to the
    simulation parameters dictionary or configuration JSON:

        "temperature": 1.5,
        "damping_coefficient": 0.03
    """

    def _initialize_custom_parameters(self) -> None:
        """
        Read SGPE-specific parameters from the per-simulation parameters dict.

        temperature : float
            k_B T / (ħ ω_ho) — dimensionless thermal energy of the reservoir.
            Feeds directly into the noise amplitude:
                σ = √(γ · kT · Δτ / δV)
            Larger T → stronger fluctuations → higher condensate fraction depleted.

        damping_coefficient : float
            γ — dimensionless coupling to the reservoir.
            The (1−iγ) prefactor in the SGPE damps modes above μ and amplifies
            modes below μ, thermalising the system to temperature T.

        chemical_potential : float or None
            μ — if not provided in parameters, it is computed once from the
            ground-state wavefunction at the start of _main_simulation_loop.
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
        """
        SGPE time-evolution loop.

        Each iteration applies three operations in sequence:

        1.  SGPE deterministic step  (see GPELibrary.sgpe_step)
            Implements Strang splitting of the damped Hamiltonian:
                exp(−(i+γ)·Δτ·(H_mf − μ))
            Modes with H_mf > μ lose energy to the reservoir (damped).
            Modes with H_mf < μ gain energy from the reservoir (amplified).
            This is the mechanism that drives condensate growth at finite T.

        1b. Three-body loss  (see BaseBEC._apply_three_body_loss)
            Strang-split as its own operator, half a step either side of the
            propagator, when k3 is non-zero. It is a *separate* channel from
            the reservoir coupling above, not a duplicate of it: γ exchanges
            atoms between the condensate and the thermal cloud, while
            three-body recombination ejects them from the trap entirely, so a
            finite-temperature run generally wants both.

        2.  Stochastic noise injection  (see GPELibrary.generate_thermal_noise)
            Adds a complex Gaussian noise field η satisfying:
                ⟨η*(r,t) η(r',t')⟩ = 2γ k_BT · δ(r−r') · δ(t−t')
            Only applied when T > 0.  Setting T = 0 with γ > 0 gives a
            purely dissipative (imaginary-time-like) damped GPE.

        The norm is deliberately *not* reset
            ∫|ψ|² dV is left free to evolve.  This is what makes the ensemble
            grand-canonical: μ enters the propagator only as the constant shift
            (H_mf − μ), so forcing the norm back to 1 after every step divides
            that factor straight back out and reduces the run to a
            number-conserving damped GPE in which μ has no effect at all.
            With the norm free, N(t) = N₀·‖ψ‖² and the reservoir sets the atom
            number through μ.  The initial ground state is very nearly a fixed
            point of the damped propagator, so ‖ψ‖ stays close to 1 unless the
            state is genuinely out of equilibrium with the reservoir.

        Chemical potential μ
            Computed once from the ground-state wavefunction at the first
            iteration if not provided in the parameters.  μ is then held
            fixed throughout the run, representing the static thermal reservoir.
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
