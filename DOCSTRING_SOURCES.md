# GPE Acceleration — Docstring Sources for Documentation Generation

> **How to use this file**
> Give this file to Claude along with `DOCS_PROMPT.md` and the instruction:
> _"Using the docstrings and API descriptions in DOCSTRING_SOURCES.md, populate the
> Astro/Starlight MDX files described in DOCS_PROMPT.md. For each new or changed symbol,
> replace the placeholder description with text derived from the docstring."_

All docstrings below are copied verbatim from the Python source so that Claude can
produce accurate, complete documentation without needing to read the repository files.

---

## `src/library/gpe_library.py` — GPELibrary (new methods)

### `GPELibrary.calculate_chemical_potential`

```python
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
        p_grid (tuple): (p1, p2, p3) 3-D momentum meshgrids.
        d_x (float): Grid cell volume (product of dx in each dimension).

    Returns:
        float: Chemical potential μ in units of ħ·ω_ho.
    """
```

### `GPELibrary.generate_thermal_noise`

```python
@staticmethod
def generate_thermal_noise(
    shape: tuple,
    gamma: float,
    kT: float,
    dtau: float,
    d_x: float,
    device: torch.device
) -> torch.Tensor:
    """
    Generate a complex Gaussian noise field for one SGPE time step.

    The SGPE noise must satisfy the fluctuation-dissipation theorem:

        ⟨η*(r,t) η(r',t')⟩ = 2γ·k_BT · δ(r−r') · δ(t−t')

    Discretising on a grid with cell volume δV = d_x and time step δt = dtau:

        noise amplitude = √(γ · kT · dtau / d_x)

    so that ⟨|Δψ_noise|²⟩ = 2·γ·kT·dtau/d_x per grid point, matching the
    continuous fluctuation-dissipation relation.

    Args:
        shape (tuple): Grid shape (n1, n2, n3).
        gamma (float): Dimensionless damping coefficient γ.
        kT (float): Dimensionless temperature k_B·T / (ħ·ω_ho).
        dtau (float): Dimensionless time step ω_ho·dt.
        d_x (float): Grid cell volume (product of dx in each dimension).
        device (torch.device): Computation device.

    Returns:
        torch.Tensor: Complex noise tensor of shape (n1, n2, n3).
    """
```

### `GPELibrary.sgpe_step`

```python
@staticmethod
def sgpe_step(
    psi: torch.Tensor,
    utot: torch.Tensor,
    mu: float,
    gamma: float,
    dtau: float,
    p_sq: torch.Tensor,
    d_x: float
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
        4. Normalise ψ

    where V_eff = u|ψ|² + V_ext is frozen at the start of the step.

    Args:
        psi (torch.Tensor): Wavefunction (n1, n2, n3), complex double.
        utot (torch.Tensor): Total mean-field potential V_ext + u|ψ|².
        mu (float): Reservoir chemical potential μ in units of ħ·ω_ho.
        gamma (float): Dimensionless damping coefficient γ.
        dtau (float): Dimensionless time step ω_ho·dt.
        p_sq (torch.Tensor): Squared momentum grid |p|².
        d_x (float): Grid cell volume used for normalisation.

    Returns:
        torch.Tensor: Updated, normalised wavefunction.
    """
```

---

## `src/models/base_BEC.py` — BaseBEC

### Module docstring

```
Base BEC class with common functionality.
This class provides the core functionality for BEC simulations.
Extend this class and override methods as needed for custom simulations.
```

### Class docstring

```python
class BaseBEC:
    """
    Base class for Bose-Einstein Condensate simulations.

    This class provides common functionality including:
    - Ground state initialization
    - Time evolution using split-step Fourier method
    - Parameter management
    - Data output and logging

    To create a custom simulation:
    1. Inherit from this class
    2. Override _main_simulation_loop() to implement custom physics
    3. Override _initialize_custom_parameters() for simulation-specific parameters
    4. Override _write_custom_outputs() for additional output files
    """
```

### `__init__`

```python
def __init__(self, parameters: Dict[str, Any], system: Any, app: Any, simulation_name: str) -> None:
    """
    Initialize the BEC simulation.

    Args:
        parameters: dict, simulation-specific parameters
        system: System object containing grid, potential, etc.
        app: Application object with device, logger, etc.
        simulation_name: str, name of this simulation
    """
```

### `_find_ground_state`

```python
def _find_ground_state(self) -> None:
    """
    Finds the ground state for the BEC in the system.
    If it exists, it just reads the file.
    The required format is `{n1}x{n2}x{n3}_{fx}_{fy}_{fz}Hz_ground_state.dat`
    If a ground state file does not exist, it is computed.
    """
```

### `_initialise`

```python
def _initialise(self) -> None:
    """
    Reads the ground state file and initialises the wavefunction to ground state.

    Override this method if you need custom initialization logic.
    """
```

### `_step`

```python
def _step(self, utot: torch.Tensor, dtau: float, p_sq: torch.Tensor, d_x: float) -> None:
    """
    Performs a single time step evolution of the BEC
    following the split-step Fourier method.

    Args:
        utot: torch.Tensor, total potential (interaction + external)
        dtau: float, time step size
        p_sq: torch.Tensor, squared momentum grid
        d_x: float, spatial grid spacing
    """
```

### `_get_snapshot_interval`

```python
def _get_snapshot_interval(self) -> Optional[int]:
    """
    Returns the iteration interval used for snapshot writes.

    Returns None when snapshots are disabled.
    """
```

### `_initialize_simulation_parameters`

```python
def _initialize_simulation_parameters(self) -> None:
    """
    Initializes simulation parameters for easy access.

    This extracts commonly used parameters from the system and stores them
    as instance variables for faster access during the simulation loop.

    Override or extend this method to add custom parameters.
    """
```

### `_initialize_custom_parameters`

```python
def _initialize_custom_parameters(self) -> None:
    """
    Initialize custom simulation-specific parameters.

    Override this method in your derived class to set up additional parameters.

    Example:
        def _initialize_custom_parameters(self):
            self.my_custom_param = self.parameters.get("my_param", default_value)
            self.special_measurements = []
    """
```

### `_main_simulation_loop`

```python
def _main_simulation_loop(self) -> None:
    """
    Main loop for evolving the BEC system.

    This is a TEMPLATE method that should be overridden for custom simulations.
    The default implementation provides a basic time evolution with measurements.

    Common modifications:
    - Add custom physics (e.g., vortex imprinting, stirring, etc.)
    - Implement time-dependent potentials
    - Add special measurement/diagnostic routines
    - Implement adaptive time stepping
    """
```

### `_write_iteration_data`

```python
def _write_iteration_data(self, count: int, t: float) -> None:
    """
    Writes data for the current iteration.

    This is called at regular intervals during the simulation to save
    snapshots of the system state.

    Override or extend this to add custom measurements or outputs.

    Args:
        count: int, snapshot counter
        t: float, current time
    """
```

### `_write_simulation_outputs`

```python
def _write_simulation_outputs(self) -> None:
    """
    Writes various output files after the simulation.

    This is called once at the end of the simulation to generate
    final output files, plots, and videos.

    Override or extend this to add custom outputs.
    """
```

### `_write_custom_outputs`

```python
def _write_custom_outputs(self) -> None:
    """
    Write custom simulation-specific outputs.

    Override this method in your derived class to save additional data.

    Example:
        def _write_custom_outputs(self):
            np.save('my_data.npy', self.custom_data)
            with open('analysis.txt', 'w') as f:
                f.write(str(self.analysis_results))
    """
```

### `get_density`

```python
def get_density(self) -> torch.Tensor:
    """
    Returns the density |psi|^2 of the condensate.

    Returns:
        torch.Tensor, density of the wavefunction
    """
```

---

## `src/models/finite_temp_BEC.py` — FiniteTempBEC

### Module docstring

```
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
```

### Class docstring

```python
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
        If None, μ is computed from the initial ground-state wavefunction as:
            μ = e_kin + e_pot + 2·e_int
        and kept fixed for the entire run.  The factor of 2 on e_int arises
        because μ = ∂E/∂N and the interaction energy scales as N².

    Usage example
    -------------
    Replace BEC with FiniteTempBEC in simulation.py, and add these keys to the
    simulation parameters dictionary or configuration JSON:

        "temperature": 1.5,
        "damping_coefficient": 0.03
    """
```

### `_initialize_custom_parameters`

```python
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
```

### `_main_simulation_loop`

```python
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

    2.  Stochastic noise injection  (see GPELibrary.generate_thermal_noise)
        Adds a complex Gaussian noise field η satisfying:
            ⟨η*(r,t) η(r',t')⟩ = 2γ k_BT · δ(r−r') · δ(t−t')
        Only applied when T > 0.  Setting T = 0 with γ > 0 gives a
        purely dissipative (imaginary-time-like) damped GPE.

    3.  Renormalisation
        ψ is renormalised to enforce ∫|ψ|² dV = 1 after noise injection.
        This imposes the grand-canonical particle-number constraint.

    Chemical potential μ
        Computed once from the ground-state wavefunction at the first
        iteration if not provided in the parameters.  μ is then held
        fixed throughout the run, representing the static thermal reservoir.
    """
```

---

## `src/experimental/zng/zng_BEC.py` — ZNGBEC

### Module docstring

```
ZNGBEC: Zaremba-Nikuni-Griffin finite-temperature BEC model.

Inherits all ground-state finding, I/O, snapshotting, and split-step
machinery from BaseBEC.  Overrides only _initialize_custom_parameters()
and _main_simulation_loop() to implement the two-component ZNG dynamics.

Condensate + Thermal cloud equations (dimensionless, ħ = m = ω_ho = 1):
---------------------------------------------------------------------------

Condensate (modified GPE):
    i ∂ψ/∂t = [H_GP + i R(r,t)/2] ψ
    H_GP = −∇²/2 + V_ext + u(n_c + 2ñ)
    R(r) = 2 γ_12 [μ − V_eff(r)]          (condensate source/sink rate)
    V_eff = V_ext + 2u(n_c + ñ)            (full mean-field for thermal atoms)

Thermal cloud (classical test particles):
    dr_i/dt = p_i
    dp_i/dt = −∇U(r_i)
    U(r) = V_ext + 2u(n_c + ñ)            (= V_eff above)

The two equations are coupled via the densities n_c = |ψ|² and ñ(r)
(deposited from the test-particle positions onto the grid).

References:
    E. Zaremba, T. Nikuni, A. Griffin, J. Low Temp. Phys. 116, 277 (1999).
    T. Nikuni, E. Zaremba, A. Griffin, PRL 83, 10 (1999).
    S. Gardiner, D. Jaksch, R. Dum, J. Cirac, P. Zoller, PRA 62, 023612 (2000).
```

### Class docstring

```python
class ZNGBEC(BaseBEC):
    """
    Finite-temperature BEC simulation using the full Zaremba-Nikuni-Griffin
    (ZNG) two-component framework.

    State
    -----
    self.psi                  : condensate wavefunction ψ(r), (n1,n2,n3) cdouble
    self.particle_positions   : thermal test-particle positions,  (N_test, 3)
    self.particle_momenta     : thermal test-particle momenta,    (N_test, 3)
    self.n_tilde              : thermal density ñ(r) on the grid, (n1,n2,n3)

    Configuration parameters (add to simulation parameters dict or JSON)
    ----------------------------------------------------------------------
    temperature : float
        Dimensionless temperature k_B T / (ħ ω_ho).
        Must be > 0 (use FiniteTempBEC / SGPE for T→0 limit).

    n_test_particles : int  (default 10000)
        Number of Monte Carlo test particles representing the thermal cloud.
        Larger N_test → smoother ñ(r), less shot noise, more compute.
        Rule of thumb: N_test ≥ 10 × (n1 × n2 × n3)^(1/3) × 10.

    gamma_12 : float  (default 0.1)
        Dimensionless C_12 coupling rate between condensate and thermal cloud.
        Controls how fast atoms exchange between the two components.
        γ_12 = 0 decouples condensate and thermal cloud entirely.

    chemical_potential : float or None  (default None)
        Reservoir chemical potential μ (ħ ω_ho units).
        If None, computed once from the initial ground-state wavefunction.

    enable_c22 : bool  (default False)
        Whether to apply C_22 (thermal–thermal) collisions.
        Currently a no-op stub; set True when a real C_22 is implemented.
    """
```

### `_initialize_custom_parameters`

```python
def _initialize_custom_parameters(self) -> None:
    """
    Read ZNG-specific parameters and allocate the thermal cloud arrays.

    temperature : float
        k_B T / (ħ ω_ho) — controls thermal cloud width and noise.

    n_test_particles : int
        N_test — number of test particles for the Monte Carlo thermal cloud.
        More particles → smoother ñ but slower per step.

    gamma_12 : float
        Phenomenological C_12 rate.  Drives atom exchange between condensate
        and thermal cloud on the timescale 1/γ_12.

    chemical_potential : float or None
        If not provided, μ is computed from the T=0 ground state as
        μ = e_kin + e_pot + 2·e_int and held fixed throughout.

    enable_c22 : bool
        Placeholder flag.  C_22 is the thermal–thermal Boltzmann collision
        integral; it thermalises the distribution further but is expensive.
    """
```

### `_initialise`

```python
def _initialise(self) -> None:
    """
    Load the condensate ground state and initialise the thermal cloud.

    Calls the parent _initialise() to find/read ψ₀ from disk, then
    computes μ (if not supplied) and draws the initial test-particle
    distribution from the semiclassical Boltzmann distribution.
    """
```

### `_main_simulation_loop`

```python
def _main_simulation_loop(self) -> None:
    """
    ZNG coupled time-evolution loop.

    Each iteration couples the condensate and the thermal cloud through
    their shared densities (n_c, ñ).  The sequence is:

    1.  Compute condensate density: n_c = |ψ|²
    2.  Deposit thermal cloud → grid: ñ = CIC(particle positions)
    3.  Build condensate GPE potential:
            V_GP = V_ext + u(n_c + 2ñ)
    4.  Build condensate source term (C_12 mean field):
            R = 2 γ_12 [μ − V_eff],   V_eff = V_ext + 2u(n_c + ñ)
    5.  Evolve condensate one step with modified GPE:
            i ∂ψ/∂t = [H_GP + iR/2] ψ
        using the split-step method with complex potential V_GP + iR/2.
    6.  Compute thermal particle potential:
            U = V_ext + 2u(n_c_new + ñ)
    7.  Advance test particles by leapfrog under U.
    8.  Apply C_12 stochastic collisions (absorption + emission).
    9.  Apply C_22 (no-op stub unless enable_c22 is True).
    10. Update ñ from new particle positions.

    The condensate is re-normalised after the split step to enforce the
    grand-canonical particle-number constraint (same as SGPE).

    Why the source term enters as +iR/2
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The modified GPE is written as:
        i ∂ψ/∂t = H_GP ψ + i(R/2) ψ

    Multiplying through by −i:
        ∂ψ/∂t = −i H_GP ψ + (R/2) ψ

    The +R/2 (not −R/2) gives ∂|ψ|²/∂t = R|ψ|², so where R > 0 the
    condensate density grows.  In the split-step, this is implemented by
    passing a complex potential V_GP + iR/2 to the x-evolution operator:
        exp(−i·dt·(V_GP + iR/2)) = exp(dt·R/2) · exp(−i·dt·V_GP)
    The real exponential exp(dt·R/2) provides the growth/decay.
    """
```

---

## `src/models/simulation.py` — Simulations & factory

### `_get_bec_class`

```python
def _get_bec_class(model_type: str):
    """
    Return the BEC model class for the given model_type string.

    model_type is set in the configuration file and controls which physics
    model is used for the condensate evolution:

        "BEC"           → src.models.BEC.BEC
                          Zero-temperature GPE with vortex / soliton imprinting.

        "FiniteTempBEC" → src.models.finite_temp_BEC.FiniteTempBEC
                          Stochastic Projected GPE (SGPE): damped GPE + thermal
                          noise.  Requires "temperature" and optionally
                          "damping_coefficient" in the config.

        "ZNGBEC"        → src.experimental.zng.zng_BEC.ZNGBEC
                          Full Zaremba-Nikuni-Griffin two-component framework:
                          condensate GPE coupled to a Monte Carlo thermal cloud.
                          Requires "temperature", "n_test_particles", "gamma_12".

    Imports are deferred so that the experimental ZNG module is only loaded
    when explicitly requested.

    Args:
        model_type (str): One of "BEC", "FiniteTempBEC", "ZNGBEC".

    Returns:
        type: The BEC model class.

    Raises:
        ValueError: If model_type is not a recognised string.
    """
```

### `Simulations` class

```python
class Simulations:
    """
    Class that holds all simulations to be run.
    For every simulation, a new BEC is created and initialised.
    Then it is let to evolve.
    """
    def __init__(self, system, app):
        ...

    def run_simulations(self):
        """
        For every simulation, it creates a new BEC.
        Then runs the simulation.

        Instantiates the correct model class based on model_type in config.
        Defaults to "BEC" (zero-temperature GPE) when the key is absent.
        """
```

---

## `src/models/BEC.py` — BEC (updated dark-soliton API)

### `_initialize_dark_soliton_parameters`

```python
def _initialize_dark_soliton_parameters(self) -> None:
    """
    Initializes dark-soliton parameters from the simulation configuration.

    Expected keys (all lists of the same length):
      - ``soliton_positions``: centre positions in dimensionless units.
      - ``soliton_widths``: characteristic widths (healing length scale).
      - ``soliton_axes``: which axis each soliton is perpendicular to (1 or 3).
      - ``soliton_greyness`` (optional): grey-soliton angle in radians per soliton.
      - ``soliton_imprint_time``: snapshot index at which the soliton is imprinted.
    """
```

### `_initialize_vortex_parameters`

```python
def _initialize_vortex_parameters(self) -> None:
    """
    Initializes vortex-related parameters from the simulation configuration.

    Reads: vortex_charge, vortex_position_x/y, initial_imprint_time,
    imprint_every, max_imprints, imprinting_charge, repetitive,
    imprint_times, imprint_position_x/y.

    When repetitive=True, also builds imprinting_vortices_dictionary
    mapping imprint_times → vortex array.
    """
```

---

## `src/library/ground_state.py` — GroundState (updated docstrings)

### `find_ground_state`

```python
@staticmethod
def find_ground_state(sim_params, system, file_name, device):
    """Compute and persist the stationary ground-state wavefunction.

    The initial state is seeded with a Thomas-Fermi profile and refined via
    imaginary-time steepest descent until the residual norm or relative
    energy change reaches the stopping criterion.

    Parameters
    ----------
    sim_params : dict
        Simulation configuration for the run.
    system : object
        Simulation system carrying ``simulation_parameters`` and the
        external potential in ``system.uext.potential``.
    file_name : str
        Output path used by write_psi to store the converged state.
    device : str or torch.device
        Device on which the tensors and FFTs are evaluated.

    Returns
    -------
    torch.Tensor
        Normalized complex tensor containing the converged ground state.
    """
```

### `steepest_descent`

```python
@staticmethod
def steepest_descent(psi, dtau, p_sq, uext, d_x, u):
    """Advance the imaginary-time solver by one steepest-descent step.

    Parameters
    ----------
    psi : torch.Tensor
        Current condensate wavefunction.
    dtau : float
        Imaginary-time step size.
    p_sq : torch.Tensor
        Squared momentum grid used to apply the kinetic-energy operator in
        Fourier space.
    uext : torch.Tensor
        External trapping potential sampled on the spatial grid.
    d_x : float
        Volume element used for normalization and expectation values.
    u : float
        Contact-interaction strength.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Updated normalized wavefunction, total energy estimate, residual
        norm (convergence metric), and chemical potential.
    """
```

### `read_ground_state`

```python
@staticmethod
def read_ground_state(data, n1, n2, n3):
    """Load a serialized ground-state wavefunction from disk.

    Parameters
    ----------
    data : str or path-like
        Text file containing the complex wavefunction values written as
        modulus/phase columns.
    n1, n2, n3 : int
        Grid resolution used to reshape the flattened data.

    Returns
    -------
    torch.Tensor
        Complex tensor of shape (n1, n2, n3) on the CPU.
    """
```

---

## Auto-documentation tools recommendation

Rather than maintaining this file manually, consider using one of these Python packages:

### `pdoc` (recommended for simplicity)
```bash
pip install pdoc
pdoc src/library src/models src/utils --output-dir docs/api
```
Generates clean HTML documentation from all docstrings. Supports NumPy / Google styles. No config file needed.

### `pydoc-markdown` (recommended for Markdown / Astro output)
```bash
pip install pydoc-markdown
pydoc-markdown -I src -m library.gpe_library -m models.BEC -m models.base_BEC \
    -m models.finite_temp_BEC > docs/api_auto.md
```
Converts docstrings directly to Markdown — compatible with Astro/Starlight MDX files.

### `mkdocstrings` + `mkdocs`
```bash
pip install mkdocs mkdocstrings[python]
```
Full MkDocs site from docstrings with a `mkdocs.yml` config. More setup but best for a full standalone docs site.

### `sphinx` + `autodoc`
```bash
pip install sphinx sphinx-autodoc-typehints
sphinx-quickstart docs/
```
Most feature-complete; supports cross-references, LaTeX math, and custom themes. Steeper learning curve.

> **For this project**: `pydoc-markdown` produces Markdown that can be pasted directly
> into the Astro/Starlight MDX files, making it the best fit for your existing workflow.
