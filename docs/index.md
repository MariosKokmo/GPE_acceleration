# BAQS — Bose–Einstein condensate simulation

GPU-accelerated solvers for the Gross–Pitaevskii equation and its
finite-temperature extensions, in Cartesian and cylindrical geometry.

All quantities are dimensionless, in harmonic-oscillator units
$\hbar = m = \omega_{ho} = 1$, with the wavefunction normalised so that
$\int |\psi|^2\, dV = 1$ over the whole grid.

## Models

| `model_type` | Class | Physics |
| --- | --- | --- |
| `"BEC"` | {py:class}`src.models.BEC.BEC` | Zero-temperature GPE, vortex and dark-soliton imprinting, three-body loss |
| `"FiniteTempBEC"` | {py:class}`src.models.finite_temp_BEC.FiniteTempBEC` | Stochastic projected GPE — damped evolution against a thermal reservoir |
| `"ZNGBEC"` | `src.experimental.zng.zng_BEC.ZNGBEC` | Two-component Zaremba–Nikuni–Griffin condensate plus Monte-Carlo thermal cloud (experimental) |

## Where the physics lives

- {py:mod}`src.library.gpe_library` — the Cartesian operators: grids, the
  split-step propagator, energies, the SGPE step, vortices, solitons and the
  3-D diagnostics.
- {py:mod}`src.library.gpe_cylindrical_library` — the same operators in
  $(r, \varphi, z)$, where the radial kinetic term is diagonalised once in a
  $\sqrt{r}$-symmetrised eigenbasis rather than transformed by FFT.
- {py:mod}`src.library.ground_state` and
  {py:mod}`src.library.ground_state_cylindrical` — imaginary-time
  steepest-descent solvers that produce the initial state.
- {py:mod}`src.library.potentials` — trap shapes, ramps and absorbing
  boundaries.

## Guides

```{toctree}
:maxdepth: 1
:caption: Guides

configuration
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/src
```

## Conventions worth knowing

Several choices in the library are easy to misread, and each is explained in
the docstring of the function concerned:

- **Volume elements.** Every integral carries the cell volume `d_x`
  (or `r dr dφ dz` in cylindrical geometry). Energies and expectation values
  would otherwise scale with the grid spacing.
- **Velocity comes from ψ, never from its phase.** `angle(ψ)` has a $2\pi$
  branch cut out of every vortex core, and a spectral derivative of that
  discontinuity corrupts the whole field. See
  {py:meth}`src.library.gpe_library.GPELibrary.superfluid_velocity`.
- **The norm is not forced back to 1.** For a real potential the propagator is
  unitary and conserves it anyway; when the potential is complex — three-body
  loss, an absorber — the norm is *meant* to decay, and in the SGPE a forced
  norm cancels the chemical potential outright. See
  {py:meth}`src.library.gpe_library.GPELibrary.sgpe_step`.

## Indices

- {ref}`genindex`
- {ref}`modindex`
