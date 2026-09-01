# Configuration and running

Everything a run needs is described by two JSON files: `configuration_file.json`,
which defines the physical system and the scenarios to simulate, and
`appConfig.json`, which controls application-level settings such as logging and
which extra outputs are written. This guide covers installing the solver, the
three ways to start a run, the units the solver works in, and then the complete
reference for both files.

```bash
pip install -e .
baqs configuration_file.json appConfig.json --check --run -v 1
```

---

## Installation

The package requires **Python 3.9–3.12** and is built on PyTorch. An NVIDIA GPU
with CUDA support is strongly recommended, but not required — the same code runs
on CPU.

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install in editable mode; this also provides the `baqs` command
pip install -e .
```

Installing from `requirements.txt` instead pins the exact versions the solver is
verified against:

| Package | Pinned | Accepted range |
|---|---|---|
| `torch` | 2.2.2 | `>=2.2,<2.5` |
| `numpy` | 1.26.4 | `>=1.26,<2.0` |
| `pandas` | 2.2.3 | `>=2.1,<2.3` |
| `matplotlib` | 3.9.2 | `>=3.8,<4` |
| `opencv-python` | 4.10.0.84 | `>=4.9` |

The pinned `torch` is the CPU wheel from PyPI. For GPU runs install the build
matching your CUDA version first — see
[PyTorch Get Started](https://pytorch.org/get-started/locally/) — then install
this package without letting it pull a different torch back in.

---

## Running a simulation

There are three entry points, and they all execute the same code path: both
`run.py` and the CLI are thin wrappers around `Simulator`.

### The `baqs` command line interface

Available once the package is installed.

```bash
baqs <config_file> <app_config_file> [options]
```

| Argument | Description |
|---|---|
| `config` | Path to the simulation configuration JSON. |
| `app` | Path to the application configuration JSON. |
| `-c`, `--check` | Validate the configuration files without running anything. |
| `--run` | Execute the simulations. |
| `-v`, `--verbose` | Verbosity — `0` silent, `1` info, `2` debug (default `0`). |

```bash
# Validate the configuration only
baqs configuration_file.json appConfig.json --check -v 1

# Validate, then run
baqs configuration_file.json appConfig.json --check --run -v 1
```

Without installing, the same CLI is reachable as a module:

```bash
python -m src.cli.baqs configuration_file.json appConfig.json -c -v 2
```

### The `run.py` script

The shortest path when both configuration files sit in the working directory
under their default names (`configuration_file.json` and `appConfig.json`):

```bash
python src/run.py
```

### The `Simulator` API

The way to drive simulations programmatically — from a notebook, a parameter
sweep, or any Python script:

```python
from src.simulator import Simulator

sim = Simulator("configuration_file.json")

# Inspect the parsed grid and potential before committing to a run
print(sim.system.simulation_parameters)
print(sim.system.uext)

# Run every simulation combination defined in the config
sim.run()

# Inspect the results afterwards
print(sim.simulations.BEC)
```

---

## What a run does

1. Read `appConfig.json` and set up logging.
2. Load and validate `configuration_file.json`.
3. Compute the ground state for the grid by imaginary-time propagation — or load
   it, if a matching file already exists.
4. Run each simulation defined in the config, one after another, from that same
   ground state.
5. Write the snapshots, metadata and any requested videos to a folder per
   simulation.

<img src="static/flow.jpg">

---

## Units and derived quantities

The configuration is written in laboratory units — hertz, seconds, microns —
and the solver converts them once, at setup, into the dimensionless
harmonic-oscillator units it works in, where

$$
\hbar = m = \omega_{ho} = 1,
\qquad \int \lvert\psi\rvert^{2}\, \mathrm{d}V = 1 .
$$

The wavefunction is normalised over the whole grid, so $\lvert\psi\rvert^{2}$ is
a *fraction* of the cloud per unit volume, not an atom count.

### From the configuration to the solver

| Quantity | Definition | Set by |
|---|---|---|
| Trap frequencies | $\omega_\alpha = 2\pi f_\alpha$ | `Trapping_frequencies` (Hz) |
| Reference frequency | $\omega_{ho} = (\omega_x \omega_y \omega_z)^{1/3}$, and $(\omega_r^2 \omega_z)^{1/3}$ in cylindrical geometry | derived |
| Length unit | $a_{ho} = \sqrt{\hbar / m \omega_{ho}}$ | derived |
| Interaction strength | $u = 4\pi N a_s / a_{ho}$ | derived, see below |
| Time step | $\Delta\tau = \omega_{ho}\, \Delta t$ | `dt` (s) |
| Iterations | $k_\mathrm{max} = \lfloor T_\mathrm{tot} / \Delta t \rfloor$ | `Total_simulation_time`, `dt` |
| Snapshot interval | $\max\!\left(1, \lfloor k_\mathrm{max} / N_\mathrm{shots} \rfloor\right)$ iterations | `snapshots` |

Grid limits given in microns are divided by $a_{ho}$, so every position, width
and soliton coordinate in the config is interpreted in units of $a_{ho}$ once
the box has been converted. The spacings follow from the limits and the
resolution:

$$
\mathrm{d}x_\alpha = \frac{x^{\max}_\alpha - x^{\min}_\alpha}{n_\alpha},
\qquad
\mathrm{d}p_\alpha = \frac{2\pi}{x^{\max}_\alpha - x^{\min}_\alpha}
\qquad \text{(Cartesian)},
$$

$$
\mathrm{d}r = \frac{r_{\max}}{n_r},
\quad r_i = \left(i + \tfrac{1}{2}\right) \mathrm{d}r,
\qquad
\mathrm{d}\varphi = \frac{2\pi}{n_\varphi},
\qquad
\mathrm{d}z = \frac{z_{\max} - z_{\min}}{n_z}
\qquad \text{(cylindrical)}.
$$

The half-point radial layout is what keeps the vortex core at $r = 0$ resolved
rather than sitting exactly on a grid point.

### Atom number, mass and scattering length

These are **not** in the configuration file. The interaction strength $u$ is
built from the constants in `src/library/parameters.py`:

| Constant | Default | Meaning |
|---|---|---|
| `nat` | $5 \times 10^{4}$ | Atom number $N$ |
| `m1` | $87\,\mathrm{amu}$ | Atomic mass $m$ (rubidium-87) |
| `ascat` | $99\,a_0$ | s-wave scattering length $a_s$ |
| `k3` | $1000$ | Dimensionless three-body loss rate $K_3$ |

> **Note** — changing the atom number or the species means editing that file,
> and it changes $u$ for every simulation. The ground state is cached per grid
> and trap only, so delete the cached `*_ground_state.dat` after changing a
> constant, or the run will start from a state computed with the old $u$.

---

## The two configuration files

### `configuration_file.json`

**One configuration file describes one grid and one potential.** The
per-simulation keys are lists, so any number of *evolution scenarios* can be run
in sequence on that grid — but a different grid or potential means a different
file, since every scenario in a file shares a single ground state.

Running two different grids at the same time therefore means two processes in
separate working directories. Every input is validated before the simulations are
set up; the complete reference is in the sections below.

### `appConfig.json`

The second file controls application-level settings. It is read once at startup
and is independent of the simulation configuration.

```json
{
    "logfile":       "log.txt",
    "configFile":    "configuration_file.json",
    "write_velocity": false,
    "phase_imaging":  false
}
```

| Key | Type | Description |
|---|---|---|
| `"logfile"` | string | Name of the top-level log file. |
| `"configFile"` | string | Path to the simulation configuration file. |
| `"write_velocity"` | bool | If `true`, write velocity-field data and create a velocity video after each simulation. |
| `"phase_imaging"` | bool | If `true`, save phase snapshots at every snapshot step. |

---

## Coordinate system

The solver supports two coordinate systems. The coordinate system is read from
`configuration_file.json` using the following logic:

1. **Explicit key** (recommended): add `"coordinates": "cylindrical"` or
   `"coordinates": "cartesian"` to the config file.
2. **Auto-detection fallback**: if the `"coordinates"` key is absent, the system
   checks whether `"r_max"` is present. If it is, cylindrical is assumed; otherwise
   Cartesian is assumed.

```json
"coordinates": "cylindrical"
```

| | Cartesian | Cylindrical |
|---|---|---|
| Coordinates | $(x, y, z)$ | $(r, \varphi, z)$ |
| Grid keys | `Grid_negative_limits`, `Grid_positive_limits` | `r_max`, `z_min`, `z_max` |
| `Grid_resolution` | $[n_x, n_y, n_z]$ | $[n_r, n_\varphi, n_z]$ |
| `Trapping_frequencies` | $[f_x, f_y, f_z]$ (Hz) | $[f_r, f_z]$ (Hz) |
| Vortex `position_x` | grid index (integer, can be negative) | radial distance $r \ge 0$, in units of $a_{ho}$ |
| Vortex `position_y` | grid index (integer, can be negative) | azimuthal angle $\varphi \in [0, 2\pi]$ (radians) |
| Typical geometry | cigar / isotropic | pancake (disk-shaped cloud) |

---

## BEC model type

The `"model_type"` key in `configuration_file.json` selects the physics model used for time
evolution. It defaults to `"BEC"` when absent.

```json
"model_type": "BEC"
```

### `"BEC"` — zero-temperature GPE (default)

Standard split-step Fourier method with no thermal effects,

$$
i\, \frac{\partial \psi}{\partial t}
    = \left[ -\tfrac{1}{2}\nabla^{2} + V_\mathrm{ext}
             + u \lvert\psi\rvert^{2} \right] \psi .
$$

Supports vortex and dark-soliton imprinting. No extra keys required.

### `"FiniteTempBEC"` — Stochastic Projected GPE (SGPE)

Adds a damping term and thermal noise to the GPE,

$$
\frac{\partial \psi}{\partial t}
    = -(i + \gamma)\left(H_\mathrm{mf} - \mu\right) \psi
      + \eta(\mathbf{r}, t),
\qquad
H_\mathrm{mf} = -\tfrac{1}{2}\nabla^{2} + V_\mathrm{ext}
    + u \lvert\psi\rvert^{2},
$$

where the noise obeys the fluctuation-dissipation relation

$$
\left\langle \eta^{*}(\mathbf{r}, t)\, \eta(\mathbf{r}', t') \right\rangle
    = 2\gamma k_B T\, \delta(\mathbf{r} - \mathbf{r}')\, \delta(t - t') .
$$

Modes above $\mu$ are damped and modes below it are amplified, which is what
drives condensate growth at finite temperature.

| Key | Type | Default | Description |
|---|---|---|---|
| `"temperature"` | float | `0.0` | Reservoir temperature $k_B T / (\hbar\omega_{ho})$. `0` gives pure damping (no noise). |
| `"damping_coefficient"` | float | `0.03` | Dimensionless damping rate $\gamma$. Typical range $0.01$–$0.1$. |
| `"chemical_potential"` | float or null | `null` | Reservoir $\mu$. If `null`, computed from the initial ground state. |

### `"ZNGBEC"` — Zaremba–Nikuni–Griffin (experimental)

Full two-component framework: the condensate GPE carries a source term $R$ and
is coupled to a Monte Carlo cloud of test particles moving in the thermal mean
field,

$$
i\, \frac{\partial \psi}{\partial t}
    = \left[ -\tfrac{1}{2}\nabla^{2} + V_\mathrm{ext}
             + u\,(n_c + 2\tilde{n})
             + \tfrac{i}{2} R \right] \psi,
$$

$$
R = 2\gamma_{12}\left(\mu - V_\mathrm{eff}\right),
\qquad
V_\mathrm{eff} = V_\mathrm{ext} + 2u\,(n_c + \tilde{n}),
$$

with $n_c = \lvert\psi\rvert^{2}$ the condensate density and $\tilde{n}$ the
thermal density deposited from the test-particle positions onto the grid.

| Key | Type | Default | Description |
|---|---|---|---|
| `"temperature"` | float | `0.0` | Temperature $k_B T / (\hbar\omega_{ho})$; must be positive for a ZNG run. |
| `"n_test_particles"` | int | `10000` | Number of Monte Carlo test particles. More particles give a smoother $\tilde{n}$ at more compute. |
| `"gamma_12"` | float | `0.1` | Condensate–thermal coupling $\gamma_{12}$; `0` decouples the two components. |
| `"chemical_potential"` | float or null | `null` | Reservoir $\mu$. If `null`, computed from the initial ground state. |
| `"enable_c22"` | bool | `false` | Enable the $C_{22}$ thermal–thermal collision term (currently a no-op stub). |

The sample is split between the two components at startup. The initial thermal
fraction comes from the ideal-Bose result for a 3-D harmonic trap,
$k_B T_c = \hbar\omega_{ho} (N/\zeta(3))^{1/3}$, so that

$$
f = \left(\frac{T}{T_c}\right)^{3} = \frac{T^{3}\, \zeta(3)}{N},
$$

capped at 1. The condensate is then scaled to hold the remaining $1 - f$ of the
sample, which is what puts $n_c$ and $\tilde{n}$ on a common scale. By default
the condensate is renormalised after every step, so $R$ reshapes it but its atom
number stays pinned.

> **Note** — the model class also recognises `zng_condensate_exchange`,
> `zng_thermal_fraction_mode` and `zng_thermal_fraction`. They are parsed from
> the configuration file but are not forwarded to the model, so setting them in
> the config currently has no effect.

---

## Cartesian configuration

> **Important** — each configuration file is strictly for one grid and one
> potential. Multiple simulations can be run in sequence, but they all share the
> same grid, potential and ground state.

### Required grid keys

| Key | Type | Example | Description |
|---|---|---|---|
| `"Grid_positive_limits"` | list of 3 floats | `[60, 1.5, 60]` | Upper box half-extents along each axis, in microns. |
| `"Grid_negative_limits"` | list of 3 floats | `[-60, -1.5, -60]` | Lower box half-extents, usually symmetric. |
| `"Grid_resolution"` | list of 3 ints | `[512, 16, 512]` | Grid points along each axis; the example is a pancake BEC. |
| `"Trapping_frequencies"` | list of 3 floats | `[20, 300, 20]` | Trap frequencies $[f_x, f_y, f_z]$ in Hz. |

### Required time and potential keys

| Key | Type | Example | Description |
|---|---|---|---|
| `"Potential_type"` | string | `"harmonic"` | Trap shape; see the list of available potentials below. |
| `"Total_simulation_time"` | float | `150e-3` | Total evolution time $T_\mathrm{tot}$ in seconds. |
| `"dt"` | float | `5e-7` | Time step $\Delta t$ in seconds. |
| `"snapshots"` | int | `150` | Number of snapshot files to write. |
| `"SwitchOff_time"` | float | `0` | Snapshot index after which the trap is switched off, for a time-of-flight release. |

### Optional absorber keys

The complex absorbing potential (CAP) damps the wavefunction near the box
boundaries, so outgoing matter waves are swallowed instead of being wrapped
around by the periodicity of the FFT. It turns on at a fixed fraction of the
half-box and grows as a power law towards the edge,

$$
c_\alpha(\mathbf{r}) =
    \left[ \mathrm{clamp}\!\left(
        \frac{\lvert\alpha\rvert - \alpha_\mathrm{start}}
             {\alpha_{\max} - \alpha_\mathrm{start}}, 0, 1
    \right) \right]^{p},
\qquad
V_\mathrm{cap} = \eta \max\left(c_x, c_y, c_z\right),
$$

and enters the evolution as $-i\, g(t)\, V_\mathrm{cap}$, which decays the
density rather than shifting its phase. The maximum over the three axes is what
makes the corners of the box absorb as strongly as the faces.

| Key | Type | Default | Description |
|---|---|---|---|
| `"Absorber_enabled"` | bool (`0`/`1` or `false`/`true`) | `false` | Master switch for the boundary absorber. |
| `"Absorber_strength"` | float $\ge 0$ | `0.0` | Damping prefactor $\eta$; the CAP is skipped when it is not positive. |
| `"Absorber_start_ratio"` | float in $[0, 1)$ | `0.8` | Fraction $\alpha_\mathrm{start} / \alpha_{\max}$ of the half-box at which damping starts. |
| `"Absorber_power"` | float $\ge 1$ | `2` | Exponent $p$, i.e. the smoothness of the ramp. |
| `"Absorber_tinit"` | float | `0.0` | Dimensionless time at which the absorber starts to turn on. |
| `"Absorber_tfinal"` | float or absent | absent | End of the linear ramp-up. If omitted, the absorber turns on instantly at `Absorber_tinit`. |

### Vortex position convention (Cartesian)

`vortex_position_x` and `vortex_position_y` are **grid indices** measured from the grid centre.

- Valid range: $-n/2$ to $+n/2$ along the respective axis.
- `(0, 0)` places the vortex at the trap centre.
- Example: `vortex_position_x = [10]` places the vortex 10 grid points from centre along $x$.

---

## Cylindrical configuration

Use this when the trap is axially symmetric (pancake geometry, $f_r \ll f_z$).
The grid is defined in $(r, \varphi, z)$ coordinates with the half-point radial
layout $r_i = (i + \tfrac{1}{2})\,\mathrm{d}r$, so the vortex core at $r = 0$ is
always resolved.

### Required grid keys

| Key | Type | Example | Description |
|---|---|---|---|
| `"r_max"` | float | `60.0` | Outer radial boundary, in microns. |
| `"z_min"` | float | `-10.0` | Lower axial bound in microns; must be negative. |
| `"z_max"` | float | `10.0` | Upper axial bound in microns; must be positive. |
| `"Grid_resolution"` | list of 3 ints | `[256, 32, 64]` | Grid points $[n_r, n_\varphi, n_z]$. |
| `"Trapping_frequencies"` | list of 2 floats | `[10, 100]` | Trap frequencies $[f_r, f_z]$ in Hz. The three-element form $[f_r, f_r, f_z]$ is also accepted, for config-file compatibility. |

The time, potential and absorber keys are the same as in the Cartesian case.

### Vortex position convention (cylindrical)

`vortex_position_x` and `vortex_position_y` use **physical coordinates**, not grid indices.

- `vortex_position_x` → radial distance $r \ge 0$, in the same dimensionless
  units as `r_max`, i.e. units of the harmonic-oscillator length $a_{ho}$.
  - `0` places the vortex on the symmetry axis.
  - Validated against `r_max`; values exceeding it are rejected.
- `vortex_position_y` → azimuthal angle $\varphi \in [0, 2\pi]$ in radians.
  - `0` aligns the vortex with the positive $x$-axis of the underlying Cartesian frame.
  - Validated against $[0, 2\pi]$; values outside this range are rejected.
  - The pair is converted to Cartesian coordinates
    $(x_0, y_0) = (r\cos\varphi,\; r\sin\varphi)$ internally before phase
    imprinting.

The same key names (`position_x` / `position_y`) are reused so that the rest of the simulation
pipeline stays coordinate-agnostic.

### Resolving the vortex core

Because the azimuthal arc length grows with radius, the effective local spacing
at a core sitting a distance $r_0$ from the axis is

$$
\mathrm{d}r_\mathrm{eff} = \max\left(\mathrm{d}r,\; r_0\, \mathrm{d}\varphi\right),
$$

and the core counts as resolved when at least two cells span one healing length,

$$
\frac{r_\mathrm{core}}{\mathrm{d}r_\mathrm{eff}} \ge 2 .
$$

A resolution warning naming the offending vortex — and whether the radial or the
azimuthal direction is the bottleneck — is printed automatically when this
condition is violated.

---

## Vortex excitation keys (both coordinate systems)

Two global switches turn the feature on:

| Key | Type | Description |
|---|---|---|
| `"vortex_excitation"` | `0` / `1` | Enable vortex imprinting. |
| `"repetitive"` | `0` / `1` | `1` for repetitive re-imprinting, `0` for a single initial imprint. |

Every remaining key is a list with **one entry per simulation**; each entry may
itself be a list, describing that simulation's vortices.

| Key | Description |
|---|---|
| `"vortex_charge"` | Initial charges per simulation, e.g. `[[1], [1, -1]]`. |
| `"imprinting_charge"` | Charges for the subsequent imprints, e.g. `[[1], [-1]]`. |
| `"vortex_position_x"` | Initial positions; see the coordinate-system section above for the meaning. |
| `"vortex_position_y"` | Initial positions; see the coordinate-system section above for the meaning. |
| `"initial_imprint_time"` | Snapshot index at which the initial vortices are imprinted. |
| `"imprint_position_x"` | Positions for the re-imprinted vortices, same convention as `vortex_position_x`. |
| `"imprint_position_y"` | Positions for the re-imprinted vortices. |
| `"imprint_every"` | Re-imprint interval in snapshots, used when `imprint_times` is empty. |
| `"imprint_times"` | Exact snapshot indices for each re-imprint; `[]` auto-generates them from `imprint_every`. |
| `"max_imprints"` | Maximum number of re-imprints per simulation. |

The example values above would give two simulations.

> **Note** — `"imprint_times"` overrides `"imprint_every"`. Pass an empty list
> for `"imprint_times"` when the times should be generated automatically from
> `"imprint_every"`.

---

## Dark solitons

Dark solitons are configured **per simulation**, exactly like vortices: each key
is a top-level list with one entry per simulation. The `"dark_soliton"` flag is a
single global on/off switch, mirroring `"vortex_excitation"`, and the
per-simulation lists must all have the same length as the other per-simulation
lists such as `vortex_charge`.

| Key | Description |
|---|---|
| `"dark_soliton"` | Global flag, `1` to enable dark-soliton imprinting. |
| `"soliton_positions"` | Per simulation, a list of soliton centre positions in microns, e.g. `[[0.0], [-10.0, 10.0]]` — simulation 0 has one soliton at 0, simulation 1 a pair at $\pm 10$. An empty list means that simulation imprints no soliton. |
| `"soliton_widths"` | Per simulation, a list of widths on the healing-length scale, e.g. `[[1.0], [0.8, 0.8]]`. |
| `"soliton_axes"` | Per simulation, a list of axes: `1` for $x_1$, `3` for $x_3$ (i.e. $z$), e.g. `[[3], [3, 3]]`. |
| `"soliton_greyness"` | *Optional.* Per simulation, values in $[0, \pi/2)$: `0` is fully dark, larger values are grey. Defaults to all zeros. |
| `"soliton_imprint_time"` | Per simulation, the snapshot index at which that simulation imprints its solitons, e.g. `[5, 10]`. |

Within a simulation, `soliton_positions[i]`, `soliton_widths[i]` and
`soliton_axes[i]` must have the same length — one value per soliton.

Solitons may be combined with vortices (both flags on, lists of equal length per
simulation) or used on their own (`"vortex_excitation": 0`, `"dark_soliton": 1`),
in which case the number of simulations is taken from the soliton lists.

### Example: a single black soliton, then a grey pair

```json
{
  "dark_soliton": 1,
  "vortex_excitation": 0,
  "soliton_positions": [[0.0], [-10.0, 10.0]],
  "soliton_widths":    [[1.0], [0.8, 0.8]],
  "soliton_axes":      [[3],   [3, 3]],
  "soliton_greyness":  [[0.0], [0.3, 0.3]],
  "soliton_imprint_time": [5, 10]
}
```

> **Note** — earlier versions documented a nested `"dark_soliton_params"`
> dictionary shared across all simulations. That form is no longer used;
> solitons are now per-simulation flat lists as shown above.

---

## Three-body loss

Setting `"three-body-losses"` to a truthy value adds three-body recombination,
which removes atoms from the densest part of the cloud,

$$
i\, \frac{\partial \psi}{\partial t}
    = \left[ -\tfrac{1}{2}\nabla^{2} + V_\mathrm{ext}
             + u \lvert\psi\rvert^{2}
             - i K_3 \lvert\psi\rvert^{4} \right] \psi .
$$

| Key | Type | Default | Description |
|---|---|---|---|
| `"three-body-losses"` | bool (`0`/`1`) | `0` | Enable the loss term. The rate itself is the constant `k3` in `src/library/parameters.py`, not a config key. |

The norm decays as a result — that is the physics, not a numerical artefact —
so runs with losses enabled should not be compared against a fixed atom number.

---

## Examples — Cartesian

### Example 1: repetitive imprinting

```json
{
  "Grid_positive_limits": [60, 1.5, 60],
  "Grid_negative_limits": [-60, -1.5, -60],
  "Grid_resolution": [512, 16, 512],
  "Trapping_frequencies": [20, 300, 20],
  "Potential_type": "harmonic",
  "Total_simulation_time": 0.15,
  "dt": 5e-7,
  "snapshots": 150,
  "vortex_excitation": 1,
  "repetitive": 1,
  "vortex_charge": [2, 5],
  "imprinting_charge": [1, 3],
  "vortex_position_x": [0, 0],
  "vortex_position_y": [0, 0],
  "imprint_position_x": [0, 0],
  "imprint_position_y": [0, 0],
  "initial_imprint_time": [0, 0],
  "imprint_every": [20, 50],
  "imprint_times": [[10, 20, 50], []],
  "max_imprints": [3, 2]
}
```

*First simulation*: start with a vortex of charge 2 at `(0, 0)`. Re-imprint charge 1 at the exact
snapshots 10, 20, 50 (`imprint_times` overrides `imprint_every`).

*Second simulation*: start with a vortex of charge 5 at `(0, 0)`. Auto-generate re-imprint times
from `imprint_every = 50` → times 50, 100, 150; but `max_imprints = 2` so only snapshots 50 and
100 are used.

### Example 2: multi-vortex, single imprint

```json
{
  "vortex_charge": [[5, 5], [1, 1, 1, 1, 1, 1, 1]],
  "imprinting_charge": [[[-5, -5]], [[0, 0, 0, 0, 0, 0, 0]]],
  "vortex_position_x": [[0, 0], [0, 0, 0, 0, 0, 0, 0]],
  "vortex_position_y": [[-20, 20], [-30, -20, -10, 0, 10, 20, 30]],
  "initial_imprint_time": [4, 4],
  "imprint_position_x": [[[0, 0]], [[0, 0, 0, 0, 0, 0, 0]]],
  "imprint_position_y": [[[-20, 20]], [[0, 0, 0, 0, 0, 0, 0]]],
  "imprint_every": [[], []],
  "imprint_times": [[50], [50]],
  "max_imprints": [1, 0]
}
```

*First simulation*: imprint two $+5$ vortices at `(0, -20)` and `(0, +20)` at snapshot 4.
Re-imprint two $-5$ vortices at the same positions at snapshot 50.

*Second simulation*: imprint seven $+1$ vortices at $x = 0$, distributed from $y = -30$ to
$y = +30$. No re-imprinting (`max_imprints = 0`).

---

## Examples — Cylindrical

### Example 1: on-axis vortex (pancake BEC)

```json
{
  "Grid_resolution": [256, 32, 64],
  "r_max": 60.0,
  "z_min": -10.0,
  "z_max": 10.0,
  "Trapping_frequencies": [20, 200],
  "Potential_type": "harmonic",
  "Total_simulation_time": 0.15,
  "dt": 5e-7,
  "snapshots": 150,
  "vortex_excitation": 1,
  "repetitive": 0,
  "vortex_charge": [[1]],
  "vortex_position_x": [[0.0]],
  "vortex_position_y": [[0.0]],
  "initial_imprint_time": [0],
  "imprinting_charge": [[]],
  "imprint_position_x": [[]],
  "imprint_position_y": [[]],
  "imprint_every": [],
  "imprint_times": [],
  "max_imprints": []
}
```

A single charge-1 vortex placed on the symmetry axis ($r = 0$, where $\varphi$ is
irrelevant).

### Example 2: off-axis vortex pair

```json
{
  "vortex_charge": [[1, -1]],
  "vortex_position_x": [[5.0, 5.0]],
  "vortex_position_y": [[0.0, 3.14159]],
  "initial_imprint_time": [0],
  ...
}
```

Two vortices at the same radial distance $r = 5\, a_{ho}$, placed at $\varphi = 0$
and $\varphi = \pi$, i.e. diametrically opposite. In Cartesian terms these sit at
$(+5, 0)$ and $(-5, 0)$.

---

## List of available potentials

`"Potential_type"` accepts the following values. The two coordinate systems share
most of them, but not all — the last two exist on one side only.

| `Potential_type` | Cartesian | Cylindrical | Description |
|---|:---:|:---:|---|
| `"harmonic"` | ✅ | ✅ | Harmonic trap built from the config trapping frequencies |
| `"constant"` | ✅ | ✅ | Uniform potential across the entire grid |
| `"ramp"` | ✅ | ✅ | Uniform potential ramped linearly from an initial to a final amplitude |
| `"rampharmonic"` | ✅ | ✅ | Harmonic trap whose amplitude ramps linearly in time |
| `"rotating"` | ✅ | ✅ | Rotating trap for stirring. Cartesian rotates the harmonic trap about a configurable axis; cylindrical rotates an anisotropic trap at angular frequency $\Omega$ with ellipticity $\epsilon$ |
| `"gaussianbeam"` | — | ✅ | Focused Gaussian laser beam along $z$: a repulsive obstacle, stirring beam or barrier |
| `"custom"` | ✅ | — | No potential is built; construct a `CustomPot` yourself and assign its shape and time dependence |

All of them accept the optional absorber (CAP) keys described above.
