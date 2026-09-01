# Configuration and running

Everything a run needs is described by two JSON files: `configuration_file.json`,
which defines the physical system and the scenarios to simulate, and
`appConfig.json`, which controls application-level settings such as logging and
which extra outputs are written. This guide covers installing the solver, the
three ways to start a run, and then the complete reference for both files.

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
| Grid keys | `Grid_negative_limits`, `Grid_positive_limits` | `r_max`, `z_min`, `z_max` |
| `Grid_resolution` | `[n_x, n_y, n_z]` | `[n_r, n_phi, n_z]` |
| `Trapping_frequencies` | `[fx, fy, fz]` (Hz) | `[fr, fz]` (Hz) |
| Vortex `position_x` | grid index (integer, can be negative) | radial distance r ≥ 0 (dimensionless units) |
| Vortex `position_y` | grid index (integer, can be negative) | azimuthal angle φ ∈ [0, 2π] (radians) |
| Typical geometry | cigar / isotropic | pancake (disk-shaped cloud) |

---

## BEC model type

The `"model_type"` key in `configuration_file.json` selects the physics model used for time
evolution. It defaults to `"BEC"` when absent.

```json
"model_type": "BEC"
```

### `"BEC"` — zero-temperature GPE (default)

Standard split-step Fourier method with no thermal effects. Supports vortex and dark-soliton
imprinting. No extra keys required.

### `"FiniteTempBEC"` — Stochastic Projected GPE (SGPE)

Adds a damping term and thermal noise to the GPE:

    ∂ψ/∂t = −(i + γ)(H_mf − μ)ψ + η(r, t)

Required/optional extra keys:

| Key | Type | Default | Description |
|---|---|---|---|
| `"temperature"` | float | `0.0` | Reservoir temperature k_BT / (ħω_ho). `0` gives pure damping (no noise). |
| `"damping_coefficient"` | float | `0.03` | Dimensionless damping rate γ. Typical range 0.01–0.1. |
| `"chemical_potential"` | float or null | `null` | Reservoir μ. If `null`, computed from the initial ground state. |

### `"ZNGBEC"` — Zaremba-Nikuni-Griffin (experimental)

Full two-component framework coupling the condensate GPE to a Monte Carlo thermal cloud.

| Key | Type | Description |
|---|---|---|
| `"temperature"` | float | Temperature in dimensionless units. |
| `"n_test_particles"` | int | Number of Monte Carlo test particles (default `10000`). |
| `"gamma_12"` | float | Condensate–thermal coupling coefficient (default `0.1`). |
| `"enable_c22"` | bool | Enable C22 collision term (default `false`). |

---

## Cartesian configuration

**Important** -- Each configuration file is strictly for one grid and potential configuration.
Multiple simulations can be run in sequence but all have to be on the same grid and potential
with the same ground state.

### Required grid keys
- `"Grid_positive_limits"`: upper box half-extents along each axis in microns, e.g. `[60, 1.5, 60]`
- `"Grid_negative_limits"`: lower box half-extents (usually symmetric), e.g. `[-60, -1.5, -60]`
- `"Grid_resolution"`: number of grid points along each axis, e.g. `[512, 16, 512]` for a pancake BEC
- `"Trapping_frequencies"`: trap frequencies in Hz `[fx, fy, fz]`, e.g. `[20, 300, 20]`

### Required time / potential keys
- `"Potential_type"`: e.g. `"harmonic"` — for a full list see below
- `"Total_simulation_time"`: total evolution time in seconds, e.g. `150e-3`
- `"dt"`: simulation time step in seconds, e.g. `5e-7`
- `"snapshots"`: number of snapshot files to write, e.g. `150`

### Optional absorber keys
- `"Absorber_enabled"`: boolean (0/1 or false/true). Enables a boundary complex absorber
- `"Absorber_strength"`: non-negative float. Damping strength at the grid boundary
- `"Absorber_start_ratio"`: float in [0, 1). Fraction of half-box size where damping starts, e.g. `0.8`
- `"Absorber_power"`: float ≥ 1. Smoothness/order of the absorber ramp, e.g. `2`
- `"Absorber_tinit"`: float. Start time (dimensionless) for turning on the absorber
- `"Absorber_tfinal"`: float. End time for linear absorber ramp-up. If omitted, absorber turns on instantly at `Absorber_tinit`

### Vortex position convention (Cartesian)
`vortex_position_x` and `vortex_position_y` are **grid indices** measured from the grid centre.
- Valid range: `−n/2` to `+n/2` along the respective axis.
- `(0, 0)` places the vortex at the trap centre.
- Example: `vortex_position_x = [10]` places the vortex 10 grid points from centre along x.

---

## Cylindrical configuration

Use this when the trap is axially symmetric (pancake geometry: `fr << fz`).
The grid is defined in `(r, φ, z)` coordinates with half-point radial layout
(`r_i = (i + 0.5) · dr`), so the vortex core at `r = 0` is always resolved.

### Required grid keys
- `"r_max"`: outer radial boundary in microns, e.g. `60.0`
- `"z_min"`: lower axial bound in microns (must be negative), e.g. `-10.0`
- `"z_max"`: upper axial bound in microns (must be positive), e.g. `10.0`
- `"Grid_resolution"`: `[n_r, n_phi, n_z]`, e.g. `[256, 32, 64]`
- `"Trapping_frequencies"`: `[fr, fz]` in Hz, e.g. `[10, 100]`.
  A three-element form `[fr, fr, fz]` is also accepted for config-file compatibility.

### Required time / potential keys
Same as Cartesian: `"Potential_type"`, `"Total_simulation_time"`, `"dt"`, `"snapshots"`.

### Optional absorber keys
Same as Cartesian.

### Vortex position convention (cylindrical)
`vortex_position_x` and `vortex_position_y` use **physical coordinates**, not grid indices.

- `vortex_position_x` → radial distance **r ≥ 0** in the same dimensionless units as `r_max`
  (i.e. units of `a_ho` — the harmonic oscillator length).
  - `0` places the vortex on the symmetry axis.
  - Validated against `r_max`; values exceeding `r_max` are rejected.
- `vortex_position_y` → azimuthal angle **φ ∈ [0, 2π]** in radians.
  - `0` aligns the vortex with the positive x-axis (in the underlying Cartesian frame).
  - Validated against `[0, 2π]`; values outside this range are rejected.
  - The combination `(r, φ)` is converted to Cartesian `(x₀, y₀) = (r cosφ, r sinφ)`
    internally before phase imprinting.

The same key names (`position_x` / `position_y`) are reused so that the rest of the simulation
pipeline stays coordinate-agnostic.

### Resolution note
Because the azimuthal arc length grows with radius, the effective resolution at large `r` is
`dr_eff = r · dφ`. The vortex core (healing length `r_core`) must satisfy
`r_core / max(dr, r · dφ) ≥ 2` at every vortex location.
A resolution warning is printed automatically when this condition is violated.

---

## Vortex excitation keys (both coordinate systems)

- `"vortex_excitation"`: `1` to enable vortex imprinting, `0` to skip
- `"repetitive"`: `1` for repetitive (re-)imprinting, `0` for a single initial imprint

For the rest it is expected that the lists contain as many elements as the simulations we want to run. Each element of the list (could be a list itself) is a separate configuration:

- `"vortex_charge"`: list of initial charges per simulation, e.g. `[[1], [1, -1]]`
- `"imprinting_charge"`: charges for subsequent imprints, e.g. `[[1], [-1]]`
- `"vortex_position_x"`: see coordinate-system section above for the meaning
- `"vortex_position_y"`: see coordinate-system section above for the meaning
- `"initial_imprint_time"`: snapshot index at which the initial vortices are imprinted
- `"imprint_position_x"`: positions for re-imprinted vortices (same convention as `vortex_position_x`)
- `"imprint_position_y"`: positions for re-imprinted vortices
- `"imprint_every"`: re-imprint interval in snapshots (used when `imprint_times` is empty)
- `"imprint_times"`: exact snapshot indices for each re-imprint; use `[]` to auto-generate from `imprint_every`
- `"max_imprints"`: maximum number of re-imprints per simulation

The above configuration would give 2 simulations.

**NOTE: When we give "imprint_times", those override the "imprint_every" parameter. We need to give an empty list for the "imprint_times" if we want them to be automatically calculated based on the "imprint_every" parameter**

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

_First simulation_: start with a vortex of charge 2 at `(0, 0)`. Re-imprint charge 1 at the exact
snapshots 10, 20, 50 (`imprint_times` overrides `imprint_every`).

_Second simulation_: start with a vortex of charge 5 at `(0, 0)`. Auto-generate re-imprint times
from `imprint_every = 50` → times 50, 100, 150; but `max_imprints = 2` so only snapshots 50 and
100 are used.

### Example 2: multi-vortex, single imprint

```json
{
  "vortex_charge": [[5, 5], [1, 1, 1, 1, 1, 1, 1]],
  "imprinting_charge": [[[−5, −5]], [[0, 0, 0, 0, 0, 0, 0]]],
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

_First simulation_: imprint two `+5` vortices at `(0, -20)` and `(0, +20)` at snapshot 4.
Re-imprint two `-5` vortices at the same positions at snapshot 50.

_Second simulation_: imprint 7 `+1` vortices at x = 0, distributed from y = -30 to y = +30.
No re-imprinting (`max_imprints = 0`).

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

A single charge-1 vortex placed on the symmetry axis (`r = 0`, `φ` irrelevant at r = 0).

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

Two vortices at the same radial distance `r = 5 a_ho`, placed at `φ = 0` and `φ = π`
(diametrically opposite). In Cartesian terms these are at `(+5, 0)` and `(−5, 0)`.

---

## Dark Solitons
Dark solitons are configured **per simulation**, exactly like vortices: each key
is a top-level list with **one entry per simulation**. The `"dark_soliton"` flag
is a single global on/off switch (mirroring `"vortex_excitation"`); the
per-simulation lists must all have the same length as the other per-simulation
lists (e.g. `vortex_charge`).

- `"dark_soliton": 1` — global flag enabling dark-soliton imprinting,
- `"soliton_positions"`: list (one entry per simulation); each entry is a list of
  soliton centre positions in microns, e.g. `[[0.0], [-10.0, 10.0]]` (sim 0 has a
  single soliton at 0, sim 1 has a pair at ±10). An empty list `[]` means that
  simulation imprints no soliton,
- `"soliton_widths"`: per-simulation list of widths (healing-length scale), e.g.
  `[[1.0], [0.8, 0.8]]`,
- `"soliton_axes"`: per-simulation list of axes (1 = x1, 3 = x3/z), e.g.
  `[[3], [3, 3]]`,
- `"soliton_greyness"`: (optional) per-simulation list in [0, π/2). `0` = fully
  dark, values > 0 = grey. Defaults to all zeros,
- `"soliton_imprint_time"`: per-simulation list of ints — snapshot index at which
  each simulation imprints its solitons, e.g. `[5, 10]`.

Within a simulation, `soliton_positions[i]`, `soliton_widths[i]` and
`soliton_axes[i]` must have the same length (one value per soliton).

Solitons may be combined with vortices (both flags on, lists of equal length per
simulation) or used on their own (`"vortex_excitation": 0`, `"dark_soliton": 1`),
in which case the number of simulations is taken from the soliton lists.

### Example (two simulations: a single black soliton, then a grey pair):
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

> Note: earlier versions documented a nested `"dark_soliton_params"` dictionary
> shared across all simulations. That form is no longer used — solitons are now
> per-simulation flat lists as shown above.

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
| `"rotating"` | ✅ | ✅ | Rotating trap for stirring. Cartesian rotates the harmonic trap about a configurable axis; cylindrical rotates an anisotropic trap at angular frequency Ω with ellipticity ε |
| `"gaussianbeam"` | — | ✅ | Focused Gaussian laser beam along z: a repulsive obstacle, stirring beam or barrier |
| `"custom"` | ✅ | — | No potential is built; construct a `CustomPot` yourself and assign its shape and time dependence |

All of them accept the optional absorber (CAP) keys described above.
