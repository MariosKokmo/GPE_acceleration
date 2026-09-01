# GPE Acceleration / BAQS

> **GPU-Accelerated Gross-Pitaevskii Equation Solver for Bose-Einstein Condensates**

![Python](https://img.shields.io/badge/Python-3.9%E2%80%933.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%E2%80%932.4%20%7C%20CUDA-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

**GPE Acceleration** (internal package name `baqs`) is a high-performance Python package designed to simulate the dynamics of Bose-Einstein Condensates (BECs) by solving the Gross-Pitaevskii Equation (GPE).

Built on **PyTorch**, this software leverages **GPU acceleration** to perform rapid Split-Step Fourier Method simulations. It is specifically tailored for studying topological excitations, such as **vortices**, allowing complex imprinting scenarios like linear arrays, isolated vortices, and repetitive imprinting protocols.

The current implementation focuses on solutions using the phase imprinting method on quasi-2D condensates. However, the
use of evolving potentials and the extensibility and configurability of the software make it useful for more general topological simulations in BEC even in 3D and with the classic methods of rotating potentials.

The software simulates the evolution of a BEC when topological excitations are imprinted in the condensate.
Current supported excitations are **vortices** and **dark solitons**. Repetitive imprinting is supported for vortex scenarios.

The vortices are assumed to be printed on the n1-n3 (i.e. x-z plane). For the simulations to have physical meaning, it is assumed that the BEC is adequately flat on the n2 (y axis) so that the vortices are assumed to not bend and traverse the whole BEC along the y axis.

Beyond zero-temperature GPE dynamics, the package includes two **finite-temperature models** under `src/experimental/`:

- **SGPE** (`src/models/finite_temp_BEC.py`) — Stochastic Projected GPE: adds damping and thermal noise to the condensate wavefunction equation, suitable for near-equilibrium finite-temperature dynamics.
- **ZNG** (`src/experimental/zng/`) — Zaremba-Nikuni-Griffin framework: couples the condensate (modified GPE) to an explicit thermal cloud represented by Monte Carlo test particles, giving access to the full two-component dynamics.

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Finite-Temperature Models](#finite-temperature-models)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [CI/CD](#cicd)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Key Features

*   **🚀 GPU Acceleration**: Utilizes PyTorch and CUDA for massively parallelized computations on the GPU.
*   **🌀 Vortex Imprinting**: Specialized tools for imprinting phase singularities (vortices) with configurable charge, position, and timing.
*   **🌊 Dark Solitons**: Supports black and grey soliton imprinting with configurable position, width, axis, and imprint time.
*   **⚡ Automated Ground State**: Automatically calculates the ground state using Imaginary Time Evolution if it doesn't exist.
*   **🔄 Repetitive & Array Configurations**: Support for dynamic simulation scenarios defined via simple JSON arrays.
*   **🧭 Rotating Potentials**: Includes a rotating harmonic potential with configurable rotation axis and angular frequency.
*   **🧱 Dissipative Extensions**: Optional three-body loss and boundary absorber (CAP) support.
*   **📐 3D Utilities**: Includes helpers for vortex rings/lines, 3D velocity, angular momentum, and density slices.
*   **🌡️ Finite-Temperature (SGPE)**: Stochastic Projected GPE with damping and thermal noise for near-equilibrium finite-T dynamics.
*   **🔬 Finite-Temperature (ZNG)**: Full two-component Zaremba-Nikuni-Griffin framework with Monte Carlo thermal cloud and stochastic C₁₂ collisions.
*   **📹 Visualization**: Integrated utilities for generating videos from simulation snapshots.
*   **📊 Flexible Configuration**: Complete control over grid resolution, trapping potentials, and simulation physics via JSON.

## Installation

### Prerequisites
*   Python 3.9–3.12
*   NVIDIA GPU with CUDA support (strongly recommended for performance)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/GPE_acceleration.git
    cd GPE_acceleration
    ```

2.  **Install the package:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    
    # Install the requirements
    pip install -r requirements.txt
    ```
    or

    ```bash
    # Install in editable mode to enable the 'baqs' CLI command
    pip install -e .
    ```

    *Note: This will automatically install all dependencies listed in `requirements.txt` (and `setup.py`). Ensure you have the correct version of PyTorch installed for your CUDA version. See [PyTorch Get Started](https://pytorch.org/get-started/locally/).*

## Quick Start

### Using the Script

The easiest way to run a simulation is using the provided `run.py` script.

1.  **Configure your simulation** by editing `configuration_file.json`.
2.  **Run the solver:**
    ```bash
    python src/run.py
    ```

### Using the Simulator API (object-oriented)

`Simulator` is the recommended way to drive simulations programmatically — from a notebook, a parameter sweep, or any Python script — without touching the CLI.

```python
from src.simulator import Simulator

sim = Simulator("configuration_file.json")

# Inspect before running
print(sim.system.simulation_parameters)
print(sim.system.uext)

# Run all simulation combinations defined in the config
sim.run()

# Inspect results after running
print(sim.simulations.BEC)
```

`run.py` is a thin wrapper around `Simulator`, so the CLI and the API always execute the same code path.

### Using the Command Line Interface (CLI)

For more control, you can use the CLI tool `baqs`.

**Installation:**
If you haven't already, install the package in editable mode:
```bash
pip install -e .
```

**Usage:**
```bash
baqs <config_file> <app_config_file> [options]
```

**Examples:**
*   **Check configuration only:**
    ```bash
    baqs configuration_file.json appConfig.json --check -v 1
    ```
*   **Run simulation:**
    ```bash
    baqs configuration_file.json appConfig.json --run -v 1
    ```

**Arguments:**
*   `config`: Path to the simulation configuration JSON.
*   `app`: Path to the application configuration JSON.
*   `-c, --check`: Validate configuration files without running.
*   `--run`: Execute the simulations (performs validation first).
*   `-v, --verbose`: Set output verbosity (0: Silent, 1: Info, 2: Debug).

The system will:
1.  Initialize the application based on `appConfig.json`.
2.  Load the physics parameters from `configuration_file.json`.
3.  Calculate (or load) the ground state.
4.  Run the time-evolution simulations defined in the config.
5.  Save results (density snapshots, metadata) to the output directory.

## Configuration

The simulations are controlled by two JSON files.

### 1. Simulation Config (`configuration_file.json`)
This file defines the physical system (Grid, Potential) and the specific scenarios to simulate.

**Important**: One config file corresponds to one Grid/Potential setup. Lists in the parameters below allow you to run multiple *evolution scenarios* sequentially on that same grid.

**Global Parameters (Grid & Physics):**
*   `Grid_[positive/negative]_limits`: Spatial extent of the simulation box (microns).
*   `Grid_resolution`: Number of grid points `[Nx, Ny, Nz]`.
*   `Trapping_frequencies`: Harmonic trap frequencies `[fx, fy, fz]` (Hz).
*   `Potential_type`: Type of external potential (`"harmonic"`, `"constant"`, `"ramp"`, `"rampharmonic"`, `"rotating"`).
*   `dt`: Time step (seconds).
*   `Total_simulation_time`: Duration of simulation (seconds).
*   `three-body-losses`: Enable/disable three-body loss term.

**Optional absorber (CAP) parameters:**
*   `Absorber_enabled`
*   `Absorber_strength`
*   `Absorber_start_ratio`
*   `Absorber_power`
*   `Absorber_tinit`
*   `Absorber_tfinal`

**Scenario Parameters (Lists for multiple runs):**
*   `vortex_charge`: List of vortex charges to imprint.
*   `vortex_position_[x/y]`: Initial positions of vortices.
*   `imprint_times`: Specific times to imprint new vortices.
*   `repetitive`: Boolean flag for repetitive imprinting modes.

**Optional dark-soliton parameters:**
*   `dark_soliton`
*   `soliton_positions`
*   `soliton_widths`
*   `soliton_axes`
*   `soliton_greyness`
*   `soliton_imprint_time`

For a detailed and complete configuration reference, see `HOW_TO_CONFIG_FILE.md`.

**Example:**
```json
{
    "Grid_positive_limits": [60, 1.5, 60],
    "Trapping_frequencies": [20, 300, 20],
    "Grid_resolution": [512, 16, 512],
    "Total_simulation_time": 0.15,
    "dt": 5e-7,
    "vortex_charge": [[1], [2]], 
    "vortex_position_x": [[0], [10]]
}
```
*This example configures a grid and runs 2 separate simulations: one with a charge-1 vortex at x=0, and another with a charge-2 vortex at x=10.*

### 2. Application Config (`appConfig.json`)
Controls application-level settings.
```json
{
    "logfile": "simulation.log",
    "configFile": "configuration_file.json",
    "write_velocity": false,   // Output velocity field data
    "phase_imaging": false     // generate phase images
}
```

To check the configuration file locally, you can run
```python
python -m src.cli.baqs config_test.json appConfig.json -c -v 2
```

## Finite-Temperature Models

Both finite-temperature models inherit from `BaseBEC`, so all existing I/O,
ground-state finding, snapshotting, and energy measurements work unchanged.
All quantities use **dimensionless units**: ħ = m = ω_ho = 1, so temperatures
are expressed as k_B T / (ħ ω_ho).

---

### SGPE — Stochastic Projected GPE

**File:** `src/models/finite_temp_BEC.py` · **Class:** `FiniteTempBEC`

The SGPE modifies the GPE to include coupling to a thermal reservoir through two extra terms:

```
∂ψ/∂t = −(i + γ)(H_mf − μ)ψ + η(r, t)
```

| Symbol | Meaning |
|--------|---------|
| γ | Dimensionless damping coefficient (energy exchange rate with reservoir) |
| μ | Reservoir chemical potential — computed from the ground state if not supplied |
| η(r,t) | Complex Gaussian noise satisfying ⟨η\*(r,t) η(r′,t′)⟩ = 2γ kT δ(r−r′) δ(t−t′) |

**Physics:** modes with H_mf > μ are damped (energy removed); modes with H_mf < μ are amplified (energy drawn from reservoir). Together with η, this drives the system to a thermal equilibrium state at temperature T.

Setting `temperature = 0` recovers a purely dissipative (damped) GPE, useful as an alternative ground-state finder.

**New config keys:**

```json
"temperature": 1.5,
"damping_coefficient": 0.03,
"chemical_potential": null
```

**Usage:**

```python
from src.experimental.zng.zng_BEC import ZNGBEC
# or for SGPE:
from src.models.finite_temp_BEC import FiniteTempBEC

# Swap into simulation.py line 59, or instantiate directly:
bec = FiniteTempBEC(parameters, system, app, simulation_name)
bec.evolve()
```

**New library functions** (added to `GPELibrary` in `src/library/gpe_library.py`):

| Function | Purpose |
|----------|---------|
| `sgpe_step` | Split-step with `exp(−(i+γ)·dt·(H_mf−μ))` — damps/amplifies modes relative to μ |
| `generate_thermal_noise` | Complex Gaussian field with amplitude √(γ kT Δτ / δV) |
| `calculate_chemical_potential` | μ = e_kin + e_pot + 2·e_int from the current wavefunction |

---

### ZNG — Zaremba-Nikuni-Griffin Framework

**Directory:** `src/experimental/zng/` · **Class:** `ZNGBEC`

ZNG is a two-component model that explicitly tracks both the condensate and the thermal cloud:

**Condensate (modified GPE):**
```
i ∂ψ/∂t = [H_GP + i R(r,t)/2] ψ
H_GP = −∇²/2 + V_ext + u(n_c + 2ñ)
R(r) = 2 γ_12 [μ − V_eff(r)]
```

**Thermal cloud (N_test classical test particles):**
```
dr_i/dt = p_i
dp_i/dt = −∇U(r_i),    U = V_ext + 2u(n_c + ñ)
```

The condensate and thermal cloud are coupled at every time step through their shared densities n_c = |ψ|² and ñ(r) (deposited from particles onto the grid via CIC interpolation).

**Per-step loop:**

1. Build condensate GPE potential V_GP = V_ext + u(n_c + 2ñ)
2. Compute source term R from C₁₂ mean-field approximation
3. Evolve ψ one split-step with complex potential V_GP + iR/2
4. Advance test particles one leapfrog step under U
5. Apply C₁₂ stochastic collisions (absorption + emission)
6. Apply C₂₂ thermal–thermal scattering (stub — see below)
7. Deposit updated particle positions → new ñ

**New config keys:**

```json
"temperature": 1.5,
"n_test_particles": 10000,
"gamma_12": 0.1,
"chemical_potential": null,
"enable_c22": false
```

**File layout:**

| File | Contents |
|------|---------|
| `zng_library.py` | Stateless physics functions: CIC deposition, spectral gradient, mean-field potentials, R term, initial-cloud sampling |
| `monte_carlo.py` | Particle dynamics: leapfrog integrator, C₁₂ stochastic collisions, C₂₂ stub |
| `zng_BEC.py` | `ZNGBEC(BaseBEC)` — wires the above into the simulation loop |

**C₂₂ note:** Thermal–thermal scattering is implemented as a documented stub in `monte_carlo.apply_c22_collisions`. The function signature, docstring, and DSMC/BGK implementation options are in place; the body returns inputs unchanged until a full collision algorithm is added.

---

### Choosing a model

| Scenario | Recommended model |
|----------|------------------|
| Zero temperature, topological excitations | `BEC` (default) |
| Near-equilibrium finite T, single wavefunction | `FiniteTempBEC` (SGPE) |
| Explicit thermal cloud dynamics, two-component | `ZNGBEC` (ZNG) |
| Alternative ground-state search | `FiniteTempBEC` with T=0, γ>0 |

## Project Structure

```
GPE_acceleration/
├── src/
│   ├── application.py         # App configuration & logging
│   ├── run.py                 # CLI entry point (delegates to Simulator)
│   ├── simulator.py           # Simulator class — OO API for programmatic use
│   ├── cli/                   # Command Line Interface
│   │   ├── baqs.py            # CLI entry point
│   │   └── functions.py       # CLI helper functions
│   ├── library/               # Core Physics Library
│   │   ├── gpe_library.py     # Split-Step Fourier, energy, SGPE functions
│   │   ├── ground_state.py    # Imaginary time evolution
│   │   ├── parameters.py      # Physical constants
│   │   └── potentials.py      # Potential definitions (harmonic, rotating, absorber)
│   ├── models/                # Simulation models
│   │   ├── base_BEC.py        # Abstract base class — ground state, I/O, loop template
│   │   ├── BEC.py             # Zero-temperature GPE with vortex/soliton imprinting
│   │   ├── finite_temp_BEC.py # SGPE finite-temperature model
│   │   ├── system.py          # System wrapper (grid + potential)
│   │   └── simulation.py      # Runs a list of simulation combinations
│   ├── experimental/          # Research-stage models (not in main pipeline)
│   │   └── zng/               # Zaremba-Nikuni-Griffin two-component framework
│   │       ├── zng_library.py # Stateless physics: CIC deposition, potentials, R term
│   │       ├── monte_carlo.py # Leapfrog integrator, C₁₂ collisions, C₂₂ stub
│   │       └── zng_BEC.py     # ZNGBEC model class
│   └── utils/
│       ├── read_write_utils.py  # I/O operations
│       ├── setup_simulations.py # Config loading and validation
│       └── video_creation.py    # Visualization tools
├── tests/                     # Unit tests
├── .pipelines/                # CI/CD YAML pipelines
├── .environments/             # Environment-specific pipeline variables
├── configuration_file.json    # Physics config
├── appConfig.json             # App settings
└── requirements.txt           # Python dependencies
```

## How It Works

### Zero-temperature GPE (`BEC`)

1.  **Ground State**: Checks for a cached ground-state file; computes it via **Imaginary Time Propagation (ITP)** if absent.
2.  **Time Evolution**: Real-time dynamics solved with the **Split-Step Fourier Method** — alternating half-steps in real space (potential) and full steps in momentum space (kinetic).
3.  **Vortex / Soliton Imprinting**: Phase or amplitude masks are applied to ψ at specified snapshot indices.
4.  **Hardware**: All FFTs and element-wise operations run on GPU via PyTorch tensors.

### Finite temperature — SGPE (`FiniteTempBEC`)

The SGPE replaces the standard split-step propagator with a damped one and adds stochastic noise at each step:

- **Damped propagator**: `exp(−(i+γ)·dt·(H_mf−μ))` — the (i+γ) factor damps modes above μ and amplifies modes below μ, thermalising the condensate.
- **Thermal noise**: a complex Gaussian field η added after each step, with variance 2γ kT Δτ/δV (fluctuation-dissipation theorem).
- **Chemical potential** μ is computed once from the ground-state wavefunction at t=0 and held fixed, representing the thermal reservoir.

### Finite temperature — ZNG (`ZNGBEC`)

ZNG tracks two coupled components at every time step:

- **Condensate**: evolved under a modified GPE `i∂ψ/∂t = [H_GP + iR/2]ψ` where R is the C₁₂ source term and H_GP includes the thermal back-reaction `2uñ`.
- **Thermal cloud**: N_test classical test particles advanced by a leapfrog integrator under the Hartree-Fock mean-field potential `U = V_ext + 2u(n_c + ñ)`.
- **C₁₂ coupling**: stochastic absorption (low-energy thermal particles join the condensate) and emission (condensate atoms scatter into the thermal cloud at the Thomas-Fermi edge).
- **C₂₂**: thermal–thermal scattering stub — returns particles unchanged until a DSMC or BGK implementation is added.
- **Grid ↔ particles**: Cloud-In-Cell (CIC) trilinear interpolation deposits particles onto the grid for ñ and interpolates grid forces back to particles each step.

## CI/CD

This repository includes starter Azure Pipelines files:

*   `.pipelines/ci.yaml`: Validation, unit tests, and package build.
*   `.pipelines/cd.yaml`: Build + multi-stage deployment flow (dev, staging, production).
*   `.environments/dev.yaml`, `.environments/staging.yaml`, `.environments/prod.yaml`: Environment variables/templates used by the pipelines.

Before enabling CD, replace the placeholder `deployCommand` in each environment file with your real deployment command.

## Testing

Ensuring the reliability of the simulation code is critical. **Users should always test their new features** or modifications before deploying them to ensure physical accuracy and software stability.

The project includes a suite of unit tests located in the `tests/` directory, covering everything from CLI arguments to core physics calculations.

### Running Tests

You can run the tests using Python's built-in `unittest` framework or `pytest` (recommended for better output).

**Using `unittest`:**
```bash
python -m unittest discover tests
```

**Using `pytest`:**
First, install pytest (if not already installed):
```bash
pip install pytest
```
Then run the test suite:
```bash
pytest
```

### Test Suite Structure

*   `test_baqs.py`: Tests the Command Line Interface (CLI) functionality.
*   `test_BEC.py`: Tests the main `BEC` class and high-level simulation logic.
*   `test_gpe_library.py`: Validates core mathematical functions, split-step Fourier utilities, dark solitons, and 3D helper APIs.
*   `test_potentials.py`: Verifies potential generation (harmonic/ramp/constant), plus absorber and related behavior.
*   `test_setup_simulations.py`: Checks the configuration loading and validation logic.

When contributing or adding new features, please ensure you add corresponding test cases to maintain coverage.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the project.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.
