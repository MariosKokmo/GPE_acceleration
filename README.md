# GPE Acceleration / BAQS

> **GPU-Accelerated Gross-Pitaevskii Equation Solver for Bose-Einstein Condensates**

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

**GPE Acceleration** (internal package name `baqs`) is a high-performance Python package designed to simulate the dynamics of Bose-Einstein Condensates (BECs) by solving the Gross-Pitaevskii Equation (GPE).

Built on **PyTorch**, this software leverages **GPU acceleration** to perform rapid Split-Step Fourier Method simulations. It is specifically tailored for studying topological excitations, such as **vortices**, allowing complex imprinting scenarios like linear arrays, isolated vortices, and repetitive imprinting protocols.

The current implementation focuses on solutions using the phase imprinting method on quasi-2D condensates. However, the
use of evolving potentials and the extensibility and configurability of the software make it useful for more general topological simulations in BEC even in 3D and with the classic methods of rotating potentials.

The software simulates the evolution of a BEC when topological excitations are imprinted in the condensate.
Current supported excitations are **vortices** and **dark solitons**. Repetitive imprinting is supported for vortex scenarios.

The vortices are assumed to be printed on the n1-n3 (i.e. x-z plane). For the simulations to have physical meaning, it is assumed that the BEC is adequately flat on the n2 (y axis) so that the vortices are assumed to not bend and traverse the whole BEC along the y axis.

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [CI/CD](#cicd)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Key Features

*   **🚀 GPU Acceleration**: Utilizes PyTorch and CUDA for massively parallelized computations on the GPU.
*   **🌀 Vortex Imprinting**: specialized tools for imprinting phase Singularities (vortices) with configurable charge, position, and timing.
*   **🌊 Dark Solitons**: Supports black and grey soliton imprinting with configurable position, width, axis, and imprint time.
*   **⚡ Automated Ground State**: Automatically calculates the ground state using Imaginary Time Evolution if it doesn't exist.
*   **🔄 Repetitive & Array Configurations**: Support for dynamic simulation scenarios defined via simple JSON arrays.
*   **🧭 Rotating Potentials**: Includes a rotating harmonic potential with configurable rotation axis and angular frequency.
*   **🧱 Dissipative Extensions**: Optional three-body loss and boundary absorber (CAP) support.
*   **📐 3D Utilities**: Includes helpers for vortex rings/lines, 3D velocity, angular momentum, and density slices.
*   **📹 Visualization**: Integrated utilities for generating videos from simulation snapshots.
*   **📊 Flexible Configuration**: Complete control over grid resolution, trapping potentials, and simulation physics via JSON.

## Installation

### Prerequisites
*   Python 3.7+
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

## Project Structure

```
GPE_acceleration/
├── src/
│   ├── application.py       # App configuration & logging
│   ├── run.py               # Legacy entry point script
│   ├── simulator.py         # Simulator orchestration
│   ├── cli/                 # Command Line Interface
│   │   ├── baqs.py          # CLI entry point
│   │   └── functions.py     # CLI helper functions
│   ├── library/             # Core Physics Library
│   │   ├── gpe_library.py   # Split-Step Fourier implementation
│   │   ├── ground_state.py  # Imaginary time evolution
│   │   ├── parameters.py    # Simulation parameter handling
│   │   └── potentials.py    # Potential definitions (incl. rotating and absorber support)
│   ├── models/              # Simulation Data Structures
│   │   ├── BEC.py           # Bose-Einstein Condensate object
│   │   ├── base_BEC.py      # Abstract base class for BECs
│   │   ├── system.py        # System wrapper (Grid + Potential)
│   │   └── simulation.py    # Simulation logic wrapper
│   └── utils/
│       ├── read_write_utils.py # I/O operations
│       ├── setup_simulations.py # Simulation configuration setup
│       └── video_creation.py   # Visualization tools
├── tests/                   # Unit tests
├── .pipelines/              # CI/CD YAML pipelines
├── .environments/           # Environment-specific pipeline variables
├── configuration_file.json  # Physics config
├── appConfig.json           # App settings
└── requirements.txt         # Python dependencies
```

## How It Works

1.  **Ground State**: The solver first checks if a ground state exists for the given grid and potential. If not, it computes it using **Imaginary Time Propagation (ITP)**.
2.  **Time Evolution**: The real-time dynamics are solved using the **Split-Step Fourier Method**.
3.  **Vortex Imprinting**: Phase imprinting is applied to the wavefunction at specified time steps to create vortices.
4.  **Hardware**: All dense matrix operations (FFTs, element-wise multiplications) are offloaded to the GPU via PyTorch tensors.

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
