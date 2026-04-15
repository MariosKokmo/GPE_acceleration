
## Features
**Potentials**

**Repetitive imprinting**

## Run
To run the code, you simply run the `run.py` script. This script invokes any necessary function and set-up of the simulation.

The simulations are defined in a json file called 'configuration_file.json'.

**Important** -- Each configuration file is strictly for one grid and potential configuration. Multiple simulations can be run in sequence but all have to be on the same grid and potential with the same ground state.

First the ground state for the specific grid is calculated (if it doesn't already exist). Then every simulation is run one after another and the results are stored in their respective folders.

The flow of the logic is as follows:
<img src="static/flow.jpg">

## Dependencies
the main package dependencies are:
- numpy:  1.23.5
- pandas:  1.5.3
- torch:  1.8.1
- matplotlib:  3.7.1

The software should run for Python version >=3.7

### configuration_file
One configuration file is needed for each grid and/or potential configuration. Multiple simulations can be run using this configuration file.
 
 **Important** : Currently only one configuration file can exist in a working directory. If different grids need to be run at the same time, then more python processes are needed that will run in different directories (i.e. copy the code in different folders and run it there)

 The configuration file can be built for either an array of vortices or vortices that will be repetitively imprinted.

 Checks on the inputs of the configuration file will be performed before setting up the simulations.

 # HOW TO CREATE THE CONFIGURATION FILE
 **Important** -- Each configuration file is strictly for one grid and potential configuration. Multiple simulations can be run in sequence but all have to be on the same grid and potential with the same ground state.

- "Grid_positive_limits": These are the 3D grid axes in microns e.g. [60, 1.5, 60],
- "Grid_negative_limits": Usually the grid is symmetric and the potential is considered centered e.g. [-60, -1.5, -60],
- "Grid_resolution": This is the number of points along each one of the axes e.g.[512, 16, 512] for a flat BEC,
- "Trapping_frequencies": These are the frequencies in Hz [20, 300, 20],
- "Potential_type":e.g. "harmonic" For a full list see below,
- "Absorber_enabled": optional boolean (0/1 or false/true). Enables a boundary complex absorber,
- "Absorber_strength": optional non-negative float. Damping strength at the grid boundary,
- "Absorber_start_ratio": optional float in [0, 1). Fraction of half-box size where damping starts (e.g. 0.8),
- "Absorber_power": optional float >= 1. Smoothness/order of the absorber ramp (e.g. 2),
- "Absorber_tinit": optional float. Start time (dimensionless simulation time) for turning on the absorber,
- "Absorber_tfinal": optional float. End time for linear absorber ramp-up. If omitted, absorber turns on at Absorber_tinit,
- "Total_simulation_time":150e-3, in sec,
- "dt":5e-7, the time resolution of the simulation in sec,
- "snapshots":150, the number of snapshot data to be generated. Each file can be several Mb. For more intuitive results set it the same as the total time so that each snapshot is 1ms,
- "vortex_excitation":1, This is used as a boolean that indicates the topological excitation we are simulating,
- "repetitive":1, Indicates that repetitive imprinting is used,

For the rest it is expected that the lists contain as many elements as the simulations we want to run. Each element of the list (could be a list itself) is a separate configuration
- "vortex_charge":[2, 5], List of initial charges,
- "imprinting_charge":[1, 3],
- "vortex_position_x":[0, 0],
- "vortex_position_y":[0, 0],
- "imprint_position_x":[0, 0],
- "imprint_position_y":[0, 0],
- "imprint_every":[20, 50],
- "imprint_times":[[10,20,50], []],
- "max_imprints":[3, 2]

The above configuration would give 2 simulations.

**NOTE: When we give "imprint_times", those override the "imprint_every" parameter. We need to give an empty list for the "imprint_times" if we want them to be automatically calculated based on the "imprint_every" parameter**

_First simulation_:

We start with an initial vortex of charge 2 at position (0,0). We then want to imprint a vortex of charge 1 at position (0,0). Because we give exact times to imprint, this overrides the "imprint_every" parameter which is 20, and the imprints will happen at the exact times 10, 20 and 50. Also note that the "max_imprints" are 3 for this simulation so all the imprints will happen.

_Second simulation_:

We start with an initial vortex of charge 5 at position (0,0). We then imprint a vortex of charge 3 at position (0,0). Now we don't give exact times to imprint, so the "imprint_every" value of 50 will create the times. The imprints times will be 50, 100, 150. However, because the "max_imprints" are 2, only the imprints at times 50 and 100 will occur. Note that even that we give an "imprint_every" value, we have to give an empty list for the "imprint_times" otherwise an error will be raised. The software needs to be able to align the inputs at the same length.

## Second example
- "vortex_charge":[[5,5],[1,1,1,1,1,1,1]],
- "imprinting_charge":[[[-5,-5]],[[0,0,0,0,0,0,0]]],
- "vortex_position_x":[[0,0],[0,0,0,0,0,0,0]],
- "vortex_position_y":[[-20,20],[-30,-20,-10,0,10,20,30]],
- "initial_imprint_time":[4,4],
- "imprint_position_x":[[[0,0]],[[0,0,0,0,0,0,0]]],
- "imprint_position_y":[[[-20,20]],[[0,0,0,0,0,0,0]]],
- "imprint_every":[[],[]],
- "imprint_times":[[50],[50]],
- "max_imprints":[1,0]

This would also give two simulations.
_First simulation_:
we imprint two vortices of charge +5 each at positions (0,-20) and (0,20) at time 4.
We thin imprint again at time 50, at the same positions two vortices of charges -5.

_Second simulation_:
We imprint 7 vortices of charge +1 each, all with an x coordinate equal to 0, and linearly allocated between
y=-30 and y=+30. We perform no imprints, note that the max_imprints variable is set to 0 for this simulation.

## Dark Solitons
Dark soliton imprinting is configured per-simulation inside the simulation block. To enable dark solitons, add the following keys:

- `"dark_soliton": 1` — boolean flag enabling dark soliton imprinting,
- `"dark_soliton_params"` — a dictionary with the soliton parameters:
  - `"soliton_positions"`: list of floats — positions (in microns) along the chosen axis for each soliton, e.g. `[0.0]` or `[-10.0, 10.0]`,
  - `"soliton_widths"`: list of floats — width (healing-length scale) for each soliton, e.g. `[1.0]`,
  - `"soliton_axes"`: list of ints (1 or 3) — axis each soliton is perpendicular to. `1` = x1 axis, `3` = x3/z axis,
  - `"soliton_greyness"`: (optional) list of floats in [0, π/2). `0` = fully dark (black soliton), values > 0 = grey soliton. Defaults to all zeros,
  - `"soliton_imprint_time"`: int — snapshot index at which to imprint the solitons (e.g. `5` means after the 5th snapshot).

### Example (single black soliton along z at z = 0):
```json
{
  "dark_soliton": 1,
  "dark_soliton_params": {
    "soliton_positions": [0.0],
    "soliton_widths": [1.0],
    "soliton_axes": [3],
    "soliton_greyness": [0.0],
    "soliton_imprint_time": 5
  }
}
```

### Example (two grey solitons):
```json
{
  "dark_soliton": 1,
  "dark_soliton_params": {
    "soliton_positions": [-10.0, 10.0],
    "soliton_widths": [0.8, 0.8],
    "soliton_axes": [3, 3],
    "soliton_greyness": [0.3, 0.3],
    "soliton_imprint_time": 10
  }
}
```

List of available potentials:
- **harmonic** , a harmonic potential with 3 trapping frequencies
- **constant**, a constant potential across the whole grid
- **ramp**, potential that ramps up from an initial to a final amplitude in a linear fashion
- **rampharmonic**, harmonic potential whose amplitude ramps up linearly in time
