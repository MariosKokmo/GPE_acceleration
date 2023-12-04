# GPE_acceleration
GPU accelerated code for the implementation of a GPE solver.
This package contains a simple Python implementation that accelerates the usual GPE split-step Fourier solution utilising GPUs and the PyTorch software package.

The software simulates the evolution of a BEC when a topological excitation is imprinted in the condensate.
Currently the only topological excitation supported is vortices. The vortices can either be in a linear array or isolated.
A repetitive imprinting functionality is supported.

## Run
To run the code, you simply run the `run.py` script.

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

## API Reference

### `run.py`
This is the entry point to the software and the simulations. This script is called and accepts all the necessary inputs (see configuration_file). Then everything runs in an automated fashion.

### `library.ground_state`
The ground state is calculated numerically using the imaginary time method.


For the ground state calculations, pyTorch 1.8.1 has been tested.
With later versions, when doing the conjugations of matrices, the functions might have changed.
Specifically, in later versions the `torch.conj()` performs a lazy conjugation and has to be realised
in a tensor separately. In PyTorch version 1.8.1, the one currently used it is calculated directly.
The rest of the used functions should be the same.

### `library.gpe_library`
Provides all the necessary functions to run the simulations, load/write the data, imprint the vortices and the split-step Fourier methods.

### `library.gpe_evolution`
Provides the functions that run the main loop of the simulations. Simulation parameters are needed as inputs.

### `utils.setup_simulations`
This script is responsible for reading the configuration file, performing checks on the validity of its contents and creating the simulation parameters for each simulation that will be run.

### `utils.video_creation`
Provides the utility functions to create the video from the snapshot data files

### configuration_file
One configuration file is needed for each grid and/or potential configuration. Multiple simulations can be run using this configuration file.
 
 **Important** : Currently only one configuration file can exist in a working directory. If different grids need to be run at the same time, then more python processes are needed that will run in different directories (i.e. copy the code in different folders and run it there)

 The configuration file can be built for either an array of vortices or vortices that will be repetitively imprinted.

 Checks on the inputs of the configuration file will be performed before setting up the simulations. 