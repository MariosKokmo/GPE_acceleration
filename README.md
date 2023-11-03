# GPE_acceleration
GPU accelerated code for the implementation of a GPE solver.
This package contains a simple Python implementation that accelerates the usual GPE split-step Fourier solution utilising GPUs and the PyTorch software package.
Currently the only topological excitation supported is vortices.

## Run
To run the code, you simply run the `run.py` script.

The simulations are defined in a json file called 'configuration_file.json'.

**Important** -- Each configuration file is strictly for one grid and potential configuration. Multiple simulations can be run in sequence but all have to be on the same grid and potential with the same ground state.

First the ground state for the specific grid is calculated (if it doesn't already exist). Then every simulation is run one after another and the results are stored in their respective folders.

## Dependencies
the main package dependencies are:
- numpy:  1.23.5
- pandas:  1.5.3
- torch:  1.8.1
- matplotlib:  3.7.1

The software should run for Python version >=3.7

### ground state
The ground state is calculated numerically using the imaginary time method.


For the ground state calculations, pyTorch 1.8.1 has been tested.
With later versions, when doing the conjugations of matrices, the functions might have changed.
Specifically, in later versions the `torch.conj()` performs a lazy conjugation and has to be realised
in a tensor separately. In PyTorch version 1.8.1, the one currently used it is calculated directly.
The rest of the used functions should be the same.

### configuration file
One configuration file is needed for each grid and/or potential configuration. Multiple simulations can be run using this configuration file.
 
 **Important** : Currently only one configuration file can exist in a working directory. If different grids need to be run at the same time, then more python processes are needed that will run in different directories (i.e. copy the code in different folders and run it there)