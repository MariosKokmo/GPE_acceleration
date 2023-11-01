# GPE_acceleration
GPU accelerated code for the implementation of a GPE solver.
This package contains a simple Python implementation that accelerates the usual GPE split-step Fourier solution utilising GPUs and the PyTorch software package.

## Run
To run the code, you simply run the `run.py` script.

The simulations are defined in a json file called 'configuration_file.json'.

**Important** -- Each configuration file is strictly for one grid and potential configuration. Multiple simulations can be run but all have to be on the same grid and potential with the same ground state.

## Dependencies
the main package dependencies are:
- numpy:  1.23.5
- pandas:  1.5.3
- torch:  2.1.0+cu118
- matplotlib:  3.7.1

The software should run for Python version >=3.8 
