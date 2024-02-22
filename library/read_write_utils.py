"""This module provides the utility functions to read and write data in files"""
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from .gpe_library import calculate_velocity2D

def write_psi(file_name, psi, n1, n2, n3):
    """
    Writes the wavefunction of the condensate to a file.
    Uaually used for the ground state.

    Args:
      file_name: str, the name of the file to be created
      psi: torch.Tensor, the wavefunction of the condensate
      n1, n2, n3: integer, the grid points in the 3 dimensions
    """
    with open(file_name, 'w') as f:
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    f.write(f'({psi[i,j,k].real},{psi[i,j,k].imag})\n')

def write_data(psi1, count, x1, x3, n1, n3, a_ho):
    """Writes the column density data on the x-z plane"""
    file_name = f'R-{count:003}-cd.dat'
    with open(file_name, 'w') as f:
        for i in range(n1):
            for k in range(n3):
                first = x1[i] * a_ho * 1e6 # x position
                second = x3[k] * a_ho * 1e6 # z position
                third = torch.sum(torch.abs(psi1[i,:,k])**2) # column density
                f.write(f'{first},{second},{third}\n')

def write_phase(phase, count, x1, x2, x3, n1, n2, n3, a_ho):
    """
    Writes the 3D phase in a file.
    """
    file_name = f'P-{count:003}-cd.dat'
    with open(file_name, 'w') as f:
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    first = x1[i] * a_ho * 1e6 # x position
                    second = x2[j] * a_ho * 1e6 # y position
                    third = x3[k] * a_ho * 1e6 # z position
                    fourth = phase[i,j,k]
                    f.write(f'{first},{second},{third},{fourth}\n')

def write_rms(rms_meas, SimulationName):
    """
    writes the RMS radius measurements for the BEC
    in a default file 'rms_meas.txt'
    """
    with open(f'{SimulationName}_RMS_meas.txt', 'w') as f:
        f.write("t\tr\n")
        for t, r in rms_meas.items():
            f.write(f"{t}\t{r}\n")

def write_phase2D(phase, count, x1, x3, n1, n2, n3, a_ho):
    """
    Writes the 2D phase in a file.
    It assumes the plane is the n1-n3
    and the central cross-section i.e. n2 is at its midpoint
    """
    j = n2//2
    file_name = f'P-{count:003}-cd.dat'
    with open(file_name, 'w') as f:
        for i in range(n1):
            for k in range(n3):
                first = x1[i] * a_ho * 1e6 # x position
                third = x3[k] * a_ho * 1e6 # z position
                fourth = phase[i,j,k]
                f.write(f'{first},{third},{fourth}\n')

def read_phase_file_2D(filename, n1, n3):
    """
    Reads a file that stores the phase of a 2D cross-section.
    It returns the phase as a tensor reshaped as n1 x n3.
    """
    phase = pd.read_csv(filename, header=None, names=['x1', 'x2', 'phase'])
    phase = phase.astype(np.float64)
    phase = phase.values
    phase = phase.reshape((n1, n3))
    phase = torch.from_numpy(phase)
    return phase

def write_velocity2D(phase, count, x1, x3, n1, n2, n3, a_ho, p_grid):
    """
    Writes the 2D velocity in a file.
    It assumes the plane is the n1-n3.
    The format of the file is `x1, x3, velocity magnitude, velocity phase`

    Args:
    -----
    phase: torch.Tensor, the 2D phase. Phase of a section
    count: int, the snapshot number
    x1, x3: torch.Tensor, the axes
    n1, n3: int, the grid resolution along x1 and x3
    a_ho: float, the harmonic oscillator length
    p_grid: Tuple[torch.Tensor], a tuple of tensors that stores
      the meshgrid of the momentum.

    Returns:
    --------
    None 
    """
    j = n2//2
    vel_file_name = f'V-{count:003}-cd.dat'
    velocity_mag, veloc_phase = calculate_velocity2D(phase, p_grid)
    with open(vel_file_name, 'w') as f:
        for i in range(n1):
            for k in range(n3):
                first = x1[i] * a_ho * 1e6 # x position
                second = x3[k] * a_ho * 1e6 # z position
                third = velocity_mag[i,j,k]
                fourth = veloc_phase[i,j,k]
                f.write(f'{first},{second},{third},{fourth}\n')

def save_figure_phase(phase, frame):
    """Saves a figure of the phase"""
    n1, n2, n3 = phase.shape
    if phase.dtype == torch.cdouble:
      plt.imshow((phase[:,n2//2,:].cpu().real),cmap='jet')
    else:
      plt.imshow((phase[:,n2//2,:].cpu()),cmap='jet')
    cb = plt.colorbar() 
    plt.title(f"Phase t = {frame}")
    plt.savefig(f"phase_t_{frame}.png")
    cb.remove()