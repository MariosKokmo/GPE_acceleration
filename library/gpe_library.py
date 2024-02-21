# -*- coding: utf-8 -*-
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt


###############################################################################
###############   CONSTANTS DO NOT CHANGE UNLESS REQUIRED #####################
###############################################################################
class CONSTANTS():
  pi = 3.141592653589793238
  elec=1.60217653e-19
  hbar= 1.054572e-34            # J*s
  amu= 1.66053873e-27           # Kg (atomic mass unit)
  a_bohr= 0.5291772083e-10      # m
  m1=87*amu                 # Rubidium mass Kg
  m2=41*amu                 # Calcium mass Kg
  mass_ratio=m1/m2
  g=9.81
  ascat = 99*a_bohr	# scattering length
  nat = 5e+4		# number of atoms

##############################################################################
##############    UTILITY FUNCTIONS    #######################################
##############################################################################

def read_ground_state(data, n1, n2, n3):
    """
    Loads the ground state from a text file into a torch.tensor.

    Parameters
    ----------
    data : .dat file
        Contains the ground state of the GPE
        as has been calculated for the specific potential.

    Returns
    -------
    data as a numpy matrix.

    """
    matrix = pd.read_csv(data, header=None, names=['modulus', 'phase'])
    matrix.modulus = matrix.modulus.str.strip(' (')
    matrix.phase = matrix.phase.str.strip(' )')
    matrix = matrix.astype(np.float64)

    psi1 = matrix.iloc[:,0] + matrix.iloc[:,1]*1j
    psi1 = psi1.values
    psi1 = psi1.reshape((n1,n2,n3))
    psi1 = torch.from_numpy(psi1)

    return psi1

def init_grid(x_min, x_max, dx, dp, w, n1, n2, n3, device):
    """
    Initialises the grid for x and p spaces
    and the external potential

    Returns
    -------
    x1, x2, x3: torch.Tensor, the respective real space axis
    p1, p2, p3: torch.Tensor, the respective momentum axis
    p_sq: torch.Tensor, the squared momentum grid
    space_grid: torch.Tensor, the real space grid

    """
    ##############    EMPTY MATRICES TO FIT DATA    ##############################
    x1 = torch.zeros((1,n1), dtype=torch.float64, device=device)
    x2 = torch.zeros((1,n2), dtype=torch.float64, device=device)
    x3 = torch.zeros((1,n3), dtype=torch.float64, device=device)
    p1 = torch.zeros((1,n1), dtype=torch.float64, device=device)
    p2 = torch.zeros((1,n2), dtype=torch.float64, device=device)
    p3 = torch.zeros((1,n3), dtype=torch.float64, device=device)
    p_sq = torch.zeros((n1,n2,n3), dtype=torch.float64, device=device)

    # Build space and momentum grids
    x1 = x_min[0] + torch.arange(n1, dtype=torch.float64)*dx[0] # size n1
    p1[0][:n1//2] = dp[0] * torch.arange(n1//2)
    p1[0][n1//2:] = dp[0] * (torch.arange(n1//2, n1) - n1)

    x2 = x_min[1] + torch.arange(n2, dtype=torch.float64)*dx[1]
    p2[0][:n2//2] = dp[1] * torch.arange(n2//2)
    p2[0][n2//2:] = dp[1] * (torch.arange(n2//2, n2) - n2)

    x3 = x_min[2] + torch.arange(n3, dtype=torch.float64)*dx[2]
    p3[0][:n3//2] = dp[2] * torch.arange(n3//2)
    p3[0][n3//2:] = dp[2] * (torch.arange(n3//2, n3) - n3)

    # build the p-space squared. Useful for the FFT later
    g_px, g_py, g_pz = torch.meshgrid(p1[0], p2[0], p3[0])
    p_sq = g_px**2 + g_py**2 + g_pz**2
    p_sq = p_sq.to(device=device)

    # momentum grid
    p_grid = (g_px.to(device=device), g_py.to(device=device), g_pz.to(device=device))

    # Real space grid
    g_x, g_y, g_z = torch.meshgrid(x1, x2, x3)
    space_grid = (g_x.to(device=device), g_y.to(device=device), g_z.to(device=device))
    return x1, x2, x3, p1, p2, p3, p_sq, space_grid, p_grid

def imprint_vortices(vortices, phase, x1, x2, x3, n1, n2, n3, device):
    """
    Creates the vortices on the condensate by modifying the phase of the
    ground state. The vortices are on the xz (n1,n3) plane.
    Note that the vortices are not yet imprinted, a 
    phase update needs to occur through the `update_phase` function

    Parameters
    ----------
    vortices : numpy array
        dimensions : 3 x number_of_vortices
        1st row: x position
        2nd row: y position
        3rd row: vortex charges
    phase: torch.Tensor
        the current phase of the condensate to be modified
    Returns
    -------
    updated phase of the condensate.

    """
    if vortices is None:
      return
    number_of_vortices = vortices.shape[1]

    for n in range(number_of_vortices):
        vx = vortices[0][n]
        vz = vortices[1][n]
        q = vortices[2][n]
        for i in range(n3):
          for k in range(n1):
            if ((i != (vz + n3//2)) or (k >= (vx + n1//2))):
              y = x3[i]-x3[vz+n3//2]
              t = x1[k]-x1[vx+n1//2]
              x = torch.sqrt(t**2 + y**2) + t
              phase[k,:,i] += 2 * q * torch.atan2(y, x)
            else:
              phase[k,:,i] += q*CONSTANTS.pi

    phase[phase.isnan()] = 0+0j
    return phase

def create_additive_phase(vortices, x1, x2, x3, n1, n2, n3, device):
    """
    Creates the additive repetitive imprinting phase to be stored and used.
    Creates the phase in 2D on the x1-x3 plane.
    Returns:
    --------
      torch.Tensor, the repetitive imprinting phase
    """
    if vortices is None:
      return
    number_of_vortices = vortices.shape[1]
    repetive_phase = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=device)
    for n in range(number_of_vortices):
        vx = vortices[0][n]
        vz = vortices[1][n]
        q = vortices[2][n]
        for i in range(n3):
          for k in range(n1):
            if ((i != (vz + n3//2)) or (k >= (vx + n1//2))):
              y = x3[i]-x3[vz+n3//2]
              t = x1[k]-x1[vx+n1//2]
              x = torch.sqrt(t**2 + y**2) + t
              repetive_phase[k,:,i] += 2 * q * torch.atan2(y, x)
            else:
              repetive_phase[k,:,i] += q*CONSTANTS.pi

    repetive_phase[repetive_phase.isnan()] = 0+0j
    return repetive_phase


def x_evolution(psi1, utot1, dtau, factor=0.5):
    """
    The evolution step in real space.
    Parameters
    ----------
    psi1 : torch.Tensor
        The wavefunction of the system.
    utot1 : torch.Tensor
        The trapping potential in this time step
    dtau : float
        The time evolution step.
    factor : float
        The splitting factor of the split-step
    Returns
    -------
    psi1 : The updated wavefunction.
    """
    psi1 = torch.exp(-factor * dtau*(1j) * utot1) * psi1
    return psi1

def p_evolution(psi1, dtau, p_sq):
    """
    The evolution step in momentum space.   
    Parameters
    ----------
    psi1 : torch.Tensor
        The wavefunction of the system.
    dtau : float
        The time evolution step.
    p_sq : torch.Tensor
        The momentum space grid.
    
    Returns
    -------
    psi1 : The updated wavefunction.
    """
    psiF = torch.fft.fftn(psi1, norm='forward')
    psiF = torch.exp(-(1j) * dtau * 0.5 * p_sq) * psiF
    psi1 = torch.fft.ifftn(psiF, norm='forward')
    return psi1

def normalize(phi, d_x):
    """
    Normalizes the wavefunction.
    Args:
    ----
    phi: torch.Tensor
        The wavefunction to be normalised
    d_x: int,
        The grid cell volume
    """
    phi = phi/torch.sqrt(d_x * torch.sum(torch.abs(phi)**2))
    return phi


def update_phase(psi1, phase):
  """
    Updates the phase of the wavefunction.
    
    Args:
    -----
      psi1 : torch.Tensor
      phase : torch.Tensor
    Returns:
    --------
      torch.Tensor, the updated wavefunction
  """
  psi1 = psi1 * torch.exp(phase*1j)
  return psi1

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
    file_name = f'R-{count:003}-cd.dat'
    with open(file_name, 'w') as f:
        for i in range(n1):
            for k in range(n3):
                first = x1[i] * a_ho * 1e6 # x position
                second = x3[k] * a_ho * 1e6 # z position
                third = torch.sum(torch.abs(psi1[i,:,k])**2) # column density
                f.write(f'{first},{second},{third}\n')

def extract_phase(psi):
    """
    Extracts the phase from the wave function.

    Args:
    -----
    psi : torch.Tensor
      The wavefunction of the condensate.
    
    Returns:
    -------
      torch.Tensor, the phase of the condensate
    """
    phase = torch.imag(torch.log(psi/torch.sqrt(torch.abs(psi)**2)))
    return phase

def add_phase(cur_phase, added_phase):
    """
    Adds an extra phase to our current phase. This is the effect of imprinting
    additional vortices in the current condensate.

    Parameters
    ----------
    cur_phase : torch.tensor
      The current phase of the condensate wavefunction.
    added_phase : torch.tensor
      The additional phase due to imprinting.

    Returns
    -------
    final_phase : torch.tensor
      The new updated phase.
    """
    final_phase = cur_phase + added_phase
    return final_phase

def repetitive_imprint(psi1, repetitive_phase, n1, n2, n3):
    """
    Performs a re-imprint on the wavefunction. Adds the `repetitive_phase`.
    
    Args:
    --------
      psi1: torch.tensor, the wavefunction
      repetitive_phase:
      n1, n2, n3: int, the grid
    Returns:
    --------
      torch.tensor, the updated wavefunction
    """
    # extract current phase of psi1
    cur_phase = extract_phase(psi1)
    # add the new vortices (init_phase)
    new_phase = add_phase(cur_phase, repetitive_phase)
    # update the phase
    psi1 = update_phase(psi1, new_phase, n1, n2, n3)
    return psi1

def split_step_step(psi1: torch.Tensor,\
                      utot1: torch.Tensor,\
                      dtau,\
                      p_sq: torch.Tensor,\
                      d_x) -> torch.Tensor:
    """
    Performs a step of the split-step Fourier transform.

    Args:
    -------
      psi1: torch.tensor, the wavefunction
      utot1: torch.tensor, the total potential
      dtau:
      p_sq: torch.tensor, the squared momentum grid
      d_x: int, the product of the grid dimensions
    Returns:
    --------
      torch.tensor, the updated wavefunction
    """
    # split-step evolution
    psi1 = x_evolution(psi1, utot1, dtau)
    psi1 = p_evolution(psi1, dtau, p_sq)
    psi1 = x_evolution(psi1, utot1, dtau)
    psi1 = normalize(psi1, d_x)
    return psi1

def calculate_velocity2D(phase2D, p_grid):
    """
    Calculates the velocity of the condensate.
    v = hbar/m * (grad(phase))
    For the calculation of the gradient, spectral derivative is used.
    It is assumed that the 2D plane is the n1-n3 defined plane.
    Note: the result needs to be multiplied by hbar/m
    Args:
    -----
    phase2D: torch.Tensor, the phase of the condensate wavefunction
    p_grid: Tuple[torch.Tensor], the momentum space grid with the
      i-th component being the momentum axis along ni (i=1,2,3)

    Returns:
    --------
      torch.Tensor, the grad of the phase
    """
    spect_x = p_grid[0] * torch.fft.fftn(phase2D) * 1j
    spect_y = p_grid[1] * torch.fft.fftn(phase2D) * 1j
    grad_x = torch.fft.ifftn(spect_x).real
    grad_y = torch.fft.ifftn(spect_y).real
    grad_mod = torch.sqrt(grad_x**2 + grad_y**2)
    grad_angle = torch.atan2(grad_y, grad_x)
    return grad_mod, grad_angle

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

def rms_radius(psi, center, space_grid):
    """
    Calculates the RMS radius of the condensate.
    
    RMS = {1/(N) * Sum[(r-center)**2 * |psi|**2]}**0.5

    Args:
    -----
    psi: torch.Tensor, the normalised wavefunction
    center: torch.Tensor, the centers of the space axes x1, x2, x3
    space_grid: torch.Tensor, the meshgrid of the space

    Returns:
    --------
    rms: torch.Tensor, a single value of the RMS calculation
    """
    center_x = center[0] 
    center_y = center[1]
    center_z = center[2]
    N_tot = torch.where(torch.abs(psi)**2 > 0, 1, 0).sum()
    # build the r-space squared.
    g_x, g_y, g_z = space_grid
    d_sq = (g_x-center_x)**2 + (g_y-center_y)**2 + (g_z-center_z)**2
    rms = (torch.sum(d_sq * (torch.abs(psi)**2))/(N_tot))**0.5
    return rms

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