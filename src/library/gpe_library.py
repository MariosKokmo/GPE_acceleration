# -*- coding: utf-8 -*-
import numpy as np
import torch
from .parameters import CONSTANTS

def init_grid(x_min, x_max, dx, dp, w, n1, n2, n3, device):
    """
    Initialises the grid for x and p spaces
    and the external potential

    Parameters
    ----------
    x_min:
    x_max:
    dx:
    dp:
    w:
    n1,n2,n3:
    device:
    
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

def create_vortices(vortices, x1, x2, x3, n1, n2, n3, device):
    """
    Creates the vortices on the condensate by calculating a new phase
    to be added. The vortices are on the xz (n1,n3) plane.
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
    phase = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=device)
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
    
    Parameters
    ----------
    phi: torch.Tensor
        The wavefunction to be normalised
    d_x: int,
        The grid cell volume

    Returns
    -------
     The normalized wavefunction
    """
    phi = phi/torch.sqrt(d_x * torch.sum(torch.abs(phi)**2))
    return phi

def update_phase(psi1, phase):
  """
    Updates the phase of the wavefunction.
    
    Parameters
    ----------
      psi1 : torch.Tensor
      phase : torch.Tensor
    
    Returns
    -------
      torch.Tensor, the updated wavefunction
  """
  psi1 = psi1 * torch.exp(phase*1j)
  return psi1

def extract_phase(psi):
    """
    Extracts the phase from the wave function.

    Parameters
    ----------
    psi : torch.Tensor
      The wavefunction of the condensate.
    
    Returns
    -------
      torch.Tensor, the phase of the condensate
    """
    phase = torch.imag(torch.log(psi/torch.sqrt(torch.abs(psi)**2)))
    return phase

def add_phase(cur_phase, added_phase):
    """
    Adds an extra phase to the current phase. 

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

def repetitive_imprint(psi1, repetitive_phase):
    """
    Performs a re-imprint on the wavefunction. Adds the `repetitive_phase`.
    
    Parameters
    ----------
      psi1: torch.tensor, the wavefunction
      repetitive_phase:
      n1, n2, n3: int, the grid
    
    Returns
    -------
      torch.tensor, the updated wavefunction
    """
    # update the phase
    psi1 = update_phase(psi1, repetitive_phase)
    return psi1

def split_step_step(psi1: torch.Tensor,\
                      utot1: torch.Tensor,\
                      dtau,\
                      p_sq: torch.Tensor,\
                      d_x) -> torch.Tensor:
    """
    Performs a step of the split-step Fourier transform.

    Parameters
    ----------
      psi1: torch.tensor, the wavefunction
      utot1: torch.tensor, the total potential
      dtau:
      p_sq: torch.tensor, the squared momentum grid
      d_x: int, the product of the grid dimensions
    
    Returns
    -------
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
    
    Parameters
    ----------
    phase2D: torch.Tensor, the phase of the condensate wavefunction
    p_grid: Tuple[torch.Tensor], the momentum space grid with the
      i-th component being the momentum axis along ni (i=1,2,3)

    Returns
    -------
      torch.Tensor, the grad of the phase
    """
    spect_x = p_grid[0] * torch.fft.fftn(phase2D) * 1j
    spect_y = p_grid[1] * torch.fft.fftn(phase2D) * 1j
    grad_x = torch.fft.ifftn(spect_x).real
    grad_y = torch.fft.ifftn(spect_y).real
    grad_mod = torch.sqrt(grad_x**2 + grad_y**2)
    grad_angle = torch.atan2(grad_y, grad_x)
    return grad_mod, grad_angle

def rms_radius(psi, center, space_grid):
    """
    Calculates the RMS radius of the condensate. Assumes a symmetric condensate.
    
    RMS = {1/(N) * Sum[(r-center)**2 * |psi|**2]}**0.5

    Parameters
    ----------
    psi: torch.Tensor, the normalised wavefunction
    center: torch.Tensor, the centers of the space axes x1, x2, x3
    space_grid: torch.Tensor, the meshgrid of the space

    Returns
    -------
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

def calculate_cross_section_line(psi, axis=1):
    """"
    Calculates the density on a line that crosses the condensate.
    Assuming the condensate on the x-z plane and centered, then the cross
    section line would be with respect to one of the axis.

    Parameters
    ----------
    psi: torch.Tensor, the BEC wavefunction
    axis: int, axis 1 means x and axis 2 means z, default is 1

    Returns
    -------
    cross_line: torch.Tensor, the cross section line passing 
                through the centre of the BEC
    """
    n1, n2, n3 = psi.shape
    if axis == 1:
        cross_line = torch.sum(torch.abs(psi[n1//2,:,:])**2, dim=0)
    if axis == 2:
        cross_line = torch.sum(torch.abs(psi[:,:,n3//2])**2, dim=0)
    return cross_line

def mod_grad_psi(psi, p_grid):
    """
    Returns the modulus of the gradient of the wavefunction.
    Calculates the gradient in a cartesian system and then
    returns the modulus.
    It uses the Fourier transform to perform the gradient in
    momentum space.

    Parameters
    ----------
    psi: torch.Tensor, the condensate wavefunction
    p_grid: Tuple[torch.Tensor], the momentum space grid with the
      i-th component being the momentum axis along ni (i=1,2,3)

    Returns
    -------
      torch.Tensor, the modulus gradient of the wavefunction
    """
    dim  = len(psi.shape)
    if dim == 3:
        px, py, pz = torch.meshgrid(*p_grid)
        P = torch.stack((px, py, pz))
        spec_x = P[0] * torch.fft.fftn(psi, norm='forward') * 1j
        grad_x = torch.fft.ifftn(spec_x, norm='forward').real
        spec_y = P[1] * torch.fft.fftn(psi, norm='forward') * 1j
        grad_y = torch.fft.ifftn(spec_y, norm='forward').real
        spec_z = P[2] * torch.fft.fftn(psi, norm='forward') * 1j
        grad_z = torch.fft.ifftn(spec_z, norm='forward').real
        grad_modulus = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    elif dim == 2:
        px, py = torch.meshgrid(p_grid[0],p_grid[1])
        P = torch.stack((px,py))
        spec_x = P[0] * torch.fft.fft2(psi, norm='forward') * 1j
        grad_x = torch.fft.ifft2(spec_x, norm='forward').real
        spec_y = P[1] * torch.fft.fft2(psi, norm='forward') * 1j
        grad_y = torch.fft.ifft2(spec_y, norm='forward').real
        grad_modulus = torch.sqrt(grad_x**2 + grad_y**2)
    elif dim == 1:
        spect_x = p_grid[0] * torch.fft.fft(psi, norm='forward') * (1j)
        grad_modulus = torch.fft.ifft(spect_x, norm='forward').real
    
    return grad_modulus

def calculate_energy_allocation(psi, Vext, p_grid, **parameters):
    """
    Calculates the energy allocation for the condensate at every moment.
    Specifically returns the kinetic, potential and interaction terms of the energy. 

    Parameters
    ----------
    psi: torch.Tensor, the BEC wavefunction
    Vext: torch.Tensor, the external potential
    p_grid: , the momentum grid of the system

    Returns
    -------
    energies: Dict[str, float], the energy terms 
    """
    hbar = CONSTANTS.hbar
    m = CONSTANTS.m1 # Rubidium mass
    u = parameters['u']

    # |nabla(psi)|**2
    grad_sq = torch.pow(mod_grad_psi(psi, p_grid), 2)

    e_kin = hbar**2 / (2*m) * torch.sum(grad_sq)
    e_pot = torch.sum(Vext * torch.abs(psi)**2)
    e_int = g * torch.sum(torch.abs(psi)**4)
    E_total = e_kin + e_pot + e_int
    return {'e_kin' : e_kin,
            'e_pot' : e_pot,
            'e_int' : e_int,
            'E_total' : E_total}