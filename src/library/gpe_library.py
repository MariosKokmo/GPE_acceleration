# -*- coding: utf-8 -*-
import numpy as np
import torch
from .parameters import CONSTANTS

def init_grid(x_min, x_max, dx, dp, w, n1, n2, n3, device):
    """
    Initializes the grid for x and p spaces and the external potential.

    Parameters
    ----------
    x_min : list
        Minimum values for the x-axis in each dimension.
    x_max : list
        Maximum values for the x-axis in each dimension.
    dx : list
        Grid spacing in real space for each dimension.
    dp : list
        Grid spacing in momentum space for each dimension.
    w : float
        Frequency of the external potential.
    n1, n2, n3 : int
        Number of grid points in each dimension.
    device : torch.device
        Device to allocate tensors (CPU or GPU).

    Returns
    -------
    x1, x2, x3 : torch.Tensor
        Real space axes for each dimension.
    p1, p2, p3 : torch.Tensor
        Momentum space axes for each dimension.
    p_sq : torch.Tensor
        Squared momentum grid.
    space_grid : tuple of torch.Tensor
        Real space grid as a tuple of meshgrids.
    p_grid : tuple of torch.Tensor
        Momentum space grid as a tuple of meshgrids.
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
    Creates vortices on the condensate by calculating a new phase to be added.

    Parameters
    ----------
    vortices : numpy.ndarray
        Array of shape (3, number_of_vortices) containing vortex positions and charges:
        - 1st row: x positions.
        - 2nd row: z positions.
        - 3rd row: vortex charges.
    x1, x2, x3 : torch.Tensor
        Real space axes for each dimension.
    n1, n2, n3 : int
        Number of grid points in each dimension.
    device : torch.device
        Device to allocate tensors (CPU or GPU).

    Returns
    -------
    torch.Tensor
        Updated phase of the condensate.
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
    Performs the evolution step in real space.

    Parameters
    ----------
    psi1 : torch.Tensor
        The wavefunction of the system.
    utot1 : torch.Tensor
        The trapping potential at this time step.
    dtau : float
        Time evolution step.
    factor : float, optional
        Splitting factor of the split-step method (default is 0.5).

    Returns
    -------
    torch.Tensor
        The updated wavefunction.
    """
    psi1 = torch.exp(-factor * dtau*(1j) * utot1) * psi1
    return psi1

def p_evolution(psi1, dtau, p_sq):
    """
    Performs the evolution step in momentum space.

    Parameters
    ----------
    psi1 : torch.Tensor
        The wavefunction of the system.
    dtau : float
        Time evolution step.
    p_sq : torch.Tensor
        Squared momentum grid.

    Returns
    -------
    torch.Tensor
        The updated wavefunction.
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
    phi : torch.Tensor
        The wavefunction to be normalized.
    d_x : float
        Grid cell volume.

    Returns
    -------
    torch.Tensor
        The normalized wavefunction.
    """
    phi = phi/torch.sqrt(d_x * torch.sum(torch.abs(phi)**2))
    return phi

def update_phase(psi1, phase):
    """
    Updates the phase of the wavefunction.

    Parameters
    ----------
    psi1 : torch.Tensor
        The wavefunction of the system.
    phase : torch.Tensor
        The phase to be applied.

    Returns
    -------
    torch.Tensor
        The updated wavefunction.
    """
    psi1 = psi1 * torch.exp(phase*1j)
    return psi1

def extract_phase(psi):
    """
    Extracts the phase from the wavefunction.

    Parameters
    ----------
    psi : torch.Tensor
        The wavefunction of the condensate.

    Returns
    -------
    torch.Tensor
        The phase of the condensate.
    """
    phase = torch.imag(torch.log(psi/torch.sqrt(torch.abs(psi)**2)))
    return phase

def add_phase(cur_phase, added_phase):
    """
    Adds an extra phase to the current phase.

    Parameters
    ----------
    cur_phase : torch.Tensor
        The current phase of the condensate wavefunction.
    added_phase : torch.Tensor
        The additional phase to be added.

    Returns
    -------
    torch.Tensor
        The updated phase.
    """
    final_phase = cur_phase + added_phase
    return final_phase

def repetitive_imprint(psi1, repetitive_phase):
    """
    Re-imprints the wavefunction by adding the repetitive phase.

    Parameters
    ----------
    psi1 : torch.Tensor
        The wavefunction of the system.
    repetitive_phase : torch.Tensor
        The repetitive phase to be added.

    Returns
    -------
    torch.Tensor
        The updated wavefunction.
    """
    # update the phase
    psi1 = update_phase(psi1, repetitive_phase)
    return psi1

def split_step_step(psi1, utot1, dtau, p_sq, d_x):
    """
    Performs a step of the split-step Fourier transform.

    Parameters
    ----------
    psi1 : torch.Tensor
        The wavefunction of the system.
    utot1 : torch.Tensor
        The total potential.
    dtau : float
        Time evolution step.
    p_sq : torch.Tensor
        Squared momentum grid.
    d_x : float
        Product of the grid dimensions.

    Returns
    -------
    torch.Tensor
        The updated wavefunction.
    """
    # split-step evolution
    psi1 = x_evolution(psi1, utot1, dtau)
    psi1 = p_evolution(psi1, dtau, p_sq)
    psi1 = x_evolution(psi1, utot1, dtau)
    psi1 = normalize(psi1, d_x)
    return psi1

def calculate_velocity2D(phase2D, p_grid):
    """
    Calculates the velocity of the condensate in 2D.
    Note the result needs to be multiplied by hbar/m to get the velocity.

    Parameters
    ----------
    phase2D : torch.Tensor
        The phase of the condensate wavefunction in 2D.
    p_grid : tuple of torch.Tensor
        Momentum space grid with components for each axis.

    Returns
    -------
    tuple of torch.Tensor
        - Gradient magnitude of the phase.
        - Gradient angle of the phase.
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
    Calculates the RMS radius of the condensate.

    Parameters
    ----------
    psi : torch.Tensor
        The normalized wavefunction.
    center : list or torch.Tensor
        Centers of the space axes (x1, x2, x3).
    space_grid : tuple of torch.Tensor
        Meshgrid of the space.

    Returns
    -------
    torch.Tensor
        RMS radius of the condensate.
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
    """
    Calculates the density on a line that crosses the condensate.

    Parameters
    ----------
    psi : torch.Tensor
        The BEC wavefunction.
    axis : int, optional
        Axis along which the cross-section is calculated (1 for x, 2 for z).
        Default is 1.

    Returns
    -------
    torch.Tensor
        Cross-section line passing through the center of the BEC.
    """
    n1, n2, n3 = psi.shape
    if axis == 1:
        cross_line = torch.sum(torch.abs(psi[n1//2,:,:])**2, dim=0)
    if axis == 2:
        cross_line = torch.sum(torch.abs(psi[:,:,n3//2])**2, dim=0)
    return cross_line

def mod_grad_psi(psi, p_axes):
    """
    Calculates the modulus of the gradient of the wavefunction.

    Parameters
    ----------
    psi : torch.Tensor
        The condensate wavefunction.
    p_axes : list of torch.Tensor
        Momentum space grid with components for each axis.

    Returns
    -------
    torch.Tensor
        Modulus of the gradient of the wavefunction.
    """
    dim  = len(psi.shape)
    if dim == 3:
        px, py, pz = torch.meshgrid(p_axes[0], p_axes[1], p_axes[2])
        P = torch.stack((px,py,pz))
        spec_x = P[0] * torch.fft.fftn(psi, norm='forward') * 1j
        grad_x = torch.fft.ifftn(spec_x, norm='forward').real
        spec_y = P[1] * torch.fft.fftn(psi, norm='forward') * 1j
        grad_y = torch.fft.ifftn(spec_y, norm='forward').real
        spec_z = P[2] * torch.fft.fftn(psi, norm='forward') * 1j
        grad_z = torch.fft.ifftn(spec_z, norm='forward').real
        grad_modulus = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    elif dim == 2:
        px, py = torch.meshgrid(p_axes[0], p_axes[1])
        P = torch.stack((px,py))
        spec_x = P[0] * torch.fft.fft2(psi, norm='forward') * 1j
        grad_x = torch.fft.ifft2(spec_x, norm='forward').real
        spec_y = P[1] * torch.fft.fft2(psi, norm='forward') * 1j
        grad_y = torch.fft.ifft2(spec_y, norm='forward').real
        grad_modulus = torch.sqrt(grad_x**2 + grad_y**2)
    elif dim == 1:
        spect_x = p_axes[0] * torch.fft.fft(psi, norm='forward') * (1j)
        grad_modulus = torch.fft.ifft(spect_x, norm='forward').real
    
    return grad_modulus

def calculate_energy_allocation(psi, Vext, p_grid, **parameters):
    """
    Calculates the energy allocation for the condensate.

    Parameters
    ----------
    psi : torch.Tensor
        The BEC wavefunction.
    Vext : torch.Tensor
        The external potential.
    p_grid : tuple of torch.Tensor
        Momentum space grid.
    **parameters : dict
        Additional parameters (e.g., interaction strength).

    Returns
    -------
    dict
        Energy terms: kinetic, potential, interaction, and total energy.
    """
    hbar = CONSTANTS.hbar
    m = CONSTANTS.m1 # Rubidium mass
    u = parameters['u']

    # |nabla(psi)|**2
    grad_sq = torch.pow(mod_grad_psi(psi, p_grid), 2)

    e_kin = (1/2) * torch.sum(grad_sq)
    e_pot = torch.sum(Vext * torch.abs(psi)**2)
    e_int = (u/2) * torch.sum(torch.abs(psi)**4)
    E_total = e_kin + e_pot + e_int
    return {'e_kin' : e_kin,
            'e_pot' : e_pot,
            'e_int' : e_int,
            'E_total' : E_total}