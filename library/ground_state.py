"""Module to find the ground state numerically using the imaginary time
propagation method. """
import torch
from .gpe_library import normalize, write_psi, init_grid, write_data
from .gpe_library import CONSTANTS


def find_ground_state(sim_params, system, file_name, device):
    """
    Args:
        sims_params: dictionary, holds the parameters
            for the specific simulation
        file_name: str, the name of the ground state file
        device: the device to run the code

    Returns:
        torch.Tensor, the ground state for the system
    """
    n1, n2, n3 = system.simulation_parameters["Grid_resolution"]
    device = device
    d_x = system.simulation_parameters["d_x"]
    dx = system.simulation_parameters["dx"]
    dp = system.simulation_parameters["dp"]
    w = system.simulation_parameters["w"]
    x_min = system.simulation_parameters["x_min"]
    x_max = system.simulation_parameters["x_max"]
    dtau = 0.05*min(dx)**2
    a_ho = system.simulation_parameters["a_ho"]

    # This will store the external potential
    uext = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=device)
    # This will store the ground state
    psi1 = torch.zeros((n1,n2,n3), dtype=torch.cdouble, device=device)

    uext = system.uext.potential
    x1, x2, x3, p1, p2, p3, p_sq, _, _ = init_grid(x_min, x_max, dx, dp, w, n1, n2, n3, device)
    # initialise some parameters
    energy = 0
    energy_old = 0
    iter = 0
    mu = 0
    tol = 0
    done = False

    u = 4.*CONSTANTS.pi * CONSTANTS.nat * CONSTANTS.ascat/ a_ho
    mu_TF = 0.5 * (15/(4 * CONSTANTS.pi) * u)**(2/5)
    
    # where mutf > uext: psi = sqrt(mu - uext)
    psi1 = torch.where(mu_TF > uext, torch.sqrt(mu_TF - uext) + 0j, psi1)
    psi1 = normalize(psi1, d_x)
    
    psi1, energy, tol, mu= steepest_descent(psi1, dtau, p_sq, uext, d_x, u)
    print("{iter}\t{energy}\t{mu}\t{dtau:}\t{tol}".format(iter="iter",energy="energy",mu="mu",dtau="dtau",tol="tol"))
    
    energy_old = energy
    
    while not done:
        iter = iter + 1
        psi1, energy, tol, mu = steepest_descent(psi1, dtau, p_sq, uext, d_x, u)
        test_e = (energy_old - energy)/energy

        if test_e < 0:
            print(f"Changing dtau : {dtau}, {test_e}")
            dtau = dtau/2
        energy_old = energy

        if iter%50 == 0:
            print(f"{iter:10}\t{energy:10}\t{mu:10}\t{dtau:10}\t{tol:10}")
        if (tol < 1e-5) or (test_e == 0):
            done = True
    write_psi(file_name, psi1, n1, n2, n3)
    return psi1


def steepest_descent(psi, dtau, p_sq, uext, d_x, u):
    """
    Calculates a step of the steepest descent algorithm.

    Args:
        psi: torch.Tensor, the wavefunction of the condensate.
             Transferred to GPU if available
        dtau:
        p_sq: torch.Tensor, the squared momentum grid
        uext: torch.Tensor, the external potential
        d_x:
        u: float, the interaction strength
    
    Returns:
    """
    dpsi = psi
    psiF = torch.fft.fftn(dpsi, norm='forward')
    psiF =  0.5 * p_sq * psiF
    dpsi = torch.fft.ifftn(psiF, norm='forward')

    utot = u*torch.abs(psi)**2 + uext
    dpsi = dpsi + utot*psi

    psi_conj = torch.conj(psi)
    mu = d_x * torch.sum(psi_conj*dpsi)
    mu = mu.abs()

    interaction = 0.5 * u * d_x * torch.sum(torch.abs(psi)**4).item()
    dpsi = dpsi - mu * psi
    tol = d_x * torch.sum(torch.abs(dpsi)**2)
    
    energy = mu - interaction
    psi = psi - dtau * dpsi
    psi = normalize(psi, d_x)
    
    return psi, energy, tol, mu
