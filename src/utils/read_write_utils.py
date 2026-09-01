"""This module provides the utility functions to read and write data in files"""
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.library.gpe_library import GPE2DLibrary as gpe2d

def write_psi(file_name, psi, n1, n2, n3):
    """
    Writes the wavefunction of the condensate to a file.
    Usually used for the ground state.

    One line per grid point, `(real,imag)`, in row-major order over
    (n1, n2, n3) — the format `GroundState.read_ground_state` expects.

    Written in a single vectorised pass. The previous element-by-element loop
    cost a device round-trip per point, which ran to minutes for a full 3-D
    grid. `%.17g` round-trips float64 exactly, so the stored state is bit-for-bit
    recoverable.

    Args:
      file_name: str, the name of the file to be created
      psi: torch.Tensor, the wavefunction of the condensate
      n1, n2, n3: integer, the grid points in the 3 dimensions

    Raises:
      ValueError: if psi does not hold exactly n1*n2*n3 points.
    """
    values = psi.detach().cpu().numpy().reshape(-1)
    expected = n1 * n2 * n3
    if values.size != expected:
        raise ValueError(
            f"psi has {values.size} points but the {n1}x{n2}x{n3} grid needs {expected}"
        )
    columns = np.empty((values.size, 2), dtype=np.float64)
    columns[:, 0] = values.real
    columns[:, 1] = values.imag
    np.savetxt(file_name, columns, fmt='(%.17g,%.17g)')

def write_data(psi1, count, x1, x3, n1, n3, a_ho, dx):
    """
    Writes the column density data on the x-z plane
    """
    dy = dx[1] # the grid spacing along the y direction
    file_name = f'R-{count:003}-cd.dat'
    with open(file_name, 'w') as f:
        for i in range(n1):
            for k in range(n3):
                first = x1[i] * a_ho * 1e6 # x position
                second = x3[k] * a_ho * 1e6 # z position
                third = torch.sum(torch.abs(psi1[i,:,k])**2) * dy # column density
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

def write_velocity2D(psi, count, x1, x3, n1, n2, n3, a_ho, p_grid):
    """
    Writes the 2D velocity in a file.
    It assumes the plane is the n1-n3.
    The format of the file is `x1, x3, velocity magnitude, velocity direction`

    Takes the wavefunction rather than its phase: the velocity is
    Im(psi* grad psi)/|psi|^2, which is free of the 2*pi branch cuts that make
    a phase-derived velocity field meaningless around a vortex.

    Args:
        psi (torch.Tensor): the condensate wavefunction (n1, n2, n3).
        count (int): the snapshot number.
        x1, x3 (torch.Tensor): the axes.
        n1, n3 (int): the grid resolution along x1 and x3.
        a_ho (float): the harmonic oscillator length.
        p_grid (tuple): momentum meshgrids.

    Returns:
        None
    """
    j = n2//2
    vel_file_name = f'V-{count:003}-cd.dat'
    velocity_mag, veloc_phase = gpe2d.calculate_velocity2D(psi, p_grid)
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

def save_rms_figure(title):
    """
    Reads a file delimited by tabs with the first
    column being the time and the second column being
    the quantity of interest i.e. RMS value
    """
    data = np.loadtxt(title, skiprows=1, delimiter="\t")
    plt.figure()
    plt.plot(data[:,0],data[:,1])
    plt.title(f"{title[:-4]}")
    plt.ylabel("RMS")
    plt.xlabel("time")
    plt.savefig(f"RMS_{title[:-3]}.png")

def save_cross_section_line_figure(cross_line_data):
    """
    Assumes data of shape (snapshots, position)
    Data consists of rows where each row is a snapshot of the
    density values across the section
    """
    shots, dim = cross_line_data.shape
    ax = plt.figure(figsize=(12,16)).add_subplot(projection='3d')
    x = np.arange(0, shots, 1)
    y = np.arange(0, dim, 1)

    # Plot a sin curve using the x and y axes.
    for t in range(shots):
        data = cross_line_data[t, :].numpy()
        x = np.ones(dim)*t
        ax.plot(x,y,data)
    ax.set_xlabel('time')
    ax.set_ylabel('space')
    ax.set_zlabel('density')
    plt.savefig(f"cross_section_line.png")


def save_tensor_to_csv(tensor, filename):
    tensor_np = tensor.numpy() #convert to Numpy array
    df = pd.DataFrame(tensor_np) #convert to a dataframe
    df.to_csv(filename,index=False, header=None) #save to file

def write_energy_terms(energies, filename):
    """
    Writes the energy allocation in a file.

    Args:
        energies (list[dict]): One dictionary per snapshot, holding the energy
            split across the kinetic, potential and interaction terms along
            with the total for that timestamp.
        filename (str): Destination path.
    """
    with open(filename, 'w') as f:
        for energy in energies:
            e_kin = energy['e_kin']
            e_pot = energy['e_pot']
            e_int = energy['e_int']
            E_total = energy['E_total']
            f.write(f"{e_kin},{e_pot},{e_int},{E_total}\n")


# =============================================================================
# Cylindrical coordinate I/O
# =============================================================================
#
# Reused without modification from above:
#   write_psi, write_rms, read_phase_file_2D,
#   save_rms_figure, save_cross_section_line_figure,
#   save_tensor_to_csv, write_energy_terms
#
# New functions below mirror their Cartesian counterparts but operate on the
# cylindrical grid (r, φ, z) with shape (n_r, n_phi, n_z).
# =============================================================================


def write_data_cylindrical(psi, count, r, phi, n_r, n_phi, a_ho, dz):
    """
    Write the z-integrated column density n(r, φ) to a snapshot file.

    Suited for pancake-geometry BECs where z is the tightly confined axis.
    Integrating out z gives the 2-D density in the r-φ plane:

        n(r, φ) = Σ_k |ψ[i, j, k]|² · dz

    File format (CSV, one row per (r, φ) point):
        r_μm, phi_rad, n(r,phi)

    Args:
        psi        : wavefunction tensor of shape (n_r, n_phi, n_z).
        count      : snapshot index used in the filename.
        r          : 1-D radial grid (n_r,) in dimensionless units.
        phi        : 1-D azimuthal grid (n_phi,) in radians.
        n_r, n_phi : grid point counts.
        a_ho       : harmonic oscillator length in metres (converts to µm).
        dz         : axial grid spacing (dimensionless).
    """
    file_name = f'R-{count:003}-cd.dat'
    col_density = torch.sum(torch.abs(psi) ** 2, dim=2) * dz   # (n_r, n_phi)
    with open(file_name, 'w') as f:
        for i in range(n_r):
            for j in range(n_phi):
                r_phys = float(r[i]) * a_ho * 1e6
                phi_val = float(phi[j])
                f.write(f'{r_phys},{phi_val},{float(col_density[i, j])}\n')


def write_phase2D_cylindrical(phase, count, r, phi, n_r, n_phi, a_ho, z_idx=None):
    """
    Write the wavefunction phase on the r-φ plane at a fixed z index.

    File format (CSV):
        r_μm, phi_rad, phase

    Args:
        phase        : phase tensor of shape (n_r, n_phi, n_z).
        count        : snapshot index.
        r            : 1-D radial grid in dimensionless units.
        phi          : 1-D azimuthal grid in radians.
        n_r, n_phi   : grid point counts.
        a_ho         : harmonic oscillator length in metres.
        z_idx        : index along the z axis to slice at.
                       Defaults to the midpoint (n_z // 2).
    """
    n_z = phase.shape[2]
    if z_idx is None:
        z_idx = n_z // 2
    file_name = f'P-{count:003}-cd.dat'
    with open(file_name, 'w') as f:
        for i in range(n_r):
            for j in range(n_phi):
                r_phys = float(r[i]) * a_ho * 1e6
                phi_val = float(phi[j])
                ph = phase[i, j, z_idx]
                if torch.is_tensor(ph):
                    ph = ph.real.item() if ph.is_complex() else ph.item()
                f.write(f'{r_phys},{phi_val},{ph}\n')


def write_radial_profile(psi, count, r, n_r, a_ho, dphi, dz):
    """
    Write the 1-D radial density profile n(r) to a file.

    n(r) = ∫∫ |ψ(r, φ, z)|² dφ dz  ≈  Σ_{j,k} |ψ[i,j,k]|² · dφ · dz

    This diagnostic has no direct Cartesian counterpart and is natural to
    cylindrical geometry (e.g. for checking the Thomas-Fermi radius).

    File format (CSV):
        r_μm, n(r)

    Args:
        psi   : wavefunction tensor of shape (n_r, n_phi, n_z).
        count : snapshot index.
        r     : 1-D radial grid in dimensionless units.
        n_r   : number of radial grid points.
        a_ho  : harmonic oscillator length in metres.
        dphi  : azimuthal grid spacing.
        dz    : axial grid spacing.
    """
    file_name = f'Rad-{count:003}-profile.dat'
    with open(file_name, 'w') as f:
        for i in range(n_r):
            r_phys = float(r[i]) * a_ho * 1e6
            n_r_val = torch.sum(torch.abs(psi[i, :, :]) ** 2) * dphi * dz
            f.write(f'{r_phys},{float(n_r_val)}\n')


def save_figure_phase_cylindrical(phase, frame, z_idx=None):
    """
    Save an image of the wavefunction phase on the r-φ plane at a fixed z index.

    Args:
        phase : phase tensor of shape (n_r, n_phi, n_z).
        frame : snapshot index used in the title and filename.
        z_idx : index along z to slice at. Defaults to the midpoint (n_z // 2).
    """
    n_z = phase.shape[2]
    if z_idx is None:
        z_idx = n_z // 2
    slice_2d = phase[:, :, z_idx]
    if slice_2d.is_complex():
        data = slice_2d.cpu().real
    else:
        data = slice_2d.cpu()
    plt.imshow(data, cmap='jet', aspect='auto', origin='lower')
    cb = plt.colorbar()
    plt.title(f"Phase (r-φ, z_idx={z_idx})  t = {frame}")
    plt.xlabel("φ index")
    plt.ylabel("r index")
    plt.savefig(f"phase_t_{frame}.png")
    cb.remove()


def write_velocity_cylindrical(psi, count, r, phi, n_r, n_phi, a_ho, dr, m_modes, z_idx=None):
    """
    Write the superfluid velocity field (vr, v_φ) on the r-φ plane at a fixed z index.

    Suited for pancake-geometry BECs.  Velocity components in dimensionless
    units (ħ/m = 1):

        v_r(r, φ) = Im(ψ* ∂ψ/∂r) / |ψ|²         — central finite differences in r
        v_φ(r, φ) = Im(ψ* (1/r) ∂ψ/∂φ) / |ψ|²   — spectral (DFT in φ)

    Grid points whose density is below 1e-12 of the peak are set to zero. The
    cut is relative because ψ is normalised over the whole grid, so an absolute
    threshold means something different at every resolution.

    File format (CSV):
        r_μm, phi_rad, vr, v_phi, |v|

    Args:
        psi          : wavefunction (n_r, n_phi, n_z).
        count        : snapshot index.
        r            : 1-D radial grid in dimensionless units.
        phi          : 1-D azimuthal grid in radians.
        n_r, n_phi   : grid point counts.
        a_ho         : harmonic oscillator length in metres.
        dr           : radial grid spacing.
        m_modes      : azimuthal mode indices (n_phi,) from init_grid.
        z_idx        : index along z to slice at. Defaults to n_z // 2.
    """
    n_z = psi.shape[2]
    if z_idx is None:
        z_idx = n_z // 2

    density_full = torch.abs(psi) ** 2   # (n_r, n_phi, n_z)
    density_threshold = 1e-12 * torch.max(density_full)

    # --- radial velocity: central finite differences in r ---
    dpsi_dr = torch.empty_like(psi)
    dpsi_dr[1:-1] = (psi[2:] - psi[:-2]) / (2.0 * dr)
    dpsi_dr[0]    = (psi[1]  - psi[0])   / dr
    dpsi_dr[-1]   = (psi[-1] - psi[-2])  / dr
    vr_full = torch.where(
        density_full > density_threshold,
        torch.imag(psi.conj() * dpsi_dr) / density_full,
        torch.zeros_like(density_full),
    )
    vr = vr_full[:, :, z_idx].real   # (n_r, n_phi)

    # --- azimuthal velocity: spectral φ-derivative divided by r ---
    psi_m = torch.fft.fft(psi, dim=1, norm='ortho')
    dpsi_dphi = torch.fft.ifft(
        1j * m_modes.reshape(1, n_phi, 1) * psi_m, dim=1, norm='ortho'
    )
    r_w = r.reshape(-1, 1, 1)
    vphi_full = torch.where(
        density_full > density_threshold,
        torch.imag(psi.conj() * dpsi_dphi) / (density_full * r_w),
        torch.zeros_like(density_full),
    )
    vphi = vphi_full[:, :, z_idx].real   # (n_r, n_phi)

    vel_file_name = f'V-{count:003}-cd.dat'
    with open(vel_file_name, 'w') as f:
        for i in range(n_r):
            for j in range(n_phi):
                r_phys = float(r[i]) * a_ho * 1e6
                phi_val = float(phi[j])
                v_r   = vr[i, j].item()
                v_phi = vphi[i, j].item()
                v_mag = (v_r ** 2 + v_phi ** 2) ** 0.5
                f.write(f'{r_phys},{phi_val},{v_r},{v_phi},{v_mag}\n')
