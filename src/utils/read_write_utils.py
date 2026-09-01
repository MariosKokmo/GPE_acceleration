r"""
Utility functions to read and write simulation data in text files.

The module covers both coordinate systems. The Cartesian helpers write
snapshots on the :math:`(x, z)` plane; the cylindrical ones, grouped in the
second half of the file, mirror them on the :math:`(r, \varphi)` plane of a
grid of shape :math:`(n_r, n_\varphi, n_z)`. All files are CSV with one row per
grid point, and positions are converted to micrometres through the
harmonic-oscillator length ``a_ho``.
"""
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.library.gpe_library import GPE2DLibrary as gpe2d

def write_psi(file_name, psi, n1, n2, n3):
    r"""
    Write the condensate wavefunction to a file, usually the ground state.

    The format is one line per grid point, ``(real,imag)``, in row-major order
    over :math:`(n_1, n_2, n_3)` — what
    :meth:`GroundState.read_ground_state <src.library.ground_state.GroundState.read_ground_state>`
    expects.

    The write is a single vectorised pass. The previous element-by-element loop
    cost a device round-trip per point, which ran to minutes for a full 3-D
    grid. ``%.17g`` round-trips float64 exactly, so the stored state is
    bit-for-bit recoverable.

    Args:
        file_name (str): Name of the file to create.
        psi (torch.Tensor): Condensate wavefunction.
        n1 (int): Number of grid points along the first dimension.
        n2 (int): Number of grid points along the second dimension.
        n3 (int): Number of grid points along the third dimension.

    Raises:
        ValueError: If ``psi`` does not hold exactly
            :math:`n_1 n_2 n_3` points.
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
    r"""
    Write the column density on the :math:`(x, z)` plane to a snapshot file.

    The density is integrated along :math:`y`,

    .. math::

        n(x, z) = \int \lvert \psi \rvert^{2}\, \mathrm{d}y
            \approx \sum_j \bigl\lvert \psi[i, j, k] \bigr\rvert^{2}\,
              \mathrm{d}y,

    and written to ``R-{count}-cd.dat`` as ``x_μm, z_μm, n(x,z)``.

    Args:
        psi1 (torch.Tensor): Condensate wavefunction of shape
            ``(n1, n2, n3)``.
        count (int): Snapshot index used in the file name.
        x1 (torch.Tensor): Axis along x, in dimensionless units.
        x3 (torch.Tensor): Axis along z, in dimensionless units.
        n1 (int): Number of grid points along x.
        n3 (int): Number of grid points along z.
        a_ho (float): Harmonic-oscillator length in metres, used to convert the
            positions to micrometres.
        dx (sequence): Grid spacings; only ``dx[1]``, the spacing along y, is
            used.
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
    r"""
    Write the full 3-D phase to a file.

    The format is ``x_μm, y_μm, z_μm, phase``, one row per grid point, written
    to ``P-{count}-cd.dat``.

    Args:
        phase (torch.Tensor): Phase :math:`\arg \psi` of shape
            ``(n1, n2, n3)``.
        count (int): Snapshot index used in the file name.
        x1 (torch.Tensor): Axis along x, in dimensionless units.
        x2 (torch.Tensor): Axis along y, in dimensionless units.
        x3 (torch.Tensor): Axis along z, in dimensionless units.
        n1 (int): Number of grid points along x.
        n2 (int): Number of grid points along y.
        n3 (int): Number of grid points along z.
        a_ho (float): Harmonic-oscillator length in metres.
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
    r"""
    Write the RMS radius measurements of the BEC to a text file.

    The output is tab-delimited with a ``t\tr`` header, written to
    ``{SimulationName}_RMS_meas.txt`` — the format
    :func:`save_rms_figure` reads back.

    Args:
        rms_meas (dict): RMS radius per snapshot, keyed by time or snapshot
            index.
        SimulationName (str): Base name of the simulation, used for the file
            name.
    """
    with open(f'{SimulationName}_RMS_meas.txt', 'w') as f:
        f.write("t\tr\n")
        for t, r in rms_meas.items():
            f.write(f"{t}\t{r}\n")

def write_phase2D(phase, count, x1, x3, n1, n2, n3, a_ho):
    r"""
    Write the 2-D phase to a file.

    The plane is assumed to be :math:`(n_1, n_3)` at the central cross-section,
    i.e. the midpoint :math:`n_2 / 2` of the second axis. The format is
    ``x_μm, z_μm, phase``, written to ``P-{count}-cd.dat``.

    Args:
        phase (torch.Tensor): Phase :math:`\arg \psi` of shape
            ``(n1, n2, n3)``.
        count (int): Snapshot index used in the file name.
        x1 (torch.Tensor): Axis along x, in dimensionless units.
        x3 (torch.Tensor): Axis along z, in dimensionless units.
        n1 (int): Number of grid points along x.
        n2 (int): Number of grid points along y, used only to locate the
            midplane.
        n3 (int): Number of grid points along z.
        a_ho (float): Harmonic-oscillator length in metres.
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
    r"""
    Read a file storing the phase of a 2-D cross-section.

    Args:
        filename (str): Path of the file to read, as written by
            :func:`write_phase2D`.
        n1 (int): Number of grid points along the first axis.
        n3 (int): Number of grid points along the second axis.

    Returns:
        torch.Tensor: The phase, reshaped to ``(n1, n3)``.
    """
    phase = pd.read_csv(filename, header=None, names=['x1', 'x2', 'phase'])
    phase = phase.astype(np.float64)
    phase = phase.values
    phase = phase.reshape((n1, n3))
    phase = torch.from_numpy(phase)
    return phase

def write_velocity2D(psi, count, x1, x3, n1, n2, n3, a_ho, p_grid):
    r"""
    Write the 2-D velocity field to a file.

    The plane is assumed to be :math:`(n_1, n_3)`, and the file format is
    ``x_μm, z_μm, velocity magnitude, velocity direction``.

    The function takes the wavefunction rather than its phase, because the
    velocity

    .. math::

        \mathbf{v} = \frac{\operatorname{Im}
            \bigl(\psi^{*} \nabla \psi\bigr)}{\lvert \psi \rvert^{2}}

    is free of the :math:`2\pi` branch cuts that make a phase-derived velocity
    field meaningless around a vortex.

    Args:
        psi (torch.Tensor): Condensate wavefunction of shape
            ``(n1, n2, n3)``.
        count (int): Snapshot index used in the file name.
        x1 (torch.Tensor): Axis along x, in dimensionless units.
        x3 (torch.Tensor): Axis along z, in dimensionless units.
        n1 (int): Number of grid points along x.
        n2 (int): Number of grid points along y, used only to locate the
            midplane.
        n3 (int): Number of grid points along z.
        a_ho (float): Harmonic-oscillator length in metres.
        p_grid (tuple): Momentum meshgrids.
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
    r"""
    Save an image of the phase at the :math:`y` midplane.

    Args:
        phase (torch.Tensor): Phase array of shape ``(n1, n2, n3)``; a complex
            array is reduced to its real part.
        frame (int): Snapshot index used in the title and the file name.
    """
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
    r"""
    Plot a time series from a tab-delimited file and save it as a PNG.

    The file is expected to have one header row, the time in the first column
    and the quantity of interest, i.e. the RMS value, in the second — the
    format :func:`write_rms` produces.

    Args:
        title (str): Path of the file to read; its stem is reused for the title
            and the output file name.
    """
    data = np.loadtxt(title, skiprows=1, delimiter="\t")
    plt.figure()
    plt.plot(data[:,0],data[:,1])
    plt.title(f"{title[:-4]}")
    plt.ylabel("RMS")
    plt.xlabel("time")
    plt.savefig(f"RMS_{title[:-3]}.png")

def save_cross_section_line_figure(cross_line_data):
    r"""
    Save a 3-D waterfall plot of the cross-section line density.

    The input has shape ``(snapshots, position)``: each row is one snapshot of
    the density values across the section, and the rows are stacked along the
    time axis of the plot.

    Args:
        cross_line_data (torch.Tensor): Line densities of shape
            ``(shots, dim)``.
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
    r"""
    Save a tensor to a CSV file, without an index or a header.

    Args:
        tensor (torch.Tensor): Tensor to save; it must live on the CPU.
        filename (str): Destination path.
    """
    tensor_np = tensor.numpy() #convert to Numpy array
    df = pd.DataFrame(tensor_np) #convert to a dataframe
    df.to_csv(filename,index=False, header=None) #save to file

def write_energy_terms(energies, filename):
    r"""
    Write the energy allocation to a file.

    One row per snapshot, ``e_kin,e_pot,e_int,E_total``.

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
    r"""
    Write the :math:`z`-integrated column density
    :math:`n(r, \varphi)` to a snapshot file.

    This suits pancake-geometry BECs, where :math:`z` is the tightly confined
    axis. Integrating it out gives the 2-D density in the
    :math:`(r, \varphi)` plane,

    .. math::

        n(r, \varphi) = \int \lvert \psi \rvert^{2}\, \mathrm{d}z
            \approx \sum_k \bigl\lvert \psi[i, j, k] \bigr\rvert^{2}\,
              \mathrm{d}z,

    written to ``R-{count}-cd.dat`` as ``r_μm, phi_rad, n(r,phi)``, one row per
    :math:`(r, \varphi)` point.

    Args:
        psi (torch.Tensor): Wavefunction of shape ``(n_r, n_phi, n_z)``.
        count (int): Snapshot index used in the file name.
        r (torch.Tensor): Radial grid of shape ``(n_r,)``, in dimensionless
            units.
        phi (torch.Tensor): Azimuthal grid of shape ``(n_phi,)``, in radians.
        n_r (int): Number of radial grid points.
        n_phi (int): Number of azimuthal grid points.
        a_ho (float): Harmonic-oscillator length in metres, used to convert the
            radius to micrometres.
        dz (float): Axial grid spacing, in dimensionless units.
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
    r"""
    Write the wavefunction phase on the :math:`(r, \varphi)` plane at a fixed
    :math:`z` index.

    The format is ``r_μm, phi_rad, phase``, written to ``P-{count}-cd.dat``.

    Args:
        phase (torch.Tensor): Phase tensor of shape ``(n_r, n_phi, n_z)``.
        count (int): Snapshot index used in the file name.
        r (torch.Tensor): Radial grid, in dimensionless units.
        phi (torch.Tensor): Azimuthal grid, in radians.
        n_r (int): Number of radial grid points.
        n_phi (int): Number of azimuthal grid points.
        a_ho (float): Harmonic-oscillator length in metres.
        z_idx (int): Index along the :math:`z` axis to slice at. Defaults to
            the midpoint ``n_z // 2``.
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
    r"""
    Write the 1-D radial density profile :math:`n(r)` to a file.

    .. math::

        n(r) = \iint \bigl\lvert \psi(r, \varphi, z) \bigr\rvert^{2}\,
                 \mathrm{d}\varphi\, \mathrm{d}z
            \approx \sum_{j, k} \bigl\lvert \psi[i, j, k] \bigr\rvert^{2}\,
              \mathrm{d}\varphi\, \mathrm{d}z

    This diagnostic has no direct Cartesian counterpart and is natural to
    cylindrical geometry, for instance when checking the Thomas-Fermi radius.
    The format is ``r_μm, n(r)``, written to ``Rad-{count}-profile.dat``.

    Args:
        psi (torch.Tensor): Wavefunction of shape ``(n_r, n_phi, n_z)``.
        count (int): Snapshot index used in the file name.
        r (torch.Tensor): Radial grid, in dimensionless units.
        n_r (int): Number of radial grid points.
        a_ho (float): Harmonic-oscillator length in metres.
        dphi (float): Azimuthal grid spacing.
        dz (float): Axial grid spacing.
    """
    file_name = f'Rad-{count:003}-profile.dat'
    with open(file_name, 'w') as f:
        for i in range(n_r):
            r_phys = float(r[i]) * a_ho * 1e6
            n_r_val = torch.sum(torch.abs(psi[i, :, :]) ** 2) * dphi * dz
            f.write(f'{r_phys},{float(n_r_val)}\n')


def save_figure_phase_cylindrical(phase, frame, z_idx=None):
    r"""
    Save an image of the wavefunction phase on the :math:`(r, \varphi)` plane
    at a fixed :math:`z` index.

    Args:
        phase (torch.Tensor): Phase tensor of shape ``(n_r, n_phi, n_z)``; a
            complex tensor is reduced to its real part.
        frame (int): Snapshot index used in the title and the file name.
        z_idx (int): Index along :math:`z` to slice at. Defaults to the
            midpoint ``n_z // 2``.
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
    r"""
    Write the superfluid velocity field :math:`(v_r, v_\varphi)` on the
    :math:`(r, \varphi)` plane at a fixed :math:`z` index.

    This suits pancake-geometry BECs. In dimensionless units
    (:math:`\hbar / m = 1`) the components are

    .. math::

        v_r(r, \varphi)
            &= \frac{\operatorname{Im}
               \bigl(\psi^{*}\, \partial_r \psi\bigr)}
               {\lvert \psi \rvert^{2}}, \\
        v_\varphi(r, \varphi)
            &= \frac{\operatorname{Im}
               \bigl(\psi^{*}\, r^{-1} \partial_\varphi \psi\bigr)}
               {\lvert \psi \rvert^{2}},

    where the radial derivative is evaluated with central finite differences
    (one-sided at the two ends) and the azimuthal one spectrally, through a DFT
    in :math:`\varphi`.

    Grid points whose density is below :math:`10^{-12}` of the peak are set to
    zero. The cut is relative because :math:`\psi` is normalised over the whole
    grid, so an absolute threshold would mean something different at every
    resolution.

    The format is ``r_μm, phi_rad, vr, v_phi, |v|``, written to
    ``V-{count}-cd.dat``.

    Args:
        psi (torch.Tensor): Wavefunction of shape ``(n_r, n_phi, n_z)``.
        count (int): Snapshot index used in the file name.
        r (torch.Tensor): Radial grid, in dimensionless units.
        phi (torch.Tensor): Azimuthal grid, in radians.
        n_r (int): Number of radial grid points.
        n_phi (int): Number of azimuthal grid points.
        a_ho (float): Harmonic-oscillator length in metres.
        dr (float): Radial grid spacing.
        m_modes (torch.Tensor): Azimuthal mode indices of shape ``(n_phi,)``,
            as returned by ``init_grid``.
        z_idx (int): Index along :math:`z` to slice at. Defaults to
            ``n_z // 2``.
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
