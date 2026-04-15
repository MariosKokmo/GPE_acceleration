import numpy as np
import torch
from .parameters import CONSTANTS

class GPELibrary:
    @staticmethod
    def init_grid(
        x_min: list,
        x_max: list,
        dx: list,
        dp: list,
        w: float,
        n1: int,
        n2: int,
        n3: int,
        device: torch.device
    ) -> tuple:
        """
        Initializes the grid for x and p spaces and the external potential.

        Args:
            x_min (list): Minimum values for the x-axis in each dimension.
            x_max (list): Maximum values for the x-axis in each dimension.
            dx (list): Grid spacing in real space for each dimension.
            dp (list): Grid spacing in momentum space for each dimension.
            w (float): Frequency of the external potential.
            n1, n2, n3 (int): Number of grid points in each dimension.
            device (torch.device): Device to allocate tensors (CPU or GPU).

        Returns:
            tuple: (x1, x2, x3, p1, p2, p3, p_sq, space_grid, p_grid)
        """
        x1 = torch.zeros((1, n1), dtype=torch.float64, device=device)
        x2 = torch.zeros((1, n2), dtype=torch.float64, device=device)
        x3 = torch.zeros((1, n3), dtype=torch.float64, device=device)
        p1 = torch.zeros((1, n1), dtype=torch.float64, device=device)
        p2 = torch.zeros((1, n2), dtype=torch.float64, device=device)
        p3 = torch.zeros((1, n3), dtype=torch.float64, device=device)
        p_sq = torch.zeros((n1, n2, n3), dtype=torch.float64, device=device)

        x1 = x_min[0] + torch.arange(n1, dtype=torch.float64) * dx[0]
        p1[0][:n1 // 2] = dp[0] * torch.arange(n1 // 2)
        p1[0][n1 // 2:] = dp[0] * (torch.arange(n1 // 2, n1) - n1)

        x2 = x_min[1] + torch.arange(n2, dtype=torch.float64) * dx[1]
        p2[0][:n2 // 2] = dp[1] * torch.arange(n2 // 2)
        p2[0][n2 // 2:] = dp[1] * (torch.arange(n2 // 2, n2) - n2)

        x3 = x_min[2] + torch.arange(n3, dtype=torch.float64) * dx[2]
        p3[0][:n3 // 2] = dp[2] * torch.arange(n3 // 2)
        p3[0][n3 // 2:] = dp[2] * (torch.arange(n3 // 2, n3) - n3)

        g_px, g_py, g_pz = torch.meshgrid(p1[0], p2[0], p3[0])
        p_sq = g_px**2 + g_py**2 + g_pz**2
        p_sq = p_sq.to(device=device)

        p_grid = (g_px.to(device=device), g_py.to(device=device), g_pz.to(device=device))

        g_x, g_y, g_z = torch.meshgrid(x1, x2, x3)
        space_grid = (g_x.to(device=device), g_y.to(device=device), g_z.to(device=device))
        return x1, x2, x3, p1, p2, p3, p_sq, space_grid, p_grid

    @staticmethod
    def x_evolution(
        psi1: torch.Tensor,
        utot1: torch.Tensor,
        dtau: float,
        factor: float = 0.5
    ) -> torch.Tensor:
        """
        Real-space evolution step for the wavefunction.

        Args:
            psi1 (torch.Tensor): Wavefunction of the system.
            utot1 (torch.Tensor): Trapping potential at this time step.
            dtau (float): Time evolution step.
            factor (float, optional): Splitting factor (default 0.5).

        Returns:
            torch.Tensor: Updated wavefunction.
        """
        return torch.exp(-factor * dtau * 1j * utot1) * psi1

    @staticmethod
    def p_evolution(
        psi1: torch.Tensor,
        dtau: float,
        p_sq: torch.Tensor
    ) -> torch.Tensor:
        """
        Momentum-space evolution step for the wavefunction.

        Args:
            psi1 (torch.Tensor): Wavefunction of the system.
            dtau (float): Time evolution step.
            p_sq (torch.Tensor): Squared momentum grid.

        Returns:
            torch.Tensor: Updated wavefunction.
        """
        psiF = torch.fft.fftn(psi1, norm='forward')
        psiF = torch.exp(-1j * dtau * 0.5 * p_sq) * psiF
        return torch.fft.ifftn(psiF, norm='forward')

    @staticmethod
    def normalize(
        phi: torch.Tensor,
        d_x: float
    ) -> torch.Tensor:
        """
        Normalize the wavefunction.

        Args:
            phi (torch.Tensor): Wavefunction to be normalized.
            d_x (float): Grid cell volume.

        Returns:
            torch.Tensor: Normalized wavefunction.
        """
        return phi / torch.sqrt(d_x * torch.sum(torch.abs(phi) ** 2))

    @staticmethod
    def update_phase(
        psi1: torch.Tensor,
        phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Update the phase of the wavefunction.

        Args:
            psi1 (torch.Tensor): Wavefunction of the system.
            phase (torch.Tensor): Phase to be applied.

        Returns:
            torch.Tensor: Updated wavefunction.
        """
        return psi1 * torch.exp(phase * 1j)

    @staticmethod
    def extract_phase(
        psi: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract the phase from the wavefunction.

        Args:
            psi (torch.Tensor): Wavefunction of the condensate.

        Returns:
            torch.Tensor: Phase of the condensate.
        """
        return torch.imag(torch.log(psi / torch.sqrt(torch.abs(psi) ** 2)))

    @staticmethod
    def add_phase(
        cur_phase: torch.Tensor,
        added_phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Add an extra phase to the current phase.

        Args:
            cur_phase (torch.Tensor): Current phase of the condensate wavefunction.
            added_phase (torch.Tensor): Additional phase to be added.

        Returns:
            torch.Tensor: Updated phase.
        """
        return cur_phase + added_phase

    @staticmethod
    def split_step_step(
        psi1: torch.Tensor,
        utot1: torch.Tensor,
        dtau: float,
        p_sq: torch.Tensor,
        d_x: float
    ) -> torch.Tensor:
        """
        Perform a step of the split-step Fourier transform.

        Args:
            psi1 (torch.Tensor): Wavefunction of the system.
            utot1 (torch.Tensor): Total potential.
            dtau (float): Time evolution step.
            p_sq (torch.Tensor): Squared momentum grid.
            d_x (float): Product of the grid dimensions.

        Returns:
            torch.Tensor: Updated wavefunction.
        """
        psi1 = GPELibrary.x_evolution(psi1, utot1, dtau)
        psi1 = GPELibrary.p_evolution(psi1, dtau, p_sq)
        psi1 = GPELibrary.x_evolution(psi1, utot1, dtau)
        return GPELibrary.normalize(psi1, d_x)

    @staticmethod
    def mod_grad_psi(
        psi: torch.Tensor,
        p_axes: list
    ) -> torch.Tensor:
        """
        Calculate the modulus of the gradient of the wavefunction.

        Args:
            psi (torch.Tensor): Condensate wavefunction.
            p_axes (list): Momentum space grid components for each axis.

        Returns:
            torch.Tensor: Modulus of the gradient of the wavefunction.
        """
        dim = len(psi.shape)
        if dim == 3:
            px, py, pz = torch.meshgrid(p_axes[0], p_axes[1], p_axes[2])
            P = torch.stack((px, py, pz))
            spec_x = P[0] * torch.fft.fftn(psi, norm='forward') * 1j
            grad_x = torch.fft.ifftn(spec_x, norm='forward').real
            spec_y = P[1] * torch.fft.fftn(psi, norm='forward') * 1j
            grad_y = torch.fft.ifftn(spec_y, norm='forward').real
            spec_z = P[2] * torch.fft.fftn(psi, norm='forward') * 1j
            grad_z = torch.fft.ifftn(spec_z, norm='forward').real
            return torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
        elif dim == 2:
            px, py = torch.meshgrid(p_axes[0], p_axes[1])
            P = torch.stack((px, py))
            spec_x = P[0] * torch.fft.fft2(psi, norm='forward') * 1j
            grad_x = torch.fft.ifft2(spec_x, norm='forward').real
            spec_y = P[1] * torch.fft.fft2(psi, norm='forward') * 1j
            grad_y = torch.fft.ifft2(spec_y, norm='forward').real
            return torch.sqrt(grad_x ** 2 + grad_y ** 2)
        elif dim == 1:
            spect_x = p_axes[0] * torch.fft.fft(psi, norm='forward') * 1j
            return torch.fft.ifft(spect_x, norm='forward').real
        return None

    @staticmethod
    def calculate_energy_allocation(
        psi: torch.Tensor,
        Vext: torch.Tensor,
        p_grid: tuple,
        **parameters
    ) -> dict:
        """
        Calculate energy allocation for the condensate.

        Args:
            psi (torch.Tensor): BEC wavefunction.
            Vext (torch.Tensor): External potential.
            p_grid (tuple): Momentum space grid.
            **parameters: Additional parameters (e.g., interaction strength 'u').

        Returns:
            dict: Energy terms (kinetic, potential, interaction, total).
        """
        u = parameters['u']
        grad_sq = torch.pow(GPELibrary.mod_grad_psi(psi, p_grid), 2)
        e_kin = 0.5 * torch.sum(grad_sq)
        e_pot = torch.sum(Vext * torch.abs(psi) ** 2)
        e_int = 0.5 * u * torch.sum(torch.abs(psi) ** 4)
        E_total = e_kin + e_pot + e_int
        return {
            'e_kin': e_kin,
            'e_pot': e_pot,
            'e_int': e_int,
            'E_total': E_total
        }
    
    @staticmethod
    def calculate_density_peak(
        psi: torch.Tensor
    ) -> tuple:
        """
        Calculate the maximum density and its position in the wavefunction.

        Args:
            psi (torch.Tensor): BEC normalised wavefunction.

        Returns:
            tuple: (max_density, peak_indices) where max_density is a scalar tensor
                   and peak_indices is a tuple of integers (i, j, k) representing
                   the grid position of the maximum density.
        """
        density = torch.abs(psi) ** 2
        density_flat = density.flatten()
        peak_position = torch.argmax(density_flat).item()
        max_density = density_flat[peak_position]
        
        # Manual unravel_index for 3D tensor
        shape = density.shape
        k = peak_position % shape[2]
        j = (peak_position // shape[2]) % shape[1]
        i = peak_position // (shape[2] * shape[1])
        
        return max_density, (i, j, k)

class GPE2DLibrary(GPELibrary):
    @staticmethod
    def create_vortices(
        vortices: np.ndarray,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        n1: int,
        n2: int,
        n3: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Creates vortices on the condensate by calculating a new phase to be added (2D BEC).

        Args:
            vortices (np.ndarray): Shape (3, number_of_vortices), rows are x positions, z positions, vortex charges.
            x1, x2, x3 (torch.Tensor): Real space axes for each dimension.
            n1, n2, n3 (int): Number of grid points in each dimension.
            device (torch.device): Device to allocate tensors (CPU or GPU).

        Returns:
            torch.Tensor: Updated phase of the condensate.
        """
        if vortices is None:
            return None
        number_of_vortices = vortices.shape[1]
        phase = torch.zeros((n1, n2, n3), dtype=torch.cdouble, device=device)
        for n in range(number_of_vortices):
            vx = vortices[0][n]
            vz = vortices[1][n]
            q = vortices[2][n]
            for i in range(n3):
                for k in range(n1):
                    if (i != (vz + n3 // 2)) or (k >= (vx + n1 // 2)):
                        y = x3[i] - x3[vz + n3 // 2]
                        t = x1[k] - x1[vx + n1 // 2]
                        x = torch.sqrt(t ** 2 + y ** 2) + t
                        phase[k, :, i] += 2 * q * torch.atan2(y, x)
                    else:
                        phase[k, :, i] += q * CONSTANTS.pi
        phase[phase.isnan()] = 0 + 0j
        return phase

    @staticmethod
    def repetitive_imprint(
        psi1: torch.Tensor,
        repetitive_phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Re-imprint the wavefunction by adding the repetitive phase (2D BEC).

        Args:
            psi1 (torch.Tensor): Wavefunction of the system.
            repetitive_phase (torch.Tensor): Repetitive phase to be added.

        Returns:
            torch.Tensor: Updated wavefunction.
        """
        return GPELibrary.update_phase(psi1, repetitive_phase)

    @staticmethod
    def calculate_velocity2D(
        phase2D: torch.Tensor,
        p_grid: tuple
    ) -> tuple:
        """
        Calculate the velocity of the condensate in 2D (multiply result by hbar/m for velocity).

        Args:
            phase2D (torch.Tensor): Phase of the condensate wavefunction in 2D.
            p_grid (tuple): Momentum space grid (px, py).

        Returns:
            tuple: (gradient magnitude, gradient angle)
        """
        spect_x = p_grid[0] * torch.fft.fftn(phase2D) * 1j
        spect_y = p_grid[1] * torch.fft.fftn(phase2D) * 1j
        grad_x = torch.fft.ifftn(spect_x).real
        grad_y = torch.fft.ifftn(spect_y).real
        grad_mod = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_angle = torch.atan2(grad_y, grad_x)
        return grad_mod, grad_angle

    @staticmethod
    def rms_radius(
        psi: torch.Tensor,
        center: list,
        space_grid: tuple
    ) -> torch.Tensor:
        """
        Calculate the RMS radius of the condensate.

        Args:
            psi (torch.Tensor): Normalized wavefunction.
            center (list): Centers of the space axes (x1, x2, x3).
            space_grid (tuple): Meshgrid of the space.

        Returns:
            torch.Tensor: RMS radius of the condensate.
        """
        center_x, center_y, center_z = center
        # Use sum of probability densities as the normalization factor for weighted average
        total_density = torch.sum(torch.abs(psi) ** 2)
        
        g_x, g_y, g_z = space_grid
        d_sq = (g_x - center_x) ** 2 + (g_y - center_y) ** 2 + (g_z - center_z) ** 2
        
        # RMS = sqrt( sum(r^2 * density) / sum(density) )
        rms = (torch.sum(d_sq * (torch.abs(psi) ** 2)) / total_density) ** 0.5
        return rms

    @staticmethod
    def create_dark_soliton(
        x1: torch.Tensor,
        x3: torch.Tensor,
        n1: int,
        n2: int,
        n3: int,
        positions: list,
        widths: list,
        axes: list,
        greyness: list = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Create a multiplicative dark-soliton profile for the wavefunction.

        Each soliton is a stripe of suppressed density across the condensate.
        The profile for a single dark soliton along coordinate *r* is

            f(r) = cos(alpha) * tanh( cos(alpha) * (r - r0) / width ) + i * sin(alpha)

        where ``alpha = 0`` gives a stationary black soliton (full density dip
        plus a pi phase jump) and ``0 < alpha < pi/2`` gives a grey (moving)
        soliton with a shallower dip.

        When multiple solitons are requested the individual profiles are
        multiplied together.

        Args:
            x1 (torch.Tensor): 1-D real-space axis along the first grid dimension.
            x3 (torch.Tensor): 1-D real-space axis along the third grid dimension.
            n1, n2, n3 (int): Number of grid points in each dimension.
            positions (list[float]): Centre positions of each soliton (in the
                same units as the corresponding axis).
            widths (list[float]): Characteristic width of each soliton.  For a
                BEC this is typically the healing length xi.
            axes (list[int]): Axis for each soliton: ``1`` for x (stripe
                perpendicular to x1) or ``3`` for z (stripe perpendicular to x3).
            greyness (list[float], optional): Grey-soliton angle alpha for
                each soliton in radians.  ``0`` = black (default), values up
                to ``pi/2`` make the soliton progressively greyer / faster.
            device (torch.device): Device for the output tensor.

        Returns:
            torch.Tensor: Complex-valued tensor of shape ``(n1, n2, n3)`` to
            be multiplied element-wise with the wavefunction.
        """
        n_solitons = len(positions)
        if greyness is None:
            greyness = [0.0] * n_solitons

        # Build 3-D coordinate grids (only the two relevant axes matter)
        grid_x1 = x1.to(device=device, dtype=torch.float64)
        grid_x3 = x3.to(device=device, dtype=torch.float64)
        # shape: (n1, 1, 1) and (1, 1, n3)
        gx = grid_x1.reshape(n1, 1, 1).expand(n1, n2, n3)
        gz = grid_x3.reshape(1, 1, n3).expand(n1, n2, n3)

        mask = torch.ones(n1, n2, n3, dtype=torch.cdouble, device=device)

        for pos, w, ax, alpha in zip(positions, widths, axes, greyness):
            if ax == 1:
                r = gx
            elif ax == 3:
                r = gz
            else:
                raise ValueError(f"Soliton axis must be 1 (x) or 3 (z), got {ax}")

            cos_a = float(np.cos(alpha))
            sin_a = float(np.sin(alpha))
            profile = cos_a * torch.tanh(cos_a * (r - pos) / w) + 1j * sin_a
            mask = mask * profile

        return mask

    @staticmethod
    def imprint_dark_soliton(
        psi: torch.Tensor,
        soliton_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply a dark-soliton mask to the wavefunction.

        Unlike vortex imprinting (which adds pure phase), this multiplies
        the wavefunction by a complex profile that modifies both amplitude
        and phase simultaneously.

        Args:
            psi (torch.Tensor): Current wavefunction (n1, n2, n3).
            soliton_mask (torch.Tensor): Profile returned by
                :meth:`create_dark_soliton`.

        Returns:
            torch.Tensor: Updated wavefunction.
        """
        return psi * soliton_mask

    @staticmethod
    def calculate_cross_section_line(
        psi: torch.Tensor,
        axis: int = 1
    ) -> torch.Tensor:
        """
        Calculate the column density on a line that crosses the condensate.

        Args:
            psi (torch.Tensor): BEC wavefunction.
            axis (int, optional): Axis for cross-section (1 for x, 2 for z). Default is 1.

        Returns:
            torch.Tensor: Cross-section line through the center of the BEC.
        """
        n1, n2, n3 = psi.shape
        if axis == 1:
            return torch.sum(torch.abs(psi[n1 // 2, :, :]) ** 2, dim=0)
        elif axis == 2:
            return torch.sum(torch.abs(psi[:, :, n3 // 2]) ** 2, dim=0)
        return None

class GPE3DLibrary(GPELibrary):
    """
    Extensions of the GPE library for fully three-dimensional BEC simulations.

    Provides tools for 3D-specific topological structures (vortex rings,
    vortex lines), density diagnostics (column densities, 2D slices), and
    observables (superfluid velocity field, angular momentum components).
    """

    @staticmethod
    def create_vortex_ring(
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        n1: int,
        n2: int,
        n3: int,
        ring_radius: float,
        center: tuple,
        axis: int,
        charge: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Create the phase texture for a quantized vortex ring.

        The ring is a circle of radius ``ring_radius`` centred at ``center``
        lying in the plane perpendicular to ``axis``.  The phase winds by
        2π * charge around the vortex core (the ring itself) using toroidal
        coordinates:

            phi = atan2(x_axis - c_axis, rho - ring_radius)

        where rho is the cylindrical radius in the plane of the ring.

        Args:
            x1, x2, x3 (torch.Tensor): 1-D real-space axes.
            n1, n2, n3 (int): Grid point counts per dimension.
            ring_radius (float): Radius of the vortex ring.
            center (tuple): (c1, c2, c3) – 3-D centre of the ring in grid units.
            axis (int): 1, 2, or 3 – axis the ring encircles (ring lies in the
                perpendicular plane).
            charge (int): Topological charge (winding number). Default 1.
            device (torch.device): Computation device.

        Returns:
            torch.Tensor: Real phase tensor of shape (n1, n2, n3).
        """
        if axis not in (1, 2, 3):
            raise ValueError(f"axis must be 1, 2, or 3; got {axis}")

        c1, c2, c3 = center
        gx = x1.to(device=device, dtype=torch.float64).reshape(n1, 1, 1).expand(n1, n2, n3)
        gy = x2.to(device=device, dtype=torch.float64).reshape(1, n2, 1).expand(n1, n2, n3)
        gz = x3.to(device=device, dtype=torch.float64).reshape(1, 1, n3).expand(n1, n2, n3)

        if axis == 1:
            rho = torch.sqrt((gy - c2) ** 2 + (gz - c3) ** 2)
            phi = torch.atan2(gx - c1, rho - ring_radius)
        elif axis == 2:
            rho = torch.sqrt((gx - c1) ** 2 + (gz - c3) ** 2)
            phi = torch.atan2(gy - c2, rho - ring_radius)
        else:  # axis == 3
            rho = torch.sqrt((gx - c1) ** 2 + (gy - c2) ** 2)
            phi = torch.atan2(gz - c3, rho - ring_radius)

        phase = charge * phi
        phase[phase.isnan()] = 0.0
        return phase

    @staticmethod
    def create_vortex_lines(
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        n1: int,
        n2: int,
        n3: int,
        positions: list,
        charges: list,
        axis: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Create the phase texture for one or more straight vortex lines.

        Each vortex line runs parallel to ``axis``.  Its core intersects the
        perpendicular plane at the coordinate pair given in ``positions``:

        - axis=1: each position is (c2, c3)
        - axis=2: each position is (c1, c3)
        - axis=3: each position is (c1, c2)

        Args:
            x1, x2, x3 (torch.Tensor): 1-D real-space axes.
            n1, n2, n3 (int): Grid point counts.
            positions (list[tuple]): Core positions in the perpendicular plane.
            charges (list[int]): Topological charges for each line.
            axis (int): 1, 2, or 3 – axis the vortex lines run along.
            device (torch.device): Computation device.

        Returns:
            torch.Tensor: Real phase tensor of shape (n1, n2, n3).
        """
        if axis not in (1, 2, 3):
            raise ValueError(f"axis must be 1, 2, or 3; got {axis}")

        gx = x1.to(device=device, dtype=torch.float64).reshape(n1, 1, 1).expand(n1, n2, n3)
        gy = x2.to(device=device, dtype=torch.float64).reshape(1, n2, 1).expand(n1, n2, n3)
        gz = x3.to(device=device, dtype=torch.float64).reshape(1, 1, n3).expand(n1, n2, n3)

        phase = torch.zeros(n1, n2, n3, dtype=torch.float64, device=device)
        for (ca, cb), q in zip(positions, charges):
            if axis == 1:
                da, db = gy - ca, gz - cb
            elif axis == 2:
                da, db = gx - ca, gz - cb
            else:  # axis == 3
                da, db = gx - ca, gy - cb
            phase = phase + q * torch.atan2(db, da)

        phase[phase.isnan()] = 0.0
        return phase

    @staticmethod
    def column_density(
        psi: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        """
        Compute the column density by integrating |psi|² along the given axis.

        Args:
            psi (torch.Tensor): BEC wavefunction (n1, n2, n3).
            axis (int): 1, 2, or 3 – axis to integrate along.

        Returns:
            torch.Tensor: 2-D column density tensor in the remaining plane.
        """
        if axis not in (1, 2, 3):
            raise ValueError(f"axis must be 1, 2, or 3; got {axis}")
        return torch.sum(torch.abs(psi) ** 2, dim=axis - 1)

    @staticmethod
    def cross_section_plane(
        psi: torch.Tensor,
        axis: int,
        index: int = None,
    ) -> torch.Tensor:
        """
        Extract a 2-D density slice orthogonal to the given axis.

        Args:
            psi (torch.Tensor): BEC wavefunction (n1, n2, n3).
            axis (int): 1, 2, or 3 – normal axis of the slice.
            index (int, optional): Grid index along ``axis``. Defaults to the
                centre of that axis.

        Returns:
            torch.Tensor: 2-D density slice.
        """
        if axis not in (1, 2, 3):
            raise ValueError(f"axis must be 1, 2, or 3; got {axis}")
        density = torch.abs(psi) ** 2
        idx = index if index is not None else psi.shape[axis - 1] // 2
        if axis == 1:
            return density[idx, :, :]
        elif axis == 2:
            return density[:, idx, :]
        else:
            return density[:, :, idx]

    @staticmethod
    def calculate_velocity3D(
        psi: torch.Tensor,
        p_grid: tuple,
    ) -> tuple:
        """
        Compute the 3-D superfluid velocity field using spectral derivatives.

        In dimensionless units (ℏ/m = 1) the superfluid velocity is:

            v_i = Im( ψ* ∂_i ψ ) / |ψ|²

        where the spatial derivative is evaluated spectrally.  Velocity is
        set to zero wherever the density vanishes.

        Args:
            psi (torch.Tensor): BEC wavefunction (n1, n2, n3).
            p_grid (tuple): (px, py, pz) – 3-D momentum meshgrids.

        Returns:
            tuple: (v1, v2, v3) – velocity component tensors, each (n1, n2, n3).
        """
        px, py, pz = p_grid
        psi_f = torch.fft.fftn(psi, norm='forward')
        density = torch.abs(psi) ** 2
        velocities = []
        for p in (px, py, pz):
            dpsi = torch.fft.ifftn(1j * p * psi_f, norm='forward')
            numerator = torch.imag(psi.conj() * dpsi)
            v = torch.where(density > 0, numerator / density, torch.zeros_like(numerator))
            velocities.append(v)
        return tuple(velocities)

    @staticmethod
    def angular_momentum(
        psi: torch.Tensor,
        space_grid: tuple,
        p_grid: tuple,
        component: int,
    ) -> torch.Tensor:
        """
        Calculate the expectation value of one angular momentum component.

        In units of ℏ:

            ⟨L_1⟩ = ⟨ψ | -i(x2 ∂_3 - x3 ∂_2) | ψ⟩
            ⟨L_2⟩ = ⟨ψ | -i(x3 ∂_1 - x1 ∂_3) | ψ⟩
            ⟨L_3⟩ = ⟨ψ | -i(x1 ∂_2 - x2 ∂_1) | ψ⟩

        Spatial derivatives are evaluated spectrally.

        Args:
            psi (torch.Tensor): Normalised BEC wavefunction (n1, n2, n3).
            space_grid (tuple): (g_x, g_y, g_z) – 3-D real-space meshgrids.
            p_grid (tuple): (px, py, pz) – 3-D momentum meshgrids.
            component (int): 1, 2, or 3.

        Returns:
            torch.Tensor: Scalar expectation value ⟨L_component⟩ (in ℏ).
        """
        if component not in (1, 2, 3):
            raise ValueError(f"component must be 1, 2, or 3; got {component}")

        gx, gy, gz = space_grid
        px, py, pz = p_grid
        psi_f = torch.fft.fftn(psi, norm='forward')

        def _d(p_comp):
            return torch.fft.ifftn(1j * p_comp * psi_f, norm='forward')

        if component == 1:
            Lpsi = -1j * (gy * _d(pz) - gz * _d(py))
        elif component == 2:
            Lpsi = -1j * (gz * _d(px) - gx * _d(pz))
        else:  # component == 3
            Lpsi = -1j * (gx * _d(py) - gy * _d(px))

        return torch.real(torch.sum(psi.conj() * Lpsi))

