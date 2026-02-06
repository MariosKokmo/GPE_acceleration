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
        N_tot = torch.where(torch.abs(psi) ** 2 > 0, 1, 0).sum()
        g_x, g_y, g_z = space_grid
        d_sq = (g_x - center_x) ** 2 + (g_y - center_y) ** 2 + (g_z - center_z) ** 2
        rms = (torch.sum(d_sq * (torch.abs(psi) ** 2)) / N_tot) ** 0.5
        return rms

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
