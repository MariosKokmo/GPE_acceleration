"""Ground-state solver for the GPE in cylindrical coordinates.

Drop-in cylindrical counterpart to :mod:`src.library.ground_state`.

Algorithm
---------
Imaginary-time propagation via a steepest-descent (gradient-flow) scheme
identical to the Cartesian version, but every inner product and norm uses the
cylindrical volume element dV = r dr dφ dz, and the kinetic operator is
applied spectrally in (φ, z) and via the precomputed radial eigendecomposition
in r (see :class:`GPECylindricalLibrary`).

Expected ``system.simulation_parameters`` keys
-----------------------------------------------
Grid_resolution : (n_r, n_phi, n_z)
r_max           : outer radial boundary (same units as a_ho).
z_min, z_max    : axial extent.
a_ho            : harmonic oscillator length (m) — used to set u and dtau.

The cylindrical grid (r, φ, z, kz, m_modes, spacings) and the radial
eigendecomposition are built inside :meth:`find_ground_state` from the above
parameters, so no extra pre-computation is required by the caller.
"""

import math
import numpy as np
import pandas as pd
import torch
from src.library.gpe_cylindrical_library import GPECylindricalLibrary as cyl
from src.library.parameters import CONSTANTS
from src.utils.read_write_utils import write_psi


class CylindricalGroundState:

    # ------------------------------------------------------------------
    # Kinetic operator action
    # ------------------------------------------------------------------

    @staticmethod
    def apply_kinetic(
        psi: torch.Tensor,
        kz: torch.Tensor,
        m_modes: torch.Tensor,
        r: torch.Tensor,
        eigvecs_dict: dict,
        eigvals_dict: dict,
    ) -> torch.Tensor:
        """
        Apply the kinetic operator T to ψ in cylindrical coordinates.

        T ψ = T_r^m ψ + T_z ψ

        For each azimuthal mode m (obtained by DFT over φ):
          - T_r^m is applied via the precomputed eigendecomposition:
                T_r^m ψ_m = (1/√r) V_m Λ_m V_m^T (√r ψ_m)
          - T_z is applied spectrally:
                T_z ψ_m = IFFT(kz²/2 · FFT(ψ_m))

        This replaces the Cartesian IFFT(p²/2 · FFT(ψ)) used in the Cartesian
        steepest descent.

        Args:
            psi           : wavefunction (n_r, n_phi, n_z), complex.
            kz            : z-momentum grid (n_z,).
            m_modes       : azimuthal mode indices (n_phi,), DFT order.
            r             : radial grid (n_r,).
            eigvecs_dict  : {|m|: (n_r, n_r)} from build_radial_operators.
            eigvals_dict  : {|m|: (n_r,)}   from build_radial_operators.

        Returns:
            T|ψ⟩ — tensor of shape (n_r, n_phi, n_z).
        """
        n_r, n_phi, n_z = psi.shape
        sqrt_r = torch.sqrt(r).to(device=psi.device, dtype=torch.float64)

        # 1. DFT over φ → azimuthal modes
        psi_m = torch.fft.fft(psi, dim=1, norm="ortho")
        result_m = torch.zeros_like(psi_m)

        for phi_idx in range(n_phi):
            m_abs = int(abs(m_modes[phi_idx].item()))
            V = eigvecs_dict[m_abs].to(torch.cdouble)    # (n_r, n_r)
            lam = eigvals_dict[m_abs].to(torch.cdouble)  # (n_r,)

            psi_slice = psi_m[:, phi_idx, :]  # (n_r, n_z)

            # Radial kinetic: symmetrised eigenbasis application
            psi_w = sqrt_r.unsqueeze(-1).to(torch.cdouble) * psi_slice
            coeff = V.T @ psi_w                                     # project to eigenbasis
            Tr_psi = (V @ (lam.unsqueeze(-1) * coeff)               # propagate
                      ) / sqrt_r.unsqueeze(-1).to(torch.cdouble)    # remove weight

            # Axial kinetic: spectral derivative
            psi_kz = torch.fft.fft(psi_slice, dim=-1, norm="forward")
            kz2 = (kz.reshape(1, -1) ** 2).to(torch.cdouble)
            Tz_psi = torch.fft.ifft(0.5 * kz2 * psi_kz, dim=-1, norm="forward")

            result_m[:, phi_idx, :] = Tr_psi + Tz_psi

        # 2. IDFT over φ → physical space
        return torch.fft.ifft(result_m, dim=1, norm="ortho")

    # ------------------------------------------------------------------
    # Imaginary-time step
    # ------------------------------------------------------------------

    @staticmethod
    def steepest_descent(
        psi: torch.Tensor,
        dtau: float,
        kz: torch.Tensor,
        m_modes: torch.Tensor,
        r: torch.Tensor,
        eigvecs_dict: dict,
        eigvals_dict: dict,
        uext: torch.Tensor,
        dr: float,
        dphi: float,
        dz: float,
        u: float,
    ) -> tuple:
        """
        One imaginary-time steepest-descent step in cylindrical coordinates.

        Mirrors :meth:`GroundState.steepest_descent` exactly, replacing:
          - the Cartesian kinetic step IFFT(p²/2·FFT(ψ)) with
            :meth:`apply_kinetic` (cylindrical Laplacian),
          - the flat volume element d_x with r dr dφ dz in all inner products.

        The update rule is:
            ψ ← ψ − Δτ (H − μ) ψ    then re-normalise.

        Args:
            psi                     : current wavefunction (n_r, n_phi, n_z).
            dtau                    : imaginary-time step.
            kz                      : z-momentum grid.
            m_modes                 : azimuthal mode indices.
            r                       : radial grid.
            eigvecs_dict, eigvals_dict : from build_radial_operators.
            uext                    : external trapping potential.
            dr, dphi, dz            : grid spacings.
            u                       : contact interaction strength.

        Returns:
            (psi, energy, tol, mu) — updated wavefunction and scalar diagnostics.
        """
        r_w = r.reshape(-1, 1, 1)
        dV = dr * dphi * dz

        # H|ψ⟩ = T|ψ⟩ + V_tot|ψ⟩
        dpsi = CylindricalGroundState.apply_kinetic(
            psi, kz, m_modes, r, eigvecs_dict, eigvals_dict
        )
        utot = u * torch.abs(psi) ** 2 + uext
        dpsi = dpsi + utot * psi

        # μ = ⟨ψ|H|ψ⟩  (cylindrical inner product)
        mu = torch.sum(psi.conj() * dpsi * r_w) * dV
        mu = mu.abs()

        # Interaction energy  e_int = (u/2) ∫|ψ|⁴ r dr dφ dz
        interaction = 0.5 * u * torch.sum(torch.abs(psi) ** 4 * r_w) * dV

        # Residual  (H − μ)|ψ⟩
        dpsi = dpsi - mu * psi
        tol = torch.sum(torch.abs(dpsi) ** 2 * r_w) * dV

        # Energy estimate: E = μ − e_int
        energy = mu - interaction

        # Gradient-flow update
        psi = psi - dtau * dpsi
        psi = cyl.normalize(psi, r, dr, dphi, dz)

        return psi, energy, tol, mu

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    @staticmethod
    def find_ground_state(sim_params, system, file_name, device):
        """
        Compute and persist the cylindrical ground-state wavefunction.

        The initial state is seeded with a Thomas-Fermi profile and refined by
        imaginary-time steepest descent until the residual norm falls below
        1e-5 or the relative energy change vanishes.

        Expected ``system.simulation_parameters`` keys
        -----------------------------------------------
        Grid_resolution : (n_r, n_phi, n_z)
        r_max           : outer radial boundary.
        z_min, z_max    : axial extent.
        a_ho            : harmonic oscillator length (m).

        Parameters
        ----------
        sim_params : dict
            Simulation configuration (passed through; not read directly here).
        system : object
            System object with ``simulation_parameters`` dict and
            ``system.uext.potential`` giving the external potential on the
            (n_r, n_phi, n_z) cylindrical grid.
        file_name : str
            Output path for the serialised ground state.
        device : torch.device

        Returns
        -------
        torch.Tensor
            Normalised complex wavefunction of shape (n_r, n_phi, n_z).
        """
        params = system.simulation_parameters
        n_r, n_phi, n_z = params["Grid_resolution"]
        r_max = params["r_max"]
        z_min = params["z_min"]
        z_max = params["z_max"]
        a_ho = params["a_ho"]

        # Build cylindrical grid and radial operators
        r, phi, z, kz, m_modes, dr, dphi, dz, _ = cyl.init_grid(
            r_max, z_min, z_max, n_r, n_phi, n_z, device
        )
        eigvecs_dict, eigvals_dict = cyl.build_radial_operators(r, dr, m_modes, device)

        uext = system.uext.potential

        # Dimensionless interaction strength
        u = 4.0 * math.pi * CONSTANTS.nat * CONSTANTS.ascat / a_ho

        # Imaginary-time step: stability requires Δτ ≪ (min grid spacing)²
        dtau = 0.05 * min(dr, dz) ** 2

        # Thomas-Fermi seed: ψ_TF = sqrt(max(μ_TF − V_ext, 0))
        mu_TF = 0.5 * (15.0 / (4.0 * math.pi) * u) ** (2.0 / 5.0)
        psi = torch.zeros((n_r, n_phi, n_z), dtype=torch.cdouble, device=device)
        psi = torch.where(
            mu_TF > uext.real,
            torch.sqrt(torch.clamp(mu_TF - uext.real, min=0.0)) + 0j,
            psi,
        )
        psi = cyl.normalize(psi, r, dr, dphi, dz)

        # First step — initialise convergence state
        psi, energy, tol, mu = CylindricalGroundState.steepest_descent(
            psi, dtau, kz, m_modes, r, eigvecs_dict, eigvals_dict,
            uext, dr, dphi, dz, u
        )
        print(
            "{iter}\t{energy}\t{mu}\t{dtau}\t{tol}".format(
                iter="iter", energy="energy", mu="mu", dtau="dtau", tol="tol"
            )
        )
        energy_old = energy.real.item()

        done = False
        iteration = 0
        while not done:
            iteration += 1
            psi, energy, tol, mu = CylindricalGroundState.steepest_descent(
                psi, dtau, kz, m_modes, r, eigvecs_dict, eigvals_dict,
                uext, dr, dphi, dz, u
            )
            energy_value = energy.real.item()
            tol_value = tol.real.item()
            test_e = (energy_old - energy_value) / energy_value

            if test_e < 0:
                print(f"Changing dtau: {dtau}, relative change: {test_e:.3e}")
                dtau = dtau / 2.0
            energy_old = energy_value

            if iteration % 50 == 0:
                mu_value = mu.real.item()
                print(
                    f"{iteration:10}\t{energy_value:10.6f}\t{mu_value:10.6f}"
                    f"\t{dtau:10.3e}\t{tol_value:10.3e}"
                )

            if tol_value < 1e-5 or test_e == 0:
                done = True

        write_psi(file_name, psi, n_r, n_phi, n_z)
        return psi

    # ------------------------------------------------------------------
    # Load from file
    # ------------------------------------------------------------------

    @staticmethod
    def read_ground_state(data, n_r: int, n_phi: int, n_z: int) -> torch.Tensor:
        """
        Load a serialised cylindrical ground-state wavefunction from disk.

        The file format is identical to the Cartesian :meth:`GroundState.read_ground_state`:
        each row contains ``(real_part, imag_part)`` for one grid point, written
        in row-major order over (n_r, n_phi, n_z).

        Parameters
        ----------
        data : str or path-like
            Text file produced by :func:`write_psi`.
        n_r, n_phi, n_z : int
            Grid point counts used to reshape the flat data.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``(n_r, n_phi, n_z)`` on CPU.
        """
        matrix = pd.read_csv(data, header=None, names=["real", "imag"])
        matrix["real"] = matrix["real"].str.strip(" (")
        matrix["imag"] = matrix["imag"].str.strip(" )")
        matrix = matrix.astype(np.float64)

        psi = matrix.iloc[:, 0].values + 1j * matrix.iloc[:, 1].values
        psi = psi.reshape((n_r, n_phi, n_z))
        return torch.from_numpy(psi)
