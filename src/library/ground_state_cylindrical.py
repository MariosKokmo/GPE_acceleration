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

        # 2. Radial kinetic for every azimuthal mode in one batched matmul
        #    (see GPECylindricalLibrary._apply_radial_eigen); the diagonal
        #    factor is Λ itself because we apply T rather than exp(-i Δτ T).
        V_all, lam_all = cyl._stacked_radial(eigvecs_dict, eigvals_dict, m_modes)
        Tr_psi = cyl._apply_radial_eigen(
            psi_m, lam_all.to(torch.cdouble).unsqueeze(-1), V_all, sqrt_r
        )

        # 3. Axial kinetic: spectral derivative over the whole array at once
        psi_kz = torch.fft.fft(psi_m, dim=2, norm="forward")
        kz2 = (kz.reshape(1, 1, n_z) ** 2).to(torch.cdouble)
        Tz_psi = torch.fft.ifft(0.5 * kz2 * psi_kz, dim=2, norm="forward")

        # 4. IDFT over φ → physical space
        return torch.fft.ifft(Tr_psi + Tz_psi, dim=1, norm="ortho")

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

        # μ = ⟨ψ|H|ψ⟩  (cylindrical inner product).  H is Hermitian so μ is
        # real; take the real part rather than the modulus, which would
        # silently flip the sign of a negative μ and drive the descent the
        # wrong way.
        mu = torch.sum(psi.conj() * dpsi * r_w) * dV
        mu = mu.real

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
    def find_ground_state(sim_params, system, file_name, device, max_iterations=200000):
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
            Simulation configuration. Only the interaction strength ``u`` is
            read from here (falling back to the value derived from
            :class:`CONSTANTS` when absent).
        system : object
            System object with ``simulation_parameters`` dict and
            ``system.uext.potential`` giving the external potential on the
            (n_r, n_phi, n_z) cylindrical grid.
        file_name : str
            Output path for the serialised ground state.
        device : torch.device
        max_iterations : int
            Hard cap on descent iterations so a state that never meets the
            tolerance cannot spin forever.

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

        # Dimensionless interaction strength — prefer the value the
        # configuration already derived so there is one source of truth.
        u = params.get("u")
        if u is None and isinstance(sim_params, dict):
            u = sim_params.get("u")
        if u is None:
            u = 4.0 * math.pi * CONSTANTS.nat * CONSTANTS.ascat / a_ho

        # Imaginary-time step: stability requires Δτ ≪ (min grid spacing)²
        dtau = 0.05 * min(dr, dz) ** 2

        # Thomas-Fermi seed: ψ_TF = sqrt(max(μ_TF − V_ext, 0))
        mu_TF = 0.5 * (15.0 / (4.0 * math.pi) * u) ** (2.0 / 5.0)
        v_real = uext.real if torch.is_complex(uext) else uext
        psi = torch.sqrt(torch.clamp(mu_TF - v_real, min=0.0)).to(torch.cdouble)

        # Weak or absent interactions put μ_TF below the trap minimum, leaving
        # the Thomas-Fermi profile identically zero — normalising that gives a
        # field of NaNs the descent can never recover from. Fall back to a
        # Gaussian, the exact non-interacting ground state of a harmonic trap.
        if not bool(torch.any(psi.real > 0)):
            psi = torch.exp(-(v_real - v_real.min())).to(torch.cdouble)
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
            if not (np.isfinite(energy_value) and np.isfinite(tol_value)):
                # Every comparison against NaN is False, so without this the
                # loop would silently run to max_iterations.
                raise FloatingPointError(
                    f"Ground state diverged at iteration {iteration}: "
                    f"energy={energy_value}, residual={tol_value}"
                )
            # Relative change, guarded against a vanishing energy scale.
            scale = abs(energy_value) if energy_value != 0.0 else 1.0
            test_e = (energy_old - energy_value) / scale

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
            elif iteration >= max_iterations:
                print(
                    f"Ground state did not converge in {max_iterations} iterations "
                    f"(residual {tol_value:.3e}, target 1e-5). Writing the current state."
                )
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

        Raises
        ------
        ValueError
            If the file does not hold exactly ``n_r*n_phi*n_z`` points.
        """
        matrix = pd.read_csv(data, header=None, names=["real", "imag"])
        matrix["real"] = matrix["real"].str.strip(" (")
        matrix["imag"] = matrix["imag"].str.strip(" )")
        matrix = matrix.astype(np.float64)

        expected = n_r * n_phi * n_z
        if len(matrix) != expected:
            raise ValueError(
                f"Ground state file '{data}' holds {len(matrix)} points but the "
                f"{n_r}x{n_phi}x{n_z} grid needs {expected}. It was probably "
                f"written for a different grid resolution."
            )

        psi = matrix.iloc[:, 0].values + 1j * matrix.iloc[:, 1].values
        psi = psi.reshape((n_r, n_phi, n_z))
        return torch.from_numpy(psi)
