"""
Common utilities for the GPE solver, including functions for wavefunction evolution, phase manipulation, and

other shared operations across different GPE variants (e.g., finite-temperature SGPE) and all the coordinate systems.

"""
import numpy as np
import torch

class CommonUtils:
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
        Extract the phase from the wavefunction, wrapped to (-pi, pi].

        Uses ``torch.angle``, which agrees exactly with the previous
        ``Im(log(psi/sqrt(|psi|^2)))`` wherever psi is non-zero and, unlike it,
        stays finite at a node. That form evaluated 0/0 at every point where
        the density vanishes — precisely a vortex core — and a single NaN
        contaminates the whole array as soon as it passes through an FFT.

        Args:
            psi (torch.Tensor): Wavefunction of the condensate.

        Returns:
            torch.Tensor: Phase of the condensate (real-valued, 0 at a node).
        """
        return torch.angle(psi)

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
