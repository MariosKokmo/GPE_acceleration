"""
Equation objects for the GPE and its extensions.

Each class encapsulates the real-space part of the split-step operator
(the term that sits between the two half-kinetic steps).  Equation objects
are created once before the simulation loop and called every iteration:

    equation = GPE_base(params, system.uext)

    for iteration in range(kmax):
        t   = dt * iteration * omega_ho
        utot = equation(psi, t)
        psi  = split_step(psi, utot, dtau, ...)

The Potential object passed in handles all time dependence via its .evol(t)
method, so the equation itself is stateless between calls.
"""
import torch
from abc import ABC, abstractmethod


def select_equation(equation_type: str, params: dict, potential) -> "Equation":
    """
    Instantiate and return an Equation object by name.

    Available types
    ---------------
    ``"gpe"``       — standard zero-temperature GPE (requires ``"u"`` in params).
    ``"gpe_3body"`` — GPE with three-body loss (requires ``"u"`` and ``"k3"``).
    ``"custom"``    — returns None; build a CustomEquation manually.

    Args:
        equation_type: string identifier (case-insensitive).
        params:        simulation parameters dict.
        potential:     Potential object whose ``.evol(t)`` returns V_ext(t).

    Returns:
        Equation instance, or None for ``"custom"``.
    """
    available = ["gpe", "gpe_3body", "custom"]
    key = equation_type.strip().lower()
    if key not in available:
        raise ValueError(
            f"Unknown equation type '{equation_type}'. "
            f"Available: {available}"
        )
    if key == "gpe":
        return GPE_base(params, potential)
    if key == "gpe_3body":
        return GPE_3body_loss(params, potential)
    return None  # "custom": caller must construct CustomEquation directly


class Equation(ABC):
    """Base class for all GPE equation objects."""

    @abstractmethod
    def __call__(self, psi: torch.Tensor, t: float) -> torch.Tensor:
        """
        Return the real-space operator evaluated at wavefunction psi and time t.

        Args:
            psi: complex wavefunction tensor.
            t:   current simulation time (dimensionless).

        Returns:
            torch.Tensor of the same shape as psi.
        """


class GPE_base(Equation):
    """
    Standard zero-temperature GPE real-space operator.

        utot = u |ψ|² + V_ext(t)

    Args:
        params:    dict containing at least ``"u"`` (interaction strength).
        potential: Potential object whose ``.evol(t)`` returns V_ext at time t.
    """

    def __init__(self, params: dict, potential):
        try:
            self.u = params["u"]
        except KeyError:
            raise KeyError("params must contain 'u' (interaction strength).")
        self.potential = potential

    def __call__(self, psi: torch.Tensor, t: float) -> torch.Tensor:
        return self.u * torch.abs(psi) ** 2 + self.potential.evol(t)


class GPE_3body_loss(Equation):
    """
    GPE with three-body loss.

        utot = u |ψ|² + V_ext(t) − i k3 |ψ|⁴

    The imaginary term drains population from high-density regions.

    Args:
        params:    dict containing ``"u"`` and ``"k3"`` (three-body loss rate).
        potential: Potential object.
    """

    def __init__(self, params: dict, potential):
        try:
            self.u  = params["u"]
            self.k3 = params["k3"]
        except KeyError as e:
            raise KeyError(f"params must contain {e}.")
        self.potential = potential

    def __call__(self, psi: torch.Tensor, t: float) -> torch.Tensor:
        density = torch.abs(psi) ** 2
        return self.u * density + self.potential.evol(t) - 1j * self.k3 * density ** 2


class CustomEquation(Equation):
    """
    Equation defined by an arbitrary callable.

    The callable must accept (psi, t) and return a tensor of the same shape.

    Example:
        def my_operator(psi, t):
            return u * torch.abs(psi)**2 + V(t)

        equation = CustomEquation(my_operator)
        utot = equation(psi, t)
    """

    def __init__(self, operator):
        if not callable(operator):
            raise TypeError("operator must be callable with signature (psi, t).")
        self._operator = operator

    def __call__(self, psi: torch.Tensor, t: float) -> torch.Tensor:
        return self._operator(psi, t)
