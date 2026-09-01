r"""
Equation objects for the GPE and its extensions.

Each class in this module encapsulates the *real-space* part of the split-step
operator, i.e. the term :math:`U_\mathrm{tot}` that is sandwiched between the
two half-kinetic steps of one split-step Fourier iteration,

.. math::

    \psi(\mathbf{r}, t + \Delta t) =
        e^{-i \hat{T} \Delta t / 2}\,
        e^{-i U_\mathrm{tot}[\psi](\mathbf{r}, t)\, \Delta t}\,
        e^{-i \hat{T} \Delta t / 2}\, \psi(\mathbf{r}, t),

with :math:`\hat{T} = -\nabla^2 / 2` the kinetic operator in dimensionless
units (:math:`\hbar = m = \omega_\mathrm{ho} = 1`).

Equation objects are created once, before the simulation loop, and are called
once per iteration:

.. code-block:: python

    equation = GPE_base(params, system.uext)

    for iteration in range(kmax):
        t = dt * iteration * omega_ho

        utot = equation(psi, t)

        psi = split_step(psi, utot, dtau, ...)

All time dependence is handled by the Potential object that is passed in,
through its ``.evol(t)`` method, so an equation object is stateless between
calls and can be reused across runs that share the same parameters.
"""
import torch
from abc import ABC, abstractmethod


def select_equation(equation_type: str, params: dict, potential) -> "Equation":
    r"""
    Instantiate and return an :class:`Equation` object by name.

    Available equation types (case-insensitive):

    ``"gpe"``
        Standard zero-temperature GPE, :class:`GPE_base`. Requires ``"u"`` in
        ``params``.
    ``"gpe_3body"``
        GPE with three-body loss, :class:`GPE_3body_loss`. Requires ``"u"`` and
        ``"k3"`` in ``params``.
    ``"custom"``
        Returns ``None``; the caller is expected to build a
        :class:`CustomEquation` manually.

    Args:
        equation_type (str): Identifier of the equation to build, one of
            ``"gpe"``, ``"gpe_3body"`` or ``"custom"``. Case and surrounding
            whitespace are ignored.
        params (dict): Simulation parameters. Which keys are required depends
            on the equation type requested.
        potential: Potential object whose ``.evol(t)`` returns the external
            potential :math:`V_\mathrm{ext}(\mathbf{r}, t)`.

    Returns:
        Equation: The instantiated equation object, or ``None`` when
        ``equation_type`` is ``"custom"``.

    Raises:
        ValueError: If ``equation_type`` is not one of the available types.
        KeyError: If ``params`` is missing a key required by the selected
            equation.

    Example:
        .. code-block:: python

            equation = select_equation("gpe_3body", params, system.uext)
            utot = equation(psi, t)
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
    r"""
    Abstract base class for all GPE equation objects.

    A concrete subclass only has to implement :meth:`__call__`, which evaluates
    the real-space operator :math:`U_\mathrm{tot}[\psi](\mathbf{r}, t)` for the
    current wavefunction and time. Everything else about the split-step
    iteration is handled by the solver.
    """

    @abstractmethod
    def __call__(self, psi: torch.Tensor, t: float) -> torch.Tensor:
        r"""
        Evaluate the real-space operator for a wavefunction at a given time.

        Args:
            psi (torch.Tensor): Complex wavefunction of the condensate.
            t (float): Current simulation time, in dimensionless units
                (:math:`\omega_\mathrm{ho}^{-1}`).

        Returns:
            torch.Tensor: The operator :math:`U_\mathrm{tot}`, of the same
            shape as ``psi``. It is complex whenever the equation contains a
            dissipative term.
        """


class GPE_base(Equation):
    r"""
    Real-space operator of the standard zero-temperature GPE.

    .. math::

        U_\mathrm{tot}(\mathbf{r}, t)
            = u\, \lvert \psi(\mathbf{r}, t) \rvert^{2}
            + V_\mathrm{ext}(\mathbf{r}, t)

    The operator is purely real, so the evolution it generates is unitary and
    the norm of the wavefunction is conserved.

    Args:
        params (dict): Simulation parameters; must contain ``"u"``, the
            dimensionless interaction strength :math:`u`.
        potential: Potential object whose ``.evol(t)`` returns
            :math:`V_\mathrm{ext}(\mathbf{r}, t)`.

    Raises:
        KeyError: If ``params`` does not contain ``"u"``.

    Attributes:
        u (float): Interaction strength :math:`u`.
        potential: The potential object supplied at construction.
    """

    def __init__(self, params: dict, potential):
        try:
            self.u = params["u"]
        except KeyError:
            raise KeyError("params must contain 'u' (interaction strength).")
        self.potential = potential

    def __call__(self, psi: torch.Tensor, t: float) -> torch.Tensor:
        r"""
        Evaluate :math:`u \lvert \psi \rvert^{2} + V_\mathrm{ext}(t)`.

        Args:
            psi (torch.Tensor): Complex wavefunction of the condensate.
            t (float): Current simulation time, in dimensionless units.

        Returns:
            torch.Tensor: Real-valued operator of the same shape as ``psi``.
        """
        return self.u * torch.abs(psi) ** 2 + self.potential.evol(t)


class GPE_3body_loss(Equation):
    r"""
    Real-space operator of the GPE with three-body recombination loss.

    .. math::

        U_\mathrm{tot}(\mathbf{r}, t)
            = u\, \lvert \psi \rvert^{2}
            + V_\mathrm{ext}(\mathbf{r}, t)
            - i\, K_3\, \lvert \psi \rvert^{4}

    The imaginary term is non-Hermitian: it drains population from the
    high-density regions of the cloud, where three atoms are most likely to
    collide and be lost from the trap. The norm of the wavefunction therefore
    decays over time and must not be renormalised during the evolution.

    Args:
        params (dict): Simulation parameters; must contain ``"u"``, the
            dimensionless interaction strength, and ``"k3"``, the three-body
            loss rate :math:`K_3`.
        potential: Potential object whose ``.evol(t)`` returns
            :math:`V_\mathrm{ext}(\mathbf{r}, t)`.

    Raises:
        KeyError: If ``params`` is missing ``"u"`` or ``"k3"``.

    Attributes:
        u (float): Interaction strength :math:`u`.
        k3 (float): Three-body loss rate :math:`K_3`.
        potential: The potential object supplied at construction.
    """

    def __init__(self, params: dict, potential):
        try:
            self.u  = params["u"]
            self.k3 = params["k3"]
        except KeyError as e:
            raise KeyError(f"params must contain {e}.")
        self.potential = potential

    def __call__(self, psi: torch.Tensor, t: float) -> torch.Tensor:
        r"""
        Evaluate the interaction, trapping and three-body loss terms.

        Args:
            psi (torch.Tensor): Complex wavefunction of the condensate.
            t (float): Current simulation time, in dimensionless units.

        Returns:
            torch.Tensor: Complex operator of the same shape as ``psi``.
        """
        density = torch.abs(psi) ** 2
        return self.u * density + self.potential.evol(t) - 1j * self.k3 * density ** 2


class CustomEquation(Equation):
    r"""
    Equation defined by an arbitrary user-supplied callable.

    This is the escape hatch for terms that none of the built-in equations
    cover. The callable is used as-is on every iteration, so it should be
    written with PyTorch operations acting on the whole tensor rather than with
    element-wise Python loops.

    Args:
        operator (collections.abc.Callable): Callable with signature
            ``operator(psi, t)`` returning a tensor of the same shape as
            ``psi``.

    Raises:
        TypeError: If ``operator`` is not callable.

    Example:
        .. code-block:: python

            def my_operator(psi, t):
                return u * torch.abs(psi) ** 2 + V(t)

            equation = CustomEquation(my_operator)
            utot = equation(psi, t)
    """

    def __init__(self, operator):
        if not callable(operator):
            raise TypeError("operator must be callable with signature (psi, t).")
        self._operator = operator

    def __call__(self, psi: torch.Tensor, t: float) -> torch.Tensor:
        r"""
        Delegate to the wrapped callable.

        Args:
            psi (torch.Tensor): Complex wavefunction of the condensate.
            t (float): Current simulation time, in dimensionless units.

        Returns:
            torch.Tensor: Whatever the wrapped callable returns, expected to be
            of the same shape as ``psi``.
        """
        return self._operator(psi, t)
