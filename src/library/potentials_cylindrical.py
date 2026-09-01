r"""
External potentials for the GPE in cylindrical coordinates :math:`(r, \varphi, z)`.

Drop-in cylindrical counterpart to :mod:`src.library.potentials`: the classes
here expose the same ``evol(t)`` / ``zero()`` interface, so the solver does not
care which coordinate system it is running in.

A potential is the product of a static spatial profile ``self.potential`` and a
scalar time-dependent amplitude ``self.form(t)``, with an optional complex
absorbing potential (CAP) added on top,

.. math::

    V_\mathrm{ext}(r, \varphi, z, t)
        = f(t)\, V(r, \varphi, z)
        - i\, \eta\, g(t)\, V_\mathrm{cap}(r, z),

so a subclass normally only has to fill in ``self.potential`` and
``self.form``. Potentials whose *shape* changes with time
(:class:`CylindricalRotatingPot`) override :meth:`~CylindricalPotential.evol`
instead.

All quantities are dimensionless (:math:`\hbar = m = \omega_\mathrm{ho} = 1`).

Grid conventions
----------------

The same conventions as :class:`~src.library.gpe_cylindrical_library.GPECylindricalLibrary`:

:math:`r`
    Half-point grid, :math:`\mathrm{d}r/2 \ldots r_\mathrm{max} - \mathrm{d}r/2`,
    which keeps the coordinate singularity off the grid.
:math:`\varphi`
    :math:`0 \ldots 2\pi`, periodic — and therefore never absorbed.
:math:`z`
    :math:`z_\mathrm{min} \ldots z_\mathrm{max}`.

Expected ``kwargs`` keys
------------------------

``Grid_resolution``
    Tuple ``(n_r, n_phi, n_z)`` — number of grid points along each direction.
``r_max``
    Outer radial boundary.
``z_min``, ``z_max``
    Axial extent of the box.
``w``
    Trap frequencies, either ``[wr, wz]`` or ``[wr, wr, wz]``. The three-element
    form exists for config-file compatibility with the Cartesian grid; its
    middle element is ignored, so ``w[0]`` is always :math:`\omega_r` and
    ``w[-1]`` always :math:`\omega_z`.

Optional absorber keys
----------------------

Identical in meaning to the Cartesian ones:

``Absorber_enabled``
    ``bool`` — master switch; the CAP is skipped when false.
``Absorber_strength``
    ``float`` — the prefactor :math:`\eta` in :math:`-i\eta V_\mathrm{cap}`.
``Absorber_start_ratio``
    ``float`` in ``(0, 1)``, default ``0.8`` — fraction of the box at which the
    CAP starts to turn on.
``Absorber_power``
    ``float`` >= 1, default ``2`` — steepness of the CAP ramp.
``Absorber_tinit``, ``Absorber_tfinal``
    ``float`` — the CAP is ramped on linearly between these times;
    ``Absorber_tfinal = None`` switches it on instantly at ``Absorber_tinit``.
``SwitchOff_time``
    ``float`` — snapshot index after which the caller zeroes the potential.
"""

import torch
from src.library.gpe_cylindrical_library import GPECylindricalLibrary as cyl


###############################################################################
# Selector
###############################################################################

def select_potential_cylindrical(potential_type, app, **simulation_parameters):
    r"""
    Instantiate and return a :class:`CylindricalPotential` object by name.

    Mirrors :func:`~src.library.potentials.select_potential` for cylindrical
    grids. Available potential types (case-insensitive):

    ``"harmonic"``
        Axisymmetric harmonic trap
        :math:`\tfrac{1}{2}[(\omega_r r)^2 + (\omega_z z)^2]`,
        :class:`CylindricalHarmonicPot`.
    ``"constant"``
        Uniform offset across the grid, :class:`CylindricalConstPot`.
    ``"ramp"``
        Uniform profile with a linear time ramp, :class:`CylindricalRampPot`.
    ``"rampharmonic"``
        Harmonic trap with a linear time ramp,
        :class:`CylindricalRampHarmonicPot`.
    ``"rotating"``
        Rotating anisotropic harmonic trap for stirring,
        :class:`CylindricalRotatingPot`.
    ``"gaussianbeam"``
        Focused Gaussian laser beam along the z-axis,
        :class:`CylindricalGaussianBeamPot`.

    Args:
        potential_type (str): Identifier of the potential to build. Case and
            surrounding whitespace are ignored.
        app: Application object; ``app.device`` decides where the potential
            tensors live.
        **simulation_parameters: Grid, trap and absorber keys, as listed in the
            module docstring. Which of them are required depends on the
            potential type requested.

    Returns:
        CylindricalPotential: The instantiated potential object. Unlike the
        Cartesian selector, there is no ``"custom"`` escape hatch here.

    Raises:
        ValueError: If ``potential_type`` is not one of the available types.
        KeyError: If ``simulation_parameters`` is missing a key required by the
            selected potential.

    Example:
        .. code-block:: python

            potential = select_potential_cylindrical("harmonic", app, **params)
            uext = potential.evol(t)
    """
    potential_type = potential_type.strip().lower()
    available = [
        "harmonic",
        "constant",
        "ramp",
        "rampharmonic",
        "rotating",
        "gaussianbeam",
    ]
    if potential_type not in available:
        raise ValueError(
            f"Potential type '{potential_type}' not available. "
            f"Available: {available}"
        )

    mapping = {
        "harmonic":     CylindricalHarmonicPot,
        "constant":     CylindricalConstPot,
        "ramp":         CylindricalRampPot,
        "rampharmonic": CylindricalRampHarmonicPot,
        "rotating":     CylindricalRotatingPot,
        "gaussianbeam": CylindricalGaussianBeamPot,
    }
    return mapping[potential_type](app, **simulation_parameters)


###############################################################################
# Base class
###############################################################################

class CylindricalPotential:
    r"""
    Base class for external potentials in cylindrical coordinates.

    Provides :meth:`evol`, :meth:`zero` and an optional complex absorbing
    potential (CAP) applied at the outer radial boundary and at both axial
    ends. The :math:`\varphi` boundaries are periodic and never absorb.

    Subclasses must set ``self.potential`` (the static shape tensor) and may
    set ``self.form`` (a callable returning the time-dependent amplitude,
    default ``lambda t: 1.0``). A subclass whose shape changes with time
    overrides :meth:`evol` instead, and is then responsible for adding
    :meth:`_absorber_term` itself.

    Attributes:
        app: The application object supplied at construction.
        form (collections.abc.Callable): Time-dependent amplitude
            :math:`f(t)`, defaults to ``lambda t: 1.0``.
        potential (torch.Tensor): Static spatial profile, of shape
            ``(n_r, n_phi, n_z)``.
        switchOff_time (float): Snapshot index after which the caller zeroes
            the potential; ``None`` when the trap is never switched off.
        absorber_potential (torch.Tensor): Real CAP profile
            :math:`\eta V_\mathrm{cap}`, or ``None`` when the absorber is
            disabled.
        absorber_form (collections.abc.Callable): Ramp-on factor :math:`g(t)`
            of the absorber.
    """

    def __init__(self, app, **kwargs):
        r"""
        Store the application handle and configure the optional absorber.

        Args:
            app: Application object; ``app.device`` decides where the potential
                tensors live.
            **kwargs: Grid and absorber keys, as listed in the module
                docstring. Only the ``Absorber_*`` keys (and, when the absorber
                is enabled, the grid keys) are read here; the trap keys are
                consumed by the subclasses.
        """
        self.app = app
        self.form = lambda t: 1.0
        self.potential = None
        self.switchOff_time = kwargs.get("SwitchOff_time", None)
        self.absorber_potential = None
        self.absorber_form = lambda t: 1.0
        self._configure_absorber(**kwargs)

    # ------------------------------------------------------------------
    # Absorber
    # ------------------------------------------------------------------

    def _configure_absorber(self, **kwargs):
        r"""
        Build the complex absorbing potential (CAP) for cylindrical boundaries.

        The CAP damps the wavefunction near :math:`r = r_\mathrm{max}` and
        :math:`\lvert z \rvert \to z_\mathrm{max}`, so that outgoing matter
        waves are swallowed instead of being reflected back into the cloud by
        the grid edges. The radial absorber is one-sided (there is no boundary
        at :math:`r = 0`), the axial one is two-sided, and the periodic
        :math:`\varphi` direction is never absorbed:

        .. math::

            c_r(r) = \left[\mathrm{clamp}\!\left(
                \frac{r - r_\mathrm{start}}{r_\mathrm{max} - r_\mathrm{start}},
                0, 1\right)\right]^{p},

        .. math::

            c_z(z) = \left[\mathrm{clamp}\!\left(
                \frac{\lvert z \rvert - z_\mathrm{start}}
                     {z_\mathrm{extent} - z_\mathrm{start}},
                0, 1\right)\right]^{p},

        .. math::

            V_\mathrm{cap} = \eta \max(c_r, c_z),
            \qquad
            V_\mathrm{abs}(t) = -i\, g(t)\, V_\mathrm{cap}.

        The term is purely imaginary, so it decays the density instead of
        shifting its phase.

        Does nothing (and leaves ``absorber_potential`` at ``None``) when the
        absorber is disabled or its strength is not positive.

        Args:
            **kwargs: Grid keys (``Grid_resolution``, ``r_max``, ``z_min``,
                ``z_max``) and the ``Absorber_*`` keys documented in the module
                docstring.
        """
        absorber_enabled = bool(kwargs.get("Absorber_enabled", False))
        absorber_strength = float(kwargs.get("Absorber_strength", 0.0))
        if (not absorber_enabled) or absorber_strength <= 0.0:
            return

        n_r, n_phi, n_z = kwargs["Grid_resolution"]
        r_max = float(kwargs["r_max"])
        z_min = float(kwargs["z_min"])
        z_max = float(kwargs["z_max"])

        start_ratio = float(kwargs.get("Absorber_start_ratio", 0.8))
        start_ratio = max(0.0, min(start_ratio, 0.999999))
        power = float(kwargs.get("Absorber_power", 2.0))
        power = max(power, 1.0)

        _, _, _, _, _, _, _, _, (gr, gphi, gz) = cyl.init_grid(
            r_max, z_min, z_max, n_r, n_phi, n_z, self.app.device
        )
        eps = torch.finfo(torch.float64).eps

        # Radial absorber (one-sided: only at r → r_max)
        start_r = start_ratio * r_max
        cap_r = torch.clamp(
            (gr - start_r) / max(r_max - start_r, eps), min=0.0, max=1.0
        ) ** power

        # Axial absorber (two-sided: |z| → z_extent)
        z_extent = max(abs(z_min), abs(z_max))
        start_z = start_ratio * z_extent
        cap_z = torch.clamp(
            (torch.abs(gz) - start_z) / max(z_extent - start_z, eps), min=0.0, max=1.0
        ) ** power

        profile = torch.maximum(cap_r, cap_z)
        self.absorber_potential = absorber_strength * profile.to(
            dtype=torch.double, device=self.app.device
        )

        tinit = float(kwargs.get("Absorber_tinit", 0.0))
        tfinal = kwargs.get("Absorber_tfinal", None)
        if tfinal is None:
            self.absorber_form = lambda t: 1.0 if t >= tinit else 0.0
        else:
            tfinal = float(tfinal)
            if tfinal <= tinit:
                self.absorber_form = lambda t: 1.0 if t >= tinit else 0.0
            else:
                self.absorber_form = (
                    lambda t: min(max((t - tinit) / (tfinal - tinit), 0.0), 1.0)
                )

    def _absorber_term(self, t):
        r"""
        Return the absorbing term :math:`-i\,g(t)\,V_\mathrm{cap}` at time t.

        Args:
            t (float): Current simulation time, in dimensionless units.

        Returns:
            torch.Tensor: The imaginary absorbing potential, or the scalar
            ``0.0`` when no absorber is configured, so that it can be added to
            any potential unconditionally.
        """
        if self.absorber_potential is None:
            return 0.0
        return -1j * self.absorber_form(t) * self.absorber_potential

    # ------------------------------------------------------------------
    # Interface (same as Potential)
    # ------------------------------------------------------------------

    def evol(self, t: float) -> torch.Tensor:
        r"""
        Return the external potential at a specific time.

        Args:
            t (float): Current simulation time, in dimensionless units
                (:math:`\omega_\mathrm{ho}^{-1}`).

        Returns:
            torch.Tensor: The potential :math:`f(t)V` plus the absorbing term,
            of shape ``(n_r, n_phi, n_z)``. It is complex whenever the absorber
            is active.
        """
        return self.form(t) * self.potential + self._absorber_term(t)

    def zero(self) -> torch.Tensor:
        r"""
        Set the static profile to zero, releasing the trap.

        The time dependence ``self.form`` and the absorber are left untouched,
        so an active CAP keeps absorbing after the trap has been switched off.

        Returns:
            torch.Tensor: The zeroed potential, which is also stored in
            ``self.potential``.
        """
        self.potential = torch.zeros_like(self.potential)
        return self.potential


###############################################################################
# Concrete potential classes
###############################################################################

class CylindricalConstPot(CylindricalPotential):
    r"""
    Uniform potential of constant amplitude across the whole grid.

    .. math::

        V(r, \varphi, z) = A

    A constant adds a global phase to the wavefunction and has no effect on the
    density; it is mostly useful as a flat background, or as a chemical
    potential offset.

    Args:
        app: Application object.
        amplitude (float, optional): The constant value :math:`A` (default
            ``1.0``).
        **kwargs: Must contain ``Grid_resolution``; absorber keys are optional.
    """

    def __init__(self, app, amplitude: float = 1.0, **kwargs):
        super().__init__(app, **kwargs)
        n_r, n_phi, n_z = kwargs["Grid_resolution"]
        self.potential = amplitude * torch.ones(
            n_r, n_phi, n_z, dtype=torch.double, device=self.app.device
        )
        self.form = lambda t: 1.0


class CylindricalRampPot(CylindricalPotential):
    r"""
    Uniform spatial profile whose amplitude is ramped linearly in time.

    .. math::

        V(r, \varphi, z, t) = V_i + (V_f - V_i)\, \frac{t}{t_f}

    The ramp is not clamped at :math:`t_f`: the amplitude keeps growing
    linearly if the simulation runs past the end of the ramp.

    Args:
        app: Application object.
        initial (float, optional): Initial amplitude :math:`V_i` (default
            ``1.0``).
        final (float, optional): Amplitude :math:`V_f` reached at ``tfinal``
            (default ``2.0``).
        tfinal (float, optional): Time at which the ramp reaches ``final``, in
            dimensionless units (default ``1.0``).
        **kwargs: Must contain ``Grid_resolution``; absorber keys are optional.
    """

    def __init__(
        self, app, initial: float = 1.0, final: float = 2.0, tfinal: float = 1.0, **kwargs
    ):
        super().__init__(app, **kwargs)
        n_r, n_phi, n_z = kwargs["Grid_resolution"]
        self.potential = torch.ones(
            n_r, n_phi, n_z, dtype=torch.double, device=self.app.device
        )
        self.form = lambda t: initial + (final - initial) * (t / tfinal)


class CylindricalHarmonicPot(CylindricalPotential):
    r"""
    Static axisymmetric harmonic trap.

    .. math::

        V(r, z) = \frac{A}{2}\left[(\omega_r r)^2 + (\omega_z z)^2\right]

    This is the workhorse trap in cylindrical geometry: it is what the ground
    state is normally computed in, and what the other potentials here ramp or
    stir.

    Args:
        app: Application object.
        amplitude (float, optional): Overall amplitude :math:`A` of the trap
            (default ``1.0``).
        **kwargs: Must contain ``Grid_resolution``, ``r_max``, ``z_min``,
            ``z_max`` and ``w``; absorber keys are optional. ``w`` may be
            ``[wr, wz]`` or ``[wr, wr, wz]`` — the middle element of the
            three-element form is ignored.
    """

    def __init__(self, app, amplitude: float = 1.0, **kwargs):
        super().__init__(app, **kwargs)
        n_r, n_phi, n_z = kwargs["Grid_resolution"]
        r_max = float(kwargs["r_max"])
        z_min = float(kwargs["z_min"])
        z_max = float(kwargs["z_max"])
        w = kwargs["w"]

        _, _, _, _, _, _, _, _, (gr, gphi, gz) = cyl.init_grid(
            r_max, z_min, z_max, n_r, n_phi, n_z, self.app.device
        )

        wr = float(w[0])
        wz = float(w[-1])   # works for both [wr, wz] and [wr, wr, wz]

        self._gr, self._gz = gr, gz
        self._wr, self._wz = wr, wz
        self._amplitude = amplitude

        self.potential = (
            0.5 * amplitude * ((wr * gr) ** 2 + (wz * gz) ** 2)
        ).to(dtype=torch.double, device=self.app.device)
        self.form = lambda t: 1.0

    def zero_radial(self, amplitude: float = 1.0) -> torch.Tensor:
        r"""
        Release the radial confinement, keeping only the axial trap.

        .. math::

            V(z) = \frac{A}{2} (\omega_z z)^2

        This is the standard release for a radial expansion, where the cloud is
        let go in :math:`r` while the axial confinement holds it in place.

        Args:
            amplitude (float, optional): Overall amplitude :math:`A` of the
                remaining axial confinement (default ``1.0``).

        Returns:
            torch.Tensor: The new potential, which is also stored in
            ``self.potential``.
        """
        self.potential = (
            0.5 * amplitude * (self._wz * self._gz) ** 2
        ).to(dtype=torch.double, device=self.app.device)
        return self.potential


class CylindricalRampHarmonicPot(CylindricalPotential):
    r"""
    Axisymmetric harmonic trap whose amplitude is ramped linearly in time.

    .. math::

        V(r, z, t) =
            \left[V_i + (A - V_i)\, \frac{t - t_i}{t_f - t_i}\right]
            \frac{1}{2}\left[(\omega_r r)^2 + (\omega_z z)^2\right]

    The initial amplitude :math:`V_i` is normally the one the ground state was
    computed with, so that the ramp starts from an equilibrium state. The ramp
    is not clamped at :math:`t_f`: the amplitude keeps changing linearly if the
    simulation runs past the end of the ramp.

    Args:
        app: Application object.
        initial (float, optional): Initial amplitude :math:`V_i` (default
            ``1.0``).
        amplitude (float, optional): Final amplitude :math:`A` reached at
            ``tfinal`` (default ``1.0``).
        tinit (float, optional): Time at which the ramp starts (default
            ``0.0``).
        tfinal (float, optional): Time at which the ramp stops (default
            ``1.0``).
        **kwargs: Must contain ``Grid_resolution``, ``r_max``, ``z_min``,
            ``z_max`` and ``w``; absorber keys are optional.
    """

    def __init__(
        self,
        app,
        initial: float = 1.0,
        amplitude: float = 1.0,
        tinit: float = 0.0,
        tfinal: float = 1.0,
        **kwargs,
    ):
        super().__init__(app, **kwargs)
        n_r, n_phi, n_z = kwargs["Grid_resolution"]
        r_max = float(kwargs["r_max"])
        z_min = float(kwargs["z_min"])
        z_max = float(kwargs["z_max"])
        w = kwargs["w"]

        _, _, _, _, _, _, _, _, (gr, gphi, gz) = cyl.init_grid(
            r_max, z_min, z_max, n_r, n_phi, n_z, self.app.device
        )

        wr = float(w[0])
        wz = float(w[-1])

        self.potential = (0.5 * ((wr * gr) ** 2 + (wz * gz) ** 2)).to(
            dtype=torch.double, device=self.app.device
        )
        self.form = (
            lambda t: initial + (amplitude - initial) * ((t - tinit) / (tfinal - tinit))
        )


class CylindricalRotatingPot(CylindricalPotential):
    r"""
    Rotating anisotropic harmonic trap — the standard stirring geometry.

    .. math::

        V(r, \varphi, z, t) =
            \frac{A}{2}\,\omega_r^2 r^2
                \left[1 + \varepsilon \cos\bigl(2(\varphi - \Omega t)\bigr)\right]
            + \frac{A}{2}\,\omega_z^2 z^2

    The angular anisotropy :math:`\varepsilon` breaks the cylindrical symmetry
    and, when the trap rotates at :math:`\Omega`, transfers angular momentum to
    the condensate — the standard experimental mechanism for creating
    quantised vortices.

    Because the profile itself changes with time, this class overrides
    :meth:`evol` and rebuilds the trap at every call; ``self.potential`` holds
    the :math:`t = 0` snapshot and ``self.form`` stays at ``1``.

    Note:
        With :math:`\varepsilon = 0` the trap is axisymmetric and the rotation
        has no effect at all. Vortex nucleation also requires
        :math:`\Omega` above a critical value set by the trap geometry.

    Args:
        app: Application object.
        amplitude (float, optional): Overall amplitude :math:`A` (default
            ``1.0``).
        angular_frequency (float, optional): Stirring rate :math:`\Omega`, in
            units of :math:`\omega_\mathrm{ho}` (default ``1.0``).
        anisotropy (float, optional): Ellipticity :math:`\varepsilon \in [0,
            1)` of the trap (default ``0.05``).
        **kwargs: Must contain ``Grid_resolution``, ``r_max``, ``z_min``,
            ``z_max`` and ``w``; absorber keys are optional.
    """

    def __init__(
        self,
        app,
        amplitude: float = 1.0,
        angular_frequency: float = 1.0,
        anisotropy: float = 0.05,
        **kwargs,
    ):
        super().__init__(app, **kwargs)
        n_r, n_phi, n_z = kwargs["Grid_resolution"]
        r_max = float(kwargs["r_max"])
        z_min = float(kwargs["z_min"])
        z_max = float(kwargs["z_max"])
        w = kwargs["w"]

        _, _, _, _, _, _, _, _, (gr, gphi, gz) = cyl.init_grid(
            r_max, z_min, z_max, n_r, n_phi, n_z, self.app.device
        )

        self._gr, self._gphi, self._gz = gr, gphi, gz
        self._wr = float(w[0])
        self._wz = float(w[-1])
        self._amplitude = amplitude
        self._Omega = angular_frequency
        self._eps = anisotropy

        # Static potential at t = 0
        self.potential = self._compute(0.0)
        self.form = lambda t: 1.0

    def _compute(self, t: float) -> torch.Tensor:
        r"""
        Evaluate the stirred trap at time t, without the absorber.

        Args:
            t (float): Current simulation time, in dimensionless units.

        Returns:
            torch.Tensor: The trap profile at angle
            :math:`\theta = \Omega t`, of shape ``(n_r, n_phi, n_z)``.
        """
        theta = self._Omega * t
        radial = (
            0.5
            * self._amplitude
            * self._wr ** 2
            * self._gr ** 2
            * (1.0 + self._eps * torch.cos(2.0 * (self._gphi - theta)))
        )
        axial = 0.5 * self._amplitude * (self._wz * self._gz) ** 2
        return (radial + axial).to(dtype=torch.double, device=self.app.device)

    def evol(self, t: float) -> torch.Tensor:
        r"""
        Return the stirred trap at time t, recomputing the angular phase.

        Args:
            t (float): Current simulation time, in dimensionless units.

        Returns:
            torch.Tensor: The rotated trap plus the absorbing term, of shape
            ``(n_r, n_phi, n_z)``.
        """
        return self._compute(t) + self._absorber_term(t)


class CylindricalGaussianBeamPot(CylindricalPotential):
    r"""
    Focused Gaussian laser beam propagating along the z-axis.

    .. math::

        V(r, z) = A\,
            \exp\!\left(-\frac{2 r^2}{w_0^2}\right)
            \exp\!\left(-\frac{2 (z - z_0)^2}{z_R^2}\right)

    with :math:`w_0` the beam waist and :math:`z_R` the Rayleigh range.
    Setting ``rayleigh_range=None`` (the default) drops the second factor and
    gives a collimated beam with no axial profile.

    Common uses: a repulsive obstacle (blue-detuned laser), a stirring beam, or
    a localised barrier for splitting the cloud.

    Args:
        app: Application object.
        amplitude (float, optional): Peak amplitude :math:`A`; positive is
            repulsive, negative attractive (default ``1.0``).
        beam_waist (float, optional): Waist :math:`w_0`, the
            :math:`1/e^2` intensity radius at the focus (default ``1.0``).
        rayleigh_range (float, optional): Rayleigh range :math:`z_R`, the
            axial :math:`1/e^2` intensity half-length. ``None`` (default)
            means no decay along z.
        z_center (float, optional): Axial position :math:`z_0` of the beam
            focus (default ``0.0``).
        **kwargs: Must contain ``Grid_resolution``, ``r_max``, ``z_min`` and
            ``z_max``; absorber keys are optional.
    """

    def __init__(
        self,
        app,
        amplitude: float = 1.0,
        beam_waist: float = 1.0,
        rayleigh_range=None,
        z_center: float = 0.0,
        **kwargs,
    ):
        super().__init__(app, **kwargs)
        n_r, n_phi, n_z = kwargs["Grid_resolution"]
        r_max = float(kwargs["r_max"])
        z_min = float(kwargs["z_min"])
        z_max = float(kwargs["z_max"])

        _, _, _, _, _, _, _, _, (gr, gphi, gz) = cyl.init_grid(
            r_max, z_min, z_max, n_r, n_phi, n_z, self.app.device
        )

        radial_env = torch.exp(-2.0 * gr ** 2 / beam_waist ** 2)

        if rayleigh_range is None:
            axial_env = torch.ones_like(gr)
        else:
            axial_env = torch.exp(-2.0 * (gz - z_center) ** 2 / float(rayleigh_range) ** 2)

        self.potential = (amplitude * radial_env * axial_env).to(
            dtype=torch.double, device=self.app.device
        )
        self.form = lambda t: 1.0
