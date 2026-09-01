r"""
External potentials for the GPE on a Cartesian grid.

This module provides the most common experimentally used traps and beams, plus
an optional complex absorbing potential (CAP) at the box boundaries. Every
potential is a small object built once, before the simulation loop, and queried
once per iteration through :meth:`Potential.evol`:

.. code-block:: python

    potential = select_potential("harmonic", app, **simulation_parameters)

    for iteration in range(kmax):
        t = dt * iteration * omega_ho

        uext = potential.evol(t)

A potential is the product of a static spatial profile ``self.potential`` and a
scalar time-dependent amplitude ``self.form(t)``, with the absorber added on
top,

.. math::

    V_\mathrm{ext}(\mathbf{r}, t)
        = f(t)\, V(\mathbf{r})
        - i\, \eta\, g(t)\, V_\mathrm{cap}(\mathbf{r}),

so a subclass normally only has to fill in ``self.potential`` and ``self.form``.
Potentials whose *shape* changes with time (:class:`RotatingPot`) override
:meth:`~Potential.evol` instead.

All quantities are dimensionless (:math:`\hbar = m = \omega_\mathrm{ho} = 1`);
lengths are in units of the harmonic oscillator length and times in units of
:math:`\omega_\mathrm{ho}^{-1}`.

Expected ``kwargs`` keys
------------------------

``Grid_resolution``
    Tuple ``(n1, n2, n3)`` — number of grid points along each axis.
``x_min``
    Sequence ``[x_min, y_min, z_min]`` — lower corner of the box.
``dx``
    Sequence ``[dx, dy, dz]`` — grid spacing along each axis.
``w``
    Sequence ``[wx, wy, wz]`` — trap frequencies, in units of
    :math:`\omega_\mathrm{ho}`.

Optional absorber keys
----------------------

``Absorber_enabled``
    ``bool`` — master switch; the CAP is skipped when false.
``Absorber_strength``
    ``float`` — the prefactor :math:`\eta` in :math:`-i\eta V_\mathrm{cap}`.
``Absorber_start_ratio``
    ``float`` in ``(0, 1)``, default ``0.8`` — fraction of the half-box at
    which the CAP starts to turn on.
``Absorber_power``
    ``float`` >= 1, default ``2`` — steepness of the CAP ramp.
``Absorber_tinit``, ``Absorber_tfinal``
    ``float`` — the CAP is ramped on linearly between these times;
    ``Absorber_tfinal = None`` switches it on instantly at ``Absorber_tinit``.
``SwitchOff_time``
    ``float`` — snapshot index after which the caller zeroes the potential.
"""
import torch

###############################################################################
##########################   EXTERNAL POTENTIALS ##############################
###############################################################################
def select_potential(potentialType, app, **simulation_parameters):
   r"""
   Instantiate and return a :class:`Potential` object by name.

   Available potential types (case-insensitive):

   ``"harmonic"``
       Static harmonic trap, :class:`HarmonicPot`.
   ``"constant"``
       Uniform offset across the grid, :class:`ConstPot`.
   ``"ramp"``
       Uniform profile with a linear time ramp, :class:`RampPot`.
   ``"rampharmonic"``
       Harmonic trap with a linear time ramp, :class:`RampHarmonicPot`.
   ``"rotating"``
       Harmonic trap rotating about an axis, :class:`RotatingPot`.
   ``"custom"``
       Returns ``None``; the caller is expected to build a :class:`CustomPot`
       manually.

   Args:
       potentialType (str): Identifier of the potential to build. Case and
           surrounding whitespace are ignored.
       app: Application object; only ``app.device`` is used here, to place the
           potential tensor on the right device.
       **simulation_parameters: Grid, trap and absorber keys, as listed in the
           module docstring. Which of them are required depends on the
           potential type requested.

   Returns:
       Potential: The instantiated potential object, or ``None`` when
       ``potentialType`` is ``"custom"``.

   Raises:
       ValueError: If ``potentialType`` is not one of the available types.
       KeyError: If ``simulation_parameters`` is missing a key required by the
           selected potential.

   Example:
       .. code-block:: python

           potential = select_potential("harmonic", app, **simulation_parameters)
           uext = potential.evol(t)
   """
   potentialType = potentialType.strip().lower()
   available_potentials = [
      "harmonic",
      "constant",
      "ramp" ,
      "rampharmonic",
      "rotating",
      "custom"
      ]
   if potentialType not in available_potentials:
      raise ValueError(f"Potential type {potentialType} is not available. Available potentials are {available_potentials}")

   if potentialType == "harmonic":
      return HarmonicPot(app, **simulation_parameters)
   elif potentialType == "constant":
      return ConstPot(app, **simulation_parameters)
   elif potentialType == "ramp":
      return RampPot(app, **simulation_parameters)
   elif potentialType == "rampharmonic":
      return RampHarmonicPot(app, **simulation_parameters)
   elif potentialType == "rotating":
      return RotatingPot(app, **simulation_parameters)
   else:
      return None

class Potential():
    r"""
    Base class for external potentials on a Cartesian grid.

    A potential is stored as a static spatial profile multiplied by a scalar
    function of time, with an optional complex absorbing potential added on
    top,

    .. math::

        V_\mathrm{ext}(\mathbf{r}, t)
            = f(t)\, V(\mathbf{r})
            - i\, \eta\, g(t)\, V_\mathrm{cap}(\mathbf{r}).

    Subclasses must set ``self.potential`` (the static profile
    :math:`V(\mathbf{r})`) and may set ``self.form`` (the amplitude
    :math:`f(t)`, which defaults to ``1``). A subclass whose *shape* changes
    with time overrides :meth:`evol` instead, and is then responsible for
    adding :meth:`_absorber_term` itself.

    Attributes:
        app: The application object supplied at construction.
        form (collections.abc.Callable): Time-dependent amplitude
            :math:`f(t)`, defaults to ``lambda t: 1.0``.
        potential (torch.Tensor): Static spatial profile
            :math:`V(\mathbf{r})`, of shape ``Grid_resolution``.
        switchOff_time (float): Snapshot index after which the caller zeroes
            the potential; ``None`` when the trap is never switched off.
        absorber_potential (torch.Tensor): Real CAP profile
            :math:`\eta V_\mathrm{cap}(\mathbf{r})`, or ``None`` when the
            absorber is disabled.
        absorber_form (collections.abc.Callable): Ramp-on factor :math:`g(t)`
            of the absorber.
    """
    def __init__(self, app, **kwargs):
      r"""
      Store the application handle and configure the optional absorber.

      Args:
          app: Application object; ``app.device`` decides where the potential
              tensors live.
          **kwargs: Grid and absorber keys, as listed in the module docstring.
              Only the ``Absorber_*`` keys are read here; the grid keys are
              consumed by the subclasses.
      """
      self.app = app
      self.form = lambda t: 1.0
      self.potential = None
      self.switchOff_time = kwargs.get("SwitchOff_time", None)
      self.absorber_potential = None
      self.absorber_form = lambda t: 1.0
      self._configure_absorber(**kwargs)

    def _configure_absorber(self, **kwargs):
      r"""
      Build the optional complex absorbing potential (CAP).

      The CAP damps the wavefunction amplitude near the grid boundaries, so
      that outgoing matter waves are swallowed instead of being reflected back
      into the cloud by the periodicity of the FFT. The profile turns on at a
      fixed fraction of the half-box and grows as a power law towards the edge,

      .. math::

          c_\alpha(\mathbf{r}) =
              \left[
                  \mathrm{clamp}\!\left(
                      \frac{\lvert \alpha \rvert - \alpha_\mathrm{start}}
                           {\alpha_\mathrm{max} - \alpha_\mathrm{start}},
                      0, 1\right)
              \right]^{p},
          \qquad \alpha \in \{x, y, z\},

      .. math::

          V_\mathrm{cap}(\mathbf{r}) =
              \eta \max\left(c_x, c_y, c_z\right),

      taking the maximum over the three axes so that the corners of the box
      absorb as strongly as the faces. The term enters the evolution as
      :math:`-i\,g(t)\,V_\mathrm{cap}`, which decays the density rather than
      shifting its phase.

      Does nothing (and leaves ``absorber_potential`` at ``None``) when the
      absorber is disabled or its strength is not positive.

      Args:
          **kwargs: Grid keys (``Grid_resolution``, ``x_min``, ``dx``) and the
              ``Absorber_*`` keys documented in the module docstring.
      """
      absorber_enabled = bool(kwargs.get("Absorber_enabled", False))
      absorber_strength = float(kwargs.get("Absorber_strength", 0.0))
      if (not absorber_enabled) or absorber_strength <= 0.0:
         return

      n1, n2, n3 = kwargs["Grid_resolution"]
      x_min = kwargs["x_min"]
      dx = kwargs["dx"]

      start_ratio = float(kwargs.get("Absorber_start_ratio", 0.8))
      start_ratio = max(0.0, min(start_ratio, 0.999999))
      power = float(kwargs.get("Absorber_power", 2.0))
      power = max(power, 1.0)

      x1 = x_min[0] + torch.arange(n1, dtype=torch.float64, device=self.app.device) * dx[0]
      x2 = x_min[1] + torch.arange(n2, dtype=torch.float64, device=self.app.device) * dx[1]
      x3 = x_min[2] + torch.arange(n3, dtype=torch.float64, device=self.app.device) * dx[2]
      gx, gy, gz = torch.meshgrid(x1, x2, x3, indexing='ij')

      # CAP starts near the box edge and smoothly ramps to full strength.
      max_x = max(torch.max(torch.abs(x1)).item(), torch.finfo(torch.float64).eps)
      max_y = max(torch.max(torch.abs(x2)).item(), torch.finfo(torch.float64).eps)
      max_z = max(torch.max(torch.abs(x3)).item(), torch.finfo(torch.float64).eps)

      start_x = start_ratio * max_x
      start_y = start_ratio * max_y
      start_z = start_ratio * max_z

      cap_x = torch.clamp((torch.abs(gx) - start_x) / max(max_x - start_x, torch.finfo(torch.float64).eps), min=0.0, max=1.0) ** power
      cap_y = torch.clamp((torch.abs(gy) - start_y) / max(max_y - start_y, torch.finfo(torch.float64).eps), min=0.0, max=1.0) ** power
      cap_z = torch.clamp((torch.abs(gz) - start_z) / max(max_z - start_z, torch.finfo(torch.float64).eps), min=0.0, max=1.0) ** power

      profile = torch.maximum(torch.maximum(cap_x, cap_y), cap_z)
      self.absorber_potential = absorber_strength * profile.to(dtype=torch.double, device=self.app.device)

      tinit = float(kwargs.get("Absorber_tinit", 0.0))
      tfinal = kwargs.get("Absorber_tfinal", None)
      if tfinal is None:
         self.absorber_form = lambda t: 1.0 if t >= tinit else 0.0
      else:
         tfinal = float(tfinal)
         if tfinal <= tinit:
            self.absorber_form = lambda t: 1.0 if t >= tinit else 0.0
         else:
            self.absorber_form = lambda t: min(max((t - tinit) / (tfinal - tinit), 0.0), 1.0)

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

    def evol(self, t):
      r"""
      Return the external potential at a specific time.

      Args:
          t (float): Current simulation time, in dimensionless units
              (:math:`\omega_\mathrm{ho}^{-1}`).

      Returns:
          torch.Tensor: The potential :math:`f(t)V(\mathbf{r})` plus the
          absorbing term, of shape ``Grid_resolution``. It is complex whenever
          the absorber is active.
      """
      return self.form(t) * self.potential + self._absorber_term(t)

    def zero(self):
      r"""
      Set the static profile to zero, releasing the trap.

      The time dependence ``self.form`` and the absorber are left untouched, so
      an active CAP keeps absorbing after the trap has been switched off.

      Returns:
          torch.Tensor: The zeroed potential, which is also stored in
          ``self.potential``.
      """
      self.potential = torch.zeros_like(self.potential)
      return self.potential

class ConstPot(Potential):
   r"""
   Uniform potential of constant amplitude across the whole grid.

   .. math::

       V(\mathbf{r}) = A

   A constant adds a global phase to the wavefunction and has no effect on the
   density; it is mostly useful as a flat background, or as a chemical
   potential offset.

   Args:
       app: Application object.
       amplitude (float, optional): The constant value :math:`A` (default
           ``1.0``).
       **kwargs: Must contain ``Grid_resolution``; absorber keys are optional.
   """
   def __init__(self, app, amplitude=1.0, **kwargs):
      super().__init__(app, **kwargs)
      n1, n2, n3 = kwargs["Grid_resolution"]
      self.potential = amplitude * torch.ones(n1,n2,n3, dtype=torch.double, device=self.app.device)
      self.form = lambda t: 1

class RampPot(Potential):
   r"""
   Uniform spatial profile whose amplitude is ramped linearly in time.

   .. math::

       V(\mathbf{r}, t) = V_i + (V_f - V_i)\, \frac{t}{t_f}

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
   def __init__(self, app, initial=1.0, final=2.0, tfinal=1.0, **kwargs):
      super().__init__(app, **kwargs)
      n1, n2, n3 = kwargs["Grid_resolution"]
      self.potential = torch.ones(n1,n2,n3, dtype=torch.double, device=self.app.device)
      self.form = lambda t: (initial + (final - initial) * (t / tfinal))

class HarmonicPot(Potential):
    r"""
    Static anisotropic harmonic trap.

    .. math::

        V(x, y, z) = \frac{A}{2}
            \left[(\omega_x x)^2 + (\omega_y y)^2 + (\omega_z z)^2\right]

    This is the workhorse trap: it is what the ground state is normally
    computed in, and what the other Cartesian potentials ramp or rotate.

    Attributes:
        x1, x2, x3 (torch.Tensor): One-dimensional coordinate axes, kept so
            that :meth:`zero_2D` can rebuild the profile.
        w (collections.abc.Sequence): Trap frequencies ``[wx, wy, wz]``.
        pot (torch.Tensor): The profile before it is moved to the device.
    """

    def __init__(self, app, amplitude=1, **kwargs):
      r"""
      Build the static harmonic profile and set a constant time dependence.

      The shape of the trap is stored in ``self.potential`` and its time
      dependence in ``self.form``.

      Args:
          app: Application object.
          amplitude (float, optional): Overall amplitude :math:`A` of the trap
              (default ``1``).
          **kwargs: Must contain ``Grid_resolution``, ``x_min``, ``dx`` and
              ``w``; absorber keys are optional.
      """
      super().__init__(app, **kwargs)
      self.x_min = kwargs["x_min"]
      self.n1, self.n2, self.n3 = kwargs["Grid_resolution"]
      self.dx = kwargs["dx"]
      self.w = kwargs["w"]

      # Build space and momentum grids
      self.x1 = self.x_min[0] + torch.arange(self.n1, dtype=torch.float64)*self.dx[0] # size n1
      self.x2 = self.x_min[1] + torch.arange(self.n2, dtype=torch.float64)*self.dx[1]
      self.x3 = self.x_min[2] + torch.arange(self.n3, dtype=torch.float64)*self.dx[2]

      gx, gy, gz = torch.meshgrid(self.x1, self.x2, self.x3, indexing='ij')
      self.pot = 0.5 * amplitude * ((self.w[0]*gx)**2 + (self.w[1]*gy)**2 + (self.w[2]*gz)**2)
      self.potential = self.pot.to(device=self.app.device, dtype=torch.double)
      self.form = lambda t: 1

    def zero_2D(self, amplitude=1):
      r"""
      Switch the trap off along two axes, keeping the confinement along the
      second one.

      The first and third coordinates are replaced by zeros, so only the
      :math:`y` term survives,

      .. math::

          V(x, y, z) = \frac{A}{2} (\omega_y y)^2 .

      This is the release used for an effectively two-dimensional expansion,
      where the flat directions are opened up and the remaining tight axis
      keeps the cloud frozen in its transverse ground state.

      Args:
          amplitude (float, optional): Overall amplitude :math:`A` of the
              remaining confinement (default ``1``).

      Returns:
          torch.Tensor: The new potential, which is also stored in
          ``self.potential``.
      """
      gx, gy, gz = torch.meshgrid(torch.zeros_like(self.x1), self.x2, torch.zeros_like(self.x3), indexing='ij')
      self.pot = 0.5 * amplitude * ((self.w[0]*gx)**2 + (self.w[1]*gy)**2 + (self.w[2]*gz)**2)
      self.potential = self.pot.to(device=self.app.device, dtype=torch.double)
      self.form = lambda t: 1
      return self.potential


class RampHarmonicPot(Potential):
   r"""
   Harmonic trap whose amplitude is ramped linearly in time.

   .. math::

       V(x, y, z, t) =
           \left[V_i + (A - V_i)\, \frac{t - t_i}{t_f - t_i}\right]
           \frac{1}{2}
           \left[(\omega_x x)^2 + (\omega_y y)^2 + (\omega_z z)^2\right]

   The initial amplitude :math:`V_i` is always the one the ground state was
   computed with, so that the ramp starts from an equilibrium state. The ramp
   is not clamped at :math:`t_f`: the amplitude keeps changing linearly if the
   simulation runs past the end of the ramp.

   Attributes:
       pot (torch.Tensor): The unit-amplitude harmonic profile before it is
           moved to the device.
   """

   def __init__(self, app, initial=1.0, amplitude=1.0, tinit=0.0, tfinal=1.0,  **kwargs):
      r"""
      Build the harmonic profile and set its linear time dependence.

      The shape of the trap is stored in ``self.potential`` and its time
      dependence in ``self.form``.

      Args:
          app: Application object.
          initial (float, optional): Initial amplitude :math:`V_i`, i.e. the
              one used to compute the ground state (default ``1.0``).
          amplitude (float, optional): Final amplitude :math:`A` reached at
              ``tfinal`` (default ``1.0``).
          tinit (float, optional): Time at which the ramp starts (default
              ``0.0``).
          tfinal (float, optional): Time at which the ramp stops (default
              ``1.0``).
          **kwargs: Must contain ``Grid_resolution``, ``x_min``, ``dx`` and
              ``w``; absorber keys are optional.
      """
      super().__init__(app, **kwargs)
      n1, n2, n3 = kwargs["Grid_resolution"]
      x_min = kwargs["x_min"]
      dx = kwargs["dx"]
      w = kwargs["w"]
      # Build space and momentum grids
      x1 = x_min[0] + torch.arange(n1, dtype=torch.float64)*dx[0] # size n1
      x2 = x_min[1] + torch.arange(n2, dtype=torch.float64)*dx[1]
      x3 = x_min[2] + torch.arange(n3, dtype=torch.float64)*dx[2]
      gx, gy, gz = torch.meshgrid(x1, x2, x3, indexing='ij')
      self.pot = 0.5 * ((w[0]*gx)**2 + (w[1]*gy)**2 + (w[2]*gz)**2)
      self.potential = self.pot.to(device=app.device, dtype=torch.double)
      self.form = lambda t: initial + (amplitude - initial) * ((t-tinit) / (tfinal-tinit))


class RotatingPot(Potential):
   r"""
   Harmonic trap that rotates in time about a configurable axis.

   The static harmonic profile is evaluated in a co-rotating frame: the
   coordinates are rotated by an angle :math:`\theta(t) = \Omega t` before the
   trap is evaluated, so that an anisotropic trap stirs the condensate and
   transfers angular momentum to it — the standard way of nucleating
   quantised vortices. For a rotation about :math:`z`, for instance,

   .. math::

       \begin{pmatrix} x' \\ y' \\ z' \end{pmatrix} =
       \begin{pmatrix}
           \cos\theta & -\sin\theta & 0 \\
           \sin\theta & \cos\theta  & 0 \\
           0          & 0           & 1
       \end{pmatrix}
       \begin{pmatrix} x \\ y \\ z \end{pmatrix},
       \qquad
       V(\mathbf{r}, t) = \frac{A}{2}
           \left[(\omega_x x')^2 + (\omega_y y')^2 + (\omega_z z')^2\right].

   Because the profile itself changes with time, this class overrides
   :meth:`evol` and rebuilds the trap at every call; ``self.potential`` is
   therefore never used and ``self.form`` stays at ``1``.

   Note:
       A trap that is isotropic in the plane of rotation is unchanged by the
       rotation. Set different frequencies in the two transverse directions
       for the stirring to have any effect.

   Attributes:
       gx, gy, gz (torch.Tensor): The static coordinate meshes.
       amplitude (float): Overall amplitude :math:`A`.
       angular_frequency (float): Rotation rate :math:`\Omega`.
       axis (int): Rotation axis, normalised to ``1``, ``2`` or ``3``.
       w (collections.abc.Sequence): Trap frequencies ``[wx, wy, wz]``.
   """

   def __init__(self, app, amplitude=1.0, angular_frequency=1.0, axis=3, **kwargs):
      r"""
      Build the coordinate meshes and store the rotation parameters.

      Args:
          app: Application object.
          amplitude (float, optional): Overall potential amplitude :math:`A`
              (default ``1.0``).
          angular_frequency (float, optional): Rotation rate :math:`\Omega`, in
              radians per unit of dimensionless time (default ``1.0``).
          axis (int or str, optional): Rotation axis, given either as ``1``,
              ``2``, ``3`` or as ``"x"``, ``"y"``, ``"z"`` (default ``3``,
              i.e. the z-axis).
          **kwargs: Must contain ``Grid_resolution``, ``x_min``, ``dx`` and
              ``w``; absorber keys are optional.

      Raises:
          ValueError: If ``axis`` is not one of ``x``, ``y``, ``z``, ``1``,
              ``2`` or ``3``.
      """
      super().__init__(app, **kwargs)
      n1, n2, n3 = kwargs["Grid_resolution"]
      x_min = kwargs["x_min"]
      dx = kwargs["dx"]
      self.w = kwargs["w"]

      x1 = x_min[0] + torch.arange(n1, dtype=torch.float64, device=self.app.device) * dx[0]
      x2 = x_min[1] + torch.arange(n2, dtype=torch.float64, device=self.app.device) * dx[1]
      x3 = x_min[2] + torch.arange(n3, dtype=torch.float64, device=self.app.device) * dx[2]

      self.gx, self.gy, self.gz = torch.meshgrid(x1, x2, x3, indexing='ij')
      self.amplitude = amplitude
      self.angular_frequency = angular_frequency
      self.axis = self._parse_axis(axis)
      self.form = lambda t: 1.0

   @staticmethod
   def _parse_axis(axis):
      r"""
      Normalise the axis argument to an integer.

      Args:
          axis (int or str): Rotation axis, either ``1``, ``2``, ``3`` or
              ``"x"``, ``"y"``, ``"z"`` (case-insensitive, surrounding
              whitespace is ignored).

      Returns:
          int: The axis index, ``1``, ``2`` or ``3``.

      Raises:
          ValueError: If ``axis`` is none of the accepted values.
      """
      if isinstance(axis, str):
         axis_map = {"x": 1, "y": 2, "z": 3}
         normalized = axis.strip().lower()
         if normalized not in axis_map:
            raise ValueError("axis must be one of x, y, z, 1, 2, or 3")
         return axis_map[normalized]

      if axis not in (1, 2, 3):
         raise ValueError("axis must be one of x, y, z, 1, 2, or 3")
      return axis

   def _rotated_coordinates(self, theta):
      r"""
      Rotate the coordinate meshes about the selected axis.

      Args:
          theta (float): Rotation angle :math:`\theta`, in radians.

      Returns:
          tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The rotated
          coordinates :math:`(x', y', z')`, each of shape ``Grid_resolution``.
      """
      c = torch.cos(torch.tensor(theta, dtype=torch.float64, device=self.app.device))
      s = torch.sin(torch.tensor(theta, dtype=torch.float64, device=self.app.device))

      if self.axis == 1:
         x_rot = self.gx
         y_rot = c * self.gy - s * self.gz
         z_rot = s * self.gy + c * self.gz
      elif self.axis == 2:
         x_rot = c * self.gx + s * self.gz
         y_rot = self.gy
         z_rot = -s * self.gx + c * self.gz
      else:
         x_rot = c * self.gx - s * self.gy
         y_rot = s * self.gx + c * self.gy
         z_rot = self.gz

      return x_rot, y_rot, z_rot

   def evol(self, t):
      r"""
      Return the rotated harmonic trap at time t.

      Unlike the base-class implementation, the profile is rebuilt on every
      call, since the rotation angle :math:`\theta = \Omega t` changes the
      shape of the potential and not just its amplitude.

      Args:
          t (float): Current simulation time, in dimensionless units.

      Returns:
          torch.Tensor: The rotated trap plus the absorbing term, of shape
          ``Grid_resolution``.
      """
      theta = self.angular_frequency * t
      x_rot, y_rot, z_rot = self._rotated_coordinates(theta)

      potential = 0.5 * self.amplitude * (
         (self.w[0] * x_rot) ** 2
         + (self.w[1] * y_rot) ** 2
         + (self.w[2] * z_rot) ** 2
      )
      return potential.to(device=self.app.device, dtype=torch.double) + self._absorber_term(t)


class CustomPot(Potential):
   r"""
   Template for a user-defined potential.

   The coordinate meshes are built exactly as in :class:`HarmonicPot`, but the
   shape ``self.pot`` and the time dependence ``self.form`` are deliberately
   left as ``None``: fill both in before using the object, otherwise
   construction fails.

   Args:
       app: Application object.
       **kwargs: Must contain ``Grid_resolution``, ``x_min``, ``dx`` and
           ``w``; absorber keys are optional.

   Example:
       .. code-block:: python

           class MyPot(CustomPot):
               def __init__(self, app, **kwargs):
                   super().__init__(app, **kwargs)
                   self.pot = my_profile(...)
                   self.potential = self.pot.to(device=app.device, dtype=torch.double)
                   self.form = lambda t: 1.0
   """
   def __init__(self, app, **kwargs):
      super().__init__(app, **kwargs)
      n1, n2, n3 = kwargs["Grid_resolution"]
      x_min = kwargs["x_min"]
      dx = kwargs["dx"]
      w = kwargs["w"]
      # Build space and momentum grids
      x1 = x_min[0] + torch.arange(n1, dtype=torch.float64)*dx[0] # size n1
      x2 = x_min[1] + torch.arange(n2, dtype=torch.float64)*dx[1]
      x3 = x_min[2] + torch.arange(n3, dtype=torch.float64)*dx[2]
      gx, gy, gz = torch.meshgrid(x1, x2, x3, indexing='ij')
      self.pot = None
      self.potential = self.pot.to(device=app.device, dtype=torch.double)
      self.form = None
      assert (not self.pot) or (not self.potential) or (not self.form), "The potential is not configured yet."
