"""Provides some of the most common external potentials that are experimentally used"""
import torch

###############################################################################
##########################   EXTERNAL POTENTIALS ##############################
###############################################################################
def select_potential(potentialType, app, **simulation_parameters):
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
    def __init__(self, app, **kwargs):
      """
      form: is a function of time
      potential: is the initial value of the potential
      """
      self.app = app
      self.form = lambda t: 1.0
      self.potential = None
      self.switchOff_time = kwargs.get("SwitchOff_time", None)
      self.absorber_potential = None
      self.absorber_form = lambda t: 1.0
      self._configure_absorber(**kwargs)

    def _configure_absorber(self, **kwargs):
      """
      Optional complex absorbing potential (CAP) used to damp wavefunction
      amplitude near the grid boundaries and suppress reflections.
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
      if self.absorber_potential is None:
         return 0.0
      return -1j * self.absorber_form(t) * self.absorber_potential
    
    def evol(self, t):
      """
      Returns the external potential at a specific time.

      Args:
          t (float): time.

      Returns:
          torch.Tensor: the potential at time t.
      """
      return self.form(t) * self.potential + self._absorber_term(t)
    
    def zero(self):
      """
      Sets the potential to zero.
      """
      self.potential = torch.zeros_like(self.potential)
      return self.potential

class ConstPot(Potential):
   """Constant potential across the grid"""
   def __init__(self, app, amplitude=1.0, **kwargs):
      super().__init__(app, **kwargs)
      n1, n2, n3 = kwargs["Grid_resolution"]
      self.potential = amplitude * torch.ones(n1,n2,n3, dtype=torch.double, device=self.app.device)
      self.form = lambda t: 1

class RampPot(Potential):
   """
   Creates a ramp potential that evolves like
   initial + (final - initial) * (t / tfinal)
   """
   def __init__(self, app, initial=1.0, final=2.0, tfinal=1.0, **kwargs):
      super().__init__(app, **kwargs)
      n1, n2, n3 = kwargs["Grid_resolution"]
      self.potential = torch.ones(n1,n2,n3, dtype=torch.double, device=self.app.device)
      self.form = lambda t: (initial + (final - initial) * (t / tfinal))

class HarmonicPot(Potential):
    """Returns a harmonic potential of the form 
      amplitude * 1/2 * (wx*x^2 + wy*y^2 + wz*z^2)"""
    
    def __init__(self, app, amplitude=1, **kwargs):
      """
      Sets the time-dependence of the potential in the `form` parameter
      and the shape of the potential in the `potential` parameter of the class.
      
      Args:
      -----
      app: application,
      amplitude: float, the final value of the potential, default=1.0
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
      """
      shuts off the potential only on 2 dimensions. The 3rd dimension that is considered flat
      is still kept.
      Returns a new potential.
      """ 
      gx, gy, gz = torch.meshgrid(torch.zeros_like(self.x1), self.x2, torch.zeros_like(self.x3), indexing='ij')
      self.pot = 0.5 * amplitude * ((self.w[0]*gx)**2 + (self.w[1]*gy)**2 + (self.w[2]*gz)**2)
      self.potential = self.pot.to(device=self.app.device, dtype=torch.double)
      self.form = lambda t: 1
      return self.potential


class RampHarmonicPot(Potential):
   """A harmonic potential that evolves linearly in time"""
   
   def __init__(self, app, initial=1.0, amplitude=1.0, tinit=0.0, tfinal=1.0,  **kwargs):
      """
      Sets the time-dependence of the potential in the `form` parameter
      and the shape of the potential in the `potential` parameter of the class.
      The initial potential is always that used to calculate the ground state.
      Args:
      -----
      tfinal: float, the time when the ramp stops
      app: application,
      initial: float, the initial value of the potential, default=1.0
      amplitude: float, the final value of the potential, default=1.0
      tinit: float, the initial time when the ramp starts, default=0
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
   """
   Harmonic potential that rotates in time around a configurable axis.

   The static harmonic profile is evaluated in a co-rotating frame by applying
   a rotation of angle ``theta(t) = angular_frequency * t`` to coordinates.
   """

   def __init__(self, app, amplitude=1.0, angular_frequency=1.0, axis=3, **kwargs):
      """
      Args:
      -----
      app: application
      amplitude: float, overall potential amplitude
      angular_frequency: float, rotation speed in rad/(dimensionless time)
      axis: int|str, rotation axis (1/2/3 or x/y/z)
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
      """Normalize axis input and return 1, 2, or 3."""
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
      """Return rotated coordinates for the selected axis at angle theta."""
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
      """Return the rotated harmonic potential at time t."""
      theta = self.angular_frequency * t
      x_rot, y_rot, z_rot = self._rotated_coordinates(theta)

      potential = 0.5 * self.amplitude * (
         (self.w[0] * x_rot) ** 2
         + (self.w[1] * y_rot) ** 2
         + (self.w[2] * z_rot) ** 2
      )
      return potential.to(device=self.app.device, dtype=torch.double) + self._absorber_term(t)


class CustomPot(Potential):
   """
   Custom potential. The time dependence and the shape of the potential need to
   be defined.
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