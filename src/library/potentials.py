"""Provides some of the most common external potentials that are experimentally used"""
import torch

###############################################################################
##########################   EXTERNAL POTENTIALS ##############################
###############################################################################
def select_potential(potentialType, app, **simulation_parameters):
   potentialType = potentialType.strip().lower()
   # available_potentials = {
   #    "harmonic" : HarmonicPot(app, **simulation_parameters),
   #    "constant" : ConstPot(app, **simulation_parameters),
   #    "ramp" : RampPot(app, **simulation_parameters),
   #    "rampharmonic" : RampHarmonicPot(app, **simulation_parameters),
   #    "custom" : CustomPot(app, **simulation_parameters),
   # }
   if potentialType == "harmonic":
      return HarmonicPot(app, **simulation_parameters)
   elif potentialType == "constant":
      return ConstPot(app, **simulation_parameters)
   elif potentialType == "ramp":
      return RampPot(app, **simulation_parameters)
   elif potentialType == "rampharmonic":
      return RampHarmonicPot(app, **simulation_parameters)
   else:
      return None

class Potential():
    def __init__(self):
      """
      form: is a function of time
      potential: is the initial value of the potential
      """
      self.form
      self.potential
      self.switchOff_time
    
    def evol(self, t):
      """
      Returns the external potential at a specific time.
      Args:
      -------
       t: float, time
      Returns:
      -------
       torch.Tensor, the potential at time t
      """
      return self.form(t) * self.potential
    
    def zero(self):
      """
      Sets the potential to zero.
      """
      self.potential = torch.zeros_like(self.potential)
      return self.potential

class ConstPot(Potential):
   """Constant potential across the grid"""
   def __init__(self, amplitude, grid, device):
      n1, n2, n3 = grid
      self.potential = amplitude * torch.ones(n1,n2,n3, dtype=torch.double, device=device)
      self.form = lambda t: 1

class RampPot(Potential):
   """
   Creates a ramp potential that evolves like
   initial + (final - initial) * (t / tfinal)
   """
   def __init__(self, initial, final, grid, tfinal, device):
      n1, n2, n3 = grid
      self.potential = torch.ones(n1,n2,n3, dtype=torch.double, device=device)
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
      n1, n2, n3 = kwargs["Grid_resolution"]
      x_min = kwargs["x_min"]
      dx = kwargs["dx"]
      w = kwargs["w"]
      self.switchOff_time = kwargs["SwitchOff_time"]
      # Build space and momentum grids
      x1 = x_min[0] + torch.arange(n1, dtype=torch.float64)*dx[0] # size n1
      x2 = x_min[1] + torch.arange(n2, dtype=torch.float64)*dx[1]
      x3 = x_min[2] + torch.arange(n3, dtype=torch.float64)*dx[2]
      gx, gy, gz = torch.meshgrid(x1, x2, x3)
      self.pot = 0.5 * amplitude * ((w[0]*gx)**2 + (w[1]*gy)**2 + (w[2]*gz)**2)
      self.potential = self.pot.to(device=app.device, dtype=torch.double)
      self.form = lambda t: 1

   def zero_2D(self, app, amplitude=1, **kwargs):
      """
      shuts off the potential only on 2 dimensions. The 3rd dimension that is considered flat
      is still kept.
      Returns a new potential.
      """
      n1, n2, n3 = kwargs["Grid_resolution"]
      x_min = kwargs["x_min"]
      dx = kwargs["dx"]
      w = kwargs["w"]
      self.switchOff_time = kwargs["SwitchOff_time"]
      # Build space and momentum grids
      x1 = x_min[0] + torch.arange(n1, dtype=torch.float64)*dx[0] # size n1
      x2 = x_min[1] + torch.arange(n2, dtype=torch.float64)*dx[1]
      x3 = x_min[2] + torch.arange(n3, dtype=torch.float64)*dx[2]
      gx, gy, gz = torch.meshgrid(torch.zeros_like(x1), x2, torch.zeros_like(x3))
      self.pot = 0.5 * amplitude * ((w[0]*gx)**2 + (w[1]*gy)**2 + (w[2]*gz)**2)
      self.potential = self.pot.to(device=app.device, dtype=torch.double)
      self.form = lambda t: 1


class RampHarmonicPot(Potential):
   """A harmonic potential that evolves linearly in time"""
   
   def __init__(self, tfinal, app, initial=1.0, amplitude=1.0, tinit=0.0, **kwargs):
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
      n1, n2, n3 = kwargs["Grid_resolution"]
      x_min = kwargs["x_min"]
      dx = kwargs["dx"]
      w = kwargs["w"]
      # Build space and momentum grids
      x1 = x_min[0] + torch.arange(n1, dtype=torch.float64)*dx[0] # size n1
      x2 = x_min[1] + torch.arange(n2, dtype=torch.float64)*dx[1]
      x3 = x_min[2] + torch.arange(n3, dtype=torch.float64)*dx[2]
      gx, gy, gz = torch.meshgrid(x1, x2, x3)
      self.pot = 0.5 * ((w[0]*gx)**2 + (w[1]*gy)**2 + (w[2]*gz)**2)
      self.potential = self.pot.to(device=app.device, dtype=torch.double)
      self.form = lambda t: initial + (amplitude - initial) * ((t-tinit) / (tfinal-tinit))


class CustomPot(Potential):
   """
   Custom potential. The time dependence and the shape of the potential need to
   be defined.
   """
   def __init__(self, app, **kwargs):
      n1, n2, n3 = kwargs["Grid_resolution"]
      x_min = kwargs["x_min"]
      dx = kwargs["dx"]
      w = kwargs["w"]
      # Build space and momentum grids
      x1 = x_min[0] + torch.arange(n1, dtype=torch.float64)*dx[0] # size n1
      x2 = x_min[1] + torch.arange(n2, dtype=torch.float64)*dx[1]
      x3 = x_min[2] + torch.arange(n3, dtype=torch.float64)*dx[2]
      gx, gy, gz = torch.meshgrid(x1, x2, x3)
      self.pot = None
      self.potential = self.pot.to(device=app.device, dtype=torch.double)
      self.form = None
      assert (not self.pot) or (not self.potential) or (not self.form), "The potential is not configured yet."