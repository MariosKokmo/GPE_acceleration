import torch

###############################################################################
##########################   EXTERNAL POTENTIALS ##############################
###############################################################################

class Potential():
    def __init__(self):
      self.form
      self.potential
    
    def evol(self, t):
      """
      Returns the external potential at a specific time.
      Args
       t: float, time
      Returns:
       torch.Tensor, the potential at time t
      """
      return self.form(t) * self.potential

class ConstPot(Potential):
   """Constant potential across the grid"""
   def __init__(self, amplitude, grid, device):
      n1, n2, n3 = grid
      self.potential = amplitude * torch.ones(n1,n2,n3, dtype=torch.cdouble, device=device)
      self.form = lambda t: 1

class RampPot(Potential):
   """Creates a ramp potential that evolves like
   initial + t * (final - initial) / tfinal
   """
   def __init__(self, initial, final, grid, tfinal, device):
      n1, n2, n3 = grid
      self.potential = torch.ones(n1,n2,n3, dtype=torch.cdouble, device=device)
      self.form = lambda t: (initial + (final - initial) * (t / tfinal))

class HarmonicPot(Potential):
   def __init__(self, grid, x_min, dx, w, device, amplitude=1):
      n1, n2, n3 = grid
      # Build space and momentum grids
      x1 = x_min[0] + torch.arange(n1, dtype=torch.float64)*dx[0] # size n1
      x2 = x_min[1] + torch.arange(n2, dtype=torch.float64)*dx[1]
      x3 = x_min[2] + torch.arange(n3, dtype=torch.float64)*dx[2]
      gx, gy, gz = torch.meshgrid(x1, x2, x3)
      self.pot = 0.5 * amplitude * ((w[0]*gx)**2 + (w[1]*gy)**2 + (w[2]*gz)**2)
      self.potential = self.pot.to(device=device, dtype=torch.cdouble)
      self.form = lambda t: 1

class RampHarmonicPot(Potential):
   """A harmonic potential that evolves linearly in time"""
   def __init__(self, grid, x_min, dx, w, tfinal, device, initial=0, amplitude=1):
      n1, n2, n3 = grid
      # Build space and momentum grids
      x1 = x_min[0] + torch.arange(n1, dtype=torch.float64)*dx[0] # size n1
      x2 = x_min[1] + torch.arange(n2, dtype=torch.float64)*dx[1]
      x3 = x_min[2] + torch.arange(n3, dtype=torch.float64)*dx[2]
      gx, gy, gz = torch.meshgrid(x1, x2, x3)
      self.pot = 0.5 * amplitude * ((w[0]*gx)**2 + (w[1]*gy)**2 + (w[2]*gz)**2)
      self.potential = self.pot.to(device=device, dtype=torch.cdouble)
      self.form = lambda t: initial + (amplitude - initial) * (t / tfinal)
