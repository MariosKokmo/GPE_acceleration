"""
This module provides the class to build the equation of the system.
In order to accommodate for equations with various terms.
"""
import torch

class Equation:
    """Class to provide the RHS equation of the system.
    It is assume that the LHS contains only the time derivative term i dPsi/dt.
    All other terms are included in the RHS. 

    The space operator is defined as the sum of all terms that depend on the wavefunction psi and the external potential uext.
    This is the part of the split-step Fourier algorithm that sits in the middle of the time evolution step, and is applied in real space.
    """
    def __init__(self):
        self.space_operator = None

class GPE_base(Equation):
    def __init__(self, params, psi, uext):
        super().__init__()
        try:
            self.u = params["u"]
        except KeyError:
            raise KeyError("The interaction strength u is not provided in the parameters dictionary.")
        self.psi = psi
        self.uext = uext
        self.space_operator = (self.u * torch.abs(self.psi) ** 2 + self.uext)

class GPE_3body_loss(Equation):
    def __init__(self, params, psi, uext):
        super().__init__()
        try:
            self.u = params["u"]
            self.k3 = params["k3"]
        except KeyError as e:
            raise KeyError(f"The required parameter {str(e)} is not provided in the parameters dictionary.")
        self.psi = psi
        self.uext = uext
        self.loss_term = - self.k3 * torch.abs(self.psi)**4
        self.space_operator = (self.u * torch.abs(self.psi) ** 2 + self.uext) + (1j * self.loss_term)
        
class CustomEquation(Equation):
    def __init__(self, custom_space_operator):
        super().__init__()
        self.space_operator = custom_space_operator