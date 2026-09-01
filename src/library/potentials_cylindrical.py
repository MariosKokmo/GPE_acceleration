"""External potentials for the GPE in cylindrical coordinates (r, φ, z).

Drop-in cylindrical counterpart to :mod:`src.library.potentials`.

Grid conventions (same as :class:`GPECylindricalLibrary`)
----------------------------------------------------------
r   : half-point, dr/2 … r_max − dr/2
φ   : 0 … 2π  (periodic — no absorber)
z   : z_min … z_max

Expected ``kwargs`` keys
------------------------

::

    Grid_resolution : (n_r, n_phi, n_z)
    r_max           : outer radial boundary.
    z_min, z_max    : axial extent.
    w               : [wr, wz] or [wr, wr, wz] — radial/axial trap frequencies.
                      For 2-element form w[0]=wr, w[1]=wz.
                      For 3-element form w[0]=wr, w[2]=wz (middle ignored).

Optional absorber keys (same semantics as Cartesian)
-----------------------------------------------------
Absorber_enabled      : bool
Absorber_strength     : float (η in −iη·V_cap)
Absorber_start_ratio  : float ∈ (0,1), default 0.8
Absorber_power        : float ≥ 1, default 2
Absorber_tinit        : float, ramp-on start time
Absorber_tfinal       : float, ramp-on end time (None = instant on)
SwitchOff_time        : float, snapshot index after which potential → 0
"""

import torch
from src.library.gpe_cylindrical_library import GPECylindricalLibrary as cyl


###############################################################################
# Selector
###############################################################################

def select_potential_cylindrical(potential_type, app, **simulation_parameters):
    """
    Factory function mirroring :func:`select_potential` for cylindrical grids.

    Available types
    ---------------
    harmonic        : axisymmetric harmonic trap  ½(ωr r)² + ½(ωz z)²
    constant        : uniform potential
    ramp            : linearly time-ramped constant
    rampharmonic    : linearly time-ramped harmonic trap
    rotating        : rotating anisotropic harmonic (stirring)
    gaussianbeam    : focused Gaussian laser beam along z-axis
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
    """
    Base class for external potentials in cylindrical coordinates.

    Provides ``evol(t)``, ``zero()``, and an optional complex absorbing
    potential (CAP) applied at the outer radial boundary and both axial ends.
    φ boundaries are periodic and never absorb.

    Subclasses must set ``self.potential`` (the static shape tensor) and
    optionally ``self.form`` (a callable returning the time-dependent
    amplitude, default ``lambda t: 1.0``).
    """

    def __init__(self, app, **kwargs):
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
        """
        Build a complex absorbing potential (CAP) for cylindrical boundaries.

        The CAP damps the wavefunction near r = r_max and |z| → z_max,
        preventing non-physical reflections at the grid edges.  The φ
        direction is periodic and is therefore never absorbed.

        Profile:
            cap_r(r)  = clamp((r − r_start) / (r_max − r_start), 0, 1)^power
            cap_z(z)  = clamp((|z| − z_start) / (z_extent − z_start), 0, 1)^power
            V_cap     = max(cap_r, cap_z)
            V_abs     = −i · strength · V_cap
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
        if self.absorber_potential is None:
            return 0.0
        return -1j * self.absorber_form(t) * self.absorber_potential

    # ------------------------------------------------------------------
    # Interface (same as Potential)
    # ------------------------------------------------------------------

    def evol(self, t: float) -> torch.Tensor:
        """Return the potential at time t: form(t) · V + V_absorber(t)."""
        return self.form(t) * self.potential + self._absorber_term(t)

    def zero(self) -> torch.Tensor:
        """Set the potential to zero and return it."""
        self.potential = torch.zeros_like(self.potential)
        return self.potential


###############################################################################
# Concrete potential classes
###############################################################################

class CylindricalConstPot(CylindricalPotential):
    """Uniform potential of constant amplitude across the grid."""

    def __init__(self, app, amplitude: float = 1.0, **kwargs):
        super().__init__(app, **kwargs)
        n_r, n_phi, n_z = kwargs["Grid_resolution"]
        self.potential = amplitude * torch.ones(
            n_r, n_phi, n_z, dtype=torch.double, device=self.app.device
        )
        self.form = lambda t: 1.0


class CylindricalRampPot(CylindricalPotential):
    """
    Constant spatial profile with a linear time ramp.

        V(t) = initial + (final − initial) · (t / tfinal)
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
    """
    Axisymmetric harmonic trap.

        V(r, z) = ½ · amplitude · [(ωr r)² + (ωz z)²]

    ``w = [wr, wz]`` or ``w = [wr, wr, wz]`` (3-element for config-file
    compatibility; the middle element is ignored).

    Extra method
    ------------
    zero_radial() : keep only the axial confinement (releases radial trap).
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
        """Release radial confinement; keep only axial harmonic potential."""
        self.potential = (
            0.5 * amplitude * (self._wz * self._gz) ** 2
        ).to(dtype=torch.double, device=self.app.device)
        return self.potential


class CylindricalRampHarmonicPot(CylindricalPotential):
    """
    Axisymmetric harmonic trap whose amplitude is linearly ramped in time.

        V(r, z, t) = [initial + (amplitude − initial)·(t − tinit)/(tfinal − tinit)]
                     · ½[(ωr r)² + (ωz z)²]
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
    """
    Rotating anisotropic harmonic trap (standard stirring geometry).

        V(r, φ, z, t) = ½ωr²r²[1 + ε·cos(2(φ − Ωt))] + ½ωz²z²

    The angular anisotropy ε breaks the cylindrical symmetry and, when the
    trap rotates at Ω, transfers angular momentum to the condensate — the
    standard mechanism for creating quantised vortices in experiments.

    Parameters
    ----------
    amplitude         : overall amplitude (default 1).
    angular_frequency : Ω in dimensionless units ωho (default 1).
    anisotropy        : ε ∈ [0, 1) — ellipticity of the trap (default 0.05).
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
        """Return the rotated trap at time t (recomputes the angular phase)."""
        return self._compute(t) + self._absorber_term(t)


class CylindricalGaussianBeamPot(CylindricalPotential):
    """
    Focused Gaussian laser beam propagating along the z-axis.

        V(r, z) = amplitude · exp(−2r²/w0²) · exp(−2(z − z0)²/zR²)

    where ``w0`` is the beam waist and ``zR`` is the Rayleigh range.
    Setting ``rayleigh_range=None`` (default) gives a collimated beam with
    no axial profile.

    Common uses: repulsive obstacle (blue-detuned laser), stirring beam,
    localised potential barrier.

    Parameters
    ----------
    amplitude      : peak amplitude (positive = repulsive, negative = attractive).
    beam_waist     : w0 — 1/e² intensity radius at the focus.
    rayleigh_range : zR — axial 1/e² intensity half-length (None = no z decay).
    z_center       : z0 — axial position of the beam focus.
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
