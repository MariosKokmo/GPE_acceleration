
# =============================================================================
# Cylindrical Coordinate Setup
# =============================================================================
#
# Reused without modification from the Cartesian section above:
#   _load_json_from_cwd, get_application_config
#   _require_keys, _format_name_component, _ensure_simulation_directory
#   get_simulation_combinations, _simulations_repetitive, _simulations_multi_vortex
#   save_parameters_to_json
#   _perform_frequency_checks   (works for any-length frequency list)
#   _perform_reimprint_checks   (purely about imprint timing, not grid geometry)
#
# New additions below handle the cylindrical grid layout (r_max, z_min, z_max)
# and the r ≥ 0 vortex-position constraint.
# =============================================================================
import math
import numpy as np
from src.utils.setup_simulations import (
    _load_json_from_cwd, 
    _require_keys, 
    _perform_frequency_checks,
    _perform_reimprint_checks,
    REQUIRED_VORTEX_KEYS,
    REQUIRED_REPETITIVE_IMPRINTING_KEYS,
    )
from src.library.parameters import CONSTANTS

REQUIRED_CYLINDRICAL_CONFIG_KEYS = [
    "Grid_resolution",        # [n_r, n_phi, n_z]
    "r_max",                  # outer radial boundary in microns
    "z_min",                  # axial lower bound in microns (must be < 0)
    "z_max",                  # axial upper bound in microns (must be > 0)
    "Trapping_frequencies",   # [fr, fz] in Hz  — radial then axial
    "Total_simulation_time",
    "dt",
    "snapshots",
    "Potential_type",
    "SwitchOff_time",
]


def get_simulation_parameters_cylindrical(config_file_path):
    """
    Read and process simulation parameters for a cylindrical-coordinate GPE run.

    Mirrors :func:`get_simulation_parameters_cartesian` but expects a cylindrical grid
    specification (``r_max``, ``z_min``, ``z_max``) instead of symmetric
    Cartesian limits, and two trapping frequencies ``[fr, fz]`` instead of
    three.

    Derived quantities
    ------------------

    ::

        a_ho      : harmonic oscillator length √(ħ/m ω_ho) in metres.
        omega_ho  : geometric mean (ωr² ωz)^(1/3)  [rad/s].
        w         : normalised frequencies [ωr/ω_ho, ωz/ω_ho].
        r_max, z_min, z_max : grid bounds scaled to dimensionless units.
        dr, dphi, dz : grid spacings in dimensionless units.
        d_x       : dr · dphi · dz, the non-r-weighted part of the
                    cylindrical volume element; the r-weight is applied
                    inside the library.

    Parameters
    ----------
    config_file_path : str
        Path to the JSON configuration file (relative to cwd).

    Returns
    -------
    simulation_params : dict or None
        All simulation parameters.  ``None`` on validation failure.
    msg : str
        Error message on failure, empty string on success.
    """
    msg = ""

    try:
        sim_params = _load_json_from_cwd(config_file_path)
        _require_keys(sim_params, REQUIRED_CYLINDRICAL_CONFIG_KEYS, "cylindrical simulation configuration")
    except (FileNotFoundError, ValueError, OSError, TypeError) as e:
        return None, f"[FATAL] {e}"

    # --- Grid ---
    try:
        n_r, n_phi, n_z = sim_params["Grid_resolution"]
    except (TypeError, ValueError) as e:
        return None, f"[FATAL] Invalid Grid_resolution format. Expected [n_r, n_phi, n_z]: {e}"

    r_max_um = float(sim_params["r_max"])    # microns
    z_min_um = float(sim_params["z_min"])    # microns
    z_max_um = float(sim_params["z_max"])    # microns

    # --- Trapping frequencies ---
    try:
        freq_list = sim_params["Trapping_frequencies"]
        if len(freq_list) == 2:
            fr, fz = freq_list
        elif len(freq_list) == 3:
            # Accept [fr, fr, fz] for config-file compatibility
            fr, _, fz = freq_list
        else:
            return None, "[FATAL] Trapping_frequencies must have 2 or 3 elements [fr, fz]."
    except (TypeError, ValueError) as e:
        return None, f"[FATAL] Invalid Trapping_frequencies: {e}"

    wr = 2.0 * math.pi * float(fr)
    wz = 2.0 * math.pi * float(fz)
    # Geometric mean for an axially symmetric trap (ωr, ωr, ωz)
    omega_ho = (wr * wr * wz) ** (1.0 / 3.0)

    # --- Time ---
    t_evol = sim_params["Total_simulation_time"]
    dt = sim_params["dt"]
    shots = sim_params["snapshots"]
    dtau = omega_ho * dt
    kmax = int(t_evol // dt)

    # --- Physical scales ---
    a_ho = math.sqrt(CONSTANTS.hbar / CONSTANTS.m1 / omega_ho)
    u = 4.0 * math.pi * CONSTANTS.nat * CONSTANTS.ascat / a_ho
    k3 = CONSTANTS.k3 if bool(sim_params.get("three-body-losses", False)) else 0.0

    # --- Normalise frequencies and grid bounds to dimensionless units ---
    wr_norm = wr / omega_ho
    wz_norm = wz / omega_ho
    w = [wr_norm, wz_norm]

    r_max  =  r_max_um * 1e-6 / a_ho
    z_min  =  z_min_um * 1e-6 / a_ho
    z_max  =  z_max_um * 1e-6 / a_ho

    # --- Grid spacings ---
    dr   = r_max / n_r
    dphi = 2.0 * math.pi / n_phi
    dz   = (z_max - z_min) / n_z
    d_x  = dr * dphi * dz        # non-r-weighted part; library applies r-weight

    # Default vortex values — safe for non-vortex runs
    vortex_excitation    = sim_params.get("vortex_excitation", False)
    vortex_charge        = []
    imprinting_charge    = []
    vortex_position_x    = []   # radial (r) positions
    vortex_position_y    = []   # azimuthal (φ) positions
    initial_imprint_time = []
    imprint_position_x   = []
    imprint_position_y   = []
    imprint_every        = []
    imprint_times        = []
    max_imprints         = []
    repetitive           = None

    if vortex_excitation:
        _require_keys(
            sim_params,
            REQUIRED_VORTEX_KEYS,
            "vortex excitation settings",
        )
        vortex_position_x    = sim_params["vortex_position_x"]   # radial
        vortex_position_y    = sim_params["vortex_position_y"]   # azimuthal angle
        initial_imprint_time = sim_params["initial_imprint_time"]
        vortex_charge        = sim_params["vortex_charge"]

        repetitive = sim_params.get("repetitive", None)
        if repetitive is not None:
            _require_keys(sim_params, REQUIRED_REPETITIVE_IMPRINTING_KEYS, "repetitive imprinting settings")
            imprint_every     = sim_params["imprint_every"]
            max_imprints      = sim_params["max_imprints"]
            imprinting_charge = sim_params["imprinting_charge"]
            imprint_position_x = sim_params["imprint_position_x"]  # radial
            imprint_position_y = sim_params["imprint_position_y"]  # azimuthal
            imprint_times     = sim_params["imprint_times"]

        if repetitive and (len(imprint_every) != len(imprint_times)):
            return None, (
                "[FATAL] imprint_every and imprint_times have different number of simulations. "
                "Write an empty list [] when not using exact times."
            )

        if repetitive:
            for i in range(len(imprinting_charge)):
                if len(imprint_times[i]) == 0:
                    step = imprint_every[i]
                    imprint_times[i] = [step * j for j in range(1, max_imprints[i] + 1)]

    # --- Finite-temperature / SGPE / ZNG parameters ---
    model_type           = str(sim_params.get("model_type", "BEC"))
    temperature          = float(sim_params.get("temperature", 0.0))
    damping_coefficient  = float(sim_params.get("damping_coefficient", 0.03))
    n_test_particles     = int(sim_params.get("n_test_particles", 10_000))
    gamma_12             = float(sim_params.get("gamma_12", 0.1))
    enable_c22           = bool(sim_params.get("enable_c22", False))
    # Whether the condensate may actually exchange atoms with the thermal
    # cloud (free norm) or is pinned to a fixed number after every step.
    zng_condensate_exchange = bool(sim_params.get("zng_condensate_exchange", False))
    # How the thermal fraction is fixed: "temperature" derives it from T via
    # the ideal-Bose result, "explicit" takes zng_thermal_fraction directly.
    zng_thermal_fraction_mode = str(sim_params.get("zng_thermal_fraction_mode", "temperature"))
    zng_thermal_fraction = sim_params.get("zng_thermal_fraction", None)
    if zng_thermal_fraction is not None:
        zng_thermal_fraction = float(zng_thermal_fraction)
    _mu_raw              = sim_params.get("chemical_potential", None)
    chemical_potential   = float(_mu_raw) if _mu_raw is not None else None

    # --- Absorber ---
    absorber_enabled     = bool(sim_params.get("Absorber_enabled", False))
    absorber_strength    = float(sim_params.get("Absorber_strength", 0.0))
    absorber_start_ratio = float(sim_params.get("Absorber_start_ratio", 0.8))
    absorber_power       = float(sim_params.get("Absorber_power", 2.0))
    absorber_tinit       = float(sim_params.get("Absorber_tinit", 0.0))
    absorber_tfinal      = sim_params.get("Absorber_tfinal", None)

    simulation_params = {
        # Grid
        "Grid_resolution":    sim_params["Grid_resolution"],
        "r_max":              r_max,
        "z_min":              z_min,
        "z_max":              z_max,
        "dr":                 dr,
        "dphi":               dphi,
        "dz":                 dz,
        "d_x":                d_x,
        # Frequencies / scales
        "Trapping_frequencies": sim_params["Trapping_frequencies"],
        "w":                  w,
        "a_ho":               a_ho,
        "omega_ho":           omega_ho,
        # Interaction / losses
        "u":                  u,
        "k3":                 k3,
        # Time
        "dtau":               dtau,
        "dt":                 dt,
        "Total_simulation_time": t_evol,
        "kmax":               kmax,
        "shots":              shots,
        # Potential
        "Potential_type":     sim_params["Potential_type"],
        "SwitchOff_time":     sim_params["SwitchOff_time"],
        # Absorber
        "Absorber_enabled":     absorber_enabled,
        "Absorber_strength":    absorber_strength,
        "Absorber_start_ratio": absorber_start_ratio,
        "Absorber_power":       absorber_power,
        "Absorber_tinit":       absorber_tinit,
        "Absorber_tfinal":      absorber_tfinal,
        # Vortex / imprint
        "vortex_excitation":    vortex_excitation,
        "vortex_charge":        vortex_charge,
        "imprinting_charge":    imprinting_charge,
        "vortex_position_x":    vortex_position_x,
        "vortex_position_y":    vortex_position_y,
        "initial_imprint_time": initial_imprint_time,
        "imprint_position_x":   imprint_position_x,
        "imprint_position_y":   imprint_position_y,
        "imprint_every":        imprint_every,
        "imprint_times":        imprint_times,
        "max_imprints":         max_imprints,
        "repetitive":           repetitive,
        # Finite-temperature
        "model_type":           model_type,
        "temperature":          temperature,
        "damping_coefficient":  damping_coefficient,
        "n_test_particles":     n_test_particles,
        "gamma_12":             gamma_12,
        "chemical_potential":   chemical_potential,
        "enable_c22":           enable_c22,
        "zng_condensate_exchange": zng_condensate_exchange,
        "zng_thermal_fraction_mode": zng_thermal_fraction_mode,
        "zng_thermal_fraction": zng_thermal_fraction,
    }

    # Carry dark-soliton settings through unchanged so the simulation builder
    # and the BEC can read them. Only meaningful when dark_soliton is truthy;
    # required soliton keys are validated downstream in get_simulation_combinations.
    if sim_params.get("dark_soliton", False):
        simulation_params["dark_soliton"] = sim_params["dark_soliton"]
        for key in ("soliton_positions", "soliton_widths", "soliton_axes",
                    "soliton_greyness", "soliton_imprint_time"):
            if key in sim_params:
                simulation_params[key] = sim_params[key]

    _, msg = _check_simulation_parameters_cylindrical(simulation_params)
    return simulation_params, msg


def _check_simulation_parameters_cylindrical(simulation_params):
    """
    Validate cylindrical simulation parameters.

    Runs grid, frequency, vortex, and re-imprint checks.
    Frequency and re-imprint checks are reused from the Cartesian section
    since they are coordinate-independent.

    Parameters
    ----------
    simulation_params : dict

    Returns
    -------
    ok : bool
    msg : str
    """
    checks = [
        _perform_grid_checks_cylindrical,
        _perform_frequency_checks,          # reused: works for [fr, fz] too
        _perform_vortex_checks_cylindrical,
        _perform_reimprint_checks,          # reused: purely about imprint timing
    ]
    overall_msg = ""
    overall = True
    for check in checks:
        ok, msg = check(simulation_params)
        if msg:
            overall_msg += msg + "\n"
        overall = overall and ok
    return overall, overall_msg


def _perform_grid_checks_cylindrical(simulation_params):
    """
    Validate the cylindrical grid parameters.

    Checks:
    - r_max > 0
    - z_min < 0 and z_max > 0  (trap centred at z = 0)
    - z_min < z_max
    - All grid point counts are positive

    Parameters
    ----------
    simulation_params : dict

    Returns
    -------
    ok : bool
    msg : str
    """
    if not isinstance(simulation_params, dict):
        raise TypeError("simulation_params must be a dictionary")

    r_max = simulation_params["r_max"]
    z_min = simulation_params["z_min"]
    z_max = simulation_params["z_max"]

    if r_max <= 0:
        return False, "r_max must be positive."

    if z_min >= 0:
        return False, "z_min must be negative (trap centred at z = 0)."

    if z_max <= 0:
        return False, "z_max must be positive (trap centred at z = 0)."

    if z_min >= z_max:
        return False, "z_min must be less than z_max."

    if min(simulation_params["Grid_resolution"]) <= 0:
        return False, "All grid point counts (n_r, n_phi, n_z) must be positive."

    return True, ""


def _perform_vortex_checks_cylindrical(simulation_params):
    """
    Validate vortex parameters for a cylindrical grid.

    Key difference from Cartesian: radial positions (vortex_position_x) must
    be non-negative because r ≥ 0.  Angle positions (vortex_position_y) must
    lie within [0, 2*pi].

    Parameters
    ----------
    simulation_params : dict

    Returns
    -------
    ok : bool
    msg : str
    """
    r_max = simulation_params["r_max"]

    if len(simulation_params["vortex_charge"]) != len(simulation_params["vortex_position_x"]):
        return False, "Number of vortex charges does not match number of radial (x) positions."

    if len(simulation_params["vortex_position_y"]) != len(simulation_params["vortex_position_x"]):
        return False, "Number of radial (x) positions does not match number of angular (y) positions."

    if simulation_params["repetitive"] and (
        len(simulation_params["vortex_charge"]) != len(simulation_params["imprinting_charge"])
    ):
        return False, "Number of initial vortex charges does not match number of imprinted charges."

    for idx, charges in enumerate(simulation_params["vortex_charge"]):
        if not isinstance(charges, list):
            continue

        pos_r     = simulation_params["vortex_position_x"][idx]   # radial positions
        pos_theta = simulation_params["vortex_position_y"][idx]   # azimuthal angles

        if len(charges) != len(pos_r):
            return False, f"Charge / radial-position count mismatch for simulation {idx + 1}."
        if len(charges) != len(pos_theta):
            return False, f"Charge / angular-position count mismatch for simulation {idx + 1}."
        if not pos_r:
            return False, f"Radial position list is empty for simulation {idx + 1}."
        if not pos_theta:
            return False, f"Angular position list is empty for simulation {idx + 1}."

        # r ≥ 0 and within grid boundary
        if min(pos_r) < 0:
            return False, (
                f"Radial vortex positions must be ≥ 0 for simulation {idx + 1}. "
                f"Got minimum {min(pos_r)}."
            )
        if max(pos_r) > r_max:
            return False, (
                f"Maximum radial position {max(pos_r)} exceeds r_max = {r_max:.4g} "
                f"for simulation {idx + 1}."
            )

        # φ ∈ [0, 2π]
        if max(pos_theta) > 2 * np.pi:
            return False, (
                f"Maximum angular position {max(pos_theta):.4g} exceeds 2π "
                f"for simulation {idx + 1}."
            )
        if min(pos_theta) < 0:
            return False, (
                f"Minimum angular position {min(pos_theta):.4g} is less than 0 "
                f"for simulation {idx + 1}."
            )

    return True, ""