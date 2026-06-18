import math
import numpy as np
from src.utils.setup_simulations import (
    _load_json_from_cwd, 
    _require_keys, 
    _perform_frequency_checks, 
    _perform_reimprint_checks,
    REQUIRED_SIMULATION_CONFIG_KEYS
    )
from src.library.parameters import CONSTANTS

# =============================================================================
# Parameter Processing
# =============================================================================

def get_simulation_parameters_cartesian(config_file_path):
    """
    Read and process simulation parameters from configuration file.
    
    This function reads the configuration file, calculates derived parameters
    (normalizations, grid spacing, etc.), and validates the resulting parameters.
    
    Parameters
    ----------
    config_file_path : str
        Path to the configuration file.
    
    Returns
    -------
    simulation_params : dict or None
        Dictionary containing all simulation parameters. Returns None if validation fails.
    msg : str
        Error message if validation fails, empty string otherwise.
    """
    msg = ""

    try:
        # Read the configuration file
        sim_params = _load_json_from_cwd(config_file_path)
        _require_keys(sim_params, REQUIRED_SIMULATION_CONFIG_KEYS, "simulation configuration")
    except (FileNotFoundError, ValueError, OSError, TypeError) as e:
        return None, f"[FATAL] {e}"
    
    # Grid parameters
    try:
        n1, n2, n3 = sim_params["Grid_resolution"]
    except (TypeError, ValueError) as e:
        return None, f"[FATAL] Invalid Grid_resolution format. Expected 3 values: {e}"
    dim = np.array([n1, n2, n3], dtype=np.float64)
    x_min = np.array(sim_params["Grid_negative_limits"])
    x_max = np.array(sim_params["Grid_positive_limits"])
    
    # Trapping frequencies
    try:
        fx, fy, fz = sim_params["Trapping_frequencies"]
    except (TypeError, ValueError) as e:
        return None, f"[FATAL] Invalid Trapping_frequencies format. Expected 3 values: {e}"
    wx = 2 * math.pi * float(fx)
    wy = 2 * math.pi * float(fy)
    wz = 2 * math.pi * float(fz)
    w = np.array([wx, wy, wz])
    omega_ho = (wx * wy * wz) ** (1 / 3)
    
    # Time step parameters
    t_evol = sim_params["Total_simulation_time"]
    dt = sim_params["dt"]
    shots = sim_params["snapshots"]
    dtau = omega_ho * dt
    kmax = int(t_evol // dt)
    
    # Vortex parameters
    vortex_excitation = sim_params["vortex_excitation"]
    vortex_charge = sim_params["vortex_charge"]
    imprinting_charge = sim_params["imprinting_charge"]
    vortex_position_x = sim_params["vortex_position_x"]
    vortex_position_y = sim_params["vortex_position_y"]
    imprint_position_x = sim_params["imprint_position_x"]
    imprint_position_y = sim_params["imprint_position_y"]
    initial_imprint_time = sim_params["initial_imprint_time"]
    
    # Re-imprinting parameters
    repetitive = sim_params["repetitive"]
    imprint_every = sim_params["imprint_every"]
    imprint_times = sim_params["imprint_times"]
    max_imprints = sim_params["max_imprints"]

    # Validate repetitive imprinting configuration
    if repetitive and (len(imprint_every) != len(imprint_times)):
        msg = ("[FATAL]. imprint_every and imprint_times have different number of simulations. "
               "Make sure you write an empty list [] when not using exact times.")
        return None, msg
    
    # Calculate imprint times if not explicitly provided
    if repetitive:
        for i in range(len(imprinting_charge)):
            # For every simulation (i.e., charge), set the imprint times if not given
            if len(imprint_times[i]) == 0:
                time_step = imprint_every[i]
                imprint_times[i] = [time_step * j for j in range(1, max_imprints[i] + 1)]
    
    # Calculate harmonic oscillator length scale in meters
    a_ho = math.sqrt(CONSTANTS.hbar / CONSTANTS.m1 / omega_ho)  
    
    # Interaction strength
    u = 4.0 * math.pi * CONSTANTS.nat * CONSTANTS.ascat / a_ho 
    
    # 3-body losses
    if bool(sim_params.get("three-body-losses", False)):
        k3 = CONSTANTS.k3
    else:
        k3 = 0.0

    # Finite-temperature model selection (optional — defaults to zero-temperature GPE)
    # model_type selects which BEC class is instantiated in simulation.py:
    #   "BEC"          → standard zero-temperature GPE (default)
    #   "FiniteTempBEC"→ Stochastic Projected GPE (SGPE)
    #   "ZNGBEC"       → Zaremba-Nikuni-Griffin two-component framework
    model_type = str(sim_params.get("model_type", "BEC"))

    # SGPE parameters (used by FiniteTempBEC; ignored by BEC)
    temperature = float(sim_params.get("temperature", 0.0))
    damping_coefficient = float(sim_params.get("damping_coefficient", 0.03))

    # ZNG parameters (used by ZNGBEC; ignored by BEC and FiniteTempBEC)
    n_test_particles = int(sim_params.get("n_test_particles", 10_000))
    gamma_12 = float(sim_params.get("gamma_12", 0.1))
    enable_c22 = bool(sim_params.get("enable_c22", False))

    # Shared finite-temperature parameter (used by both SGPE and ZNG)
    # None means "compute from the ground-state wavefunction at runtime"
    _mu_raw = sim_params.get("chemical_potential", None)
    chemical_potential = float(_mu_raw) if _mu_raw is not None else None

    # Optional absorber (complex absorbing potential) settings
    absorber_enabled = bool(sim_params.get("Absorber_enabled", False))
    absorber_strength = float(sim_params.get("Absorber_strength", 0.0))
    absorber_start_ratio = float(sim_params.get("Absorber_start_ratio", 0.8))
    absorber_power = float(sim_params.get("Absorber_power", 2.0))
    absorber_tinit = float(sim_params.get("Absorber_tinit", 0.0))
    absorber_tfinal = sim_params.get("Absorber_tfinal", None)

    # Normalizations
    w = w / omega_ho
    x_max = x_max * 1e-6 / a_ho
    x_min = x_min * 1e-6 / a_ho

    # Grid spacing
    dx = (x_max - x_min) / dim
    dp = 2 * math.pi / (x_max - x_min)
    d_x = np.prod(dx)

    # Assemble simulation parameters dictionary
    simulation_params = {
        "Grid_resolution": [n1, n2, n3],
        "x_min": x_min,
        "x_max": x_max,
        "Trapping_frequencies": sim_params["Trapping_frequencies"],
        "Potential_type": sim_params["Potential_type"],
        "SwitchOff_time": sim_params["SwitchOff_time"],
        "w": w,
        "dx": dx,
        "dp": dp,
        "dtau": dtau,
        "Total_simulation_time": t_evol,
        "kmax": kmax,
        "u": u,
        "k3": k3,
        "Absorber_enabled": absorber_enabled,
        "Absorber_strength": absorber_strength,
        "Absorber_start_ratio": absorber_start_ratio,
        "Absorber_power": absorber_power,
        "Absorber_tinit": absorber_tinit,
        "Absorber_tfinal": absorber_tfinal,
        "a_ho": a_ho,
        "omega_ho": omega_ho,
        "d_x": d_x,
        "shots": shots,
        "dt": dt,
        "vortex_excitation": vortex_excitation,
        "vortex_charge": vortex_charge,
        "imprinting_charge": imprinting_charge,
        "vortex_position_x": vortex_position_x,
        "vortex_position_y": vortex_position_y,
        "initial_imprint_time": initial_imprint_time,
        "imprint_position_x": imprint_position_x,
        "imprint_position_y": imprint_position_y,
        "imprint_every": imprint_every,
        "imprint_times": imprint_times,
        "max_imprints": max_imprints,
        "repetitive": repetitive,
        # Finite-temperature fields — present in all configs; ignored when model_type="BEC"
        "model_type": model_type,
        "temperature": temperature,
        "damping_coefficient": damping_coefficient,
        "n_test_particles": n_test_particles,
        "gamma_12": gamma_12,
        "chemical_potential": chemical_potential,
        "enable_c22": enable_c22,
    }
    
    # Carry dark-soliton settings through unchanged so the simulation builder
    # (get_simulation_combinations) and the BEC can read them. Only meaningful
    # when dark_soliton is truthy; the required soliton keys are validated
    # downstream in get_simulation_combinations.
    if sim_params.get("dark_soliton", False):
        simulation_params["dark_soliton"] = sim_params["dark_soliton"]
        for key in ("soliton_positions", "soliton_widths", "soliton_axes",
                    "soliton_greyness", "soliton_imprint_time"):
            if key in sim_params:
                simulation_params[key] = sim_params[key]

    # Validate simulation parameters
    _, msg = _check_simulation_parameters(simulation_params)
    return simulation_params, msg


def _perform_vortex_checks(simulation_params):
    """
    Validate vortex-related parameters.
    
    Checks that:
    - Number of vortex charges matches number of positions
    - Vortex positions are within grid bounds
    - For repetitive mode: number of initial charges matches imprinting charges
    
    Parameters
    ----------
    simulation_params : dict
        Dictionary containing simulation parameters.
    
    Returns
    -------
    ok : bool
        True if validation passes, False otherwise.
    msg : str
        Error message describing the issue, empty string if validation passes.
    """
    n1, n2, n3 = simulation_params["Grid_resolution"]
    
    # Check the number of combinations
    if len(simulation_params["vortex_charge"]) != len(simulation_params["vortex_position_x"]):
        msg = "The list number of vortex charges doesn't agree with the list number of x positions"
        return False, msg
    
    if len(simulation_params["vortex_position_y"]) != len(simulation_params["vortex_position_x"]):
        msg = "The list number of x positions doesn't agree with the list number of y positions"
        return False, msg
    
    if simulation_params["repetitive"] and (len(simulation_params["vortex_charge"]) != len(simulation_params["imprinting_charge"])):
        msg = "The number of initial vortex charges doesn't agree with the number of imprinted charges"
        return False, msg
    
    for index, charges in enumerate(simulation_params["vortex_charge"]):
        if isinstance(charges, list):
            if len(charges) != len(simulation_params["vortex_position_x"][index]):
                msg = f"The number of charges doesn't agree with the number of x positions for simulation {index + 1}"
                return False, msg
            
            if len(charges) != len(simulation_params["vortex_position_y"][index]):
                msg = f"The number of charges doesn't agree with the number of y positions for simulation {index + 1}"
                return False, msg

            if len(simulation_params["vortex_position_x"][index]) == 0:
                msg = f"The x positions list is empty for simulation {index + 1}"
                return False, msg

            if len(simulation_params["vortex_position_y"][index]) == 0:
                msg = f"The y positions list is empty for simulation {index + 1}"
                return False, msg
            
            if max(simulation_params["vortex_position_x"][index]) > n1 // 2:
                msg = (f"The maximum n1 position for simulation {index + 1}, "
                       f"{max(simulation_params['vortex_position_x'][index])} is greater than "
                       f"half the grid size {n1 // 2}")
                return False, msg
            
            if max(simulation_params["vortex_position_y"][index]) > n3 // 2:
                msg = (f"The maximum n3 position for simulation {index + 1}, "
                       f"{max(simulation_params['vortex_position_y'][index])} is greater than "
                       f"half the grid size {n3 // 2}")
                return False, msg
            
            if min(simulation_params["vortex_position_x"][index]) < -n1 // 2:
                msg = (f"The minimum n1 position for simulation {index + 1}, "
                       f"{min(simulation_params['vortex_position_x'][index])} is less than "
                       f"half the grid size {-n1 // 2}")
                return False, msg
            
            if min(simulation_params["vortex_position_y"][index]) < -n3 // 2:
                msg = (f"The minimum n3 position for simulation {index + 1}, "
                       f"{min(simulation_params['vortex_position_y'][index])} is less than "
                       f"half the grid size {-n3 // 2}")
                return False, msg
    
    return True, ""

def _perform_grid_checks(simulation_params):
    """
    Validate grid-related parameters.
    
    Checks that:
    - simulation_params is a dictionary
    - Grid minimums are negative
    - Grid is symmetric (|x_min| == |x_max|)
    - Grid resolution is positive
    
    Parameters
    ----------
    simulation_params : dict
        Dictionary containing simulation parameters.
    
    Returns
    -------
    ok : bool
        True if validation passes, False otherwise.
    msg : str
        Error message describing the issue, empty string if validation passes.
    
    Raises
    ------
    TypeError
        If simulation_params is not a dictionary.
    """
    # Ensure simulation_params is a dictionary
    if not isinstance(simulation_params, dict):
        raise TypeError("simulation_params must be a dictionary")

    # The minimums should be negative
    for index, x in enumerate(simulation_params["x_min"]):
        if x > 0:
            msg = f"x_min for axis {index + 1} is not negative. Grid is assumed symmetric."
            return False, msg

    # The grid should be symmetric
    xmins = simulation_params["x_min"]
    xmaxs = simulation_params["x_max"]
    for index, x_min in enumerate(xmins):
        if abs(x_min) != abs(xmaxs[index]):
            msg = f"{index + 1} max and min are not symmetric. Grid is assumed symmetric."
            return False, msg

    if min(simulation_params["Grid_resolution"]) <= 0:
        msg = "The grid resolution must be greater than zero"
        return False, msg

    return True, ""

def _check_simulation_parameters(simulation_params):
    """
    Perform comprehensive validation of simulation parameters.
    
    This function runs multiple validation checks on the simulation parameters
    including grid, frequency, vortex, and re-imprinting checks.
    
    Parameters
    ----------
    simulation_params : dict
        Dictionary containing all simulation parameters.
    
    Returns
    -------
    ok : bool
        True if all checks pass, False otherwise.
    msg : str
        Concatenated error messages from all validation checks.
    """
    checks = [
        _perform_grid_checks,
        _perform_frequency_checks,
        _perform_vortex_checks,
        _perform_reimprint_checks,
    ]
    overall_msg = ""
    overall = True
    for check in checks:
        ok, msg = check(simulation_params)
        if msg:
            overall_msg += msg + "\n"
        overall = overall and ok
    return overall, overall_msg