"""
Utilities for setting up and configuring GPE simulations.

This module provides functions for reading configuration files, setting up simulation
parameters, validating inputs, and creating simulation directories.
"""

import json
import math
import os

import numpy as np

from src.library.parameters import CONSTANTS


# =============================================================================
# Configuration File Reading
# =============================================================================

def _read_configuration_file(config_file):
    """
    Read the simulation configuration file.
    
    Parameters
    ----------
    config_file : str
        Path to the configuration file relative to the current working directory.
    
    Returns
    -------
    dict
        Dictionary containing the simulation configuration parameters.
    
    Raises
    ------
    ValueError
        If the configuration file cannot be parsed as valid JSON.
    FileNotFoundError
        If the configuration file does not exist.
    """
    cwd = os.getcwd()  
    print(cwd)
    path_config_file = os.path.join(cwd, config_file)
    
    try:
        with open(path_config_file, 'r') as f:
            simulations = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse configuration file '{config_file}': {e}")   
    
    return simulations


def get_application_config(config_file="appConfig.json"):
    """
    Read the application configuration file.
    
    Parameters
    ----------
    config_file : str, optional
        Path to the application configuration file, by default "appConfig.json".
    
    Returns
    -------
    dict
        Dictionary containing the application configuration parameters.
    
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    json.JSONDecodeError
        If the configuration file is not valid JSON.
    """
    cwd = os.getcwd()
    print(cwd)
    path_config_file = os.path.join(cwd, config_file)
    
    with open(path_config_file, 'r') as f:
        app_configs = json.load(f)
    
    return app_configs



# =============================================================================
# Simulation Setup
# =============================================================================

def get_simulation_combinations(sims):
    """
    Create distinct simulations to be run based on configuration parameters.
    
    Generates simulation folders with appropriate names and assigns parameters
    to each simulation based on whether repetitive imprinting is enabled.
    
    Parameters
    ----------
    sims : dict
        Dictionary containing all simulation parameters including:
        - imprint_every : list of int
        - max_imprints : list of int
        - vortex_charge : list
        - imprinting_charge : list
        - repetitive : int (0 or 1)
        - vortex_position_x : list
        - vortex_position_y : list
        - vortex_excitation : various
        - initial_imprint_time : list
        - imprint_position_x : list
        - imprint_position_y : list
        - imprint_times : list of lists
    
    Returns
    -------
    list of list
        Each inner list contains [folder_name, parameters_dict] where:
        - folder_name : str, name of the simulation folder
        - parameters_dict : dict, simulation parameters
    
    Raises
    ------
    AssertionError
        If parameter list lengths are inconsistent or repetitive flag is invalid.
    """
    imprint_every = sims["imprint_every"]
    max_imprints = sims["max_imprints"]
    charges = sims["vortex_charge"]
    imprinting_charge = sims["imprinting_charge"]
    repetitive = sims["repetitive"]
    vortex_position_x = sims["vortex_position_x"]
    vortex_position_y = sims["vortex_position_y"]
    vortex_excitation = sims["vortex_excitation"]
    initial_imprint_time = sims["initial_imprint_time"]
    imprint_position_x = sims["imprint_position_x"]
    imprint_position_y = sims["imprint_position_y"]
    imprint_times = sims["imprint_times"]
    
    assert repetitive == 0 or repetitive == 1, "repetitive must be 0 or 1"
    
    if repetitive:
        max_imprints = sims['max_imprints']
        assert len(max_imprints) >= 1, "max_imprints is not correct in configuration file"
        assert len(charges) == len(imprint_every), "charges and imprint_every have different number of values"
        assert len(charges) == len(max_imprints), "charges and max_imprints have different number of values"
        
        parameters_repetitive = []
        for i in range(len(charges)):
            parameters_repetitive.append({
                "vortex_charge": charges[i],
                "vortex_position_x": vortex_position_x[i],
                "vortex_position_y": vortex_position_y[i],
                "max_imprints": max_imprints[i],
                "imprint_every": imprint_every[i],
                "repetitive": repetitive,
                "imprinting_charge": imprinting_charge[i],
                "imprint_position_x": imprint_position_x[i],
                "imprint_position_y": imprint_position_y[i],
                "imprint_times": imprint_times[i],
                "initial_imprint_time": initial_imprint_time[i],
                "vortex_excitation": vortex_excitation
            })
        simulations = _simulations_repetitive(parameters_repetitive)
    else:
        parameters_multi_vortex = []
        for i in range(len(charges)):
            parameters_multi_vortex.append({
                "vortex_charge": charges[i],
                "vortex_position_x": vortex_position_x[i],
                "vortex_position_y": vortex_position_y[i],
                "max_imprints": 0,
                "imprint_every": 0,
                "repetitive": repetitive,
                "initial_imprint_time": initial_imprint_time[i],
                "imprinting_charge": [],
                "imprint_position_x": [],
                "imprint_position_y": [],
                "vortex_excitation": vortex_excitation,
                "imprint_times": []
            })
        simulations = _simulations_multi_vortex(parameters_multi_vortex)
    
    return simulations


def _simulations_repetitive(parameters_list):
    """
    Create simulation folders for repetitive imprinting scenarios.
    
    Parameters
    ----------
    parameters_list : list of dict
        List where each dictionary contains parameters for one simulation,
        including vortex_charge, imprinting_charge, max_imprints, and imprint_times.
    
    Returns
    -------
    list of list
        Each inner list contains [simulation_name, parameters] where:
        - simulation_name : str, name of the simulation folder
        - parameters : dict, simulation parameters
    """
    simulations = []
    
    for parameters in parameters_list:
        charges = parameters["vortex_charge"]
        imprinting_charge = parameters["imprinting_charge"]
        max_imprints = parameters["max_imprints"]
        imprint_every = parameters["imprint_every"]
        
        if isinstance(charges, list):
            number_charges = len(charges)
        else:
            number_charges = 1
        
        if isinstance(charges, list) and len(charges) > 10:
            charges_str = f"_init_{charges[0]}_last_{charges[-1]}"
        elif isinstance(charges, list):
            charges_str = "_".join([str(charge) for charge in charges])
        else:
            charges_str = str(charges)

        if isinstance(imprinting_charge, list) and len(imprinting_charge) > 10:
            imprinting_charge_str = f"_init_{imprinting_charge[0]}_last_{imprinting_charge[-1]}"
        elif isinstance(imprinting_charge, list):
            imprinting_charge_str = "_".join([str(charge) for charge in imprinting_charge])
        else:
            imprinting_charge_str = str(imprinting_charge)

        imprint_times_list = parameters["imprint_times"]
        if isinstance(imprint_times_list, list) and len(imprint_times_list) > 10:
            imprint_times = f"_init_{imprint_times_list[0]}_last_{imprint_times_list[-1]}"
        else:
            imprint_times = "_".join([str(time) for time in imprint_times_list])

        simulation_name = (
            f'{number_charges}vort__initCharge{charges_str}__'
            f'imprintCharge{imprinting_charge_str}__snapshots{imprint_times}'
        )
        simulations.append([simulation_name, parameters])
        
        print("creating folder: ", simulation_name)
        if not os.path.isdir(simulation_name):
            os.mkdir(simulation_name)
        else:
            print(f"{simulation_name} folder already exists...data will be overwritten")
    
    return simulations


def _simulations_multi_vortex(parameters_list):
    """
    Create simulation folders for multi-vortex scenarios.
    
    Parameters
    ----------
    parameters_list : list of dict
        List where each dictionary contains parameters for one simulation,
        including vortex_charge and vortex positions (x, y).
    
    Returns
    -------
    list of list
        Each inner list contains [simulation_name, parameters] where:
        - simulation_name : str, name of the simulation folder
        - parameters : dict, simulation parameters
    """
    simulations = []
    
    for parameters in parameters_list:
        charges = parameters["vortex_charge"]
        all_charges = "_".join([str(c) for c in charges if c != " "])
        
        vortex_position_x = parameters["vortex_position_x"]
        vortex_position_x_str = "_".join([str(x) for x in vortex_position_x if x != " "])
        
        vortex_position_y = parameters["vortex_position_y"]
        vortex_position_y_str = "_".join([str(y) for y in vortex_position_y if y != " "])
        
        simulation_name = (
            f"{len(charges)}vortex_charges{all_charges}__"
            f"x-{vortex_position_x_str}__y-{vortex_position_y_str}"
        )
        simulations.append([simulation_name, parameters])
        
        print("creating folder: ", simulation_name)
        if not os.path.isdir(simulation_name):
            os.mkdir(simulation_name)
        else:
            print(f"{simulation_name} folder already exists...data will be overwritten")
    
    return simulations



# =============================================================================
# Parameter Processing
# =============================================================================

def get_simulation_parameters(config_file_path):
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
    pi = CONSTANTS.pi
    
    # Read the configuration file
    sim_params = _read_configuration_file(config_file_path)
    
    # Grid parameters
    n1, n2, n3 = sim_params["Grid_resolution"]
    dim = np.array([n1, n2, n3], dtype=np.float64)
    x_min = np.array(sim_params["Grid_negative_limits"])
    x_max = np.array(sim_params["Grid_positive_limits"])
    
    # Trapping frequencies
    fx, fy, fz = sim_params["Trapping_frequencies"]
    wx = 2 * pi * float(fx)
    wy = 2 * pi * float(fy)
    wz = 2 * pi * float(fz)
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
    u = 4.0 * CONSTANTS.pi * CONSTANTS.nat * CONSTANTS.ascat / a_ho 
    
    # Normalizations
    w = w / omega_ho
    x_max = x_max * 1e-6 / a_ho
    x_min = x_min * 1e-6 / a_ho

    # Grid spacing
    dx = (x_max - x_min) / dim
    dp = 2 * pi / (x_max - x_min)
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
        "repetitive": repetitive
    }
    
    # Validate simulation parameters
    ok, msg = _check_simulation_parameters(simulation_params)  
    if not ok:
        print(msg)
    
    return simulation_params, msg


def save_parameters_to_json(parameters, filepath="simulation_parameters.json"):
    """
    Save simulation parameters to a JSON file.
    
    This function is expected to be called for each simulation, saving a JSON
    file in each simulation folder.
    
    Parameters
    ----------
    parameters : dict
        Dictionary containing simulation parameters to save.
    filepath : str, optional
        Path to the output JSON file, by default "simulation_parameters.json".
    """
    def convert(x):
        """Convert numpy arrays to lists for JSON serialization."""
        if hasattr(x, "tolist"):  # numpy arrays have this
            return x.tolist()
        raise TypeError(x)
    
    with open(filepath, "w") as fp:
        json.dump(parameters, fp, indent=4, default=convert)



# =============================================================================
# Parameter Validation
# =============================================================================

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
    overall_msg = ""
    overall = True

    ok, msg = _perform_grid_checks(simulation_params)
    overall_msg += msg + "\n"
    overall = overall and ok

    ok, msg = _perform_frequency_checks(simulation_params)
    overall_msg += msg + "\n"
    overall = overall and ok

    ok, msg = _perform_vortex_checks(simulation_params)
    overall_msg += msg + "\n"
    overall = overall and ok

    ok, msg = _perform_reimprint_checks(simulation_params)
    overall_msg += msg + "\n"
    overall = overall and ok

    return overall, overall_msg


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


def _perform_frequency_checks(simulation_params):
    """
    Validate trapping frequency parameters.
    
    Checks that all trapping frequencies are positive.
    
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
    for index, freq in enumerate(simulation_params["Trapping_frequencies"]):
        if freq <= 0:
            msg = f"Frequency {index + 1} is negative or zero. Frequencies are assumed positives."
            return False, msg
    
    return True, ""



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


def _perform_reimprint_checks(simulation_params):
    """
    Validate re-imprinting parameters.
    
    Checks that:
    - Number of imprint times matches number of imprinting charges
    - Number of initial charges matches imprinting charges
    - Imprinting charge positions are specified correctly
    - Maximum imprint time is within simulation bounds
    
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
    sim_time = simulation_params["Total_simulation_time"]  # time in sec
    snapshots = simulation_params["shots"]
    
    if simulation_params["repetitive"] and (len(simulation_params["imprint_times"]) != len(simulation_params["imprinting_charge"])):
        msg = "One list of imprinting times should be given for every simulation"
        return False, msg
    
    if simulation_params["repetitive"]:
        if len(simulation_params["vortex_charge"]) != len(simulation_params["imprinting_charge"]):
            msg = (f"The list number of initial charges {len(simulation_params['vortex_charge'])}, "
                   f"doesn't agree with the list number of imprinted {len(simulation_params['imprinting_charge'])}")
            return False, msg
        
        for index, charges in enumerate(simulation_params["imprinting_charge"]):
            if isinstance(charges, list):
                if len(charges) != len(simulation_params["imprint_position_x"][index]):
                    msg = f"The number of imprinting charges doesn't agree with the number of x positions at index {index}"
                    return False, msg
                
                if len(charges) != len(simulation_params["imprint_position_y"][index]):
                    msg = f"The number of imprinting charges doesn't agree with the number of y positions at index {index}"
                    return False, msg
        
        for index, times in enumerate(simulation_params["imprint_times"]):
            # Check that the maximum imprint time is less than simulation time
            # Imprint times are given in snapshots
            if max(times) > snapshots:
                msg = f"The maximum imprint time is greater than the total simulation time for simulation {index + 1}\n"
                return False, msg
    
    return True, ""