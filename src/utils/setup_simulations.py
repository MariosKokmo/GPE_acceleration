r"""
Utilities for setting up and configuring GPE simulations.

This module reads the configuration files, expands them into the individual
simulations of a run, validates the inputs and creates the simulation
directories. A configuration may specify several vortex charges, soliton
positions and so on; each entry becomes its own simulation, with its own folder
and its own parameter dictionary.
"""

import json
import os

#: Keys every simulation configuration must define.
REQUIRED_SIMULATION_CONFIG_KEYS = [
    "Grid_resolution",
    "Grid_negative_limits",
    "Grid_positive_limits",
    "Trapping_frequencies",
    "Total_simulation_time",
    "dt",
    "snapshots",
    "Potential_type",
    "SwitchOff_time",
]

#: Keys required by the vortex excitation settings, when ``vortex_excitation``
#: is true.
REQUIRED_VORTEX_KEYS = [
    "vortex_excitation",
    "vortex_charge",
    "vortex_position_x",
    "vortex_position_y",
    "initial_imprint_time",
]

#: Keys required by repetitive imprinting scenarios, when ``repetitive`` is 1.
REQUIRED_REPETITIVE_IMPRINTING_KEYS = [
    "imprint_every",
    "max_imprints",
    "vortex_charge",
    "imprinting_charge",
    "repetitive",
    "vortex_position_x",
    "vortex_position_y",
    "vortex_excitation",
    "initial_imprint_time",
    "imprint_position_x",
    "imprint_position_y",
    "imprint_times",
]


def _require_keys(data, required_keys, context):
    r"""
    Check that every required key is present.

    Args:
        data (dict): Mapping to check.
        required_keys (list[str]): Keys that must be present.
        context (str): Short description of what is being validated, used in
            the error message.

    Raises:
        ValueError: If one or more required keys are missing.
    """
    missing = [key for key in required_keys if key not in data]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required parameter(s) in {context}: {missing_str}")


# =============================================================================
# Configuration File Reading
# =============================================================================

def _load_json_from_cwd(config_file):
    r"""
    Load and parse a JSON file located relative to the current working
    directory.

    Args:
        config_file (str): File name or relative path of the JSON file.

    Returns:
        dict: The parsed contents of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file, or the contents are not valid
            JSON.
        OSError: If the file exists but cannot be read.
    """
    path_config_file = os.path.join(os.getcwd(), config_file)

    if not os.path.exists(path_config_file):
        raise FileNotFoundError(f"Configuration file not found: '{path_config_file}'")
    if not os.path.isfile(path_config_file):
        raise ValueError(f"Configuration path is not a file: '{path_config_file}'")

    try:
        with open(path_config_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse configuration file '{config_file}': {e}") from e
    except OSError as e:
        raise OSError(f"Unable to read configuration file '{path_config_file}': {e}") from e


def get_application_config(config_file="appConfig.json"):
    r"""
    Read the application configuration file.

    Args:
        config_file (str): Path to the application configuration file, by
            default ``"appConfig.json"``.

    Returns:
        dict: The application configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is not valid JSON.
    """
    return _load_json_from_cwd(config_file)



# =============================================================================
# Simulation Setup
# =============================================================================

def _format_name_component(values):
    r"""
    Format a scalar or a list into a compact simulation-name component.

    Lists are joined with underscores, ignoring blank entries. A list of more
    than ten entries is abbreviated to its first and last values, so that the
    folder name stays a usable length.

    Args:
        values: Scalar or list of values to format.

    Returns:
        str: The formatted name component.
    """
    if isinstance(values, list):
        cleaned_values = [value for value in values if value != " "]
        if len(cleaned_values) > 10:
            return f"_init_{cleaned_values[0]}_last_{cleaned_values[-1]}"
        return "_".join(str(value) for value in cleaned_values)
    return str(values)


def _ensure_simulation_directory(simulation_name):
    r"""
    Create a simulation directory when it is missing, and report its status.

    An existing directory is left alone, with a printed warning that its data
    will be overwritten.

    Args:
        simulation_name (str): Name of the simulation directory.
    """
    if not os.path.isdir(simulation_name):
        print("creating folder: ", simulation_name)
        os.mkdir(simulation_name)
    else:
        print(f"{simulation_name} folder already exists...data will be overwritten")

def _validate_soliton_lengths(sims, n_sims):
    r"""
    Ensure the per-simulation dark-soliton lists have one entry per simulation.

    Dark solitons are configured per simulation, mirroring the vortex lists:
    ``soliton_positions``, ``soliton_widths`` and ``soliton_axes`` are required
    and must have exactly ``n_sims`` entries, while ``soliton_greyness`` and
    ``soliton_imprint_time`` are optional but, when given as lists, must match
    too.

    Args:
        sims (dict): Full configuration dictionary.
        n_sims (int): Expected number of simulations.

    Raises:
        ValueError: If a required list is missing, is not a list, or has the
            wrong length; or if an optional list is given with the wrong
            length.
    """
    for key in ("soliton_positions", "soliton_widths", "soliton_axes"):
        value = sims.get(key)
        if not isinstance(value, list) or len(value) != n_sims:
            raise ValueError(
                f"'{key}' must be a list with one entry per simulation "
                f"({n_sims}); dark solitons are configured per simulation."
            )
    for key in ("soliton_greyness", "soliton_imprint_time"):
        value = sims.get(key)
        if isinstance(value, list) and len(value) != n_sims:
            raise ValueError(
                f"'{key}', when given, must have one entry per simulation ({n_sims})."
            )


def _dark_soliton_params_for_sim(sims, index):
    r"""
    Slice out the dark-soliton parameters of one simulation.

    Args:
        sims (dict): Full configuration dictionary.
        index (int): Index of the simulation to slice.

    Returns:
        dict: The dark-soliton parameters for this simulation, or ``{}`` when
        dark solitons are disabled or this simulation has none, i.e. its
        position list is empty.
    """
    if not sims.get("dark_soliton", False):
        return {}

    def _slice(key, default):
        value = sims.get(key)
        if not isinstance(value, list) or index >= len(value):
            return default
        return value[index]

    positions = _slice("soliton_positions", [])
    if not positions:
        return {}  # this simulation imprints no solitons

    return {
        "dark_soliton": True,
        "soliton_positions": positions,
        "soliton_widths": _slice("soliton_widths", []),
        "soliton_axes": _slice("soliton_axes", []),
        "soliton_greyness": _slice("soliton_greyness", None),
        "soliton_imprint_time": _slice("soliton_imprint_time", 0),
    }


def get_simulation_combinations(sims):
    r"""
    Create the distinct simulations to be run from the configuration
    parameters.

    Simulation folders are generated with names describing their contents, and
    each simulation receives its own parameter dictionary. Which branch is
    taken depends on the configuration: vortex excitation with repetitive
    imprinting, vortex excitation without it, or a dark-soliton-only run. The
    finite-temperature settings are shared across all combinations, whereas the
    vortex and soliton settings are sliced per simulation.

    Args:
        sims (dict): All simulation parameters. The keys consulted here are
            ``imprint_every`` (list[int]), ``max_imprints`` (list[int]),
            ``vortex_charge`` (list), ``imprinting_charge`` (list),
            ``repetitive`` (0 or 1), ``vortex_position_x`` and
            ``vortex_position_y`` (lists), ``vortex_excitation``,
            ``initial_imprint_time`` (list), ``imprint_position_x`` and
            ``imprint_position_y`` (lists), ``imprint_times``
            (list of lists), and the dark-soliton and finite-temperature keys.

    Returns:
        list[list]: One entry per simulation, ``[folder_name, parameters]``,
        where ``folder_name`` is the name of the simulation folder and
        ``parameters`` its parameter dictionary.

    Raises:
        ValueError: If a required key is missing, if ``repetitive`` is neither
            0 nor 1, or if the parameter list lengths are inconsistent.
    """

    # Finite-temperature parameters — shared across all simulation combinations.
    # Passed through so that BaseBEC subclasses can read them from self.parameters.
    finite_temp_params = {
        "model_type": sims.get("model_type", "BEC"),
        "temperature": sims.get("temperature", 0.0),
        "damping_coefficient": sims.get("damping_coefficient", 0.03),
        "n_test_particles": sims.get("n_test_particles", 10_000),
        "gamma_12": sims.get("gamma_12", 0.1),
        "chemical_potential": sims.get("chemical_potential", None),
        "enable_c22": sims.get("enable_c22", False),
    }

    # Dark solitons are configured per simulation (mirroring the vortex lists),
    # so they are sliced per-simulation inside the loops below via
    # _dark_soliton_params_for_sim rather than shared globally.
    dark_soliton_enabled = bool(sims.get("dark_soliton", False))

    simulations = []
    vortex_excitation = sims.get("vortex_excitation", False)
    if vortex_excitation:
        _require_keys(
            sims,
            REQUIRED_VORTEX_KEYS,
            "vortex excitation settings",
        )
        vortex_position_x = sims["vortex_position_x"]
        vortex_position_y = sims["vortex_position_y"]
        initial_imprint_time = sims["initial_imprint_time"]
        charges = sims["vortex_charge"]

        # When solitons are also enabled they must be given for each simulation.
        if dark_soliton_enabled:
            _require_keys(sims, ["soliton_positions", "soliton_widths", "soliton_axes"],
                          "dark soliton settings")
            _validate_soliton_lengths(sims, len(charges))

        repetitive = sims.get("repetitive", None)
        if repetitive is not None:
            _require_keys(sims, REQUIRED_REPETITIVE_IMPRINTING_KEYS, "repetitive imprinting settings")
            imprint_every = sims["imprint_every"]
            max_imprints = sims["max_imprints"]
            imprinting_charge = sims["imprinting_charge"]
            imprint_position_x = sims["imprint_position_x"]
            imprint_position_y = sims["imprint_position_y"]
            imprint_times = sims["imprint_times"]

        if repetitive is not None and repetitive not in (0, 1):
            raise ValueError("repetitive must be 0 or 1")

        if repetitive:
            if len(max_imprints) < 1:
                raise ValueError("max_imprints is not correct in configuration file")
            if len(charges) != len(imprint_every):
                raise ValueError("charges and imprint_every have different number of values")
            if len(charges) != len(max_imprints):
                raise ValueError("charges and max_imprints have different number of values")

            parameters_repetitive = []
            for i in range(len(charges)):
                sim_params_i = {
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
                }
                sim_params_i.update(_dark_soliton_params_for_sim(sims, i))
                sim_params_i.update(finite_temp_params)
                parameters_repetitive.append(sim_params_i)
            simulations = _simulations_repetitive(parameters_repetitive)
        else:
            parameters_multi_vortex = []
            for i in range(len(charges)):
                sim_params_i = {
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
                }
                sim_params_i.update(_dark_soliton_params_for_sim(sims, i))
                sim_params_i.update(finite_temp_params)
                parameters_multi_vortex.append(sim_params_i)
            simulations = _simulations_multi_vortex(parameters_multi_vortex)

    elif dark_soliton_enabled:
        # Dark-soliton-only run (no vortices). One simulation per entry in the
        # per-simulation soliton lists.
        _require_keys(sims, ["soliton_positions", "soliton_widths", "soliton_axes"],
                      "dark soliton settings")
        n_sims = len(sims["soliton_positions"])
        _validate_soliton_lengths(sims, n_sims)
        soliton_sims = []
        for i in range(n_sims):
            sim_params_i = {"vortex_excitation": 0, "repetitive": 0}
            sim_params_i.update(finite_temp_params)
            sim_params_i.update(_dark_soliton_params_for_sim(sims, i))
            soliton_sims.append(sim_params_i)
        simulations = _simulations_dark_soliton(soliton_sims)

    return simulations


def _simulations_repetitive(parameters_list):
    r"""
    Create the simulation folders for repetitive imprinting scenarios.

    The folder name records the number of vortices, the initial charges, the
    imprinting charges and the imprint snapshots.

    Args:
        parameters_list (list[dict]): One dictionary per simulation,
            containing at least ``vortex_charge``, ``imprinting_charge``,
            ``max_imprints`` and ``imprint_times``.

    Returns:
        list[list]: One entry per simulation, ``[simulation_name,
        parameters]``.
    """
    simulations = []

    for parameters in parameters_list:
        charges = parameters["vortex_charge"]
        imprinting_charge = parameters["imprinting_charge"]

        if isinstance(charges, list):
            number_charges = len(charges)
        else:
            number_charges = 1

        charges_str = _format_name_component(charges)
        imprinting_charge_str = _format_name_component(imprinting_charge)
        imprint_times = _format_name_component(parameters["imprint_times"])

        simulation_name = (
            f'{number_charges}vort__initCharge{charges_str}__'
            f'imprintCharge{imprinting_charge_str}__snapshots{imprint_times}'
        )
        simulations.append([simulation_name, parameters])

        _ensure_simulation_directory(simulation_name)

    return simulations


def _simulations_multi_vortex(parameters_list):
    r"""
    Create the simulation folders for multi-vortex scenarios.

    The folder name records the number of vortices, their charges and their
    :math:`x` and :math:`y` positions.

    Args:
        parameters_list (list[dict]): One dictionary per simulation,
            containing at least ``vortex_charge``, ``vortex_position_x`` and
            ``vortex_position_y``.

    Returns:
        list[list]: One entry per simulation, ``[simulation_name,
        parameters]``.
    """
    simulations = []

    for parameters in parameters_list:
        charges = parameters["vortex_charge"]
        all_charges = _format_name_component(charges)

        vortex_position_x = parameters["vortex_position_x"]
        vortex_position_x_str = _format_name_component(vortex_position_x)

        vortex_position_y = parameters["vortex_position_y"]
        vortex_position_y_str = _format_name_component(vortex_position_y)

        simulation_name = (
            f"{len(charges)}vortex_charges{all_charges}__"
            f"x-{vortex_position_x_str}__y-{vortex_position_y_str}"
        )
        simulations.append([simulation_name, parameters])

        _ensure_simulation_directory(simulation_name)

    return simulations


def _simulations_dark_soliton(parameters_list):
    r"""
    Build the dark-soliton-only simulations, one per entry.

    These runs have no vortex excitation; the folder name records the soliton
    positions, their axes and the imprint snapshot.

    Args:
        parameters_list (list[dict]): One dictionary per simulation, each
            carrying that simulation's dark-soliton settings and the
            finite-temperature settings, with ``vortex_excitation`` set to 0.

    Returns:
        list[list]: One entry per simulation, ``[simulation_name,
        parameters]``.
    """
    simulations = []
    for parameters in parameters_list:
        positions = _format_name_component(parameters.get("soliton_positions", []))
        axes = _format_name_component(parameters.get("soliton_axes", []))
        imprint = parameters.get("soliton_imprint_time", 0)
        simulation_name = f"darksoliton__pos{positions}__ax{axes}__snap{imprint}"
        simulations.append([simulation_name, parameters])
        _ensure_simulation_directory(simulation_name)
    return simulations


def save_parameters_to_json(parameters, filepath="simulation_parameters.json"):
    r"""
    Save the simulation parameters to a JSON file.

    This is expected to be called once per simulation, writing a JSON file in
    each simulation folder. NumPy arrays are converted to lists on the way out.

    Args:
        parameters (dict): Simulation parameters to save.
        filepath (str): Path of the output JSON file, by default
            ``"simulation_parameters.json"``.
    """
    def convert(x):
        r"""
        Convert NumPy arrays to lists for JSON serialisation.

        Args:
            x: Object the JSON encoder could not serialise.

        Returns:
            list: ``x`` as a list, when it exposes ``tolist``.

        Raises:
            TypeError: If ``x`` cannot be converted.
        """
        if hasattr(x, "tolist"):  # numpy arrays have this
            return x.tolist()
        raise TypeError(x)

    with open(filepath, "w") as fp:
        json.dump(parameters, fp, indent=4, default=convert)


# =============================================================================
# Parameter Validation
# =============================================================================

def _perform_reimprint_checks(simulation_params):
    r"""
    Validate the re-imprinting parameters.

    The checks are that the number of imprint times matches the number of
    imprinting charges, that the number of initial charges matches the
    imprinting charges, that the imprinting charge positions are specified
    correctly, and that the maximum imprint time lies within the simulation
    bounds. Imprint times are given in snapshots.

    Args:
        simulation_params (dict): Simulation parameters to validate.

    Returns:
        tuple: ``(ok, msg)`` — ``ok`` is ``True`` when validation passes, and
        ``msg`` describes the problem otherwise (an empty string on success).
    """
    snapshots = simulation_params["shots"]

    if simulation_params["repetitive"] and (len(simulation_params["imprint_times"]) != len(simulation_params["imprinting_charge"])):
        msg = "One list of imprinting times should be given for every simulation"
        return False, msg

    if simulation_params["repetitive"]:
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
            if times and max(times) > snapshots:
                msg = f"The maximum imprint time is greater than the total simulation time for simulation {index + 1}\n"
                return False, msg

    return True, ""

def _perform_frequency_checks(simulation_params):
    r"""
    Validate the trapping frequency parameters.

    All trapping frequencies are checked to be strictly positive.

    Args:
        simulation_params (dict): Simulation parameters to validate.

    Returns:
        tuple: ``(ok, msg)`` — ``ok`` is ``True`` when validation passes, and
        ``msg`` describes the problem otherwise (an empty string on success).
    """
    for index, freq in enumerate(simulation_params["Trapping_frequencies"]):
        if freq <= 0:
            msg = f"Frequency {index + 1} is negative or zero. Frequencies are assumed positives."
            return False, msg

    return True, ""
