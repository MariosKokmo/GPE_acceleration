"""Reading, writing, default-building and validating the simulation config.

This module knows the *shape* of ``configuration_file.json`` but nothing about
the solver. All numeric parsing is forgiving of scientific notation
(e.g. ``5e-7``) and treats an empty string as "unset / null".
"""
import json

# Vocabulary kept in sync with HOW_TO_CONFIG_FILE.md ---------------------------
COORDINATES = ["cartesian", "cylindrical"]
POTENTIAL_TYPES = ["harmonic", "constant", "ramp", "rampharmonic"]
MODEL_TYPES = ["BEC", "FiniteTempBEC", "ZNGBEC"]

# Per-simulation keys: each is a list with one entry per simulation.
SIM_KEYS = [
    "vortex_charge",
    "imprinting_charge",
    "vortex_position_x",
    "vortex_position_y",
    "initial_imprint_time",
    "imprint_position_x",
    "imprint_position_y",
    "imprint_every",
    "imprint_times",
    "max_imprints",
]

SIM_KEY_HELP = {
    "vortex_charge": "Initial charge(s) for this simulation, e.g. [1] or [1, -1].",
    "imprinting_charge": "Charges for re-imprints, e.g. [[1,1,1]] or [].",
    "vortex_position_x": "Cartesian: grid index. Cylindrical: radial r >= 0.",
    "vortex_position_y": "Cartesian: grid index. Cylindrical: angle phi in [0, 2pi].",
    "initial_imprint_time": "Snapshot index at which initial vortices are imprinted.",
    "imprint_position_x": "Positions for re-imprinted vortices (same convention as x).",
    "imprint_position_y": "Positions for re-imprinted vortices (same convention as y).",
    "imprint_every": "Re-imprint interval in snapshots. Use [] when imprint_times is set.",
    "imprint_times": "Exact snapshot indices for re-imprints. Overrides imprint_every.",
    "max_imprints": "Maximum number of re-imprints for this simulation.",
}

# Per-simulation dark-soliton keys: each is a list with one entry per simulation
# (mirroring the vortex keys), enabled globally by the dark_soliton flag.
SOLITON_SIM_KEYS = [
    "soliton_positions",
    "soliton_widths",
    "soliton_axes",
    "soliton_greyness",
    "soliton_imprint_time",
]

SOLITON_KEY_HELP = {
    "soliton_positions": "This sim's soliton centre positions (um), e.g. [0.0] or [-2.0, 2.0]. [] = none.",
    "soliton_widths": "Width (healing-length scale) per soliton, e.g. [1.0] or [1.0, 1.0].",
    "soliton_axes": "Axis each soliton is perpendicular to: 1 (x) or 3 (z), e.g. [3] or [3, 3].",
    "soliton_greyness": "Optional greyness 0..pi/2 (0 = black) per soliton, e.g. [0.0].",
    "soliton_imprint_time": "Snapshot index at which to imprint this sim's solitons, e.g. 5.",
}


def default_config():
    """A minimal but runnable Cartesian, single-simulation config."""
    return {
        "coordinates": "cartesian",
        "Grid_positive_limits": [60, 1.5, 60],
        "Grid_negative_limits": [-60, -1.5, -60],
        "Grid_resolution": [256, 16, 256],
        "Trapping_frequencies": [20, 300, 20],
        "Potential_type": "harmonic",
        "SwitchOff_time": 9999,
        "three-body-losses": 0,
        "Total_simulation_time": 15e-3,
        "dt": 5e-7,
        "snapshots": 15,
        "model_type": "BEC",
        "temperature": 0.0,
        "damping_coefficient": 0.03,
        "chemical_potential": None,
        "n_test_particles": 10000,
        "gamma_12": 0.1,
        "enable_c22": False,
        "vortex_excitation": 1,
        "repetitive": 0,
        "vortex_charge": [[1]],
        "imprinting_charge": [[]],
        "vortex_position_x": [[0]],
        "vortex_position_y": [[0]],
        "initial_imprint_time": [0],
        "imprint_position_x": [[]],
        "imprint_position_y": [[]],
        "imprint_every": [[]],
        "imprint_times": [[]],
        "max_imprints": [0],
    }


def default_app_config():
    return {
        "logfile": "log.txt",
        "configFile": "configuration_file.json",
        "write_velocity": False,
        "phase_imaging": False,
    }


# --- IO ----------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4)
        fh.write("\n")


# --- numeric parsing helpers -------------------------------------------------
def parse_float(text):
    text = (text or "").strip()
    if text == "":
        return None
    return float(text)


def parse_int(text):
    text = (text or "").strip()
    if text == "":
        return None
    return int(float(text))  # tolerate "16.0"


def num_to_str(value):
    if value is None:
        return ""
    return str(value)


def parse_float_list(text):
    """Parse a comma/space-separated string into a list of floats ([] if empty)."""
    text = (text or "").replace(",", " ").strip()
    if not text:
        return []
    return [float(tok) for tok in text.split()]


def parse_int_list(text):
    """Parse a comma/space-separated string into a list of ints ([] if empty)."""
    text = (text or "").replace(",", " ").strip()
    if not text:
        return []
    return [int(float(tok)) for tok in text.split()]


def list_to_str(values):
    """Render a list back into a compact comma-separated string."""
    if not values:
        return ""
    return ", ".join(str(v) for v in values)


# --- validation --------------------------------------------------------------
def validate(config):
    """Return (errors, warnings) — both lists of human-readable strings.

    Errors should block a run; warnings are advisory. This mirrors the spirit
    of the solver's own checks but stays lightweight and dependency-free.
    """
    errors = []
    warnings = []

    coords = config.get("coordinates", "cartesian")
    if coords not in COORDINATES:
        errors.append(f"coordinates must be one of {COORDINATES}, got {coords!r}.")

    # Grid keys per coordinate system.
    if coords == "cartesian":
        for key, n in (
            ("Grid_positive_limits", 3),
            ("Grid_negative_limits", 3),
            ("Grid_resolution", 3),
            ("Trapping_frequencies", 3),
        ):
            _check_vector(config, key, n, errors)
    else:  # cylindrical
        for key in ("r_max", "z_min", "z_max"):
            if config.get(key) is None:
                errors.append(f"Cylindrical config requires '{key}'.")
        _check_vector(config, "Grid_resolution", 3, errors)
        tf = config.get("Trapping_frequencies")
        if not isinstance(tf, list) or len(tf) not in (2, 3):
            errors.append("Cylindrical 'Trapping_frequencies' must have 2 (or 3) entries.")

    # Time / potential.
    if config.get("Potential_type") not in POTENTIAL_TYPES:
        warnings.append(
            f"Potential_type {config.get('Potential_type')!r} is not in the known list "
            f"{POTENTIAL_TYPES}."
        )
    total = config.get("Total_simulation_time")
    dt = config.get("dt")
    if total is None or total <= 0:
        errors.append("Total_simulation_time must be a positive number.")
    if dt is None or dt <= 0:
        errors.append("dt must be a positive number.")
    if total and dt and dt >= total:
        errors.append("dt must be smaller than Total_simulation_time.")
    snaps = config.get("snapshots")
    if not isinstance(snaps, int) or snaps <= 0:
        errors.append("snapshots must be a positive integer.")

    if config.get("model_type") not in MODEL_TYPES:
        warnings.append(
            f"model_type {config.get('model_type')!r} is not in {MODEL_TYPES}."
        )

    # Dark soliton settings (per simulation, mirroring the vortex lists).
    if config.get("dark_soliton"):
        n = count_simulations(config)
        pos = config.get("soliton_positions")
        wid = config.get("soliton_widths")
        ax = config.get("soliton_axes")
        for key in ("soliton_positions", "soliton_widths", "soliton_axes"):
            val = config.get(key)
            if not isinstance(val, list) or len(val) != n:
                errors.append(
                    f"'{key}' must have one entry per simulation ({n}) when "
                    "dark_soliton is enabled."
                )
        # Per-simulation consistency of each row.
        if isinstance(pos, list) and isinstance(wid, list) and isinstance(ax, list):
            for i in range(min(len(pos), len(wid), len(ax))):
                rp, rw, ra = pos[i], wid[i], ax[i]
                if isinstance(rp, list) and isinstance(rw, list) and isinstance(ra, list):
                    if len({len(rp), len(rw), len(ra)}) > 1:
                        errors.append(
                            f"Simulation {i + 1}: soliton positions/widths/axes "
                            "have different lengths."
                        )
                    if any(a not in (1, 3) for a in ra):
                        errors.append(
                            f"Simulation {i + 1}: soliton_axes entries must be 1 (x) or 3 (z)."
                        )
        grey = config.get("soliton_greyness")
        if isinstance(grey, list) and len(grey) not in (0, n):
            errors.append(
                f"soliton_greyness, when given, must have one entry per simulation ({n})."
            )
        if isinstance(pos, list) and not any(isinstance(p, list) and p for p in pos):
            warnings.append("dark_soliton is enabled but no simulation defines any solitons.")

    # Per-simulation list-length consistency.
    lengths = {}
    for key in SIM_KEYS:
        val = config.get(key)
        if isinstance(val, list):
            lengths[key] = len(val)
    if lengths:
        n_sims = max(lengths.values())
        mismatched = [k for k, v in lengths.items() if v != n_sims and v != 0]
        if mismatched:
            warnings.append(
                "Per-simulation lists have inconsistent lengths "
                f"(expected {n_sims}): {', '.join(mismatched)}."
            )

    return errors, warnings


def _check_vector(config, key, n, errors):
    val = config.get(key)
    if not isinstance(val, list) or len(val) != n:
        errors.append(f"'{key}' must be a list of {n} numbers.")
        return
    if any(x is None for x in val):
        errors.append(f"'{key}' has empty entries.")


def count_simulations(config):
    keys = SIM_KEYS + SOLITON_SIM_KEYS
    lengths = [len(config[k]) for k in keys if isinstance(config.get(k), list)]
    return max(lengths) if lengths else 0
