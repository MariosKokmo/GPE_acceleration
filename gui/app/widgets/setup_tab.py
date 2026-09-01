"""Setup tab: the single grid / potential / time / model configuration.

Per the project rule, one config file is one grid + potential + ground state.
Those global settings live here; the per-simulation table lives in its own tab.
"""
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app import config_io
from app.widgets.fields import NumberField, VectorField


class SetupTab(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)

        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        self._build_coordinate_box(root)
        self._build_grid_boxes(root)
        self._build_time_box(root)
        self._build_model_box(root)
        self._build_absorber_box(root)
        root.addStretch(1)

        self.setWidget(container)
        self._sync_coordinate_visibility()
        self._sync_model_visibility()

    # --- construction --------------------------------------------------------
    def _build_coordinate_box(self, root):
        box = QGroupBox("Coordinate system")
        form = QFormLayout(box)
        self.coordinates = QComboBox()
        self.coordinates.addItems(config_io.COORDINATES)
        self.coordinates.currentTextChanged.connect(self._sync_coordinate_visibility)
        form.addRow("coordinates", self.coordinates)
        hint = QLabel("Cartesian uses box limits + grid indices; cylindrical uses "
                      "r_max / z bounds + physical (r, phi).")
        hint.setProperty("hint", True)
        hint.setWordWrap(True)
        form.addRow(hint)
        root.addWidget(box)

    def _build_grid_boxes(self, root):
        # Cartesian
        self.cartesian_box = QGroupBox("Cartesian grid")
        cform = QFormLayout(self.cartesian_box)
        self.grid_positive = VectorField(["x", "y", "z"])
        self.grid_negative = VectorField(["x", "y", "z"])
        self.grid_resolution_cart = VectorField(["nx", "ny", "nz"], is_int=True)
        self.trapping_cart = VectorField(["fx", "fy", "fz"])
        cform.addRow("Grid_positive_limits (um)", self.grid_positive)
        cform.addRow("Grid_negative_limits (um)", self.grid_negative)
        cform.addRow("Grid_resolution", self.grid_resolution_cart)
        cform.addRow("Trapping_frequencies (Hz)", self.trapping_cart)
        root.addWidget(self.cartesian_box)

        # Cylindrical
        self.cylindrical_box = QGroupBox("Cylindrical grid")
        zform = QFormLayout(self.cylindrical_box)
        self.r_max = NumberField(placeholder="60.0")
        self.z_min = NumberField(placeholder="-10.0")
        self.z_max = NumberField(placeholder="10.0")
        self.grid_resolution_cyl = VectorField(["nr", "nphi", "nz"], is_int=True)
        self.trapping_cyl = VectorField(["fr", "fz"])
        zform.addRow("r_max (um)", self.r_max)
        zform.addRow("z_min (um)", self.z_min)
        zform.addRow("z_max (um)", self.z_max)
        zform.addRow("Grid_resolution", self.grid_resolution_cyl)
        zform.addRow("Trapping_frequencies (Hz)", self.trapping_cyl)
        root.addWidget(self.cylindrical_box)

    def _build_time_box(self, root):
        box = QGroupBox("Time & potential")
        form = QFormLayout(box)
        self.potential_type = QComboBox()
        self.potential_type.addItems(config_io.POTENTIAL_TYPES)
        self.total_time = NumberField(placeholder="15e-3")
        self.dt = NumberField(placeholder="5e-7")
        self.snapshots = NumberField(is_int=True, placeholder="15")
        self.switchoff_time = NumberField(placeholder="9999")
        self.three_body = NumberField(placeholder="0")
        form.addRow("Potential_type", self.potential_type)
        form.addRow("Total_simulation_time (s)", self.total_time)
        form.addRow("dt (s)", self.dt)
        form.addRow("snapshots", self.snapshots)
        form.addRow("SwitchOff_time", self.switchoff_time)
        form.addRow("three-body-losses", self.three_body)
        root.addWidget(box)

    def _build_model_box(self, root):
        self.model_box = QGroupBox("Physics model")
        self._model_form = QFormLayout(self.model_box)
        self.model_type = QComboBox()
        self.model_type.addItems(config_io.MODEL_TYPES)
        self.model_type.currentTextChanged.connect(self._sync_model_visibility)
        self._model_form.addRow("model_type", self.model_type)

        self.temperature = NumberField(placeholder="0.0")
        self.damping = NumberField(placeholder="0.03")
        self.chemical_potential = NumberField(placeholder="(blank = auto / null)")
        self.n_test_particles = NumberField(is_int=True, placeholder="10000")
        self.gamma_12 = NumberField(placeholder="0.1")
        self.enable_c22 = QComboBox()
        self.enable_c22.addItems(["false", "true"])

        self._model_form.addRow("temperature", self.temperature)
        self._model_form.addRow("damping_coefficient", self.damping)
        self._model_form.addRow("chemical_potential", self.chemical_potential)
        self._model_form.addRow("n_test_particles", self.n_test_particles)
        self._model_form.addRow("gamma_12", self.gamma_12)
        self._model_form.addRow("enable_c22", self.enable_c22)

        # widget -> set of model_types for which the row is relevant
        self._model_rows = {
            self.temperature: {"FiniteTempBEC", "ZNGBEC"},
            self.damping: {"FiniteTempBEC"},
            self.chemical_potential: {"FiniteTempBEC"},
            self.n_test_particles: {"ZNGBEC"},
            self.gamma_12: {"ZNGBEC"},
            self.enable_c22: {"ZNGBEC"},
        }
        root.addWidget(self.model_box)

    def _build_absorber_box(self, root):
        self.absorber_box = QGroupBox("Boundary absorber")
        self.absorber_box.setCheckable(True)
        self.absorber_box.setChecked(False)
        form = QFormLayout(self.absorber_box)
        self.absorber_strength = NumberField(placeholder="e.g. 1.0")
        self.absorber_start_ratio = NumberField(placeholder="0.8")
        self.absorber_power = NumberField(placeholder="2")
        self.absorber_tinit = NumberField(placeholder="0")
        self.absorber_tfinal = NumberField(placeholder="(blank = instant)")
        form.addRow("Absorber_strength", self.absorber_strength)
        form.addRow("Absorber_start_ratio", self.absorber_start_ratio)
        form.addRow("Absorber_power", self.absorber_power)
        form.addRow("Absorber_tinit", self.absorber_tinit)
        form.addRow("Absorber_tfinal", self.absorber_tfinal)
        root.addWidget(self.absorber_box)

    # --- visibility toggles --------------------------------------------------
    def _sync_coordinate_visibility(self, *_):
        is_cart = self.coordinates.currentText() == "cartesian"
        self.cartesian_box.setVisible(is_cart)
        self.cylindrical_box.setVisible(not is_cart)

    def _sync_model_visibility(self, *_):
        model = self.model_type.currentText()
        for widget, models in self._model_rows.items():
            self._model_form.setRowVisible(widget, model in models)

    # --- load / dump ---------------------------------------------------------
    def load(self, config):
        coords = config.get("coordinates", "cartesian")
        self.coordinates.setCurrentText(coords if coords in config_io.COORDINATES else "cartesian")

        self.grid_positive.set_value(config.get("Grid_positive_limits", [None, None, None]))
        self.grid_negative.set_value(config.get("Grid_negative_limits", [None, None, None]))
        res = config.get("Grid_resolution", [None, None, None])
        self.grid_resolution_cart.set_value(res)
        self.grid_resolution_cyl.set_value(res)
        tf = config.get("Trapping_frequencies", [])
        self.trapping_cart.set_value(tf if len(tf) == 3 else [None, None, None])
        self.trapping_cyl.set_value(tf[:2] if len(tf) >= 2 else [None, None])

        self.r_max.set_value(config.get("r_max"))
        self.z_min.set_value(config.get("z_min"))
        self.z_max.set_value(config.get("z_max"))

        pot = config.get("Potential_type", "harmonic")
        if pot in config_io.POTENTIAL_TYPES:
            self.potential_type.setCurrentText(pot)
        self.total_time.set_value(config.get("Total_simulation_time"))
        self.dt.set_value(config.get("dt"))
        self.snapshots.set_value(config.get("snapshots"))
        self.switchoff_time.set_value(config.get("SwitchOff_time"))
        self.three_body.set_value(config.get("three-body-losses"))

        model = config.get("model_type", "BEC")
        if model in config_io.MODEL_TYPES:
            self.model_type.setCurrentText(model)
        self.temperature.set_value(config.get("temperature"))
        self.damping.set_value(config.get("damping_coefficient"))
        self.chemical_potential.set_value(config.get("chemical_potential"))
        self.n_test_particles.set_value(config.get("n_test_particles"))
        self.gamma_12.set_value(config.get("gamma_12"))
        self.enable_c22.setCurrentText("true" if config.get("enable_c22") else "false")

        absorber_on = bool(config.get("Absorber_enabled", False))
        self.absorber_box.setChecked(absorber_on)
        self.absorber_strength.set_value(config.get("Absorber_strength"))
        self.absorber_start_ratio.set_value(config.get("Absorber_start_ratio"))
        self.absorber_power.set_value(config.get("Absorber_power"))
        self.absorber_tinit.set_value(config.get("Absorber_tinit"))
        self.absorber_tfinal.set_value(config.get("Absorber_tfinal"))

    def dump(self, config):
        """Write this tab's fields into ``config`` (mutates and returns it)."""
        coords = self.coordinates.currentText()
        config["coordinates"] = coords

        if coords == "cartesian":
            config["Grid_positive_limits"] = self.grid_positive.value()
            config["Grid_negative_limits"] = self.grid_negative.value()
            config["Grid_resolution"] = self.grid_resolution_cart.value()
            config["Trapping_frequencies"] = self.trapping_cart.value()
            for key in ("r_max", "z_min", "z_max"):
                config.pop(key, None)
        else:
            config["r_max"] = self.r_max.value()
            config["z_min"] = self.z_min.value()
            config["z_max"] = self.z_max.value()
            config["Grid_resolution"] = self.grid_resolution_cyl.value()
            config["Trapping_frequencies"] = self.trapping_cyl.value()
            for key in ("Grid_positive_limits", "Grid_negative_limits"):
                config.pop(key, None)

        config["Potential_type"] = self.potential_type.currentText()
        config["Total_simulation_time"] = self.total_time.value()
        config["dt"] = self.dt.value()
        config["snapshots"] = self.snapshots.value()
        config["SwitchOff_time"] = self.switchoff_time.value()
        config["three-body-losses"] = self.three_body.value()

        config["model_type"] = self.model_type.currentText()
        config["temperature"] = self.temperature.value()
        config["damping_coefficient"] = self.damping.value()
        config["chemical_potential"] = self.chemical_potential.value()
        config["n_test_particles"] = self.n_test_particles.value()
        config["gamma_12"] = self.gamma_12.value()
        config["enable_c22"] = self.enable_c22.currentText() == "true"

        if self.absorber_box.isChecked():
            config["Absorber_enabled"] = True
            config["Absorber_strength"] = self.absorber_strength.value()
            config["Absorber_start_ratio"] = self.absorber_start_ratio.value()
            config["Absorber_power"] = self.absorber_power.value()
            config["Absorber_tinit"] = self.absorber_tinit.value()
            tfinal = self.absorber_tfinal.value()
            if tfinal is None:
                config.pop("Absorber_tfinal", None)
            else:
                config["Absorber_tfinal"] = tfinal
        else:
            config["Absorber_enabled"] = False
            for key in ("Absorber_strength", "Absorber_start_ratio", "Absorber_power",
                        "Absorber_tinit", "Absorber_tfinal"):
                config.pop(key, None)

        return config
