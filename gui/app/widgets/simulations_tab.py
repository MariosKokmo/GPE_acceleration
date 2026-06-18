"""Simulations tab: one row per simulation.

Every per-simulation key is a list with one entry per simulation. Vortex keys
(vortex_charge, imprint_times, ...) and dark-soliton keys (soliton_positions,
soliton_axes, ...) are both edited as columns of this table, so each simulation
gets its own vortices *and* its own solitons. Cells hold compact JSON
expressions (e.g. ``[1]``, ``[[1,1,1]]``, ``[0.0]``, ``[3]``) and are validated
on edit. Two global flags (vortex_excitation, dark_soliton) enable each
excitation type for the whole run, mirroring the solver.
"""
import json

from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app import config_io
from app.theme import DANGER

_INVALID_BRUSH = QBrush(QColor(DANGER))

# Sensible per-cell defaults for a freshly added simulation row.
_ROW_DEFAULTS = {
    "vortex_charge": [1],
    "imprinting_charge": [],
    "vortex_position_x": [0],
    "vortex_position_y": [0],
    "initial_imprint_time": 0,
    "imprint_position_x": [],
    "imprint_position_y": [],
    "imprint_every": [],
    "imprint_times": [],
    "max_imprints": 0,
    # dark soliton (empty by default => this simulation imprints no soliton)
    "soliton_positions": [],
    "soliton_widths": [],
    "soliton_axes": [],
    "soliton_greyness": [],
    "soliton_imprint_time": 0,
}


class SimulationsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Vortex columns first, then dark-soliton columns.
        self.columns = list(config_io.SIM_KEYS) + list(config_io.SOLITON_SIM_KEYS)
        self._col_help = dict(config_io.SIM_KEY_HELP)
        self._col_help.update(config_io.SOLITON_KEY_HELP)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Global excitation flags (mirror the solver's global on/off switches).
        flags = QHBoxLayout()
        self.vortex_excitation = QCheckBox("vortex_excitation (imprint vortices)")
        self.repetitive = QCheckBox("repetitive (re-imprinting enabled)")
        self.dark_soliton = QCheckBox("dark_soliton (imprint solitons)")
        flags.addWidget(self.vortex_excitation)
        flags.addSpacing(20)
        flags.addWidget(self.repetitive)
        flags.addSpacing(20)
        flags.addWidget(self.dark_soliton)
        flags.addStretch(1)
        root.addLayout(flags)

        hint = QLabel(
            "Each row is one simulation (all share the grid/potential from the "
            "Setup tab). Cells accept JSON, e.g. [1], [1,-1], [[1,1,1]], [0.0], [3]. "
            "Vortex columns apply when vortex_excitation is on; soliton columns "
            "(soliton_*) apply when dark_soliton is on. Hover a header for help; "
            "invalid cells turn red."
        )
        hint.setProperty("hint", True)
        hint.setWordWrap(True)
        root.addWidget(hint)

        # Table.
        self.table = QTableWidget(0, len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.table.verticalHeader().setDefaultSectionSize(34)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        for col, key in enumerate(self.columns):
            self.table.horizontalHeaderItem(col).setToolTip(self._col_help.get(key, ""))
            self.table.setColumnWidth(col, 130)
        self.table.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.table, 1)

        # Row controls.
        controls = QHBoxLayout()
        add_btn = QPushButton("+ Add simulation")
        add_btn.setProperty("accent", True)
        dup_btn = QPushButton("Duplicate")
        del_btn = QPushButton("Remove selected")
        del_btn.setProperty("danger", True)
        add_btn.clicked.connect(self.add_row)
        dup_btn.clicked.connect(self.duplicate_selected)
        del_btn.clicked.connect(self.remove_selected)
        controls.addWidget(add_btn)
        controls.addWidget(dup_btn)
        controls.addWidget(del_btn)
        controls.addStretch(1)
        self.count_label = QLabel("0 simulations")
        self.count_label.setProperty("hint", True)
        controls.addWidget(self.count_label)
        root.addLayout(controls)

    # --- row operations ------------------------------------------------------
    def add_row(self, values=None):
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = values or _ROW_DEFAULTS
        self.table.blockSignals(True)
        for col, key in enumerate(self.columns):
            text = json.dumps(values.get(key, _ROW_DEFAULTS.get(key, [])))
            self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.blockSignals(False)
        self._refresh_row_headers()
        self._validate_all()

    def duplicate_selected(self):
        row = self.table.currentRow()
        if row < 0:
            return
        values = {}
        for col, key in enumerate(self.columns):
            item = self.table.item(row, col)
            try:
                values[key] = json.loads(item.text()) if item else []
            except (ValueError, TypeError):
                values[key] = []
        self.add_row(values)

    def remove_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            row = self.table.currentRow()
            rows = [row] if row >= 0 else []
        for row in rows:
            self.table.removeRow(row)
        self._refresh_row_headers()
        self._validate_all()

    # --- validation / display ------------------------------------------------
    def _on_item_changed(self, item):
        self._validate_item(item)

    def _validate_item(self, item):
        if item is None:
            return True
        try:
            json.loads(item.text())
            valid = True
        except (ValueError, TypeError):
            valid = False
        item.setForeground(QColor("#e7e9f3") if valid else _INVALID_BRUSH)
        item.setToolTip("" if valid else "Not valid JSON, e.g. use [1], [[1,1,1]] or [0.0].")
        return valid

    def _validate_all(self):
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                self._validate_item(self.table.item(row, col))
        self.count_label.setText(f"{self.table.rowCount()} simulations")

    def has_invalid_cells(self):
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                if not self._validate_item(self.table.item(row, col)):
                    return True
        return False

    def _refresh_row_headers(self):
        for row in range(self.table.rowCount()):
            self.table.setVerticalHeaderItem(row, QTableWidgetItem(f"Sim {row + 1}"))
        self.count_label.setText(f"{self.table.rowCount()} simulations")

    # --- load / dump ---------------------------------------------------------
    def load(self, config):
        self.vortex_excitation.setChecked(bool(config.get("vortex_excitation", 1)))
        self.repetitive.setChecked(bool(config.get("repetitive", 0)))
        self.dark_soliton.setChecked(bool(config.get("dark_soliton", 0)))

        n_sims = config_io.count_simulations(config)
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for i in range(n_sims):
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, key in enumerate(self.columns):
                seq = config.get(key, [])
                if isinstance(seq, list) and i < len(seq):
                    value = seq[i]
                else:
                    value = _ROW_DEFAULTS.get(key, [])
                self.table.setItem(row, col, QTableWidgetItem(json.dumps(value)))
        self.table.blockSignals(False)
        self._refresh_row_headers()
        self._validate_all()

    def dump(self, config):
        config["vortex_excitation"] = 1 if self.vortex_excitation.isChecked() else 0
        config["repetitive"] = 1 if self.repetitive.isChecked() else 0

        # Read every column into a per-simulation list.
        columns = {key: [] for key in self.columns}
        for row in range(self.table.rowCount()):
            for col, key in enumerate(self.columns):
                item = self.table.item(row, col)
                text = item.text() if item else "[]"
                try:
                    columns[key].append(json.loads(text))
                except (ValueError, TypeError):
                    columns[key].append(text)  # keep raw; validation flags it

        # Vortex (and shared) per-simulation keys are always written.
        for key in config_io.SIM_KEYS:
            config[key] = columns[key]

        # Soliton keys only when dark_soliton is enabled.
        if self.dark_soliton.isChecked():
            config["dark_soliton"] = 1
            for key in config_io.SOLITON_SIM_KEYS:
                config[key] = columns[key]
        else:
            config["dark_soliton"] = 0
            for key in config_io.SOLITON_SIM_KEYS:
                config.pop(key, None)
        return config
