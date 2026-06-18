"""Small reusable input widgets used across the form tabs."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QWidget,
)

from app import config_io


class NumberField(QLineEdit):
    """A line edit for a single number (int or float).

    Empty text -> ``None`` (treated as "unset/null"). Accepts scientific
    notation such as ``5e-7``. Shows a red border when the text is unparseable.
    """

    def __init__(self, is_int=False, allow_empty=True, placeholder="", parent=None):
        super().__init__(parent)
        self._is_int = is_int
        self._allow_empty = allow_empty
        if placeholder:
            self.setPlaceholderText(placeholder)
        self.textChanged.connect(self._revalidate)

    def value(self):
        return config_io.parse_int(self.text()) if self._is_int else config_io.parse_float(self.text())

    def set_value(self, value):
        self.setText(config_io.num_to_str(value))

    def is_valid(self):
        text = self.text().strip()
        if text == "":
            return self._allow_empty
        try:
            self.value()
            return True
        except (ValueError, TypeError):
            return False

    def _revalidate(self, _text):
        self.setProperty("invalid", not self.is_valid())
        # Force the stylesheet to re-evaluate the dynamic property.
        self.style().unpolish(self)
        self.style().polish(self)


class VectorField(QWidget):
    """A horizontal row of NumberFields, e.g. a 3-vector ``[x, y, z]``."""

    def __init__(self, labels, is_int=False, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._edits = []
        for label in labels:
            edit = NumberField(is_int=is_int, allow_empty=False, placeholder=str(label))
            edit.setMinimumWidth(64)
            edit.setAlignment(Qt.AlignCenter)
            self._edits.append(edit)
            layout.addWidget(edit)

    def value(self):
        return [edit.value() for edit in self._edits]

    def set_value(self, values):
        values = values or []
        for edit, val in zip(self._edits, values):
            edit.set_value(val)

    def is_valid(self):
        return all(edit.is_valid() for edit in self._edits)
