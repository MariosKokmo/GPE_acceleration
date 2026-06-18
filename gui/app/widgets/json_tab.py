"""Raw JSON tab: a power-user escape hatch.

Sync with the structured tabs is explicit (Pull / Push buttons) so there is no
hidden state and no surprising overwrites.
"""
import json

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class JsonTab(QWidget):
    """Signals are handled by MainWindow via the exposed buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        hint = QLabel(
            "Live JSON view of the whole config. 'Pull from form' regenerates "
            "this text from the editor tabs; 'Push to form' parses this text "
            "back into them. 'Format' pretty-prints."
        )
        hint.setProperty("hint", True)
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.editor = QPlainTextEdit()
        mono = QFont("Consolas", 10)
        mono.setStyleHint(QFont.Monospace)
        self.editor.setFont(mono)
        root.addWidget(self.editor, 1)

        self.status = QLabel("")
        self.status.setProperty("hint", True)
        root.addWidget(self.status)

        controls = QHBoxLayout()
        self.pull_btn = QPushButton("Pull from form")
        self.push_btn = QPushButton("Push to form")
        self.push_btn.setProperty("accent", True)
        self.format_btn = QPushButton("Format")
        self.format_btn.clicked.connect(self.format_text)
        controls.addWidget(self.pull_btn)
        controls.addWidget(self.push_btn)
        controls.addWidget(self.format_btn)
        controls.addStretch(1)
        root.addLayout(controls)

    def set_text(self, config):
        self.editor.setPlainText(json.dumps(config, indent=4))
        self.status.setText("Synced from form.")

    def parse(self):
        """Return the parsed dict, or raise ValueError with a friendly message."""
        try:
            return json.loads(self.editor.toPlainText())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno}).")

    def format_text(self):
        try:
            data = self.parse()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.editor.setPlainText(json.dumps(data, indent=4))
        self.status.setText("Formatted.")
