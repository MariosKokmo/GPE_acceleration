"""Entry point for the GPE Simulation Studio GUI.

This GUI is intentionally *segregated* from the simulation code base: it never
imports the solver. It only reads/writes the two JSON files the project already
uses (``configuration_file.json`` and ``appConfig.json``) and launches the
existing ``src/run.py`` as a subprocess.

Run with:
    python gui/run_gui.py
"""
import os
import sys

# Make the sibling ``app`` package importable regardless of the launch cwd.
_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
if _GUI_DIR not in sys.path:
    sys.path.insert(0, _GUI_DIR)

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.theme import apply_theme


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("GPE Simulation Studio")
    app.setOrganizationName("GPE")
    apply_theme(app)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
