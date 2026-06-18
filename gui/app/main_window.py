"""Main window: wires the tabs together and owns load / save / run actions."""
import os

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QToolBar,
)

from app import config_io, paths
from app.runner import SimRunner
from app.widgets.json_tab import JsonTab
from app.widgets.run_tab import MODE_BAQS, MODE_MODULE, RunTab
from app.widgets.setup_tab import SetupTab
from app.widgets.simulations_tab import SimulationsTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPE Simulation Studio")
        self.resize(1080, 760)

        self._config_path = paths.DEFAULT_CONFIG_PATH
        self._app_config_path = paths.DEFAULT_APP_CONFIG_PATH

        self.runner = SimRunner(paths.PROJECT_ROOT, self)
        self.runner.output.connect(self._on_run_output)
        self.runner.started.connect(lambda: self.run_tab.set_running(True))
        self.runner.finished.connect(self._on_run_finished)

        self._build_tabs()
        self._build_toolbar()
        self.statusBar().showMessage(f"Project root: {paths.PROJECT_ROOT}")

        self._load_initial()

    # --- UI construction -----------------------------------------------------
    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.setup_tab = SetupTab()
        self.simulations_tab = SimulationsTab()
        self.run_tab = RunTab()
        self.json_tab = JsonTab()

        self.tabs.addTab(self.setup_tab, "Setup")
        self.tabs.addTab(self.simulations_tab, "Simulations")
        self.tabs.addTab(self.run_tab, "App && Run")
        self.tabs.addTab(self.json_tab, "Raw JSON")
        self.setCentralWidget(self.tabs)

        self.run_tab.run_btn.clicked.connect(self.run_simulation)
        self.run_tab.stop_btn.clicked.connect(self.runner.stop)
        self.json_tab.pull_btn.clicked.connect(self._json_pull)
        self.json_tab.push_btn.clicked.connect(self._json_push)

    def _build_toolbar(self):
        bar = QToolBar("Main")
        bar.setMovable(False)
        self.addToolBar(bar)

        def add(text, slot, shortcut=None, accent=False):
            action = QAction(text, self)
            action.triggered.connect(slot)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            bar.addAction(action)
            return action

        add("New", self.new_config)
        add("Open...", self.open_config, "Ctrl+O")
        add("Save", self.save_config, "Ctrl+S")
        add("Save As...", self.save_config_as, "Ctrl+Shift+S")
        bar.addSeparator()
        add("Validate", self.validate_config, "Ctrl+E")
        add("Run", self.run_simulation, "Ctrl+R")

        bar.addSeparator()
        self.path_label = QLabel()
        self.path_label.setProperty("hint", True)
        bar.addWidget(self.path_label)

    # --- gather / apply across tabs -----------------------------------------
    def gather_config(self):
        config = {}
        self.setup_tab.dump(config)
        self.simulations_tab.dump(config)
        return config

    def apply_config(self, config):
        self.setup_tab.load(config)
        self.simulations_tab.load(config)

    # --- file actions --------------------------------------------------------
    def _load_initial(self):
        if os.path.exists(self._config_path):
            try:
                self.apply_config(config_io.load_json(self._config_path))
            except (ValueError, OSError) as exc:
                self._warn("Could not read configuration_file.json", str(exc))
                self.apply_config(config_io.default_config())
        else:
            self.apply_config(config_io.default_config())

        if os.path.exists(self._app_config_path):
            try:
                self.run_tab.load_app_config(config_io.load_json(self._app_config_path))
            except (ValueError, OSError):
                self.run_tab.load_app_config(config_io.default_app_config())
        else:
            self.run_tab.load_app_config(config_io.default_app_config())

        self._update_path_label()

    def new_config(self):
        self.apply_config(config_io.default_config())
        self.run_tab.load_app_config(config_io.default_app_config())
        self.statusBar().showMessage("New config created (not yet saved).")

    def open_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open configuration", paths.PROJECT_ROOT, "JSON files (*.json)"
        )
        if not path:
            return
        try:
            self.apply_config(config_io.load_json(path))
        except (ValueError, OSError) as exc:
            self._warn("Could not open file", str(exc))
            return
        self._config_path = path
        self._update_path_label()
        self.statusBar().showMessage(f"Opened {path}")

    def save_config(self):
        return self._save_to(self._config_path)

    def save_config_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save configuration", self._config_path, "JSON files (*.json)"
        )
        if not path:
            return False
        self._config_path = path
        return self._save_to(path)

    def _save_to(self, path):
        config = self.gather_config()
        try:
            config_io.save_json(path, config)
            config_io.save_json(self._app_config_path, self.run_tab.dump_app_config())
        except OSError as exc:
            self._warn("Could not save", str(exc))
            return False
        self._update_path_label()
        self.statusBar().showMessage(f"Saved {path}")
        return True

    # --- validate ------------------------------------------------------------
    def validate_config(self):
        config = self.gather_config()
        errors, warnings = config_io.validate(config)
        if self.simulations_tab.has_invalid_cells():
            errors.insert(0, "One or more simulation cells contain invalid JSON (red).")

        if not errors and not warnings:
            QMessageBox.information(self, "Validation", "All good — no problems found.")
            return True

        msg = QMessageBox(self)
        msg.setWindowTitle("Validation")
        msg.setIcon(QMessageBox.Critical if errors else QMessageBox.Warning)
        parts = []
        if errors:
            parts.append("Errors (must fix):\n  - " + "\n  - ".join(errors))
        if warnings:
            parts.append("Warnings:\n  - " + "\n  - ".join(warnings))
        msg.setText("\n\n".join(parts))
        msg.exec()
        return not errors

    # --- run -----------------------------------------------------------------
    def run_simulation(self):
        if self.runner.is_running():
            return
        config = self.gather_config()
        errors, _ = config_io.validate(config)
        if self.simulations_tab.has_invalid_cells():
            errors.insert(0, "Fix invalid simulation cells (red) before running.")
        if errors:
            self._warn("Cannot run", "Resolve these first:\n\n  - " + "\n  - ".join(errors))
            self.tabs.setCurrentWidget(self.setup_tab)
            return

        working_dir = self.run_tab.current_working_dir()
        if not working_dir:
            self._warn("Cannot run", "Set a working directory in the App && Run tab.")
            self.tabs.setCurrentWidget(self.run_tab)
            return
        try:
            os.makedirs(working_dir, exist_ok=True)
        except OSError as exc:
            self._warn("Cannot run", f"Could not create working directory:\n{exc}")
            return

        # Write the canonical config files into the working directory; the solver
        # reads them from there (its current directory) in every launch mode.
        try:
            config_io.save_json(os.path.join(working_dir, paths.CONFIG_FILENAME), config)
            config_io.save_json(
                os.path.join(working_dir, paths.APP_CONFIG_FILENAME),
                self.run_tab.dump_app_config(),
            )
        except OSError as exc:
            self._warn("Could not write config before run", str(exc))
            return

        argv = self._build_launch_argv(working_dir)
        if argv is None:
            return

        self.tabs.setCurrentWidget(self.run_tab)
        self.run_tab.append_log(
            f"\n=== Launching: {' '.join(argv)} (cwd={working_dir}) ===\n"
        )
        ok = self.runner.start(argv, working_dir)
        if not ok:
            self.run_tab.set_running(False)

    def _build_launch_argv(self, working_dir):
        """Assemble the subprocess argv for the selected launch mode.

        Returns None (after warning) if the mode's prerequisites are missing.
        """
        mode = self.run_tab.current_launch_mode()
        if mode == MODE_BAQS:
            return [
                self.run_tab.current_baqs_command(),
                paths.CONFIG_FILENAME, paths.APP_CONFIG_FILENAME,
                "--check", "--run", "-v", "1",
            ]

        interpreter = self.run_tab.current_interpreter()
        if mode == MODE_MODULE:
            return [interpreter, "-u", "-m", "src.run"]

        # Script mode needs the source tree in the working directory.
        script = os.path.join(working_dir, paths.RUN_SCRIPT_REL)
        if not os.path.isfile(script):
            self._warn(
                "Cannot run",
                f"Script mode needs the source tree, but {paths.RUN_SCRIPT_REL} "
                f"was not found in the working directory:\n{working_dir}\n\n"
                "Switch to 'Module' or 'Installed (baqs)' mode, or point the "
                "working directory at the project root.",
            )
            return None
        return [interpreter, "-u", paths.RUN_SCRIPT_REL]

    def _on_run_output(self, text):
        self.run_tab.append_log(text)

    def _on_run_finished(self, code):
        self.run_tab.set_running(False)
        self.run_tab.append_log(f"\n=== Simulation finished with exit code {code} ===\n")
        self.statusBar().showMessage(f"Simulation finished (exit {code}).")

    # --- raw JSON sync -------------------------------------------------------
    def _json_pull(self):
        self.json_tab.set_text(self.gather_config())

    def _json_push(self):
        try:
            config = self.json_tab.parse()
        except ValueError as exc:
            self.json_tab.status.setText(str(exc))
            self._warn("Invalid JSON", str(exc))
            return
        self.apply_config(config)
        self.json_tab.status.setText("Pushed to form.")
        self.statusBar().showMessage("Applied raw JSON to the form.")

    # --- helpers -------------------------------------------------------------
    def _update_path_label(self):
        self.path_label.setText("  " + os.path.basename(self._config_path))

    def _warn(self, title, text):
        QMessageBox.warning(self, title, text)

    def closeEvent(self, event):
        if self.runner.is_running():
            answer = QMessageBox.question(
                self, "Simulation running",
                "A simulation is still running. Stop it and quit?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.runner.stop()
        event.accept()
