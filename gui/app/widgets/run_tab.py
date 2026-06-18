"""Run tab: app-level settings, the interpreter, and the live simulation log.

The actual launch/stop is wired up by MainWindow; this tab owns the widgets and
exposes the chosen interpreter plus the appConfig fields.
"""
import os
import sys

from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app import paths

# Launch-mode identifiers (kept as combo userData).
MODE_SCRIPT = "script"   # <interpreter> -u src/run.py   (needs the source tree)
MODE_MODULE = "module"   # <interpreter> -u -m src.run   (source tree or installed)
MODE_BAQS = "baqs"       # <baqs> config app --check --run (installed package only)


def _derive_baqs_command(interpreter):
    """Guess the baqs console-script path that sits next to an interpreter."""
    folder = os.path.dirname(interpreter)
    name = "baqs.exe" if os.name == "nt" else "baqs"
    candidate = os.path.join(folder, name)
    return candidate if os.path.isfile(candidate) else "baqs"


class RunTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        top = QHBoxLayout()
        top.addWidget(self._build_app_config_box(), 1)
        top.addWidget(self._build_launch_box(), 1)
        root.addLayout(top)

        root.addWidget(self._build_log_box(), 1)

    # --- boxes ---------------------------------------------------------------
    def _build_app_config_box(self):
        box = QGroupBox("Application settings (appConfig.json)")
        form = QFormLayout(box)
        self.logfile = QLineEdit()
        self.config_file = QLineEdit()
        self.write_velocity = QCheckBox("write velocity field + video after each run")
        self.phase_imaging = QCheckBox("save phase snapshots at every snapshot step")
        form.addRow("logfile", self.logfile)
        form.addRow("configFile", self.config_file)
        form.addRow("write_velocity", self.write_velocity)
        form.addRow("phase_imaging", self.phase_imaging)
        return box

    def _build_launch_box(self):
        box = QGroupBox("Launch")
        form = QFormLayout(box)

        # Working directory: where the config is written and the solver runs.
        # Outputs (ground state, result folders) land here. Decoupled from the
        # GUI install location so a frozen exe can target any project/results dir.
        self.working_dir = QLineEdit(paths.PROJECT_ROOT)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._choose_working_dir)
        wd_row = QHBoxLayout()
        wd_row.addWidget(self.working_dir, 1)
        wd_row.addWidget(browse_btn)
        wd_widget = QWidget()
        wd_widget.setLayout(wd_row)

        # How to launch the solver.
        self.launch_mode = QComboBox()
        self.launch_mode.addItem("Script — python src/run.py", MODE_SCRIPT)
        self.launch_mode.addItem("Module — python -m src.run", MODE_MODULE)
        self.launch_mode.addItem("Installed — baqs command", MODE_BAQS)
        self.launch_mode.currentIndexChanged.connect(self._sync_mode)

        self.interpreter = QComboBox()
        self.interpreter.setEditable(True)
        default_interp = sys.executable if not getattr(sys, "frozen", False) else "python"
        self.interpreter.addItems([default_interp, "python", "python3"])
        self.interpreter.setCurrentText(default_interp)

        self.baqs_cmd = QLineEdit(_derive_baqs_command(default_interp))

        self.run_btn = QPushButton("Run simulations")
        self.run_btn.setProperty("accent", True)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setProperty("danger", True)
        self.stop_btn.setEnabled(False)
        btns = QHBoxLayout()
        btns.addWidget(self.run_btn)
        btns.addWidget(self.stop_btn)
        btns.addStretch(1)

        self.status = QLabel("Idle.")
        self.status.setProperty("hint", True)

        form.addRow("Working directory", wd_widget)
        form.addRow("Launch mode", self.launch_mode)
        self._interp_label = QLabel("Python interpreter")
        form.addRow(self._interp_label, self.interpreter)
        self._baqs_label = QLabel("baqs command")
        form.addRow(self._baqs_label, self.baqs_cmd)
        form.addRow(btns)
        form.addRow(self.status)

        self.hint = QLabel()
        self.hint.setProperty("hint", True)
        self.hint.setWordWrap(True)
        form.addRow(self.hint)

        self._form = form
        self._sync_mode()
        return box

    def _choose_working_dir(self):
        start = self.working_dir.text().strip() or paths.PROJECT_ROOT
        chosen = QFileDialog.getExistingDirectory(self, "Select working directory", start)
        if chosen:
            self.working_dir.setText(chosen)

    def _sync_mode(self, *_):
        """Enable only the fields relevant to the selected launch mode and
        update the explanatory hint."""
        mode = self.current_launch_mode()
        is_baqs = mode == MODE_BAQS
        for w in (self._interp_label, self.interpreter):
            w.setEnabled(not is_baqs)
        for w in (self._baqs_label, self.baqs_cmd):
            w.setEnabled(is_baqs)

        if mode == MODE_SCRIPT:
            self.hint.setText("Runs <interpreter> -u src/run.py in the working "
                              "directory. The working directory must contain the "
                              "src/ source tree.")
        elif mode == MODE_MODULE:
            self.hint.setText("Runs <interpreter> -u -m src.run in the working "
                              "directory. Works with the source tree present or "
                              "with the package pip-installed in that interpreter.")
        else:
            self.hint.setText("Runs the installed baqs console script: "
                              "baqs configuration_file.json appConfig.json "
                              "--check --run. Needs only a venv with baqs "
                              "pip-installed — no source tree required.")

    def _build_log_box(self):
        box = QGroupBox("Simulation log")
        layout = QVBoxLayout(box)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        mono = QFont("Consolas", 9)
        mono.setStyleHint(QFont.Monospace)
        self.log.setFont(mono)
        layout.addWidget(self.log)
        clear_btn = QPushButton("Clear log")
        clear_btn.clicked.connect(self.log.clear)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(clear_btn)
        layout.addLayout(row)
        return box

    # --- log helpers ---------------------------------------------------------
    def append_log(self, text):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)

    def set_running(self, running):
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.status.setText("Running..." if running else "Idle.")

    def current_interpreter(self):
        return self.interpreter.currentText().strip() or "python"

    def current_launch_mode(self):
        return self.launch_mode.currentData()

    def current_working_dir(self):
        return self.working_dir.text().strip()

    def current_baqs_command(self):
        return self.baqs_cmd.text().strip() or "baqs"

    # --- load / dump (appConfig) ---------------------------------------------
    def load_app_config(self, app_config):
        self.logfile.setText(str(app_config.get("logfile", "log.txt")))
        self.config_file.setText(str(app_config.get("configFile", "configuration_file.json")))
        self.write_velocity.setChecked(bool(app_config.get("write_velocity", False)))
        self.phase_imaging.setChecked(bool(app_config.get("phase_imaging", False)))

    def dump_app_config(self):
        return {
            "logfile": self.logfile.text().strip() or "log.txt",
            "configFile": self.config_file.text().strip() or "configuration_file.json",
            "write_velocity": self.write_velocity.isChecked(),
            "phase_imaging": self.phase_imaging.isChecked(),
        }
