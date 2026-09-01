"""Launches the simulation as a subprocess and streams its output.

Using QProcess keeps the UI responsive and lets us tail stdout/stderr live.
The GUI never imports the solver — it only spawns a command (``python
src/run.py``, ``python -m src.run`` or the installed ``baqs`` console script) in
a chosen working directory, exactly as a user would from a terminal.
"""
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal


class SimRunner(QObject):
    output = Signal(str)
    started = Signal()
    finished = Signal(int)

    def __init__(self, project_root, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        self._proc = None

    def is_running(self):
        return self._proc is not None and self._proc.state() != QProcess.NotRunning

    def start(self, argv, cwd):
        """Launch ``argv`` (program + arguments) with the given working dir."""
        if self.is_running():
            return False
        if not argv:
            self.output.emit("[runner] Nothing to launch (empty command).\n")
            return False

        proc = QProcess(self)
        proc.setWorkingDirectory(cwd)
        proc.setProcessChannelMode(QProcess.MergedChannels)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")  # so we see logs as they happen
        proc.setProcessEnvironment(env)

        proc.readyReadStandardOutput.connect(self._on_ready)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)

        self._proc = proc
        program, arguments = argv[0], list(argv[1:])
        proc.start(program, arguments)
        if not proc.waitForStarted(5000):
            self.output.emit(f"[runner] Failed to start '{program}'.\n")
            self._proc = None
            return False

        self.started.emit()
        return True

    def stop(self):
        if self.is_running():
            self.output.emit("\n[runner] Stopping simulation...\n")
            self._proc.kill()

    # --- internal callbacks --------------------------------------------------
    def _on_ready(self):
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        if data:
            self.output.emit(data)

    def _on_error(self, _err):
        if self._proc is not None:
            self.output.emit(f"[runner] Process error: {self._proc.errorString()}\n")

    def _on_finished(self, code, _status):
        self.finished.emit(int(code))
        self._proc = None
