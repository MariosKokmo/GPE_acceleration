# GPE Simulation Studio (GUI)

A small, self-contained desktop app for **building the simulation config and
launching runs** for this project. It is deliberately **segregated** from the
solver: it never imports `src/`. It only

1. reads/writes `configuration_file.json` and `appConfig.json` in the project
   root, and
2. launches the existing `src/run.py` as a **subprocess**, streaming its log
   live.

That separation means the GUI stays lightweight (it does not pull in torch/CUDA)
and the solver code is unaffected by anything here.

## Layout

```
gui/
  run_gui.py              # entry point
  requirements.txt        # PySide6 only
  app/
    paths.py              # project-root / file locations
    theme.py              # modern dark Qt stylesheet
    config_io.py          # defaults, load/save, validation (knows the schema)
    runner.py             # QProcess wrapper that runs src/run.py
    widgets/
      fields.py           # reusable numeric / vector inputs
      setup_tab.py        # grid / potential / time / model / absorber
      simulations_tab.py  # one row per simulation (per-sim list editor)
      run_tab.py          # appConfig + interpreter + live log
      json_tab.py         # raw JSON escape hatch (explicit Pull/Push sync)
    main_window.py        # wires tabs + open/save/validate/run actions
```

## Install & run

The GUI needs its own dependency (Qt via PySide6).

```bash
# from the project root
python -m pip install -r gui/requirements.txt
python gui/run_gui.py
```

### Python version note

The project targets **Python 3.12** (Python 3.8 is end-of-life). PySide6 6.7/6.8
provide cp312 wheels, so a normal `pip install` works on 3.12 with no DLL games.

> Historical note: the original environment was Python **3.8.0**, whose
> `python3.dll` / `python38.dll` shipped with missing stable-ABI exports
> (`Py_CompileString`, `PyObject_GenericGetDict`), causing
> `ImportError: DLL load failed while importing Shiboken: The specified
> procedure could not be found`. That env was patched in place to 3.8.10 DLLs
> (backups `python3.dll.bak_3.8.0` / `python38.dll.bak_3.8.0` in the base Python
> dir). Moving to a fresh 3.12 install makes all of that moot.

> Tip: you can install PySide6 into the same environment that runs the solver,
> or a separate one — the GUI launches the simulation via whatever interpreter
> you pick in the **App && Run** tab (defaults to the one running the GUI).

## Using it

- **Setup** — coordinate system (Cartesian/cylindrical fields swap
  automatically), grid, potential, time stepping, physics model (model-specific
  fields show/hide per `model_type`), and an optional boundary absorber.
- **Simulations** — global excitation flags (`vortex_excitation`, `repetitive`)
  plus a checkable **Dark soliton excitation** group (`soliton_positions`,
  `soliton_widths`, `soliton_axes` [1=x / 3=z], optional `soliton_greyness`,
  `soliton_imprint_time`). Below that, one row per simulation: cells accept JSON
  (`[1]`, `[1,-1]`, `[[1,1,1]]`); invalid cells turn red. Add / Duplicate /
  Remove rows; all per-simulation lists stay the same length automatically.
  Dark-soliton settings are global (shared by every simulation), matching what
  the solver reads.
- **App && Run** — `appConfig.json` settings, a **working directory**, a
  **launch mode**, the interpreter / baqs command, Run/Stop, and a live log pane.
- **Raw JSON** — full-text view with explicit **Pull from form** / **Push to
  form** so there is no hidden state.

Toolbar shortcuts: New, Open (Ctrl+O), Save (Ctrl+S), Save As (Ctrl+Shift+S),
Validate (Ctrl+E), Run (Ctrl+R).

### Run: working directory + launch modes

On **Run**, the GUI writes the canonical `configuration_file.json` /
`appConfig.json` into the chosen **working directory** and launches the solver
there (its outputs — ground state, result folders, `log.txt` — also land there).
Three launch modes decouple *where the GUI is* from *how the solver runs*:

| Mode | Command | Needs |
|---|---|---|
| **Script** | `<interpreter> -u src/run.py` | the `src/` source tree in the working dir |
| **Module** | `<interpreter> -u -m src.run` | source tree present **or** the package pip-installed in that interpreter |
| **Installed (baqs)** | `baqs configuration_file.json appConfig.json --check --run` | only a venv with `baqs` pip-installed — **no source tree** |

The **Installed (baqs)** mode is the one to use with a frozen GUI exe pointed at
a deployment venv: `pip install .` once into a venv, point the working directory
at any (even empty) results folder, and run. The baqs command auto-fills to the
`baqs` executable sitting next to the chosen interpreter.

## Packaging to a standalone .exe (optional)

Because the GUI is decoupled, you can freeze just the GUI and let it call the
user's existing Python for the heavy solver:

```bash
python -m pip install pyinstaller
pyinstaller --noconfirm --windowed --name "GPE Simulation Studio" ^
    --paths gui gui/run_gui.py
```

Prefer the default one-dir build (faster start, easier to debug) over
`--onefile`. Do **not** try to bundle torch/CUDA into this exe — keep the GUI
thin and point it at a Python interpreter that has the solver installed.
