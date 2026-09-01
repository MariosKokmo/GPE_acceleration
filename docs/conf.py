"""Sphinx configuration for the BAQS documentation.

Design notes
------------
*Docstrings are the single source of truth.* The API reference is generated
from the code by ``sphinx-apidoc`` on every build (see ``run_apidoc`` below),
so adding a module needs no edit here — it appears in the next build.

*Both docstring styles are supported.* This codebase mixes Google style
(``Args:`` / ``Returns:``) with NumPy style (``Parameters`` / ``-------``).
``sphinx.ext.napoleon`` understands both at once, so neither has to be
rewritten.

*Heavy dependencies are mocked, not installed.* Nothing in ``src`` does real
work at import time — every torch/numpy call sits inside a function or method —
so autodoc can import the modules with the expensive third-party packages
replaced by stubs. That keeps a Read the Docs build to seconds instead of
pulling ~200 MB of PyTorch wheels, and means the docs build on machines with no
CUDA stack at all.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent

# Make ``import src...`` resolve without installing the package.
sys.path.insert(0, str(ROOT))

# -- Project information -----------------------------------------------------

project = "BAQS"
author = "Marios Kokmotos"
copyright = f"{author}"


def _find_version() -> str:
    """Read the version from src/__version__.py without importing it."""
    import re

    text = (ROOT / "src" / "__version__.py").read_text(encoding="utf-8")
    match = re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}", text)
    return match.group(0) if match else "0.0.0"


release = _find_version()
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",       # pull documentation out of docstrings
    "sphinx.ext.autosummary",   # summary tables at the top of each module page
    "sphinx.ext.napoleon",      # understand Google *and* NumPy docstring styles
    "sphinx.ext.viewcode",      # link every object to its highlighted source
    "sphinx.ext.intersphinx",   # cross-link to torch / numpy / python docs
    "sphinx.ext.mathjax",       # render the LaTeX in the physics docstrings
    "myst_parser",              # let the existing Markdown files be pages
]

templates_path = ["_templates"]
# apidoc also emits a modules.rst index; the curated toctree in index.md
# points at api/src instead, so exclude it rather than leave it orphaned.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "api/modules.rst"]

# Markdown and reStructuredText side by side, so the existing guides need no
# conversion.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3

# -- autodoc / napoleon ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    # The library leans on inheritance (GPE2DLibrary(GPELibrary),
    # GPE2DCylindricalLibrary(GPECylindricalLibrary), the BaseBEC subclasses),
    # so a page that hid inherited members would be actively misleading.
    "inherited-members": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autoclass_content = "both"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_notes = True

# Third-party packages replaced by stubs at import time. Safe here because no
# module executes them at import; see the note at the top of this file.
autodoc_mock_imports = [
    "torch",
    "cv2",
    "pandas",
    "matplotlib",
    "h5py",
    "PySide6",
    "PyQt5",
    "PySide2",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Unresolved cross-references are warnings; CI runs with -W so they fail there.
nitpicky = False

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"{project} {release}"
html_static_path = ["_static"]

# The repository keeps figures in ``static/`` and the Markdown guides refer to
# them with raw ``<img src="static/...">`` tags, which Sphinx passes through
# untouched. Staging a copy under ``_extra`` puts them at the output root so
# those paths resolve in the built site as well as on GitHub.
html_extra_path = ["_extra"]


# -- Generate the API tree from the source on every build --------------------

def run_apidoc(_app) -> None:
    """Regenerate ``docs/api`` from ``src`` before the build starts.

    Running this from ``conf.py`` rather than as a separate command means the
    local build, the CI build and the Read the Docs build all produce the same
    tree, and a newly added module is picked up without anyone remembering to
    rerun a script.
    """
    from sphinx.ext import apidoc

    output = HERE / "api"
    apidoc.main([
        "--force",           # overwrite, so deleted modules do not linger
        "--separate",        # one page per module
        "--module-first",    # module docstring above its members
        "--implicit-namespaces",
        "-o", str(output),
        str(ROOT / "src"),
        # Entry points and scripts: no useful API surface.
        str(ROOT / "src" / "run.py"),
    ])


def stage_static_figures(_app) -> None:
    """Copy ``static/`` next to the built HTML so the guides' figures resolve."""
    import shutil

    source = ROOT / "static"
    if not source.is_dir():
        return
    destination = HERE / "_extra" / "static"
    destination.parent.mkdir(exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


# Illustrative JSON in the guides is not valid JSON (it uses a typographic
# minus sign and "..." elisions), so Pygments cannot lex it. Harmless.
suppress_warnings = ["misc.highlighting_failure"]

# ``|psi|`` and friends are modulus bars, not reStructuredText substitutions.
# Anything between a pair of vertical bars is otherwise read as a reference to
# a substitution that was never defined, which is where the bulk of the build's
# warnings came from. Escaping them as the docstring is handed to the parser
# keeps the physics notation readable in the source and correct in the output.
_MODULUS_BARS = re.compile(r"(?<![\\|`])\|([^\s|][^|\n]{0,30}?)\|(?![|`])")


def escape_modulus_bars(_app, _what, _name, _obj, _options, lines) -> None:
    """Escape ``|x|`` so docutils reads it as text rather than a substitution."""
    in_literal_block = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        # Leave literal blocks (```` :: ```` followed by an indented run) alone:
        # they are reproduced verbatim and need no escaping.
        if in_literal_block:
            if stripped and not line.startswith((" ", "\t")):
                in_literal_block = False
            else:
                continue
        if stripped.endswith("::"):
            in_literal_block = True
            continue
        if "|" in line:
            lines[index] = _MODULUS_BARS.sub(r"\\|\1\\|", line)


def setup(app):
    app.connect("builder-inited", run_apidoc)
    app.connect("builder-inited", stage_static_figures)
    app.connect("autodoc-process-docstring", escape_modulus_bars)
