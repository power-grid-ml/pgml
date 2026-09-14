"""Sphinx configuration for the pgml documentation.

The pages live under ``docs/pgml/``, with ``docs/index.md`` as the landing page. The API
reference is generated from each subpackage's ``__all__``, so the docstrings are the
reference. ``pandapower`` is mocked at autodoc time; ``torch`` and ``pydantic`` are real.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup. src-layout: add `src` so `import pgml` works without an install step.
# An installed (editable or wheel) package takes over when `src` is absent.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Fix __module__ for re-exported symbols so Sphinx registers them under the public
# package path (``pgml.assembly.YBus`` rather than ``pgml.assembly.index.YBus``).
# Without this, autodoc emits a duplicate object description for every re-exported
# class reachable from two automodule directives.
# ---------------------------------------------------------------------------
_REEXPORTS = {
    "pgml.assembly": ["NodePhaseIndex", "YBus"],
    "pgml.solver": ["PowerFlowResult", "HarmonicFlowResult"],
    "pgml.scenarios": ["SampledScenarios", "ScenarioResult"],
    "pgml.geometry": [
        "series_impedance",
        "internal_impedance",
        "potential_coefficients",
        "kron_reduce",
        "line_constants",
        "i0_over_i1",
        "positive_sequence_z",
        "skin_resistance_multiplier",
        "fit_equivalent_rdc",
        "two_conductor_geometry",
        "two_conductor_loop_z",
        "phase_to_sequence",
        "sequence_impedances",
        "carson_earth_resistance",
        "zero_sequence_harmonic_z",
        "sequence_to_phase_z",
        "sequence_aware_phase_z",
        "synthesize_line_geometry",
        "synthesize_grid_geometry",
        "positive_sequence_resistance_model",
        "apply_positive_sequence_harmonic_model",
        "apply_sequence_aware_harmonic_model",
        "apply_default_harmonic_model",
    ],
}


def _patch_module(pkg_path: str, names: list) -> None:
    """Set ``__module__ = pkg_path`` for each name imported into that package."""
    import importlib

    pkg = importlib.import_module(pkg_path)
    for name in names:
        obj = getattr(pkg, name, None)
        if obj is not None and hasattr(obj, "__module__"):
            obj.__module__ = pkg_path


for _module, _names in _REEXPORTS.items():
    _patch_module(_module, _names)

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "pgml"
author = "power-grid-ml contributors"
copyright = "2024-2026, power-grid-ml contributors"

import pgml as _pkg

release = _pkg.__version__
version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# Autosummary: generate stub .rst files automatically
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False

# Napoleon: NumPy-style docstrings (Google style also parsed)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Autodoc defaults
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
    "inherited-members": False,
}
autodoc_member_order = "bysource"
add_module_names = False

# ---------------------------------------------------------------------------
# Mock the one heavyweight optional dependency. `pandapower` imports cleanly but is
# slow and unnecessary for autodoc, since only its converter's docstrings are
# published. Every other dependency is installed alongside Sphinx and imports for real.
# ---------------------------------------------------------------------------
autodoc_mock_imports = [
    "pandapower",
]

# ---------------------------------------------------------------------------
# MyST parser settings
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

# ---------------------------------------------------------------------------
# Link checking. The PyTorch documentation numbers its anchors differently from the
# objects inventory intersphinx resolves against, so an anchor check on those generated
# links reports failures that are not broken links.
# ---------------------------------------------------------------------------
linkcheck_anchors_ignore_for_url = [
    "https://docs.pytorch.org/.*",
    "https://pytorch.org/docs/.*",
]

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = f"{project} {release}"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# ---------------------------------------------------------------------------
# Other settings
# ---------------------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# nitpicky off: cross-references into the mocked dependency cannot resolve.
nitpicky = False

# - toc.not_included: README.md is excluded from the build, so it needs no toctree entry.
# - ref.python: pgml.schemas re-exports the three schema modules, so both
#   pgml.schemas.Phase and pgml.schemas.grid_schema.Phase exist. That is intentional.
suppress_warnings = [
    "toc.not_included",
    "ref.python",
]


# ---------------------------------------------------------------------------
# Silence "duplicate object description" for re-exported schema types.
#
# The pydantic schema models are defined in pgml.schemas.{grid,result,scenario}_schema and
# re-exported through pgml.schemas via ``__all__``. With both the package and its submodules
# documented, the same fully-qualified name is registered twice, which emits a warning that
# carries no type code and therefore cannot be suppressed through ``suppress_warnings``.
# Keep the first registration and drop the duplicate.
# ---------------------------------------------------------------------------
def _patch_py_domain_silent_overwrite(app) -> None:  # noqa: ANN001
    """Keep the first registration of a fully-qualified name, drop later duplicates."""
    from sphinx.domains.python import PythonDomain

    _orig_note = PythonDomain.note_object

    def _note_object_silent(
        self,
        fullname,
        objtype,
        node_id,  # noqa: ANN001
        aliased=False,
        location=None,
    ):
        if fullname in self.objects:
            return
        _orig_note(self, fullname, objtype, node_id, aliased=aliased, location=location)

    PythonDomain.note_object = _note_object_silent


def setup(app) -> None:  # noqa: ANN001
    """Sphinx setup hook."""
    app.connect("builder-inited", _patch_py_domain_silent_overwrite)
