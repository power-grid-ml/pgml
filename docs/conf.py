"""Sphinx configuration for the power-grid-ml suite documentation.

The suite publishes ONE documentation site, assembled from the ``docs/<pkg>/`` tree of
every package repository (pgml, pgl, pgg, pghub, pgd). This configuration serves both
build modes with the same settings:

- a **package repository** builds only its own ``docs/<pkg>/`` tree (the CI gate; the
  landing ``docs/index.md`` is that package's toctree), and
- the org-level ``docs`` repository builds the assembled site (every ``docs/<pkg>/`` tree
  next to the suite landing page and ``getting-started/``).

``PACKAGES`` lists which package trees the current source tree contains; cross-package
``{doc}`` references to a package that is NOT built here resolve to the published site.
"""

from __future__ import annotations

import posixpath
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Which packages this build documents (edit per repository)
# ---------------------------------------------------------------------------
#: This repository documents its own package only.
PACKAGES = ('pgml',)

#: All packages of the suite, in dependency order; the published site carries them all.
SUITE_PACKAGES = ("pgml", "pgl", "pgg", "pghub", "pgd")
SUITE_DOCS_URL = "https://power-grid-ml.readthedocs.io/en/latest/"

# ---------------------------------------------------------------------------
# Path setup — src-layout; add `src` so `import <pkg>` works without install (a package
# repository). In the assembled build the packages are installed (editable) instead.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Fix __module__ for re-exported symbols so Sphinx registers them under the
# public package path (e.g. pgml.assembly.NodePhaseIndex) rather than the
# private sub-module path (pgml.assembly.index.NodePhaseIndex).  Without this
# autodoc emits "duplicate object description" warnings for every re-exported
# dataclass/class attribute.
# ---------------------------------------------------------------------------
def _patch_module(pkg_path: str, names: list) -> None:
    """Set __module__ = pkg_path for each name imported into that package."""
    import importlib

    pkg = importlib.import_module(pkg_path)
    for name in names:
        obj = getattr(pkg, name, None)
        if obj is not None and hasattr(obj, "__module__"):
            obj.__module__ = pkg_path


_REEXPORTS = {
    "pgml": {
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
    },
    # pgd (the dashboard/service layer) has the same aggregator-package shape: pgd.core and
    # pgd.storage each re-export classes defined in several private submodules
    # (core/{gridstore,jobs,errors}.py; storage/{duck,interface}.py) via __all__, and the
    # rest of pgd's own docstrings cross-reference them under the aggregator's short name
    # (e.g. ``pgd.core.GridStore``, ``pgd.storage.DuckDBStore``) — patch so autodoc registers
    # them there instead of under the private submodule path.
    "pgd": {
        "pgd.core": [
            "GridStore",
            "GridRecord",
            "GridRevision",
            "BranchLimit",
            "GridNotFoundError",
            "GridEditError",
            "GridEditValidationError",
            "EstimationUnavailable",
            "JobManager",
            "JobRecord",
            "JobCancelled",
        ],
        "pgd.storage": [
            "DuckDBStore",
            "TimeseriesStore",
            "DatasetInfo",
            "SeriesSpec",
            "DiagramSpec",
        ],
    },
}
for _pkg in PACKAGES:
    for _module, _names in _REEXPORTS.get(_pkg, {}).items():
        _patch_module(_module, _names)

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "pgml"
author = "power-grid-ml contributors"
copyright = "2024-2026, power-grid-ml contributors"
# Version from the package (the source tree, no install step).
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

# Napoleon: Google and NumPy-style docstrings
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
# Mock heavy/optional dependencies so the docs build does NOT need the full
# runtime stack. `pandapower` imports cleanly but is heavyweight and unneeded
# for autodoc (only its converter's docstrings are published); all other heavy
# deps install alongside Sphinx.
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
# Intersphinx — link to upstream docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

# ---------------------------------------------------------------------------
# HTML output — furo theme
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

# nitpicky off — avoids noise from cross-refs into mocked/optional deps and into suite
# packages that are documented in another repository's build.
nitpicky = False

# Suppress specific recurring warning categories.
# - toc.not_included: README.md is excluded, no need for it in a toctree.
# - ref.python (duplicate cross-refs): pgml.schemas re-exports grid_schema symbols,
#   so both pgml.schemas.Phase and pgml.schemas.grid_schema.Phase exist; this is
#   intentional for user convenience.
suppress_warnings = [
    "toc.not_included",
    "ref.python",
]


# ---------------------------------------------------------------------------
# Silence "duplicate object description" warnings for re-exported schema types.
#
# The pydantic schema models are defined in pgml.schemas.{grid,result,scenario}
# _schema and re-exported through pgml.schemas via ``__all__``. With autosummary
# documenting both the submodule and the package, the same field/class gets
# registered twice under the same fully-qualified name, producing "duplicate
# object description" warnings that cannot be suppressed via suppress_warnings
# (the warning carries no type= code). This is structural (re-export + pydantic
# field descriptors), NOT a docstring issue, so it cannot be fixed by editing the
# schema docstrings. We patch the Python domain's note_object to keep the first
# registration and silently drop duplicates.
# ---------------------------------------------------------------------------
def _patch_py_domain_silent_overwrite(app) -> None:  # noqa: ANN001
    """Silence duplicate-object warnings for re-exported schema types.

    The frozen pydantic schema models are re-exported from :mod:`pgml.schemas`
    via ``__all__``, so autosummary registers the same fully-qualified name twice
    (once under the submodule, once under the package), emitting an unsuppressible
    "duplicate object description" warning (no ``type=`` code, so
    ``suppress_warnings`` cannot catch it). We patch ``PythonDomain.note_object``
    to keep the first registration and silently drop the duplicate.
    """
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
        # If already registered, keep the first entry silently.
        if fullname in self.objects:
            return
        _orig_note(self, fullname, objtype, node_id, aliased=aliased, location=location)

    PythonDomain.note_object = _note_object_silent


# ---------------------------------------------------------------------------
# Cross-package ``{doc}`` references.
#
# A package's pages may point at another package's pages (a pgl page at
# ``/pgml/api/provenance``, a pghub page at ``../pgg/workflow``). In the assembled
# site every target exists; in a per-repository build the target package is not
# part of the source tree, so the reference would be "unknown document" — a
# warning that fails the strict build. Resolve such references to the published
# site instead of dropping them, keeping the strict gate meaningful for genuinely
# broken intra-package links.
# ---------------------------------------------------------------------------
def _resolve_suite_doc_refs(app, env, node, contnode):  # noqa: ANN001
    """Turn an unresolved ``{doc}`` ref into another suite package into a site link."""
    if node.get("refdomain") != "std" or node.get("reftype") != "doc":
        return None
    target = node.get("reftarget", "")
    if target.startswith("/"):
        docname = target.lstrip("/")
    else:
        docname = posixpath.normpath(posixpath.join(posixpath.dirname(node.get("refdoc", "")), target))
    head = docname.split("/", 1)[0]
    if head not in SUITE_PACKAGES or head in PACKAGES:
        return None
    from docutils import nodes

    ref = nodes.reference("", "", internal=False, refuri=f"{SUITE_DOCS_URL}{docname}.html")
    ref.append(contnode)
    return ref


def setup(app) -> None:  # noqa: ANN001
    """Sphinx setup hook — apply runtime patches."""
    app.connect("builder-inited", _patch_py_domain_silent_overwrite)
    app.connect("missing-reference", _resolve_suite_doc_refs)
