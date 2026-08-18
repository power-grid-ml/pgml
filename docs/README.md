# Documentation (contributor notes)

This is the Sphinx source of the `pgml` documentation. It is **not** itself a built page
(it is excluded from the build); it explains how to build and how the docs are laid out.

The power-grid-ml suite publishes ONE documentation site
(<https://power-grid-ml.readthedocs.io>), assembled by the org-level `docs` repository
from the `docs/<pkg>/` tree of every package repository plus the suite landing page and
`getting-started/`. This repository owns the `pgml` tree; `index.md` here is only the
standalone landing toctree used by the per-repository build (the CI gate).

## Building locally

Use the `docs` pixi environment (Sphinx + furo + myst-parser + a CPU torch for autodoc):

```bash
pixi run -e docs docs            # standard build -> docs/_build/html
pixi run -e docs docs-clean      # force full rebuild
pixi run -e docs docs-strict     # warnings-as-errors (mirrors CI)
pixi run -e docs docs-linkcheck  # check external links
```

Open `docs/_build/html/index.html` after building.

## Layout

```
docs/
  index.md                  standalone landing toctree (the suite landing lives in the docs repo)
  conf.py                   Sphinx configuration (shared shape across the suite; PACKAGES = ("pgml",))
  pgml/                     the simulation engine
    index.md                pgml overview
    concepts.md             modeling conventions (the short version)
    public-api.md           the stable public facade
    examples.md             example scripts + result figures
    modeling/               modeling decisions (the long version)
      *.md                  conventions, asymmetric, transformer, line model, DER, ...
      references/           external-library briefs (opendss/, pandapower/, power-grid-model/)
    api/                    autodoc API reference (one .rst per subpackage)
  _static/figures/         committed figures embedded in the docs
```

The API reference is generated from each subpackage's `__init__.py` `__all__`, so the
**docstrings are the docs**. Keep new docstrings valid reStructuredText. `pandapower` is
mocked at autodoc time (`autodoc_mock_imports`); `pydantic`/`torch` are real.

Cross-package `{doc}` references (a page pointing at another suite package's page) resolve
natively in the assembled site and to the published site's URL in this per-repository build
(a `missing-reference` handler in `conf.py`).
