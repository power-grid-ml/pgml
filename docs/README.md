# Documentation (contributor notes)

This is the Sphinx source for the published documentation. It is **not** itself a built
page (it is excluded from the build); it explains how to build and how the docs are laid
out. The published entry point is `index.md`.

## Building locally

Use the `docs` pixi environment (Sphinx + furo + myst-parser):

```bash
pixi run -e docs docs            # standard build -> docs/_build/html
pixi run -e docs docs-clean      # force full rebuild
pixi run -e docs docs-strict     # warnings-as-errors (mirrors CI)
pixi run -e docs docs-linkcheck  # check external links
```

Open `docs/_build/html/index.html` after building.

## Read-the-Docs

RTD is configured via `.readthedocs.yaml` at the repo root. It installs only
`docs/requirements.txt` (Sphinx + furo + myst-parser + pydantic); the heavy runtime
dependencies are mocked via `autodoc_mock_imports` in `conf.py`. `pydantic` must be a real
install (not mocked) because `pgml.schemas` subclasses `pydantic.BaseModel` and autodoc
resolves the class hierarchy at import time. `pandapower` is mocked (its NumPy-2 import
break).

## Layout

```
docs/
  index.md                  landing page: suite overview + dependency diagram + navigation
  conf.py                   Sphinx configuration
  requirements.txt          RTD pip requirements
  getting-started/          install.md, quickstart.md
  pgml/                     the simulation engine
    index.md                pgml overview
    concepts.md             modeling conventions (the short version)
    public-api.md           the stable public facade
    examples.md             example scripts + result figures
    modeling/               modeling decisions (the long version)
      *.md                  conventions, asymmetric, transformer, line model, DER, ...
      references/           external-library briefs (opendss/, pandapower/, power-grid-model/)
    api/                    autodoc API reference (one .rst per subpackage)
  pgl/                      the learning framework (state estimation): index.md, io-schema.md
  pgg/                      the grid-generation package: index.md
  _static/figures/         committed figures embedded in the docs
```

The API reference is generated from each subpackage's `__init__.py` `__all__`, so the
**docstrings are the docs**. Keep new docstrings valid reStructuredText.
