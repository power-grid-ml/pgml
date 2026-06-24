# Documentation

## Building locally (recommended)

Use the `docs` pixi environment, which includes Sphinx, furo, and myst-parser:

```bash
# Standard build
pixi run --environment docs sphinx-build -b html docs docs/_build/html

# Clean build (force regenerate all pages)
pixi run --environment docs sphinx-build -b html -E docs docs/_build/html

# Strict build (warnings treated as errors, same as CI)
pixi run --environment docs sphinx-build -b html -W --keep-going docs docs/_build/html
```

Or use the pixi tasks:

```bash
pixi run --environment docs docs          # standard build
pixi run --environment docs docs-clean    # force full rebuild
pixi run --environment docs docs-strict   # CI-equivalent
pixi run --environment docs docs-linkcheck  # check external links
```

Open `docs/_build/html/index.html` in a browser after building.

## Read-the-Docs

RTD is configured via `.readthedocs.yaml` at the repo root.  It installs only
`docs/requirements.txt` (Sphinx + furo + myst-parser + pydantic) — the heavy
runtime dependencies are mocked via `autodoc_mock_imports` in `docs/conf.py`.

`pydantic` must be a real install (not mocked) because `pgml.schemas` subclasses
`pydantic.BaseModel` and Sphinx autodoc resolves the class hierarchy at import
time.

## Structure

```
docs/
  conf.py           Sphinx configuration
  index.md          Landing page + top-level toctree
  install.md        Installation and quick start
  concepts.md       Modelling conventions
  architecture.md   Includes references/ARCHITECTURE.md via MyST
  examples.md       Example scripts in examples/
  api/
    index.md        API reference overview
    schemas.rst     pgml.schemas (frozen contracts)
    assembly.rst    pgml.assembly
    solver.rst      pgml.solver
    geometry.rst    pgml.geometry
    scenarios.rst   pgml.scenarios
    evaluation.rst  pgml.evaluation
    convert.rst     pgml.convert
    config.rst      pgml.config
  requirements.txt  RTD pip requirements
  README.md         This file
```
