# Documentation sources

Sphinx sources for the `pgml` documentation. This file is excluded from the build and only
explains the layout.

## Building

The `docs` pixi environment carries Sphinx, furo, myst-parser and a CPU torch, which autodoc
needs because the build imports the package.

```bash
pixi run -e docs docs            # build HTML into docs/_build/html
pixi run -e docs docs-clean      # force a full rebuild
pixi run -e docs docs-strict     # warnings as errors, what CI runs
pixi run -e docs docs-linkcheck  # check external links
```

Open `docs/_build/html/index.html` afterwards.

## Layout

```
docs/
  index.md            landing page: what it is, install, first example, links
  conf.py             Sphinx configuration
  requirements.txt    extra build requirements for Read the Docs
  pgml/
    concepts.md          the model in brief
    differentiability.md what the gradients are for, with runnable examples
    public-api.md        the stable entry points
    examples.md          the shipped example scripts and the validation figures
    modeling/            modelling decisions, plus reference-tool briefs
    api/                 generated API reference, one page per subpackage
  _static/figures/    committed SVG figures embedded in the pages
  _build/             output, not tracked
```

The API reference is generated from each subpackage's `__all__`, so the docstrings are the
reference. Keep them valid reStructuredText. `pandapower` is mocked at autodoc time, while
`pydantic` and `torch` are imported for real.

Read the Docs is configured by `.readthedocs.yaml` at the repository root. It builds with
`fail_on_warning`, the same strictness as `docs-strict`.
