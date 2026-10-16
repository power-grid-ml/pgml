# pgml architecture

Differentiable, GPU-ready, vectorized harmonic power flow. The harmonic system
`Y(h)·V(h) = I(h)` is the core; the fundamental load flow is one differentiable output of it.
This file is the map of the repository: where things are, the rules that hold everywhere, and
which file to open next. The published documentation (concepts, modelling decisions, API
reference) is under `docs/`.

Signatures: docstrings / `API_INDEX.md` (generated from `__all__` by `tools/api_index.py`;
never edit by hand).

## Two hard constraints

1. **Differentiable.** Gradients flow `grid parameters → Y-bus → solve → outputs`. No
   `.item()`, `.detach()`, `.numpy()`, no in-place ops on tracked tensors and no Python control
   flow on tensor values in the differentiable path. The one sanctioned `.detach()` is the IFT
   adjoint in `solver/power_flow.py`.
2. **GPU-ready.** Every core op runs on CPU and CUDA unchanged, honours the input device and
   dtype, supports complex64/complex128 and is batched: no Python loop over nodes, branches,
   harmonics or scenarios on the tape.

A change that breaks the float64 gradcheck (`tests/differentiability`) or the device/dtype
parity test (`tests/gpu`) is not done.

## Where pgml sits

pgml is the base layer: it owns the physics engine and the data contracts (the `Grid`, result
and scenario schemas). It imports no other package of the suite, and knows nothing about its
consumers beyond the public API, the `power-grid-ml` version range they pin, and the
`SCHEMA_VERSION` stamped into persisted datasets. State estimation, the scenario recipes a
learning task needs (device populations, calibrated emission ranges, load profiles) and grid
synthesis are out of scope here; a downstream generator plugs into `run_scenarios` through the
`ScenarioSpec` protocol (an object with `sample(grid)`).

## Why all-PyTorch

Per harmonic order the system is a linear complex solve, whose adjoint is cheap, and the
nonlinear fundamental is differentiated through the implicit-function theorem instead of
through the Newton iterations. PyTorch provides complex tensors, autograd, batched
factorizations and CUDA, so one autograd tape runs from the grid parameters to a downstream
loss.

## Pipeline

```
grid (schemas) ──▶ assembly ──▶ solver ──▶ result        ◀── evaluation (plots, oracles)
      │              ▲   │         ▲                       ◀── scenarios (batched inputs)
      │              │   └─ geometry (Carson Z(h), Yc(h))  ◀── convert (pandapower, OpenDSS, pgm)
      └─ defaults (modelling defaults, shipped data/ tables) ┘
```

The subpackage map is `src/pgml/CONTEXT.md`; each subpackage has its own `CONTEXT.md`.

## Codemap

- `src/pgml/` — the library (`src/pgml/CONTEXT.md`).
- `tests/` — tests and the gates (`tests/CONTEXT.md`).
- `docs/` — standalone Sphinx site (`docs/index.md`, pages under `docs/pgml/`, API reference
  from the docstrings via autodoc over `__all__`; `.readthedocs.yaml` builds it).
  `docs/pgml/modeling/` holds the modelling decisions, `limitations.md` what is not modelled.
- `run/examples/pgml/` — runnable studies and benchmarks (each self-documenting);
  `run/configs/` — config templates; `run/readme/render.py` redraws the README figures from
  `assets/readme/` without solving.
- `assets/` — `PERFORMANCE.md` (results), `PERFORMANCE_RECORD.md` (measurement record),
  `CONFORMANCE.md`, and the README figure data with its provenance.
- `tools/api_index.py` — writes and checks `API_INDEX.md` (`pixi run -e cpu api-index`,
  `api-index-check`).
- `.githooks/pre-commit` — ruff on staged Python, API index check when `src/` changes
  (`git config core.hooksPath .githooks`). `.github/workflows/ci.yml` — lint, API index,
  tests, strict docs, package build.

## Configuration: three buckets

1. **Shipped library data**: modelling defaults and standards tables, read-only, versioned with
   the code and loaded through `importlib.resources`: `pgml.defaults` over `src/pgml/data/`.
   Not user run-config.
2. **Run-config schemas**: serializable pydantic contracts, one config plus `seed` reproduces a
   run: `pgml.scenarios.config` (`python -m pgml.scenarios.config --json-schema|--example`;
   templates in `run/configs/`).
3. **Run-config instances and outputs**: the user's YAML, datasets and tracking. Never tracked;
   they live under the experiments root (`PGML_EXPERIMENTS`, default `./data`,
   `pgml.experiments_root()`), one folder per package.

## Frozen contract

`src/pgml/schemas/` (grid, result, scenario) is the single source of truth for every package
of the suite. Everything conforms to it. A schema change needs the maintainer's sign-off and a
`SCHEMA_VERSION` bump, and a public API or schema change needs a search of all consumers.

## Conventions (defined in the schemas; do not reinvent)

- Phase domain, SI units. Reactive elements store `L` and `C` (`X(h) = 2πh·f0·L`,
  `B(h) = 2πh·f0·C`); reactances and susceptances are never stored.
- Every branch is a pi-form primitive admittance stamp; transformers use the
  winding-incidence vector group with a complex tap.
- Results store phasors as (real, imag) and index by `frequency_hz`.
- Float/tensor duality: physical schema fields accept floats or tensors and pass them through
  untouched, so autograd flows through one `assemble_ybus(grid)` call.
- Compact node-phase indexing: one matrix row per existing `(node, phase)`
  (`assembly.node_phase_index`), not a padded A/B/C/N layout.
- `Node.u_rated_v` is line-to-line for three-phase nodes; the solver's per-phase voltages are
  line-to-neutral (`assembly._params.phase_voltage_magnitude`). The cross-tool record is
  `docs/pgml/modeling/conventions.md`.
- Reference-tool divergence: add an option and state the physics verdict; change a default
  only when pgml lacks the option.

## Version

`src/pgml/__init__.py` holds `__version__`; `pixi.toml` and `CITATION.cff` mirror it.
Dependents pin `power-grid-ml>=X.Y,<X.Y+1`.

## Commands

- Run: `pixi run -e cpu python …`; add a dependency with `pixi add`.
- Tests: `pixi run -e cpu pytest -q` (quick: `-m "not slow"`); gates `tests/differentiability`
  and `tests/gpu` (CUDA host).
- Lint: `pixi run -e cpu ruff check src tests run` and `ruff format`.
- Docs: `pixi run -e docs docs-strict`.
- API index: `pixi run -e cpu api-index` (write), `api-index-check` (CI).
