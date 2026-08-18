# pgml — power-grid-ml

**Differentiable, GPU-ready, vectorized harmonic power flow for power grids.**

`pgml` is a PyTorch library that (1) loads or builds power grids, (2) simulates **harmonic
power quality in steady state** (harmonic power flow at integer harmonic orders, typically
1–50), and (3) exposes every result as a differentiable tensor, so the same code is a
forward simulator, a differentiable physics engine for machine learning, and an inverse /
parameter-recovery tool. The defining requirement is **end-to-end differentiability**:
gradients flow from grid parameters — down to line geometry — through Y-bus assembly and
the complex solve to the outputs.

Harmonic power flow decouples per harmonic into a *linear* complex solve
`Y(h)·V(h) = I(h)`, whose adjoint is cheap — so end-to-end gradients flow without
differentiating Newton iterations. PyTorch provides complex tensors, batched solves, GPU,
and native PyTorch-Geometric integration for the ML layer.

`pgml` is the **base package of the power-grid-ml suite**; the learning framework
(`pgl` / `power-grid-learn`), the grid generator (`pgg` / `power-grid-gen`), the dataset hub
(`pghub` / `power-grid-hub`) and the dashboard (`pgd` / `power-grid-dash`) live in sibling
repositories under the same organization and build on this package's public API. The
combined documentation is published from the org `docs` repository; the `suite` repository
aggregates everything for development.

## Highlights

- **Phase-domain, fully asymmetric.** Per-phase, per-harmonic complex Y-bus; WYE / DELTA
  / grounded-neutral loads; two-winding transformers with real **vector groups** (a Dyn
  delta correctly traps zero-sequence / triplen harmonics).
- **Differentiable + GPU.** Every core op runs on CPU and CUDA unchanged, honors input
  device/dtype, and is vectorized/batched (no Python loops over nodes/branches/harmonics).
  `float64` `gradcheck` and a CPU/CUDA parity suite are gates.
- **Carson/Deri line model.** Conductor geometry → frequency-dependent `Z(h)`/`Yc(h)`,
  **bit-exact vs OpenDSS**, with R/X → geometry synthesis for sequence-defined feeders.
- **Validated against oracles.** Y-bus and voltages compared to pandapower /
  power-grid-model (load flow) and OpenDSS (harmonics) on IEEE-33 and CIGRE LV.
- **Two power-flow solvers + diagnostics.** Current-injection fixed point and Newton
  (linear const-Z warm start; converges near the loadability nose), both with IFT
  gradients. Non-convergence is actionable: `ConvergenceDiagnostics` and a
  `loadability_limit` continuation that reports the margin, the critical bus, and the
  limiting load.
- **Reproducible batched scenarios.** QMC / cartesian sampling → operating points →
  batched solves, for ML training-data generation.

## Installation

```bash
pip install power-grid-ml                 # core differentiable engine (import name: pgml)
pip install "power-grid-ml[convert,viz]"  # + reference-library converters and plotting
pip install "power-grid-ml[all]"          # everything except the docs/dev tooling
```

| Extra | Adds | For |
|-------|------|-----|
| `convert` | pandapower, power-grid-model | converting reference grids into a `Grid` |
| `scenarios` | polars, pyarrow | batched scenario sampling + parquet datasets |
| `viz` | matplotlib, plotly, networkx | the `pgml.evaluation` comparison plots |
| `opendss` | opendssdirect | the OpenDSS harmonic path / oracle |
| `oracles` | pandapower, power-grid-model, opendss | the reference oracles used in validation |
| `docs` / `dev` | sphinx stack / pytest, ruff | building the docs, running the tests |

The distribution is named `power-grid-ml` because `pgml` is taken on PyPI by an unrelated
project; the import name stays `pgml`. Python 3.13.

Development uses [pixi](https://pixi.sh) (conda-based, pinned environments; `PYTHONPATH=src`
is set on activation, so nothing needs installing):

```bash
pixi run -e cpu pytest -q          # run the test suite (CPU)
pixi run -e cpu python run/examples/pgml/evaluate_ieee33.py
pixi run -e docs docs-strict       # the strict Sphinx build (mirrors CI)
```

## Quickstart

```python
import pgml
from pgml.convert.pandapower import to_grid
from pgml.schemas import Phase
from pgml.geometry import apply_default_harmonic_model
import pandapower.networks as pn

grid, _ = to_grid(pn.create_cigre_network_lv())   # reference net -> pgml Grid
apply_default_harmonic_model(grid)                 # frequency-dependent line model

config = pgml.SimulationConfig(calculation="harmonic", harmonic_orders=[1, 3, 5, 7])
state = pgml.simulate(grid, config)                # -> a differentiable SolvedState

v = state.node_voltages()              # complex [orders, nodes]; gradients flow to params
currents = state.branch_currents()     # per-branch terminal currents (lazy, differentiable)
thd = state.thd(node_id=1, phase=Phase.A)

bundle = state.to_result_set()         # JSON-serializable ResultBundle (REST / dashboard)
bundle.model_dump_json()
```

### Entry points (which one to use)

| You want… | Use |
|---|---|
| One front door: full, differentiable **solved grid state** | `pgml.simulate(grid, config) -> SolvedState` |
| The same, but **JSON** out (REST / dashboard / persist) | `pgml.simulate_serializable(...) -> ResultBundle` |
| Raw differentiable **tensors** at minimal overhead (ML) | `pgml.solver.solve_power_flow` / `solve_harmonic_flow` |
| **Batched** training-data generation → filesystem | `pgml.scenarios.run_scenarios(...)` + `write_dataset` |

`SimulationConfig` is the serializable *definition* of what to simulate; `device`/`dtype`
are execution kwargs on `simulate`. Errors form a small hierarchy
(`pgml.PgmError` → `InputError` / `ComputationError`, e.g. `ConvergenceError`), each with
an `http_status` hint for a REST layer; schema-validation errors stay as pydantic
`ValidationError`.

## Architecture

```
grid (schemas) ──▶ assembly ──▶ solver ──▶ result        ◀── evaluation (plots vs refs)
      │              ▲   │         ▲                       ◀── scenarios (batched inputs)
      │              │   └─ geometry (Carson Z(h)/Yc(h))   ◀── convert (pandapower/OpenDSS/pgm)
      └─ config (documented modeling defaults) ────────────┘
```

- **`schemas/`** — frozen, framework-free contracts (`Grid`, `Node`, `Branch`,
  `Appliance`, `Result`, `Scenario`). Physical fields accept plain floats *or* tensors.
- **`assembly/`** — per-phase, per-harmonic, batched, differentiable Y-bus + injections.
- **`solver/`** — complex batched linear solve; nonlinear const-P/ZIP via the
  implicit-function theorem; harmonic flow.
- **`geometry/`** — differentiable Carson/Deri line constants + R/X → geometry synthesis.
- **`convert/`** — pandapower / power-grid-model / OpenDSS → `Grid`.
- **`scenarios/`** — reproducible config-driven batched sampling.
- **`evaluation/`** — comparison plots and reference oracles.
- **`data/` + `defaults.py`** — documented modeling defaults and standards tables (single
  source of truth; `defaults.yaml`, shipped in the wheel).

Each subpackage has a `CONTEXT.md` interface ledger; the package map is the root
`CONTEXT.md`; status and open work live in `src/pgml/STATUS.md`.

## Documentation

The full, human-facing documentation — concepts, modeling decisions, examples, and the API
reference — is published with Sphinx / Read-the-Docs as part of the suite site
(<https://power-grid-ml.readthedocs.io>). This repository holds the pgml pages under
`docs/pgml/` (start at `docs/pgml/index.md`); build them locally with

```bash
pixi run -e docs docs            # build HTML into docs/_build/html
```

Contributor orientation: `CONTEXT.md` (the package map), `src/pgml/STATUS.md` (status +
open work), each subpackage's `CONTEXT.md` (interface ledger), `tests/CONTEXT.md` (the
gates). If you use pgml in research, please cite it (`CITATION.cff`).

## License

See [LICENSE](LICENSE).
