# pgml navigation index (read this first)

Differentiable, GPU-ready, vectorized **harmonic power flow for power grids** — the base
package of the power-grid-ml suite. This is the top-level map for contributors and agents:
what the package does, where the contracts live, and which file to open next.

**Read order.** This file → `src/pgml/CONTEXT.md` (the subpackage map + interface ledgers)
→ the subpackage `CONTEXT.md` you are touching → the code. How to *work* here (the hard
constraints, code style, commands, delegation): `CLAUDE.md`. Current status + open work:
`src/pgml/STATUS.md`. The published, human-facing documentation (concepts, modeling
decisions, API reference): `docs/pgml/` (Sphinx; standalone landing `docs/index.md`).

## Two hard constraints (every line of core code)

1. **DIFFERENTIABLE** — gradients flow `grid params → Y-bus → solve → outputs`. No
   `.item()/.detach()/.numpy()`, no in-place on tracked tensors, no Python control flow on
   tensor values in the differentiable path. (The only sanctioned `.detach()` is the IFT
   adjoint in `solver/power_flow.py`.)
2. **GPU-READY** — every core op runs on CPU and CUDA unchanged; honor input device/dtype;
   complex dtypes; vectorized/batched (no Python loop over nodes/branches/harmonics/
   scenarios on the tape).

A change that breaks float64 `gradcheck` or the GPU device/dtype test is not done.

## The suite (one repository per package, one-way dependencies)

`pgml` is the **base** package; the others are one-way dependents that import only `pgml`'s
public API and never its internals. `pgml` imports none of them and knows nothing about
them beyond this table. Every dependent pins a `power-grid-ml` version range and reads
`SCHEMA_VERSION` from persisted datasets — the schemas are the cross-package data contract.

| package | distribution | role | repository |
|---|---|---|---|
| **pgml** | `power-grid-ml` | differentiable, GPU-ready harmonic power flow (the gradient engine) | this one |
| **pgl** | `power-grid-learn` | harmonic state-estimation models + training | `pgl` |
| **pgg** | `power-grid-gen` | QD synthesis of LV grids (CVT-MAP-Elites + differentiable repair) | `pgg` |
| **pghub** | `power-grid-hub` | real grid datasets → `pgml.Grid`, structural metrics, embeddings | `pghub` |
| **pgd** | `power-grid-dash` | FastAPI backend + web SPA over simulation, estimation, live measurements | `pgd` |
| — | `power-grid-suite` | developer aggregation (submodules, one pixi env, cluster jobs) + meta-package | `suite` |
| — | — | the published documentation site, assembled from every package's `docs/<pkg>/` | `docs` |

## Why all-PyTorch

Harmonic power flow decouples per harmonic into a LINEAR complex solve `Y(h)·V(h)=I(h)`. A
linear solve has a clean, cheap adjoint, so end-to-end gradients flow without differentiating
Newton iterations. PyTorch gives complex tensors and autograd, batched `torch.linalg.solve`,
GPU, and native PyTorch-Geometric integration for the ML layer — one autograd tape end to
end. (JAX is a fallback only if PyTorch complex/sparse autograd proves insufficient. Do not
start there.)

## The pgml pipeline (data flow, left → right; gradients flow end-to-end)

1. **schemas** define the `Grid` (physical params, possibly tensors) + the result / scenario
   contracts.
2. **assembly** builds the per-frequency complex nodal admittance `Y(f)` — lines via explicit
   R/L/C or the **geometry** (Carson/Deri) path.
3. **solver** solves `Y(f)·V(f)=I(f)` (linear) or the nonlinear const-P/ZIP problem
   (current-injection fixed point or Newton, IFT gradients) → phasor **result**.
4. **convert** turns pandapower / OpenDSS / power-grid-model nets into a `Grid`; **scenarios**
   batches the inputs (ML training data); **evaluation** compares results to those reference
   libraries.

The subpackage-by-subpackage map, with each interface ledger, is `src/pgml/CONTEXT.md`.

## Where things live

| Need… | Open |
|---|---|
| How to *work* here (constraints, style, commands, the frozen-schema rule) | `CLAUDE.md` |
| The package map (subpackages + interface ledgers) | `src/pgml/CONTEXT.md` |
| Status + open work | `src/pgml/STATUS.md` |
| Published human docs (concepts, modeling decisions, API reference) | `docs/pgml/` (`docs/pgml/index.md`) |
| Modeling decisions (conventions, transformer, line model, DER, asymmetric) | `docs/pgml/modeling/` |
| Cross-tool conventions + reference-library briefs | `docs/pgml/modeling/conventions.md`, `docs/pgml/modeling/references/` |
| Runnable studies + config templates | `run/examples/pgml/`, `run/configs/` |
| **Configuration** (three buckets, see below) | `src/pgml/data/CONTEXT.md`, `run/configs/` |
| Testing conventions + the gates | `tests/CONTEXT.md` |

## Configuration: three buckets

Don't conflate "config". There are three kinds:

1. **Shipped library data** — modeling defaults + standards tables (physical constants),
   read-only, versioned with the code, loaded via `importlib.resources` so they ship in the
   wheel: `pgml.defaults` over `src/pgml/data/` (`defaults.yaml` + `standards/`). *Not*
   user run-config.
2. **Run-config schemas** — serializable pydantic contracts; one config + `seed` reproduces a
   run: `pgml.scenarios.config` (data generation). Inspect with
   `python -m pgml.scenarios.config --json-schema|--example`; templates in `run/configs/`.
   (Dependents follow the same convention: `pgl.config` for training, `pgg.config` for
   generation.)
3. **Run-config instances + outputs** — the user's own YAML + datasets/checkpoints/tracking.
   Never tracked here; live under the **experiments root** (`PGML_EXPERIMENTS`, default
   `./data`; `pgml.experiments_root()`), organised per package (`data/pgml/`, `data/pgl/`, …).

## Frozen-contract rule

`src/pgml/schemas/` (grid/result/scenario) is the single source of truth. Import it; do NOT
edit it as a subagent (orchestrator-only, and only after asking the user). Everything else
— here and in every dependent repository — conforms to it. The full behavioral rule is in
`CLAUDE.md`.

## Key conventions (defined in the schemas; do not reinvent)

- Phase-domain, SI base units; reactive elements store **L and C** (`X(h)=2πhf0·L`,
  `B(h)=2πhf0·C`); reactances/susceptances are NEVER stored.
- Every branch → pi-form primitive admittance stamp; transformers add a complex tap / use
  the winding-incidence vector group.
- Results store phasors as (real, imag); index by `frequency_hz`.
- **Float/tensor duality**: physical schema fields accept python floats OR tensors, passed
  through untouched, so autograd flows through one `assemble_ybus(grid)` call.
- Compact node-phase indexing: one matrix row per existing `(node, phase)`
  (`assembly.node_phase_index`), NOT a padded A/B/C/N grid.
- Voltage base: `Node.u_rated_v` is line-to-line (≥3φ); the per-phase voltages the solver
  uses are line-to-neutral via `assembly._params.phase_voltage_magnitude`. See
  `docs/pgml/modeling/conventions.md`.

## Roadmap

0. Schemas frozen (grid / result / scenario). **Done.**
1. Differentiable load flow: Y-bus assembly + complex solve; validate vs OpenDSS &
   pandapower; gradcheck + GPU. **Done** (linear + nonlinear const-P/ZIP).
2. Geometry → impedance differentiable path (Carson/Deri, skin effect). **Done** — bit-exact
   vs OpenDSS.
3. Full harmonic range; validate harmonic results vs OpenDSS (IEEE-33 + CIGRE LV). **Done.**
4. Batching/scale (`pgml.scenarios`): reproducible QMC/cartesian + correlated + EN 50160 +
   parquet. **Done** — including cross-grid batching (`pgml.multigrid.merge_grids`
   disjoint-union solves) and switch-state batching (`branch_states`); the production
   GPU data-generation scale decision remains open (`src/pgml/STATUS.md` §A).
5. Harmonic state estimation + the inverse (parameter recovery) path — the `pgl`
   repository builds on this package. **In development there.**

## Commands

- Run / install: `pixi run -e cpu python ...` / `pixi add <pkg>`.
- Tests: `pixi run -e cpu pytest -q`. Differentiability gate: `tests/differentiability`;
  GPU gate: `tests/gpu`.
- Lint/format: `ruff check src tests run && ruff format src tests run`. Docs: see `CLAUDE.md`.
