# pgml navigation index (read this first)

Differentiable, GPU-ready, vectorized **harmonic power flow for power grids**. This is the
top-level map for contributors: what the package does, where the contracts live, and which
file to open next.

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

## Where pgml sits

pgml is designed as the base layer of a larger power-grid ecosystem: this repository owns
the physics engine and the data contracts (the `Grid` / result / scenario schemas).
Downstream tools — state estimation, grid synthesis, dataset and dashboard applications —
build on pgml's public API only, pin a `power-grid-ml` version range, and read
`SCHEMA_VERSION` from persisted datasets; pgml imports none of them and knows nothing about
their internals beyond that contract.

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
   declares and solves a BATCH of input deltas on one grid and persists it (ML training
   data); **evaluation** compares results to those reference libraries.

The subpackage-by-subpackage map, with each interface ledger, is `src/pgml/CONTEXT.md`.

## GitHub publication landing page

README leads with capabilities, a seven-library comparison, then ONE throughput
figure, the conformance and resistance-recovery
figures before installation. Figure inputs and provenance are in `assets/readme/`;
`run/readme/render.py` redraws every recorded figure without running a solve. Each
figure's measurement date, source hashes and validation
scope accompany its data; the README caption identifies the measured hardware and
precision. The throughput comparison gives every engine the SAME allocation: one
L40S against eight physical CPU cores, on which pgml, pandapower and OpenDSS each
get eight single-threaded worker processes and power-grid-model eight threads.
pgml's single batched CPU call is a second configuration of the same engine and
belongs on the performance page, not in the README figure.
`assets/PERFORMANCE.md` documents the protocol and carries the grid-size,
harmonic, memory and cost figures, the crossover table, and where pgml loses.
Installation instructions target the GitHub repository.

## Where things live

| Need… | Open |
|---|---|
| How to *work* here (constraints, style, commands, the frozen-schema rule) | `CLAUDE.md` |
| The package map (subpackages + interface ledgers) | `src/pgml/CONTEXT.md` |
| Status + open work | `src/pgml/STATUS.md` |
| Published human docs (concepts, modeling decisions, API reference) | `docs/` (landing page `docs/index.md`, pages under `docs/pgml/`) |
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
   (A downstream package that adds its own run-config schema follows the same convention.)
3. **Run-config instances + outputs** — the user's own YAML + datasets/checkpoints/tracking.
   Never tracked here; live under the **experiments root** (`PGML_EXPERIMENTS`, default
   `./data`; `pgml.experiments_root()`), organised per package (`data/pgml/` for this one).

## Frozen-contract rule

`src/pgml/schemas/` (grid/result/scenario) is the single source of truth. Import it; a
schema change needs the maintainer's sign-off (a breaking change to a published data
contract). Everything else — here and in every downstream consumer — conforms to it. The
full behavioral rule is in `CLAUDE.md`.

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
4. Batching/scale (`pgml.scenarios`): a declared batch of input deltas — reproducible
   QMC/cartesian sampling, explicit values (`batch_from_values`), the excitation sweeps, the
   standards-referenced emission, parquet persistence. **Done** — including cross-grid
   batching (`pgml.multigrid.merge_grids` disjoint-union solves) and switch-state batching
   (`branch_states`); the production GPU data-generation scale decision remains open
   (`src/pgml/STATUS.md` §A).
5. Harmonic state estimation and the inverse (parameter recovery) path build on this
   package's public API. **Out of scope for this repository** — including the scenario
   RECIPES a learning task needs (device populations, calibrated emission ranges, load
   profiles). A downstream generator plugs into `run_scenarios` through the `ScenarioSpec`
   protocol (an object with `sample(grid)`), so the engine stays free of them.

## Commands

- Run / install: `pixi run -e cpu python ...` / `pixi add <pkg>`.
- Tests: `pixi run -e cpu pytest -q`. Differentiability gate: `tests/differentiability`;
  GPU gate: `tests/gpu`.
- Lint/format: `ruff check src tests run && ruff format src tests run`. Docs: see `CLAUDE.md`.
