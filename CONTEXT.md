# pgml — agent navigation index (read this first)

Differentiable, GPU-ready, vectorized **harmonic power-flow + ML for power grids**.
This file is the map: what each package does, where the contracts live, and which
`CONTEXT.md` to open next. New agents: read this, then the `CONTEXT.md` of the package
you're touching, then the relevant code. Big-picture prose: `references/ARCHITECTURE.md`.
Human-facing overview + diagram + open work: `HANDOFF.md` and `TODO.md`.

## TWO HARD CONSTRAINTS (every line of core code)
1. **DIFFERENTIABLE** — gradients flow `grid params -> Y-bus -> solve -> outputs`. No
   `.item()/.detach()/.numpy()`, no in-place on tracked tensors, no Python control flow
   on tensor values in the differentiable path. (The only sanctioned `.detach()` is the
   IFT adjoint inside `solver/power_flow.py`.)
2. **GPU-READY** — every core op runs on CPU and CUDA unchanged; honor input
   device/dtype; complex dtypes; vectorized/batched (no Python loop over
   nodes/branches/harmonics/scenarios in the tape).
A change that breaks float64 `gradcheck` or the GPU device/dtype test is not done.

## The pipeline (data flow)
```
grid (schemas) ──▶ assembly ──▶ solver ──▶ result        ◀── evaluation (plots vs refs)
      │              ▲   │         ▲                       ◀── scenarios (batched inputs)
      │              │   └─ geometry (Carson Z(h)/Yc(h))   ◀── convert (pandapower/OpenDSS/pgm → grid)
      └─ equations (residual laws, the source of physics) ─┘
```
- A `Grid` (physical params) is assembled into a per-frequency complex nodal admittance
  `Y(f)`; the solver solves `Y(f)·V(f)=I(f)` (linear) or a nonlinear const-P/ZIP fixed
  point (IFT gradients) for `V`; results are phasors. Scenarios batch the inputs;
  evaluation compares to reference libraries.

## Where things live (open the package CONTEXT.md for the interface ledger)
| Need… | Package | CONTEXT |
|---|---|---|
| Input/output **contracts** (Grid, Node, Branch, Appliance, Result, Scenario) — FROZEN | `src/pgml/schemas/` | `schemas/CONTEXT.md` |
| The **physics equations** (residual `0=a-b` registry + torch evaluators; skin/seq laws) | `src/pgml/equations/` | `equations/CONTEXT.md` |
| **Y-bus assembly** (per-phase/per-harmonic/batched stamps; linear + network + injections) | `src/pgml/assembly/` | `assembly/CONTEXT.md` |
| **Solve** (complex batched linear; nonlinear const-P/ZIP via IFT; harmonic flow) | `src/pgml/solver/` | `solver/CONTEXT.md` |
| **Geometry → impedance** (differentiable Carson/Deri + skin; R/X→geometry synthesis) | `src/pgml/geometry/` | `geometry/CONTEXT.md` |
| **Converters** from pandapower / power-grid-model / OpenDSS → our `Grid` | `src/pgml/convert/` | `convert/CONTEXT.md` |
| **Batched scenario sampling** (QMC/cartesian, reproducible; ML training data) | `src/pgml/scenarios/` | `scenarios/CONTEXT.md` + `scenarios/ROADMAP.md` |
| **Evaluation plots** (Y-bus heatmaps, voltage/harmonic profiles, 3D, refs-vs-ours) | `src/pgml/evaluation/` | `evaluation/CONTEXT.md` |
| Reference-library briefs (conventions, gotchas) | `references/` | `references/*/CONTEXT.md`, `references/opendss/{harmonics,carson}.md` |
| Tests (oracle comparisons, differentiability, GPU) | `tests/` | `tests/CONTEXT.md` |

## Frozen-contract rule
`src/pgml/schemas/` (grid/result/scenario) is the single source of truth. Import it;
do NOT edit it as a subagent (orchestrator-only). Everything else conforms to it.

## Key conventions (defined in the schemas; do not reinvent)
- Phase-domain, SI base units; reactive elements store **L and C** (X(h)=2πhf0·L,
  B(h)=2πhf0·C); reactances/susceptances are NEVER stored.
- Every branch → pi-form primitive admittance stamp; transformers add a complex tap.
- Results store phasors as (real, imag); index by `frequency_hz`.
- **Float/tensor duality**: physical schema fields accept python floats OR tensors,
  passed through untouched, so autograd flows through one `assemble_ybus(grid)` call.
- Compact node-phase indexing: one matrix row per existing `(node, phase)`
  (`assembly.node_phase_index`), NOT a padded A/B/C/N grid.

## Commands
- Run code / install: `pixi run -e cpu python ...` / `pixi add <pkg>`.
- Tests: `pixi run -e cpu pytest -q` (165 passing, 9 GPU-skipped as of 2026-06-18).
- Differentiability gate: `pytest -q tests/differentiability`; GPU gate: `tests/gpu`.
- Lint/format: `ruff check src tests && ruff format src tests`.

## Status (2026-06-18)
Phases 0–3 + geometry done & validated: linear + nonlinear (const-P/ZIP) power flow,
harmonic flow, differentiable Carson/Deri geometry (bit-exact vs OpenDSS), scenario
batching (increment 1), evaluation plots. Next big rocks: production batching for
GPU training-data generation, and the PyG **harmonic state-estimation** ML layer
(physics-guided). See `TODO.md` and `HANDOFF.md`.
