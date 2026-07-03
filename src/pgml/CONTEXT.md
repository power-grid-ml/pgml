# pgml — package map (interface ledger index)

The differentiable, GPU-ready harmonic power-flow engine — the base of the suite. This file
maps pgml's subpackages and points at each one's `CONTEXT.md` interface ledger. For the
suite-level picture (the other packages, the rationale, the roadmap) see the root
`CONTEXT.md`; for how to work here (the two hard constraints, style, commands) see `CLAUDE.md`;
for status + open work see `STATUS.md`; for the published human docs see `docs/pgml/`.

## The pipeline (gradients flow end-to-end, left → right)

```
grid (schemas) ──▶ assembly ──▶ solver ──▶ result        ◀── evaluation (plots vs refs)
      │              ▲   │         ▲                       ◀── scenarios (batched inputs)
      │              │   └─ geometry (Carson Z(h)/Yc(h))   ◀── convert (pandapower/OpenDSS/pgm)
      └─ defaults (modeling defaults; shipped `data/` tables) ┘
```

## Where things live (open the subpackage CONTEXT.md for the interface ledger)

| Need… | Subpackage | CONTEXT |
|---|---|---|
| Input/output **contracts** (Grid, Node, Branch, Appliance, Result, Scenario) — FROZEN | `schemas/` | `schemas/CONTEXT.md` |
| **Modeling defaults** (documented values + model choices; explicit > defaults > converter) — internal, not user run-config | `defaults.py` (loader) + shipped `data/` | `data/CONTEXT.md` |
| **Run-config schemas** (serializable; one config + seed reproduces a run) — user-facing | `scenarios/config.py` (data gen) | `scenarios/CONTEXT.md` |
| **Experiments root** (where run outputs + config instances live; `PGML_EXPERIMENTS`, default `./experiments`) | `paths.py` | `experiments/README.md` |
| **Y-bus assembly** (per-phase/per-harmonic/batched stamps; network + device injections; node-phase index; branch currents; vector-group transformer; control laws) | `assembly/` | `assembly/CONTEXT.md` |
| **Solve** (complex batched linear; nonlinear const-P/ZIP via IFT; Newton; harmonic flow; diagnostics + loadability) | `solver/` | `solver/CONTEXT.md` |
| **Geometry → impedance** (differentiable Carson/Deri + skin; R/X → geometry synthesis; sequence-aware harmonic line models) | `geometry/` | `geometry/CONTEXT.md` |
| **Converters** from pandapower / power-grid-model / OpenDSS → our `Grid` | `convert/` | `convert/CONTEXT.md` |
| **Batched scenario sampling** (QMC/cartesian, reproducible; parquet I/O; ML training data) | `scenarios/` | `scenarios/CONTEXT.md` |
| **Topology bookkeeping** (slack anchor, branch edges, distance-from-slack Dijkstra) — stdlib-only, safe for lean training imports | `topology.py` | — |
| **Benchmark input grids** (IEEE-33 / CIGRE LV via pandapower; the canonical example/training feeders) | `grids.py` | — |
| **Evaluation plots** (Y-bus heatmaps, voltage/harmonic profiles, 3D, refs-vs-ours) + the optional reference oracles | `evaluation/` | `evaluation/CONTEXT.md` |
| The **public API** (`simulate`, `SolvedState`, `SimulationConfig`, `ResultBundle`) | `simulation.py` | `docs/pgml/public-api.md` |
| The **error hierarchy** (`PgmError` → `InputError` / `ComputationError`, http_status hints) | `errors.py` | — |

## Frozen-contract rule

`schemas/` (grid/result/scenario) is the single source of truth — import and conform; never
edit it as a subagent (orchestrator-only, ask the user first). `SCHEMA_VERSION` stamps the
contract version into persisted datasets. Full rule: `CLAUDE.md`.

## Key conventions (do not reinvent)

- Phase-domain, SI; store **L/C** not X/B (`X(h)=2πhf0·L`, `B(h)=2πhf0·C`); phasors as
  (real, imag); index by `frequency_hz`.
- **Float/tensor duality** — physical schema fields accept python floats OR tensors, passed
  through untouched, so autograd flows through one `assemble_ybus(grid)` call.
- Compact node-phase indexing (`assembly.node_phase_index`): one row per existing
  `(node, phase)`, not a padded A/B/C/N grid.
- `Node.u_rated_v` is line-to-line (≥3φ); the solver's per-phase voltages are line-to-neutral
  via `assembly._params.phase_voltage_magnitude`. The full cross-tool convention record is
  `docs/pgml/modeling/conventions.md`.

## Tests

Oracle comparisons (`tests/reference`), the differentiability gate (`tests/differentiability`,
float64 gradcheck), and the GPU device/dtype gate (`tests/gpu`). See `tests/CONTEXT.md`.
