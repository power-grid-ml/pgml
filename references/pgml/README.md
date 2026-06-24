# pgml — power-grid-machine-learning: the differentiable physics base

`pgml` is the foundation of the suite: a single PyTorch library that (1) loads/generates
power grids, (2) simulates **harmonic power quality in steady state** (harmonic power flow,
orders 1–50+), and (3) is **end-to-end differentiable + GPU-ready** so gradients flow from
grid parameters (down to line geometry) through Y-bus assembly and the complex solve to the
outputs. It is simultaneously a forward simulator, a differentiable physics engine for ML
(`pgl`), and an inverse / generation tool (`pgg`).

This is the **base package** the rest of the suite depends on; it depends on nothing in the
suite. The detailed references live alongside this file:

| topic | document |
|---|---|
| Big-picture rationale (why all-PyTorch, the model, the roadmap) | `references/ARCHITECTURE.md` |
| Cross-tool conventions (base voltage L-L/L-N, transformer referral, earth return) | `references/conventions.md` |
| DER / PV / storage modeling | `references/der_pv_storage_modeling.md` |
| Fully-asymmetric modeling | `references/asymmetric_modeling.md` |
| Per-node harmonic "error" source | `references/error_injection.md` |
| Analytic harmonic line model | `references/positive_sequence_harmonic_line_model.md` |
| Reference-library briefs + OpenDSS harmonics/Carson | `references/{opendss,pandapower,power-grid-model}/` |
| Architecture map + package CONTEXTs | root `CONTEXT.md`, `src/pgml/*/CONTEXT.md` |
| Orientation + open work | root `HANDOFF.md` |
| How to work here (constraints, style, commands) | `CLAUDE.md` |

## The public API `pgl` / `pgg` build on (the contract)

`pgl` and `pgg` import **only** these — never `pgml`'s package internals (`pgml.*._*`):

- **Schema (frozen)** — `pgml.schemas`: `Grid` / `Node` / `Branch` / `Appliance`,
  `ResultSet`, `Scenario`. The single source of truth; importers conform, never edit it.
- **Forward (differentiable)** — `pgml.simulate(grid, config) -> SolvedState`,
  `pgml.solver.solve_harmonic_flow(...)`, `pgml.solver.solve_power_flow(...)`. Gradients
  flow grid params → Y(h) → V. The `pgml.errors` hierarchy.
- **Scenarios / data** — `pgml.scenarios`: `ScenarioConfig` / `CoherentSpectrumConfig` /
  `run_scenarios` (batched solve), `write_dataset` / `read_dataset` (parquet I/O).
- **Topology** — `pgml.assembly.node_phase_index` (the `N` row layout), branch parameters,
  and `pgml.evaluation.topology` (networkx graph, distance-from-slack). The PyG `Data`/
  `Batch` builder is a `pgl` concern built on these.
- **Physics residual** — `pgml.equations.registry` (`Y(h)V − I`) for a physics-consistency
  loss term.

## Two hard constraints (every line of `pgml` core code)

1. **DIFFERENTIABLE** — gradients flow `grid params → Y-bus → solve → outputs`; no
   `.item()/.detach()/.numpy()`, in-place on tracked tensors, or Python control flow on
   tensor values on the tape.
2. **GPU-READY** — runs on CPU and CUDA unchanged; honors device/dtype; complex dtypes;
   vectorized/batched.

These constraints are what make `pgl`'s physics-informed losses and `pgg`'s
generation-by-gradient possible. A change that breaks float64 `gradcheck` or the GPU
device/dtype test is not done.
