# pgml — the differentiable physics engine

`pgml` (*power-grid-machine-learning*) is the base package of the suite: a single PyTorch
library that loads or generates power grids, simulates **harmonic power quality in steady
state** (harmonic power flow, orders 1–50+), and is **differentiable + GPU-ready** so
gradients flow from grid parameters — down to line geometry — through Y-bus assembly and the
complex solve to the outputs. It is simultaneously a forward simulator, a differentiable
physics engine for machine learning, and an inverse / parameter-recovery tool.

## Why all-PyTorch

Harmonic power flow decouples per harmonic into a **linear** complex solve
$Y(h)\,V(h) = I(h)$. A linear solve has a clean, cheap adjoint, so end-to-end gradients
flow without differentiating any Newton iteration. PyTorch supplies the rest: complex
tensors and complex autograd, batched `torch.linalg.solve`, GPU execution, and native
PyTorch-Geometric integration for the learning layer — one autograd tape from end to end.

## The model

- **Phase-domain, fully asymmetric.** Each harmonic builds a complex per-phase $Y(h)$ and
  solves for node voltages, from which currents and powers are derived. There is no implicit
  sequence-domain conversion in the core; sequence inputs are decomposed at the converter
  boundary.
- **Norton sources.** Harmonic sources are complex current injections; loads and generators
  are a current source (spectrum) in parallel with a frequency-dependent shunt admittance.
- **Batched** over harmonics, scenarios/steps, and (optionally) leading dimensions, for
  machine-learning-scale data generation.

See [concepts](concepts.md) for the conventions in brief and the
[modeling decisions](modeling/index.md) for the full derivations.

## Validated against oracles

OpenDSS is the harmonic ground truth; pandapower and power-grid-model are
fundamental-frequency load-flow oracles. pgml compares its **assembled Y-bus** to OpenDSS's
exported system Y and its **node voltages / branch flows** to the load-flow oracles, on
small IEEE feeders (IEEE-33, CIGRE LV).

```{figure} ../_static/figures/ybus_heatmaps.svg
:alt: Assembled Y-bus compared to pandapower and OpenDSS
:width: 95%

The IEEE-33 nodal admittance assembled by pgml (left) versus pandapower's network Y
(centre) and OpenDSS's exported system Y (right). They agree to floating-point precision;
the [examples](examples.md) page shows the difference map.
```

The validation and scenario [examples](examples.md) reproduce this and the other comparison
figures.

## The public API other packages build on

`pgl` and `pgg` import **only** the public surface below — never `pgml`'s package internals:

- **Schema (frozen)** — `pgml.schemas`: `Grid` / `Node` / `Branch` / `Appliance`,
  `ResultSet`, `Scenario`. The single source of truth; importers conform to it and never
  edit it.
- **Forward (differentiable)** — `pgml.simulate(grid, config) → SolvedState`,
  `pgml.solver.solve_harmonic_flow`, `pgml.solver.solve_power_flow`; gradients flow grid
  params → $Y(h)$ → $V$. Plus the `pgml.errors` hierarchy.
- **Scenarios / data** — `pgml.scenarios`: `ScenarioConfig` / `CoherentSpectrumConfig` /
  `run_scenarios` (batched solve), `write_dataset` / `read_dataset` (parquet I/O).
- **Topology** — `pgml.assembly.node_phase_index` (the row layout), branch parameters, and
  `pgml.topology` (dependency-free: `slack_node_id`, `branch_edges`,
  `distance_from_slack` — the graph features a training process needs, without pulling in
  matplotlib/plotly/networkx). `pgml.evaluation.topology` adds the one networkx view
  (`grid_graph`) used by the plotting stack. The PyTorch-Geometric `Data` / `Batch` builder
  is a `pgl` concern built on these. `pgml.topology.connectivity_report` /
  `energized_subgrid` back the pre-solve connectivity check (`pgml.errors.ConnectivityError`)
  and the solver's `on_disconnected` handling.
- **Reference grids** — `pgml.grids`: the canonical IEEE-33 / CIGRE LV benchmark builders
  (pandapower → `Grid`, with synthesized Carson geometry and converter harmonic spectra),
  plus `add_pv_systems` and `se_benchmark_scenario_config` for the state-estimation
  benchmark recipe (requires the `convert` extra, pandapower), and
  `synthetic_feeder` — a schema-only synthetic radial MV feeder of any size, with no
  external dependency, for solver-scaling and switch-state-batching studies.

## Two hard constraints

Every line of pgml core code satisfies two non-negotiable properties:

1. **Differentiable** — gradients flow `grid params → Y-bus → solve → outputs`. No
   `.item()` / `.detach()` / `.numpy()`, no in-place operations on tracked tensors, no
   Python control flow on tensor values on the tape.
2. **GPU-ready** — every core operation runs on CPU and CUDA unchanged, honors the input
   device/dtype, uses complex dtypes, and is vectorized/batched.

A change that breaks the `float64` `gradcheck` or the GPU device/dtype test is not done.
These constraints are what make `pgl`'s physics-informed losses and `pgg`'s
generation-by-gradient possible.
