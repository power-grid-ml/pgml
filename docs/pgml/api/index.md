# API Reference

The `pgml` public API is organised into the following subpackages.  Each
subpackage exports its public surface via `__all__` in its `__init__.py`.

```{toctree}
:maxdepth: 2

simulation
errors
schemas
assembly
solver
geometry
topology
scenarios
evaluation
convert
grids
defaults
paths
```

---

## High-level entry points

| Symbol | Purpose |
|--------|---------|
| {mod}`pgml.simulation` | `simulate`, `simulate_serializable`, `SimulationConfig`, `SolvedState`, `ResultBundle` — the stable public facade |
| {mod}`pgml.errors` | `PgmlError`, `InputError`, `ComputationError`, and leaf exception classes |

## Subpackage overview

| Package | Purpose |
|---------|---------|
| {mod}`pgml.schemas` | Frozen data contracts: `Grid`, `Node`, `Branch`, `ResultSet`, `Scenario` |
| {mod}`pgml.assembly` | Differentiable, batched, per-frequency Y-bus assembly; `branch_currents` / `BranchCurrent` |
| {mod}`pgml.solver` | Complex batched linear solve and nonlinear power-flow |
| {mod}`pgml.geometry` | Differentiable Carson/Deri line constants (geometry → Z(h)/Yc(h)) |
| {mod}`pgml.topology` | Dependency-free slack anchor, branch edges, electrical distance from slack |
| {mod}`pgml.scenarios` | Reproducible config-driven batched scenario sampling |
| {mod}`pgml.evaluation` | Comparison plots (references vs our solve) |
| {mod}`pgml.convert` | Converters from pandapower / power-grid-model / OpenDSS |
| {mod}`pgml.grids` | Reference grid builders (IEEE-33, CIGRE LV) + the state-estimation benchmark recipe |
| {mod}`pgml.defaults` | Documented modelling defaults (shipped in the package, importlib.resources) |
| {mod}`pgml.paths` | Experiments-root convention (`PGML_EXPERIMENTS`) |
