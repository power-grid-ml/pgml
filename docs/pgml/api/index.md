# API reference

Which module to open for what. Each page is generated from that subpackage's `__all__`, so
what you see here is the public surface.

| Module | Open it for |
|---|---|
| {mod}`pgml.simulation` | `simulate`, `SimulationConfig`, `SolvedState`, the facade most code uses |
| {mod}`pgml.schemas` | `Grid`, `Node`, `Branch`, `Appliance`, results and scenarios, the data contracts |
| {mod}`pgml.solver` | `solve_harmonic_flow`, `solve_power_flow`, loadability, diagnostics, equilibration, prepared systems, switch-state batching |
| {mod}`pgml.assembly` | `assemble_ybus`, device injections, branch currents, the node-phase row layout, the bus-fusion map |
| {mod}`pgml.geometry` | Carson/Deri line constants from conductor geometry, and the analytic harmonic line models |
| {mod}`pgml.topology` | Slack anchors, branch edges, distance from the slack, connectivity reports, grid fingerprints |
| {mod}`pgml.scenarios` | Batched sampling of operating points, sweeps, the batched solve, parquet datasets |
| {mod}`pgml.convert` | Readers for pandapower, power-grid-model and OpenDSS networks |
| {mod}`pgml.grids` | Benchmark builders and the synthetic feeder |
| {mod}`pgml.multigrid` | Merge a grid ensemble into one solvable grid, then split the result per member |
| {mod}`pgml.dispatch` | State-of-charge integration and the realized power sequence a `Storage` element describes |
| {mod}`pgml.evaluation` | Comparison plots and the reference oracles |
| {mod}`pgml.errors` | The exception hierarchy and its HTTP status hints |
| {mod}`pgml.defaults` | The shipped modelling defaults and standards tables |
| {mod}`pgml.paths` | Where run outputs go |
| {mod}`pgml.provenance` | The commit and version stamp to write beside a generated artifact |

```{toctree}
:maxdepth: 2

simulation
schemas
solver
assembly
geometry
topology
scenarios
convert
grids
multigrid
dispatch
evaluation
errors
defaults
paths
provenance
```
