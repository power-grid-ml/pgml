# Public API

One function covers most use. `pgml.simulate` takes a grid and a configuration and returns a
differentiable `SolvedState`. The subpackages underneath are public too, for raw tensors,
batched sampling and converters.

| Symbol | Module | Purpose |
|---|---|---|
| `pgml.simulate(grid, config)` | `pgml.simulation` | Solve, return a differentiable {class}`~pgml.simulation.SolvedState` |
| `pgml.simulate_serializable(grid, config)` | `pgml.simulation` | Solve, return a JSON-ready {class}`~pgml.simulation.ResultBundle` |
| `pgml.SimulationConfig` | `pgml.simulation` | Serializable definition of what to simulate |
| `pgml.Grid` | `pgml.schemas` | The input grid, a frozen pydantic contract |
| `pgml.PgmlError` and subclasses | `pgml.errors` | Exception hierarchy with HTTP status hints |

## What to simulate, and how to run it

`SimulationConfig` is a pydantic model, so it serialises into a configuration file or a REST
request body. It carries the physics of the study.

| Field | Meaning |
|---|---|
| `calculation` | `"harmonic"` by default, or `"power_flow"` |
| `harmonic_orders` | Integer orders, default `[1, 3, 5, 7, 9, 11, 13]` |
| `slack` | `"ideal"` or `"norton"` |
| `symmetry` | `None`, `"auto"`, `"symmetric"` or `"asymmetric"`, see {doc}`concepts` |
| `load_shunt` | The harmonic device Norton shunt, `"none"`, `"opendss"` or `"motor"`; `None` resolves the documented default |
| `operating_point` | Per-appliance P and Q overrides |
| `enforce_q_limits` | Whether a voltage-regulating generator's reactive limits bound its output |
| `tol`, `tol_update_pu`, `s_base_va`, `max_iter` | Convergence control for the nonlinear solve |

`tol` is the primary convergence tolerance: the largest nodal apparent-power mismatch in per
unit of `s_base_va`, default 1e-8 pu on a 1e6 VA base. `tol_update_pu` is the secondary one,
the largest per-row voltage update in per unit of the node's line-to-neutral rated voltage,
also 1e-8 pu. Both criteria must hold. Each is capped by what the working precision can
resolve, and a tighter request logs a warning naming the floor, which then governs.
`load_shunt` lives in the config because it is a modelling choice.

Execution concerns stay out of the config and are keyword arguments of
{func}`~pgml.simulation.simulate`. `device` and `dtype` are the two common ones, so the same
stored config runs on a CPU or a GPU unchanged.

`precision` is an execution keyword as well. `"full"` solves at `dtype`; `"mixed"` factors at
complex64 and refines against complex128 residuals, which keeps complex128 accuracy at
single-precision factorization cost and requires `dtype="complex128"`.

`equilibrate` is the diagonal equilibration of every linear system the solve factors: `None`
for the documented default `solver.equilibration.mode`, or `"symmetric"`, `"row_column"` or
`"off"`. It is an execution keyword rather than a config field because it changes the
conditioning of the factorizations and not the model. The result, its units and its gradients
are the same either way. See {doc}`modeling/solver-performance`.

`param_overrides` is another keyword rather than a config field, because it carries live
tensors instead of serialisable values. It maps
`(component_kind, element_id, field_name)` to a tensor and substitutes individual grid
parameters for a parameter-recovery loop. It applies to `calculation="power_flow"` and
raises for `"harmonic"` rather than being ignored. The returned state's lazy branch
accessors reuse the same overrides, so voltages and currents always describe one network.

`on_disconnected` decides what happens when part of the grid has no galvanic path to a
source, after an open switch or an out-of-service line. It applies to both calculations.

- `"raise"`, the default. {class}`~pgml.errors.ConnectivityError` names the de-energized
  nodes, the separating branches and the available fixes.
- `"zero"`. Solve the energized sub-grid and report exactly 0 V on the de-energized rows at
  every order, keeping the full row layout so every id stays addressable. This is what an
  operational tool wants. {meth}`~pgml.simulation.SolvedState.thd` is undefined on such a
  row.
- `"ignore"`. Skip the check. A de-energized area then shows up as a singular factorization
  or as non-convergence.

## Reading a result

`SolvedState` holds the solved voltages eagerly and derives everything else on demand. All
accessors return tensors on the autograd tape.

| Accessor | Returns |
|---|---|
| `node_voltages()` | `[*batch, H, N]` complex node-phase voltages |
| `voltage(node_id, phase)` | `[*batch, H]` phasor at one node and phase |
| `spectrum_at(node_id, phase)` | The same values read as a spectrum |
| `thd(node_id, phase)` | Voltage THD, needs order 1 among the solved orders |
| `branch_currents()` | Per-branch terminal currents |
| `branch_flows()` | Per-branch complex power at each terminal |
| `to_result_set()` | A serializable `ResultBundle`, detached from the tape |
| `fusion` | The bus-fusion map, when the grid had a zero-impedance branch |

Voltages and the row index are always the grid's own layout, so nothing downstream changes when
a grid contains an ideal switch or coupler. `SolvedState.fusion` records which node-phases were
solved as one row, and it is what lets `branch_currents()` report the current through a fused
branch. It is `None` on a grid without such a branch. See the ideal-branch section of
{doc}`modeling/conventions`.

## Four calls

```python
import pgml

state = pgml.simulate(grid)                         # harmonic, orders 1..13, complex128, CPU
v = state.node_voltages()                           # [H, N] complex, on the tape
```

```python
cfg = pgml.SimulationConfig(calculation="power_flow")
state = pgml.simulate(grid, cfg)                    # fundamental load flow only
```

```python
bundle = pgml.simulate_serializable(grid)
body = bundle.model_dump_json()                     # REST response or file
```

```python
cfg = pgml.SimulationConfig(harmonic_orders=[1, 3, 5, 7, 11, 13])
state = pgml.simulate(grid, cfg, device="cuda", dtype="complex64")
```

## Underneath the facade

- {mod}`pgml.solver`. `solve_harmonic_flow` and `solve_power_flow` for direct tensor work.
  Both run the connectivity and zero-impedance checks, accept `branch_states` for
  differentiable switch states and switch-state batching, and take the same factorization
  options: `linear_solver` for the sparse, dense and block backends, `block_rows`,
  `equilibrate`, `criticality` and `branch_states_method`. `solve_power_flow` additionally
  takes a prepared `system=` handle that reuses one factorization across solves, and
  `enforce_q_limits` for a voltage-regulating generator's reactive bounds.
  `loadability_limit` walks the loading parameter to the largest solvable value. See
  {doc}`api/solver`.
- {mod}`pgml.scenarios`. Reproducible batched sampling of operating points, parameter and
  injection sweeps, a batched solve through `run_scenarios`, and parquet persistence with
  `write_dataset` and `read_dataset`.
- {mod}`pgml.assembly`. `assemble_ybus`, `device_current_injections`, `branch_currents` and
  the node-phase index, for control over a single step.
- {mod}`pgml.geometry`. Carson/Deri line constants from conductor geometry, and the analytic
  harmonic line models for lines given as R and X.
- {mod}`pgml.convert`. Readers for pandapower, power-grid-model and OpenDSS networks.
- {mod}`pgml.grids`. The benchmark builders used throughout these pages, plus
  `synthetic_feeder` for a radial feeder of any size with no external dependency.
- {mod}`pgml.multigrid`. Merge an ensemble of grids into one solvable grid with a
  block-diagonal admittance, then split the solved state back per member.
- {mod}`pgml.dispatch`. State-of-charge integration and the realized power sequence a
  `Storage` element's energy-state fields describe. The dispatch rule is the caller's ordinary
  Python; `integrate_soc` realizes a requested power sequence under the reserve, the capacity
  and the power rating, so a gradient with respect to the setpoint value is available.
- {mod}`pgml.schemas`. The frozen data contracts.

Errors form a small hierarchy. `PgmlError` splits into `InputError` and `ComputationError`,
with leaves such as `ConvergenceError` and `ConnectivityError`, each carrying an
`http_status` hint for a service layer. Schema validation keeps raising pydantic's own
`ValidationError`.

The full generated reference is {doc}`api/index`.
