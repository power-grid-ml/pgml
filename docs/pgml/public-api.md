# Public API

`pgml` exposes a single stable entry point for most users. The lower-level
subpackages are available for advanced use (raw tensors, scenario batching,
converter pipelines) but the facade below is the recommended starting point.

## Entry points at a glance

| Symbol | Where | Purpose |
|--------|-------|---------|
| `pgml.simulate(grid, config)` | `pgml.simulation` | Run a simulation; return differentiable {class}`~pgml.simulation.SolvedState` |
| `pgml.simulate_serializable(grid, config)` | `pgml.simulation` | As above but return JSON-ready {class}`~pgml.simulation.ResultBundle` |
| `pgml.SimulationConfig` | `pgml.simulation` | Serializable definition of WHAT to simulate |
| `pgml.Grid` | `pgml.schemas` | The input grid (frozen pydantic contract) |
| `pgml.PgmlError` and subclasses | `pgml.errors` | Exception hierarchy with HTTP status hints (`PgmError` is a deprecated alias) |

## `SimulationConfig` vs execution kwargs

`SimulationConfig` is a pydantic model — a clean, JSON-serialisable body for a
REST handler or configuration file.  It carries:

- `calculation` — `"harmonic"` (default) or `"power_flow"`
- `harmonic_orders` — list of harmonic orders (default `[1, 3, 5, 7, 9, 11, 13]`)
- `slack` — `"ideal"` or `"norton"`
- `symmetry` — `None` / `"auto"` / `"symmetric"` / `"asymmetric"` (see {doc}`concepts`)
- `operating_point` — per-appliance P/Q overrides
- `tol`, `max_iter` — convergence tolerances

**Execution concerns** (`device`, `dtype`) are passed directly to {func}`~pgml.simulation.simulate`
as keyword arguments, not stored in the config.  This keeps the config portable and
lets the same spec run on CPU or GPU without modification.

## Quick examples

### Harmonic flow (default)

```python
import pgml
from pgml.schemas import Grid, ...   # build your grid

# Defaults: harmonic, orders [1,3,5,7,9,11,13], complex128, CPU
state = pgml.simulate(grid)

V = state.node_voltages()           # [H, N] complex — on the autograd tape
thd = state.thd(node_id=1, phase=Phase.A)
```

### Power flow only

```python
from pgml import simulate, SimulationConfig

cfg = SimulationConfig(calculation="power_flow")
state = simulate(grid, cfg, dtype="complex128")
v_fund = state.node_voltages()      # [1, N] complex (one harmonic = fundamental)
```

### Serialise for REST / persistence

```python
bundle = pgml.simulate_serializable(grid)
json_body = bundle.model_dump_json()   # standard pydantic JSON export
```

### Custom harmonic orders and GPU execution

```python
cfg = pgml.SimulationConfig(harmonic_orders=[1, 3, 5, 7, 11, 13])
state = pgml.simulate(grid, cfg, device="cuda", dtype="complex64")
```

## Lower-level surfaces

When the high-level facade is not enough:

- {mod}`pgml.solver` — raw `solve_harmonic_flow` / `solve_power_flow` for
  scenarios where you want to manage tensors directly (minimal overhead). Both
  run a pre-solve connectivity check by default (`on_disconnected="raise"`
  raises {class}`~pgml.errors.ConnectivityError`; `"zero"` solves the
  energized sub-grid; `"ignore"` skips it), accept `branch_states` for
  differentiable topology / switch-state batching, and `solve_power_flow`
  accepts `linear_solver="auto"` (sparse on large CPU systems, dense on GPU)
  and a `system=` handle from `prepare_power_flow` to reuse one factorization
  across repeated solves of the same grid. See {doc}`api/solver`.
- {mod}`pgml.scenarios` — `run_scenarios` / `run_node_injection_sweep` for
  reproducible batched training-data generation.
- {mod}`pgml.assembly` — `assemble_ybus` / `device_current_injections` /
  `branch_currents` for per-step Y-bus control.
- {mod}`pgml.convert` — converters from pandapower / power-grid-model / OpenDSS.
- {mod}`pgml.schemas` — frozen pydantic contracts (the single source of truth
  for all data types).

See the {doc}`api/index` for the full API reference.
