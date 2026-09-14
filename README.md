# pgml

`pgml` simulates the harmonic power flow of an electrical grid in PyTorch. Every result is a
differentiable tensor, so one backward pass returns the gradient of any output with respect to
the grid parameters that produced it, on CPU or GPU.

The engine solves the complex nodal system `Y(h)·V(h) = I(h)` per harmonic order in the phase
domain, with a nonlinear load flow at the fundamental. Grids come from the included benchmark
builders, or from a pandapower, OpenDSS or power-grid-model network.

## Install

The distribution is named `power-grid-ml` and the import name is `pgml`. It is not on PyPI
yet, so install it from the repository:

```bash
pip install "power-grid-ml @ git+https://github.com/power-grid-ml/pgml.git"
```

After the first release the same install becomes `pip install power-grid-ml`, with optional
extras for the parts a minimal install leaves out.

| Extra | Adds | For |
|---|---|---|
| `convert` | pandapower, power-grid-model | reading reference grids into a `Grid` |
| `scenarios` | polars, pyarrow | batched scenario sampling and parquet datasets |
| `viz` | matplotlib, plotly, networkx | the comparison plots in `pgml.evaluation` |
| `opendss` | opendssdirect | the OpenDSS harmonic path |
| `oracles` | pandapower, power-grid-model, opendssdirect | the reference oracles used in validation |

Python 3.13. Development uses [pixi](https://pixi.sh), which sets `PYTHONPATH=src` on
activation so nothing needs installing.

```bash
pixi run -e cpu pytest -q                                  # the test suite
pixi run -e cpu python run/examples/pgml/evaluate_ieee33.py # an example study
pixi run -e docs docs-strict                               # the strict docs build
```

## A first result, and its gradient

```python
import torch
import pgml
from pgml.grids import ieee33_geometry_grid
from pgml.schemas import Phase

grid, _ = ieee33_geometry_grid()               # IEEE 33-bus feeder with converter loads
load = next(a for a in grid.appliances if a.id == 83)
p = torch.tensor(float(load.p_nom_w), dtype=torch.float64, requires_grad=True)
load.p_nom_w = p                               # a grid parameter that carries gradients

state = pgml.simulate(grid)                    # harmonic power flow, orders 1, 3, ... 13
v = state.voltage(node_id=18, phase=Phase.A)   # complex phasor per order [V]
print(f"fundamental   {v.abs()[0]:8.1f} V")
print(f"5th harmonic  {v.abs()[2]:8.1f} V")
print(f"voltage THD   {state.thd(node_id=18, phase=Phase.A):8.2%}")

v.abs()[2].backward()                          # one backward pass
print(f"d|V5|/dP      {1e3 * p.grad:8.3f} V per kW of converter load")
```

```text
fundamental    11562.2 V
5th harmonic     142.5 V
voltage THD      2.10%
d|V5|/dP         0.971 V per kW of converter load
```

Any physical field of a grid accepts a tensor in place of a float, so `p` above is an ordinary
autograd leaf. The documentation shows what that buys: fitting line parameters to noisy
measurements, every load sensitivity of one voltage from a single backward pass, curtailment
that removes an overvoltage, and ranking lines by the benefit of reinforcing them.

## What it covers

- Steady state at integer harmonic orders. A non-integer order is rejected rather than
  approximated.
- Phase domain throughout, with WYE, grounded WYE and DELTA connections and a neutral
  conductor where the grid has one.
- Lines from explicit R/L/C or from conductor geometry through a Carson/Deri model that
  agrees with OpenDSS to 4.8e-8 relative on the same geometry.
- Two-winding transformers with real vector groups, so a Dyn delta traps triplen harmonics.
- Generators as PQ injections or as voltage-regulating terminals with reactive limits, which
  is what makes the MATPOWER transmission benchmarks importable.
- Inverter control laws, storage, shunts and switches, with switch states as a differentiable
  continuous parameter and an ideal closed switch solved by exact bus fusion.
- Two nonlinear power-flow solvers with actionable diagnostics, per-unit convergence criteria
  and a mixed-precision factorization, including a continuation that reports the loadability
  margin, the critical bus and the limiting load.
- Reproducible batched scenarios for data generation, and one implementation that runs on CPU
  and CUDA in complex64 or complex128.

Results are checked against OpenDSS for harmonics and against pandapower and power-grid-model
at the fundamental, on the IEEE 33-bus feeder and the CIGRE LV network. Known modelling gaps
are documented next to the model they affect.

Optional generator/storage harmonic impedances retain measured or specified passive
R/L without deriving damping from signed power. Native OpenDSS impedance laws and
voltage-source initialization are explicit reference choices; unknown impedance stays
absent. Harmonic batches can use exact Woodbury updates for sparse changing shunts,
with a checked direct-solve fallback when the update rank or conditioning is unsuitable.

## Entry points

| You want | Use |
|---|---|
| A full differentiable solved state | `pgml.simulate(grid, config)` |
| The same as JSON, for a service or a file | `pgml.simulate_serializable(...)` |
| Raw differentiable tensors at minimal overhead | `pgml.solver.solve_power_flow`, `solve_harmonic_flow` |
| Batched generation to disk | `pgml.scenarios.run_scenarios(...)` with `write_dataset` |

`SimulationConfig` is the serializable definition of what to simulate, while `device` and
`dtype` are execution keywords on `simulate`. Errors form a small hierarchy under
`PgmlError`, each with an HTTP status hint for a service layer.

## Architecture

```
grid (schemas) ──▶ assembly ──▶ solver ──▶ result        ◀── evaluation (plots vs references)
      │              ▲   │         ▲                     ◀── scenarios (batched inputs)
      │              │   └─ geometry (Carson Z(h), Yc(h)) ◀── convert (pandapower, OpenDSS, pgm)
      └─ defaults (documented modelling defaults) ────────┘
```

- `schemas/` frozen, framework-free contracts. Physical fields accept floats or tensors.
- `assembly/` per-phase, per-harmonic, batched, differentiable admittance and injections.
- `solver/` complex batched linear solve, nonlinear load flow through the implicit function
  theorem, harmonic flow.
- `geometry/` differentiable Carson/Deri line constants and R/X to geometry synthesis.
- `convert/` readers for pandapower, power-grid-model and OpenDSS.
- `scenarios/` reproducible batched sampling and parquet persistence.
- `evaluation/` comparison plots and reference oracles.
- `data/` with `defaults.py` the shipped modelling defaults and standards tables.

## Documentation

The documentation sources are in `docs/`, starting at `docs/index.md`. Build them with

```bash
pixi run -e docs docs
```

and open `docs/_build/html/index.html`.

`pgml` is the engine of a wider set of tools for machine learning on power grids. Those
packages are not part of this release.

## Citing and license

If you use `pgml` in research, please cite it. `CITATION.cff` carries the metadata and GitHub
renders it as a citation button.

Licensed under the Mozilla Public License 2.0. See [LICENSE](LICENSE).
