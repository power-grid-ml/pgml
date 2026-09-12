# pgml

`pgml` simulates the harmonic power flow of an electrical grid in PyTorch. Every result is a
differentiable tensor, so one backward pass returns the gradient of any output with respect
to the grid parameters that produced it, on CPU or GPU.

The engine solves the complex nodal system $Y(h)\,V(h) = I(h)$ for each harmonic order in
the phase domain, with a nonlinear load flow at the fundamental. Grids come from the
included benchmark builders, or from a pandapower, OpenDSS or power-grid-model network.

## Install

The distribution is named `power-grid-ml` and the import name is `pgml`. It is not on PyPI
yet, so install it from the repository:

```bash
pip install "power-grid-ml @ git+https://github.com/power-grid-ml/pgml.git"
```

After the first release the same two lines become a plain install, the second one adding the
readers for pandapower, power-grid-model and OpenDSS files:

```bash
pip install power-grid-ml
pip install "power-grid-ml[convert]"
```

Python 3.13. The benchmark grid used below needs the `convert` extra.

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

Any physical field of a grid accepts a tensor in place of a float, so `p` above is an
ordinary autograd leaf. The gradient travels back through the branch currents, the solve and
the assembled admittance to that one load parameter.

## Where to go next

- {doc}`pgml/differentiability` shows what the gradients are good for: fitting line
  parameters to measurements, sensitivities of a voltage to every load at once, curtailment
  that removes an overvoltage, and ranking lines by the benefit of reinforcing them.
- {doc}`pgml/concepts` explains the model: phase domain, SI units, Norton harmonic sources,
  and how a grid becomes an admittance matrix.
- {doc}`pgml/public-api` is the stable front door, `simulate` and what it returns.
- {doc}`pgml/modeling/index` records the modelling decisions and their limits, from
  transformer vector groups to the frequency dependence of a line.
- {doc}`pgml/examples` lists the runnable example scripts and the validation figures.
- {doc}`pgml/api/index` is the generated API reference.

## Scope

- Steady state at integer harmonic orders. A non-integer order is rejected rather than
  approximated, so interharmonics, flicker and transients are out of scope.
- Phase domain throughout, per phase and per harmonic, with WYE, grounded WYE and DELTA
  connections and a neutral conductor where the grid has one.
- Lines from explicit R/L/C or from conductor geometry (Carson/Deri), two-winding
  transformers with real vector groups, loads, generators as PQ injections or as
  voltage-regulating terminals with reactive limits, inverter control laws, storage, shunts
  and switches.
- One implementation runs on CPU and CUDA, in complex64 or complex128, batched over
  harmonics and over scenarios.
- Results are checked against OpenDSS for harmonics and against pandapower and
  power-grid-model at the fundamental, on the IEEE 33-bus feeder, the CIGRE LV network and
  the MATPOWER transmission benchmarks.
- Known gaps are listed where they belong, next to the model they affect, in
  {doc}`pgml/modeling/index`.

`pgml` is the engine of a wider set of tools for machine learning on power grids. Those
packages are not part of this release.

```{toctree}
:hidden:

pgml/concepts
pgml/differentiability
pgml/public-api
pgml/examples
pgml/modeling/index
pgml/api/index
```
