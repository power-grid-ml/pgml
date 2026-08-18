# pgml — differentiable harmonic power flow

**A differentiable, GPU-ready PyTorch engine for harmonic power-flow simulation on power
grids** — the base package of the power-grid-ml suite.

pgml simulates **harmonic power quality** in steady state — full per-phase, per-harmonic
complex power flow — and is **differentiable end to end**: gradients flow from grid
parameters (down to line geometry) through Y-bus assembly and the complex solve to every
output. The same code is therefore a forward simulator, a differentiable physics engine
for machine learning, and an inverse / parameter-recovery tool.

These pages document the `pgml` package. The **suite documentation** — the getting-started
guide, this package next to the learning framework (`pgl`), the grid generator (`pgg`), the
dataset hub (`pghub`) and the dashboard (`pgd`) — is published at
<https://power-grid-ml.readthedocs.io>.

- **Want to understand the model?** Read the [concepts](pgml/concepts.md), then the
  [modeling decisions](pgml/modeling/index.md).
- **Want the stable front door?** See the [public API](pgml/public-api.md) and the
  [examples](pgml/examples.md).
- **Looking for a function or class?** See the [API reference](pgml/api/index.md).

```{toctree}
:hidden:
:caption: pgml — simulation

pgml/index
pgml/concepts
pgml/public-api
pgml/examples
pgml/modeling/index
pgml/api/index
```
