# API Reference

The `pgml` public API is organised into the following subpackages.  Each
subpackage exports its public surface via `__all__` in its `__init__.py`.

```{toctree}
:maxdepth: 2

schemas
assembly
solver
geometry
scenarios
evaluation
convert
equations
config
```

---

## Package overview

| Package | Purpose |
|---------|---------|
| {mod}`pgml.schemas` | Frozen data contracts: `Grid`, `Node`, `Branch`, `ResultSet`, `Scenario` |
| {mod}`pgml.assembly` | Differentiable, batched, per-frequency Y-bus assembly |
| {mod}`pgml.solver` | Complex batched linear solve and nonlinear power-flow |
| {mod}`pgml.geometry` | Differentiable Carson/Deri line constants (geometry → Z(h)/Yc(h)) |
| {mod}`pgml.scenarios` | Reproducible config-driven batched scenario sampling |
| {mod}`pgml.evaluation` | Comparison plots (references vs our solve) |
| {mod}`pgml.convert` | Converters from pandapower / power-grid-model / OpenDSS |
| {mod}`pgml.equations` | Residual-form physical-law registry |
| {mod}`pgml.config` | Documented modelling defaults |
