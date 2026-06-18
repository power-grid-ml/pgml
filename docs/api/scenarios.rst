pgml.scenarios
==============

Reproducible, config-driven batched scenario sampling.

The scenarios package lets you define a :class:`~pgml.scenarios.ScenarioConfig`
(+ seed) that **deterministically** produces a batch of realised operating
points.  The sampling strategies are:

- **Independent** (default) — each dimension sampled independently.
- **Sobol QMC** — quasi-Monte-Carlo for better space filling (recommended for
  ML training data).
- **Latin hypercube sampling (LHS)** — stratified random sampling.
- **Cartesian product** — explicit grid over named axes via
  :class:`~pgml.scenarios.CartesianConfig`.

The full batch is solved at once via the batched solver (``run_scenarios``),
producing a :class:`~pgml.scenarios.ScenarioResult` that can be fed directly
to a GNN or written to parquet for offline training.

.. automodule:: pgml.scenarios
   :members:
   :show-inheritance:
