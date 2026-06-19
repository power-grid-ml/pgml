pgml.scenarios
==============

Reproducible, config-driven batched scenario sampling.

A serialisable :class:`~pgml.scenarios.ScenarioConfig` (+ seed) **deterministically**
produces a batch of realised operating points; :func:`~pgml.scenarios.sample` draws
them and :func:`~pgml.scenarios.run_scenarios` solves the whole batch in a single
vectorised call through the batched solver.  The goal is generating ML training
data in controlled distributions, reproducibly — the same config and seed always
yield the same dataset.

Sampling strategies
-------------------

Three strategies share a common ``icdf``-based transform path: unit-cube samples
``U in [0, 1]^(B, D)`` are drawn then each column is mapped through the
parameter's ``distribution.icdf``.  This means QMC and plain random sampling
use identical downstream transform code.

- **Sobol QMC** (``method="sobol"``, recommended for ML training data) —
  quasi-Monte-Carlo low-discrepancy sequence for better space filling.
- **Latin hypercube** (``method="lhs"``) — stratified random sampling.
- **Independent** (``method="independent"``) — plain seeded RNG (each dimension
  independently uniform).
- **Cartesian product** (via :class:`~pgml.scenarios.CartesianConfig`) —
  explicit deterministic grid over named axes; no RNG.

Parameter targeting and distributions
--------------------------------------

A :class:`~pgml.scenarios.ParameterSpec` pairs a :class:`~pgml.scenarios.Selector`
(which appliances to vary) with a :class:`~pgml.scenarios.Distribution` and
controls how the sample is applied:

- ``field`` — ``"p"`` / ``"q"`` (one quantity) or ``"pq"`` (P and Q by the same
  scale factor, preserving power factor; requires ``mode="scale"``).
- ``mode`` — ``"scale"`` (multiply the nominal P/Q) or ``"absolute"`` (the sampled
  value is set directly in W / VAr).
- ``per`` — ``"each"`` (one independent sample per matched component) or
  ``"shared"`` (one sample broadcast to all matched components).  Ignored when
  ``correlation`` is set.

Correlated sampling
-------------------

Real grids have correlated loading: all PV generators in a district share the
same cloud-cover driver.  :class:`~pgml.scenarios.LatentFactor` and
:class:`~pgml.scenarios.Correlation` model this without losing individual
marginal distributions.

**Declare factors** in :class:`~pgml.scenarios.ScenarioConfig`::

    ScenarioConfig(
        n_samples=512,
        method="sobol",
        factors=[LatentFactor(name="solar")],
        parameters=[
            ParameterSpec(
                name="pv_output",
                selector=Selector(component="generator", consumer_type="pv"),
                distribution=Uniform(low=0.2, high=1.0),
                correlation=Correlation(factor="solar", rho=0.8),
            ),
        ],
    )

Each declared factor occupies **one column** of the unit-cube ``U`` (the
leftmost columns, one per declared factor in declaration order).

**Single-factor Gaussian copula.** For each matched component ``i``::

    Z_i = sqrt(rho) * Z_factor + sqrt(1 - rho) * eps_i

where ``Z_factor = Phi^{-1}(u_factor)`` and ``eps_i = Phi^{-1}(u_i)`` are
independent standard-normal scores.  The score ``Z_i`` is mapped back through
the standard-normal CDF and then through ``distribution.icdf``, preserving the
marginal distribution exactly.  When ``rho=0`` this reproduces ``per="each"``
(independent); when ``rho=1`` it reproduces ``per="shared"`` (identical).

Per-phase symmetry
------------------

:class:`~pgml.scenarios.ParameterSpec` carries a ``symmetry`` field that controls
how the sampled value is distributed across the phases of each matched appliance:

- ``"balanced"`` (default) — one value per component written as a scalar total
  (``p_w`` / ``q_var``, split equally across phases downstream).
- ``"independent"`` — each phase drawn **independently**; writes per-phase
  overrides ``p_per_phase_w`` / ``q_per_phase_var``.
- ``"small_imbalance"`` — a balanced base multiplied by ``(1 + delta_ph)`` where
  ``delta_ph`` is a small per-phase Gaussian perturbation of fractional standard
  deviation ``imbalance`` (must be ``> 0``).  Also writes per-phase overrides.

The latter two options write per-phase operating-point keys and thereby
**automatically promote the solve to asymmetric** (``symmetry="auto"``
resolution in the solver).  The ``symmetry`` argument of
:func:`~pgml.scenarios.run_scenarios` forwards this to the solver; the default
``None`` / ``"auto"`` lets per-phase samples promote the solve without any
extra configuration.

.. note::

   ``"independent"`` requires all matched components to have the same phase
   count.  Split into separate :class:`~pgml.scenarios.ParameterSpec` instances
   if the selector matches components with differing phase counts.

   ``correlation`` is incompatible with ``symmetry="independent"`` (there is no
   component-level value to correlate when every phase is drawn independently).

Unit-cube layout
----------------

The ``D``-dimensional unit-cube ``U[B, D]`` is laid out as:

1. **Factor columns** — one column per declared :class:`~pgml.scenarios.LatentFactor`
   (in declaration order).
2. **Per-spec blocks** — for each :class:`~pgml.scenarios.ParameterSpec` in order:

   a. **Base block** — component-level draws.  Width: number of matched
      components (for ``per="each"`` or correlated), 1 (for ``per="shared"``),
      or 0 (for ``symmetry="independent"``).
   b. **Per-phase block** — additional columns for per-phase variation.  Width:
      total phase count across matched components (for ``"small_imbalance"`` and
      for ``"independent"`` with ``per="each"``), first component's phase count
      (for ``"independent"`` with ``per="shared"``), or 0 (for ``"balanced"``).

The same config and seed always produce the same ``U`` and therefore the same
batch (deterministic).

Running scenarios
-----------------

:func:`~pgml.scenarios.run_scenarios` accepts a :class:`~pgml.scenarios.ScenarioConfig`,
:class:`~pgml.scenarios.CartesianConfig`, or a pre-built
:class:`~pgml.scenarios.SampledScenarios` and solves the batch::

    result = run_scenarios(
        grid,
        ScenarioConfig(
            n_samples=256,
            method="sobol",
            parameters=[...],
        ),
        calculation="harmonic",
        harmonic_orders=[1, 5, 7, 11, 13],
        symmetry=None,  # "auto": per-phase samples promote to asymmetric
        dtype=torch.complex128,
    )
    # result.v  shape [256, 5, N] for B=256 scenarios, H=5 harmonics, N nodes
    # result.sampled.operating_point  — the realised batch (reproducible input record)

The ``spec`` argument may also be a pre-built :class:`~pgml.scenarios.SampledScenarios`
so that sampling and solving can be separated (e.g. inspect the batch before
solving it).

.. automodule:: pgml.scenarios
   :members:
   :show-inheritance:
