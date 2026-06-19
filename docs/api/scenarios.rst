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

Harmonic spectrum sampling
--------------------------

Beyond varying fundamental-frequency P/Q, the sampler can produce batched harmonic
injection spectra — the current-injection vector ``I_h`` that drives
``solve_harmonic_flow``.

**Per-order random sampling via** ``ParameterSpec``

A :class:`~pgml.scenarios.ParameterSpec` with ``field="h_mag"`` or
``field="h_phase"`` draws random per-order injection magnitudes or phases as part of
the ordinary :class:`~pgml.scenarios.ScenarioConfig` batch.  Harmonic specs require
an ``orders`` list (harmonic orders >= 2) and write to
:attr:`~pgml.scenarios.SampledScenarios.harmonic_injection` instead of an operating
point::

    from pgml.scenarios import (
        ParameterSpec, Selector, ScenarioConfig, Uniform, run_scenarios
    )

    cfg = ScenarioConfig(
        n_samples=128,
        method="sobol",
        parameters=[
            ParameterSpec(
                name="h5_mag",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.0, high=1.0),
                field="h_mag",
                orders=[5, 7, 11, 13],
                harmonic_reference="en50160",   # sample as fraction of EN 50160 limit
            ),
        ],
    )
    result = run_scenarios(grid, cfg, calculation="harmonic",
                           harmonic_orders=[1, 5, 7, 11, 13])
    # result.v  shape [128, 5, N]

Key rules for harmonic :class:`~pgml.scenarios.ParameterSpec`:

- ``field="h_mag"`` requires ``mode="scale"`` (relative to stored spectrum, default)
  or ``mode="absolute"`` (absolute pu).  When ``harmonic_reference="en50160"`` the
  sampled value is a fraction of the per-order DIN EN 50160 limit; combine with a
  ``[0, 1]`` distribution (e.g. ``Uniform(0, 1)``).
- ``field="h_phase"`` sets the injection phase in degrees; requires
  ``mode="absolute"``.
- ``correlation`` and per-phase ``symmetry`` (other than ``"balanced"``) are not
  supported for harmonic specs.

DIN EN 50160 compatibility limits
----------------------------------

Two helpers load the per-order harmonic voltage limits defined by DIN EN 50160.  The
data lives in ``config/max_harmonic_values_din-en50160.yaml`` at the repo root:

- :func:`~pgml.scenarios.en50160_limits` — returns ``{order: max_pu}`` for all
  tabulated orders (cached).
- :func:`~pgml.scenarios.en50160_limit` — returns the limit for a single order;
  raises ``KeyError`` if the order is absent.

The active file is resolved in priority order: an explicit ``path`` argument, then
the ``PGML_EN50160`` environment variable, then the first
``config/max_harmonic_values_din-en50160.yaml`` found by walking up from the package
root.  Example::

    from pgml.scenarios import en50160_limits, en50160_limit

    limits = en50160_limits()        # {2: 0.02, 3: 0.05, 5: 0.06, 7: 0.05, ...}
    cap_h5 = en50160_limit(5)        # 0.06 (6 % of fundamental)

Node-coherent harmonic fingerprints
-------------------------------------

For training harmonic state estimators the key challenge is that each node must have
a *recognisable, time-varying* injection signature — so the estimator can attribute
a harmonic pattern to a node even under noise.
:class:`~pgml.scenarios.CoherentSpectrumConfig` and
:func:`~pgml.scenarios.sample_coherent_spectra` implement this model:

**Concept.** Each matched device draws ``n_modes`` base spectra ("operating states",
e.g. a washing machine in heating vs spin cycle).  Over ``n_steps`` time steps it
**sticks** to a mode (Markov dwell probability ``dwell``) and **wanders** around it
(AR(1) temporal jitter with stickiness ``ar1_rho``), with magnitudes clamped to the
DIN EN 50160 per-order limits.  The result is a ``[B, T]`` batch of harmonic
injections where each node keeps its characteristic fingerprint while varying
realistically over time.

**Output shape.** Passing a :class:`~pgml.scenarios.CoherentSpectrumConfig` to
:func:`~pgml.scenarios.run_scenarios` forces ``calculation="harmonic"`` and produces
node voltages ``v`` of shape ``[B, T, H, N]`` (B scenarios, T steps, H harmonic
orders, N nodes)::

    from pgml.scenarios import (
        CoherentSpectrumConfig, Selector, Uniform, run_scenarios
    )

    cfg = CoherentSpectrumConfig(
        selector=Selector(component="load"),
        orders=[3, 5, 7, 11, 13],
        n_steps=24,          # one day at hourly resolution
        n_scenarios=64,      # B independent sequences
        n_modes=3,
        seed=42,
        dwell=0.9,           # P(stay in mode) per step
        ar1_rho=0.8,         # temporal stickiness of AR(1) jitter
        jitter_mag=0.05,     # fractional std of magnitude jitter
        jitter_phase_deg=5.0,
        step_size_s=3600.0,
        harmonic_reference="en50160",   # clamp to EN 50160
    )
    result = run_scenarios(grid, cfg)
    # result.v  shape [64, 24, 6, N]  (H=6: fundamental + 5 harmonics)
    # result.sampled.samples["time_s"]  shape [24]  (timestamps in seconds)

**Sample record keys** written to ``sampled.samples`` (``name`` defaults to
``"harmonics"``):

- ``"<name>_mode"`` — ``[B, n_dev, T]`` active mode index (int), the ML attribution
  label (ground truth for which fingerprint is active).
- ``"<name>_mag"`` / ``"<name>_phase"`` — ``[B, n_dev, n_ord, T]`` realized injection
  magnitudes and phases.
- ``"<name>_mode_base_mag"`` — the per-device fingerprint spectra (base magnitudes
  per mode, before jitter).
- ``"<name>_device_ids"`` — ``[n_dev]`` integer device IDs in selector order.
- ``"time_s"`` — ``[T]`` timestamps in seconds (``step_size_s`` * step index).

``harmonic_injection`` in the returned :class:`~pgml.scenarios.SampledScenarios` maps
``{device_id: {order: (mag[B, T], phase[B, T])}}``; this is passed directly to
``solve_harmonic_flow`` by :func:`~pgml.scenarios.run_scenarios`.

Perturbation sweep
------------------

:func:`~pgml.scenarios.perturbation_sweep` implements roadmap use-case 1: "inject
a specific error ONCE at each selected node and measure how it spreads."  Rather
than sampling a distribution, it builds a *diagonal* batch of ``B = #targets``
scenarios in which scenario ``j`` perturbs exactly target ``j``'s operating point
(P / Q injection) while every other target stays at its nominal value::

    from pgml.scenarios import (
        Perturbation, Selector, perturbation_sweep, run_scenarios
    )

    sweep = perturbation_sweep(
        grid,
        selector=Selector(component="load"),
        perturbation=Perturbation(
            name="load_error",
            field="p",
            mode="scale",
            value=1.10,          # +10 % active-power error
        ),
    )
    # sweep.n_samples == number of matched loads (one scenario per target)
    result = run_scenarios(grid, sweep)

The :class:`~pgml.scenarios.Perturbation` config controls what is perturbed:

- ``field`` — ``"p"`` / ``"q"`` (one quantity) or ``"pq"`` (both at constant power
  factor; requires ``mode="scale"``).
- ``mode`` — ``"scale"`` (multiply the nominal by ``value``), ``"delta"`` (add
  ``value`` as an absolute offset in W / var), or ``"set"`` (replace the operating
  point with ``value`` directly).
- ``value`` — the perturbation magnitude.

Ground truth and sample record
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~pgml.scenarios.perturbation_sweep` records the full perturbation ground
truth in two places:

- :attr:`~pgml.scenarios.SampledScenarios.perturbations` — a list of
  :class:`~pgml.schemas.scenario_schema.ParameterPerturbation` objects (one per
  perturbed (scenario, field) pair), each carrying the ``scenario_id``,
  ``component_id``, ``nominal_value``, ``perturbed_value``, and ``unit_short``.
  This is the ML ground truth — it says exactly which component was perturbed and
  by how much.
- :attr:`~pgml.scenarios.SampledScenarios.samples` — two tensors keyed by the
  perturbation ``name``:

  - ``"<name>_target_id"`` — shape ``[B]`` (``long``): the perturbed component id
    in scenario ``j``.
  - ``"<name>_perturbed_<field>"`` — shape ``[B]``: the applied value (one entry
    per perturbed field; ``"pq"`` produces both ``_p`` and ``_q`` keys).

For non-perturbation batches (random, cartesian, coherent),
:attr:`~pgml.scenarios.SampledScenarios.perturbations` is always an empty list.

Scope note
~~~~~~~~~~~

The sweep perturbs **operating-point** quantities (P / Q injection at a load or
generator).  Perturbing a **network parameter** (line or transformer impedance) —
the inverse / parameter-recovery use case — is deferred to a later phase: it
requires a branch-aware selector and matrix-valued ground truth that the schema's
scalar ``nominal_value`` / ``perturbed_value`` fields cannot represent.

Running scenarios
-----------------

:func:`~pgml.scenarios.run_scenarios` accepts a :class:`~pgml.scenarios.ScenarioConfig`,
:class:`~pgml.scenarios.CartesianConfig`, :class:`~pgml.scenarios.CoherentSpectrumConfig`,
or a pre-built :class:`~pgml.scenarios.SampledScenarios` and solves the batch::

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

When ``spec`` is a :class:`~pgml.scenarios.CoherentSpectrumConfig` the runner
automatically forces ``calculation="harmonic"`` and sets ``harmonic_orders`` to
``[1, *config.orders]`` if not provided, so the minimal invocation is::

    result = run_scenarios(grid, CoherentSpectrumConfig(...))
    # result.v  shape [B, T, H, N]

.. automodule:: pgml.scenarios
   :members:
   :show-inheritance:
