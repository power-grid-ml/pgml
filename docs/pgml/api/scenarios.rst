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

:class:`~pgml.scenarios.Selector`'s ``component`` targets ``"load"`` (default),
``"generator"``, ``"storage"``, or ``"source"`` (the slack, for a ``field="u_ref"``
spec).  ``"storage"`` varies the SIGNED :class:`~pgml.schemas.grid_schema.Storage`
setpoint — positive discharging/injecting, negative charging, the same sign
convention a :class:`~pgml.schemas.grid_schema.Generator` uses — so a ``[low, high]``
distribution spanning zero sweeps both charge and discharge in one
:class:`~pgml.scenarios.ParameterSpec`.

Shared stationary noise
-----------------------

:func:`~pgml.scenarios.ar1_noise` draws a seeded stationary standard-normal AR(1)
process along the last tensor axis. It is the common primitive for downstream
time-series recipes that need the same recurrence without copying its implementation::

    import torch
    from pgml.scenarios import ar1_noise

    generator = torch.Generator().manual_seed(17)
    noise = ar1_noise((128, 24), 0.9, generator)  # scenarios × hourly steps

The output defaults to ``torch.float64`` on ``generator.device``. ``rho`` may be a
tensor broadcastable against the axes before the step axis, and gradients with respect
to a tensor ``rho`` are retained. ``dtype`` and ``device`` are keyword-only; an explicit
device must match the generator.

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
  or ``mode="absolute"`` (absolute pu).  ``harmonic_reference`` turns the sampled
  value into a FRACTION of a per-order limit — combine with a ``[0, 1]`` distribution
  (e.g. ``Uniform(0, 1)``):

  - ``"iec61000-3-2"`` — the physically correct reference for a device current
    fingerprint: the IEC 61000-3-2 appliance harmonic-CURRENT emission limit for
    that device (see `IEC 61000-3-2 appliance current-emission limits`_ below).
  - ``"en50160"`` — the DIN EN 50160 supply-VOLTAGE compatibility level (see
    `DIN EN 50160 compatibility limits`_ below); a background-distortion SHAPE, not
    an appliance emission model, kept for backward compatibility.
  - ``None`` (default for :class:`~pgml.scenarios.ParameterSpec`) — the sampled
    value is an absolute pu magnitude (or a ``scale`` of the stored spectrum).
- ``emission_class`` — an IEC 61000-3-2 equipment class ``"A"``/``"B"``/``"C"``/``"D"``,
  or ``"auto"`` (default) to resolve one per device from its ``consumer_type`` and
  nominal power (:func:`~pgml.scenarios.resolve_emission_class`).  Only valid with
  ``harmonic_reference="iec61000-3-2"``.
- ``field="h_phase"`` sets the injection phase in degrees; requires
  ``mode="absolute"``.
- ``field="h_param"`` draws a free per-device, per-order value; requires
  ``mode="absolute"``.  It is recorded under ``samples[spec.name]`` and written
  nowhere.  Use it for an emission model of your own, for instance a spectrum that
  depends on how heavily a device is loaded.  Declare the model's per-device quantities
  as ``h_param`` specs so they share the batch's cube, seed and persisted record, and
  apply them to ``harmonic_injection`` in the ``sample(grid)`` of your own
  :class:`~pgml.scenarios.ScenarioSpec`.
- ``correlation`` and per-phase ``symmetry`` (other than ``"balanced"``) are not
  supported for harmonic specs.

**Realized-injection audit columns.** Beside the raw draw under ``samples[spec.name]``
(which for a referenced spec is only a FRACTION of a per-device emission limit — the
magnitude that actually reaches the solver exists only once that reference is applied),
an ``h_mag`` spec additionally records what it REALLY injected: ``"<spec.name>_mag"`` /
``"<spec.name>_phase"`` (``[B, n_dev, n_ord]``, per unit of each device's own fundamental
current / degrees, i.e. exactly the values ``solve_harmonic_flow`` receives) and
``"<spec.name>_device_ids"`` (``[n_dev]``). This is what a persisted dataset is audited
against later, which is cheaper and more direct than reconstructing
``I(h) = Y(h)·V(h)`` from the solved state.

IEC 61000-3-2 appliance current-emission limits
---------------------------------------------------

:mod:`pgml.scenarios.iec61000_3_2` is the default reference for a device harmonic
current draw. It bounds what a device is PERMITTED TO INJECT into the supply. The DIN EN
50160 limits below bound the supply VOLTAGE distortion instead, so they are not an
appliance-emission model.  The data ships inside the package
at ``pgml/data/standards/iec61000_3_2.yaml`` and is read via :mod:`importlib.resources`;
an explicit ``path`` argument or the ``PGML_IEC61000_3_2`` environment variable
overrides it.

The standard defines four equipment classes with different native units — Class A
(balanced three-phase / general catch-all, absolute amperes), Class B (portable tools,
1.5x Class A), Class C (lighting, percent of the device fundamental current — the 3rd
harmonic scaled by the circuit power factor), and Class D (75-600 W equipment with a
special wave shape, mA per watt of active power):

- :func:`~pgml.scenarios.iec61000_3_2_limits` — the raw per-class limit table (Class B
  expanded to explicit amperes).
- :func:`~pgml.scenarios.iec61000_3_2_fraction` — converts one class/order limit into a
  fraction of the device's own fundamental current ``I1 = p_w / (u_ln_v * power_factor)``
  — the same convention ``harmonic_injection`` magnitudes use — clamped to ``<= 1.0``.
- :func:`~pgml.scenarios.resolve_emission_class` — maps a device's
  :class:`~pgml.schemas.grid_schema.ConsumerType` (and, for the 600 W Class-D window,
  its nominal power) to a concrete class letter; this is what ``emission_class="auto"``
  calls internally.
- :func:`~pgml.scenarios.iec61000_3_2_device_caps` — builds the full
  ``{device_id: {order: fraction}}`` cap table for a set of grid appliances in one call
  (per-device fundamental current from nominal power and the node's line-to-neutral
  voltage), which is the per-device clamp a sampled harmonic draw is held to.
- :func:`~pgml.scenarios.iec61000_3_2_provenance` — identifies the ACTIVE table
  (``{"source", "override", "sha256"}``); a ``PGML_IEC61000_3_2`` override silently changes
  every generated emission fraction, so an artifact that does not record which table it used
  cannot be compared against another one.

::

    from pgml.scenarios import iec61000_3_2_limits, iec61000_3_2_fraction

    iec61000_3_2_limits("A")["limits"][3]        # 2.30 A (Class A, 3rd harmonic)
    iec61000_3_2_fraction(
        3, emission_class="A", p_w=1500.0, u_ln_v=230.0,
    )                                             # fraction of I1, clamped to [0, 1]

All four helpers work with plain floats — the caps are a sampling BOUND, off the
autograd tape, not a differentiable quantity.

DIN EN 50160 compatibility limits
----------------------------------

Two helpers load the per-order harmonic voltage limits defined by DIN EN 50160.  The
data ships inside the package at ``pgml/data/standards/en50160.yaml`` and is read via
:mod:`importlib.resources`, resolving identically from a source checkout and an installed
wheel:

- :func:`~pgml.scenarios.en50160_limits` — returns ``{order: max_pu}`` for all
  tabulated orders (cached).
- :func:`~pgml.scenarios.en50160_limit` — returns the limit for a single order;
  raises ``KeyError`` if the order is absent.
- :func:`~pgml.scenarios.en50160_provenance` — identifies the ACTIVE table
  (``{"source", "override", "sha256"}``), the DIN EN 50160 counterpart of
  :func:`~pgml.scenarios.iec61000_3_2_provenance` above.

The active file is resolved in priority order: an explicit ``path`` argument, then the
``PGML_EN50160`` environment variable, then the packaged table.  Example::

    from pgml.scenarios import en50160_limits, en50160_limit

    limits = en50160_limits()        # {2: 0.02, 3: 0.05, 5: 0.06, 7: 0.05, ...}
    cap_h5 = en50160_limit(5)        # 0.06 (6 % of fundamental)

Per-node harmonic "error"-source sweep (``run_node_injection_sweep``)
-----------------------------------------------------------------------

:func:`~pgml.scenarios.run_node_injection_sweep` sweeps a per-node harmonic
"error" source (see :doc:`solver` and :doc:`/pgml/modeling/error-injection`) over a set of
nodes: **scenario** ``i`` places a :class:`~pgml.solver.NodeHarmonicSource` at
node ``i`` ONLY.  This builds the harmonic-domain analogue of a network
sensitivity map — "inject a disturbance source at each node, measure how the
spectrum spreads."

Unlike :func:`~pgml.scenarios.spectrum_sweep` (which scales a load's existing
harmonic injection), this source is injected at **any** node (no load required),
at a user-defined strength ``source_power_va``, and only at ``h > 1`` (the
fundamental power flow is preserved exactly).

**Config** (:class:`~pgml.scenarios.NodeInjectionSweepConfig`)

.. code-block:: python

    from pgml.scenarios import NodeInjectionSweepConfig, run_node_injection_sweep

    cfg = NodeInjectionSweepConfig.from_spectrum(
        spectrum={5: (0.04, 0.0), 7: (0.03, 0.0)},
        source_power_va=1e6,          # 1 MVAsc Thévenin source
        kind="voltage",
        # node_ids=None  →  sweep ALL nodes in the grid
    )
    result = run_node_injection_sweep(grid, cfg, slack="norton")
    # result.v  shape [B, H, N]  where B = number of swept nodes, H includes the
    #           fundamental (order 1) + the harmonic orders from the config.
    # result.sampled.samples["injection_node_id"]  shape [B]  (swept node per scenario)

:meth:`~pgml.scenarios.NodeInjectionSweepConfig.from_spectrum` builds the
config from a ``{order: (magnitude_pu, phase_deg)}`` dict (order 1 is the
implicit reference and must **not** be listed).

**Direct use** (without a config)

For ad-hoc sweeps you can also call :func:`~pgml.scenarios.run_node_injection_sweep`
with a manually constructed config::

    cfg = NodeInjectionSweepConfig(
        node_ids=[0, 1, 5],           # subset of nodes
        orders=[5, 7],
        magnitudes_pu=[0.04, 0.03],
        phases_deg=[0.0, 0.0],
        source_power_va=500e3,        # 500 kVAsc
        kind="current",               # Norton (ideal current injection)
    )

**Key differences vs** :func:`~pgml.scenarios.spectrum_sweep`

- **What is swept** — ``spectrum_sweep`` targets a load/generator device;
  ``run_node_injection_sweep`` targets any node (no load required).
- **Source physics** — ``spectrum_sweep`` uses a Norton current scaled by the
  device's fundamental current; ``run_node_injection_sweep`` uses a
  Thévenin/Norton source at a fixed ``source_power_va`` strength.
- **Fundamental** — ``spectrum_sweep`` affects the harmonic model (the device's
  ``I1`` term); ``run_node_injection_sweep`` preserves the fundamental exactly
  (``h > 1`` only).
- **Config class** — ``SpectrumSweepConfig`` vs ``NodeInjectionSweepConfig``.
- **run_scenarios compatibility** — ``SpectrumSweepConfig`` is accepted directly
  by :func:`~pgml.scenarios.run_scenarios` (auto-forces harmonic calculation);
  ``NodeInjectionSweepConfig`` is NOT — call
  :func:`~pgml.scenarios.run_node_injection_sweep` directly.

.. note::

   Because a voltage-kind source modifies ``Y(h)`` (adds a shunt to the diagonal),
   the batched ``Y`` differs for each scenario.  The sweep therefore calls
   :func:`~pgml.solver.solve_harmonic_flow` once per node and stacks the results.
   For current-kind sources the cost is identical (``Y(h)`` is also built fresh per
   call to avoid coupling).  For large grids with hundreds of nodes, profile first.

Per-node injection sweep (spectrum_sweep)
-----------------------------------------

:func:`~pgml.scenarios.spectrum_sweep` is the harmonic analogue of
:func:`~pgml.scenarios.perturbation_sweep`: it builds a *diagonal* batch of
``B = #targets`` scenarios in which scenario ``i`` injects a given harmonic
spectrum at target device ``i`` ONLY — every other device in the selector is
silent.  Use this to measure how a single injected spectrum propagates through
the network node by node::

    from pgml.scenarios import (
        Selector, SpectrumSweepConfig, spectrum_sweep, run_scenarios
    )

    # Build directly from a spectrum dict (order 1 = fundamental, excluded)
    sweep = spectrum_sweep(
        grid,
        selector=Selector(component="load"),
        spectrum={5: (0.06, 0.0), 7: (0.05, 0.0), 11: (0.035, 0.0)},
    )
    # sweep.n_samples == number of matched loads
    result = run_scenarios(grid, sweep, calculation="harmonic",
                           harmonic_orders=[1, 5, 7, 11])
    # result.v  shape [B, 3, N] — one scenario per injection target

The config form :class:`~pgml.scenarios.SpectrumSweepConfig` is serializable
and can be passed directly to :func:`~pgml.scenarios.run_scenarios`::

    cfg = SpectrumSweepConfig.from_spectrum(
        selector=Selector(component="load"),
        spectrum={5: (0.06, 0.0), 7: (0.05, 0.0)},
        name="load_sweep",
    )
    result = run_scenarios(grid, cfg)
    # run_scenarios auto-sets calculation="harmonic" and harmonic_orders=[1, 5, 7]

The returned :class:`~pgml.scenarios.SampledScenarios` carries:

- ``harmonic_injection`` — diagonal ``{device_id: {order: (mag[B], phase[B])}}``:
  magnitude is the spectrum value at the device's own scenario index, zero elsewhere.
- ``samples["<name>_id"]`` — shape ``[B]`` (``long``): the injected device id in
  each scenario.

.. note::

   The sweep perturbs only the **harmonic injection** (Norton current at the device
   terminal); fundamental P/Q stays at the nominal operating point for all scenarios.

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

For every other kind of batch
:attr:`~pgml.scenarios.SampledScenarios.perturbations` is an empty list.

Scope note
~~~~~~~~~~~

The sweep perturbs **operating-point** quantities (P / Q injection at a load or
generator).  Perturbing a **network parameter** (line or transformer impedance) —
the inverse / parameter-recovery use case — is deferred to a later phase: it
requires a branch-aware selector and matrix-valued ground truth that the schema's
scalar ``nominal_value`` / ``perturbed_value`` fields cannot represent.

Building a batch from values
-----------------------------

A batch does not have to come from a sampler. :func:`~pgml.scenarios.batch_from_values`
takes the tensors you already have — a measured load curve, an optimiser's iterate, a
sequence produced elsewhere — and returns the same
:class:`~pgml.scenarios.SampledScenarios` the samplers do::

    from pgml.scenarios import batch_from_values, run_scenarios

    batch = batch_from_values(
        grid,
        n_samples=2,
        p_w={load_id: torch.tensor([5.0e3, 2.0e4])},
    )
    result = run_scenarios(grid, batch)

A component no mapping names, and a field a mapping leaves out, keep the grid's nominal
value; there is no fill value to get wrong. Component ids are resolved against the grid, so
a typo raises instead of being silently ignored. The values pass through untouched, so
gradients flow from them into the solve and their device and dtype are the solver's.

``n_steps`` declares a STEP axis. With ``n_steps=1`` (the default) the batch is a set of
independent snapshots and the result is ``[B, N]`` or ``[B, H, N]``; with ``n_steps=T`` the
per-step values are ``[B, T]`` and the result is ``[B, T, H, N]``. The step count is
declared rather than inferred, because ``[B, T, N]`` and ``[B, H, N]`` have the same rank.

Plugging in your own generator
--------------------------------

:func:`~pgml.scenarios.run_scenarios` accepts any :class:`~pgml.scenarios.ScenarioSpec`: an
object with a ``sample(grid)`` method returning a
:class:`~pgml.scenarios.SampledScenarios`, and optionally a ``harmonic_orders`` hint that
declares the spec only makes sense as a harmonic calculation. The serializable configs
satisfy the protocol themselves, so a generator that lives outside this package plugs into
the same run and persistence path without the engine knowing its recipe.

Records that are not per scenario
-----------------------------------

:attr:`~pgml.scenarios.SampledScenarios.samples` holds one row per scenario.
:attr:`~pgml.scenarios.SampledScenarios.shared_samples` holds the records that have no
leading scenario axis — a step-time vector, a device-id column, a per-device constant. The
split is declared because a record whose length happens to equal the batch size cannot be
told apart by shape, and a misfiled record reads back with the wrong shape.
:attr:`~pgml.scenarios.SampledScenarios.all_samples` is the merged view, which is also what
:func:`~pgml.scenarios.read_dataset` returns.

Running scenarios
-----------------

:func:`~pgml.scenarios.run_scenarios` accepts any
:class:`~pgml.scenarios.ScenarioSpec` — :class:`~pgml.scenarios.ScenarioConfig`,
:class:`~pgml.scenarios.CartesianConfig`, :class:`~pgml.scenarios.SpectrumSweepConfig`,
:class:`~pgml.scenarios.NodeInjectionSweepConfig` or a spec of your own — or a pre-built
:class:`~pgml.scenarios.SampledScenarios`, and solves the batch::

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

When a spec declares ``harmonic_orders``, as
:class:`~pgml.scenarios.SpectrumSweepConfig` and
:class:`~pgml.scenarios.NodeInjectionSweepConfig` do, the runner forces
``calculation="harmonic"`` and sets ``harmonic_orders`` to ``[1, *spec.harmonic_orders]``
if none was given, so the minimal invocation is::

    result = run_scenarios(grid, SpectrumSweepConfig(...))
    # result.v  shape [B, H, N] (B = number of matched targets)

``load_shunt`` selects the harmonic device Norton shunt for the whole batch. A shunt derived
from a PER-SCENARIO operating point makes ``Y(h)`` scenario-dependent, so every scenario
factors its own matrix per order instead of sharing one factorization; ``"none"`` keeps the
shared one.

Convergence and failed scenarios
-----------------------------------

A batched run never raises on a single scenario that fails to converge — its
best-effort voltages are still returned so a large sweep yields data plus diagnosable
failures, not a total loss.
:attr:`~pgml.scenarios.ScenarioResult.converged` is ``True`` iff EVERY scenario
converged; :attr:`~pgml.scenarios.ScenarioResult.failed_states` lists the SCENARIO
indices (along the ``B`` axis of ``v``) that did not, and the solver logs the
residual and likely cause for each. For stepped (``[B, T, H, N]``) data a scenario counts as
failed when ANY of its ``T`` steps failed — the per-step solver failure index maps to its
scenario by integer division by ``T`` — so ``failed_states`` is always indexed the same way
regardless of ``calculation`` or whether :func:`~pgml.scenarios.run_scenarios` chunked the
batch with ``chunk_size``.

Persistence (parquet training data)
------------------------------------

:func:`~pgml.scenarios.write_dataset` and :func:`~pgml.scenarios.read_dataset`
persist a :class:`~pgml.scenarios.ScenarioResult` to a **self-describing dataset
directory** suitable for use as ML training data.  A dataset is fully reproducible:
the ``meta.json`` sidecar embeds the serialized config, seed, node/phase index,
frequencies, and full shape information, so the exact batch that produced it can be
regenerated from the config and seed alone.  ``meta.json`` also records
:attr:`~pgml.scenarios.ScenarioResult.converged` and the (possibly empty)
``failed_scenarios`` index list, so a reload can filter out non-converged rows
without re-solving; :func:`~pgml.scenarios.write_dataset` logs a warning naming the
first few failed indices whenever any scenario did not converge.

.. note::

   Persistence is result I/O, not the differentiable core — tensors are detached and
   moved to CPU before writing.  Do not use ``write_dataset`` inside a gradient tape.

Dataset directory layout
~~~~~~~~~~~~~~~~~~~~~~~~~

Each call to :func:`~pgml.scenarios.write_dataset` creates three files under the
target directory:

- ``voltages.parquet`` — node voltages in the chosen layout (see below).
- ``samples.parquet`` — the realized sampled inputs (B-leading per-scenario records,
  dtype-preserving; absent if there are no per-scenario samples).
- ``meta.json`` — sidecar: serialized config + seed + ``frequencies_hz`` + node/phase
  index + shape dims + ``ParameterPerturbation`` ground-truth rows.
- ``voltages.csv`` — (optional) tidy long-format CSV, written when ``also_csv=True``
  is passed.  Convenience for manual inspection; not the primary format and not read
  back by :func:`~pgml.scenarios.read_dataset`.

Voltage layouts
~~~~~~~~~~~~~~~

Two interchangeable on-disk layouts for ``voltages.parquet`` are supported.
:func:`~pgml.scenarios.read_dataset` is layout-agnostic and always restores an
identical :attr:`~pgml.scenarios.LoadedDataset.v` tensor regardless of which layout
was used to write.

**Wide layout** (``layout="wide"``, default)
    One row per ``(scenario, step)`` with ``v_re`` / ``v_im`` as compact
    fixed-size-array columns flattened over ``[H*N]``.  This is the fast
    training-loop tensor cache — minimal deserialization overhead when streaming
    batches into a model.

**Long layout** (``layout="long"``)
    A tidy table with one row per ``(scenario, step, frequency, node-phase)`` and
    scalar ``v_re`` / ``v_im`` columns (the ``result_schema`` phasor convention).
    Larger on disk, but joins cleanly by ``node_id`` and ``frequency_hz`` — the
    preferred format for exploratory analysis with DuckDB or polars.

Usage example::

    from pgml.scenarios import write_dataset, read_dataset, run_scenarios

    result = run_scenarios(grid, cfg, calculation="harmonic",
                           harmonic_orders=[1, 3, 5, 7])

    # Persist (wide is the default; use layout="long" for analysis)
    # also_csv=True writes voltages.csv for quick manual inspection
    dataset_path = write_dataset(result, "data/ieee33_harmonic", layout="wide",
                                 also_csv=True)

    # Reload — identical tensor, layout-agnostic
    ds = read_dataset(dataset_path)
    # ds.v         complex tensor, shape [B, H, N] (harmonic) or [B, N] (power flow)
    # ds.samples   dict of tensors — the sampled inputs record
    # ds.config    reconstructed ScenarioConfig / CartesianConfig / ...
    # ds.frequencies_hz  [H] real tensor (None for power flow)
    # ds.node_ids  int64 [N], ds.phase_codes  int64 [N]
    # ds.perturbations   list of ground-truth dicts (perturbation_sweep only)
    # ds.converged        True iff every scenario converged; None for a dataset
    #                      written before convergence metadata was persisted
    # ds.failed_scenarios tuple of non-converged scenario indices (empty if none)
    # ds.meta      full sidecar dict

Generation provenance
~~~~~~~~~~~~~~~~~~~~~~~

Beyond the config and seed, ``meta.json`` records what a reload alone cannot reconstruct:
``config_hash`` and :func:`~pgml.scenarios.generation_provenance`'s fields.

- :func:`~pgml.scenarios.config_hash` — a 16-hex-character fingerprint of the serialized
  config (or its pre-serialized JSON string). Two configs producing the same hash describe
  the same batch — this is what a reuse gate (e.g. an ablation driver deciding whether to
  regenerate a dataset) compares instead of a field-by-field diff.
- :func:`~pgml.scenarios.generation_provenance` — what pgml itself contributes:
  ``provenance`` (:func:`~pgml.provenance.code_provenance` — the commit, dirty flag, and
  library versions; see :doc:`provenance`) and ``standards`` (the ACTIVE EN 50160 /
  IEC 61000-3-2 tables, via :func:`~pgml.scenarios.en50160_provenance` /
  :func:`~pgml.scenarios.iec61000_3_2_provenance`).

Two datasets written from a byte-identical config and seed can still differ numerically
because the code changed or a ``PGML_*`` environment variable silently swapped a standards
table for the run — neither of which the config or seed alone would show.
:func:`~pgml.scenarios.read_dataset` needs none of these keys to reload a dataset; a dataset
written before they existed simply lacks them.

``write_dataset(..., provenance={...})`` records the caller's own generation stamps under
``extra_provenance`` in ``meta.json``. This is where a generator records what its config does
not capture, such as the revision of a calibrated recipe or of a device library, since those
change the data without changing the config.

``read_dataset(path, config_types={"MyConfig": MyConfig})`` reconstructs a config class this
package does not define. Reconstruction is explicit because reading a dataset must not import
a module the file chose; the sidecar records ``config_module`` and ``config_class`` so the
owner is identifiable either way, and without the mapping such a config reads back as a dict.
``pgml.scenarios.SCENARIO_CONFIG_TYPES`` is the set this package reconstructs on its own.

``meta.json`` additionally carries the time axis as data — ``n_steps``, ``step_size_s``,
``t0_unix_s`` — so a consumer does not have to parse a generator-specific config to find it.

Changes in 0.5.1
----------------

The load-dependent emission law is no longer part of this package.  It is a calibrated
model of a device population, which a study defines, and not a property of the solver.

- Removed: ``affine_emission_correction``, ``phase_slope_shift``, ``LOADING_FLOOR``, the
  module ``pgml.scenarios.emission``, the :class:`~pgml.scenarios.ParameterSpec` fields
  ``"h_floor"`` / ``"h_floor_phase"`` / ``"h_slope"``, the property
  ``ParameterSpec.is_emission_law``, the constant ``EMISSION_LAW_FIELDS`` and the
  ``"<spec>_loading"`` sample record.
- Added: ``field="h_param"``, a per-device, per-order draw that is recorded and written
  nowhere (see `Harmonic spectrum sampling`_), and ``ParameterSpec.is_free_parameter``.
  A generator with its own emission model declares the model's quantities as ``h_param``
  specs and applies them in its ``sample(grid)``.
- A config stored with one of the removed fields no longer validates as a
  :class:`~pgml.scenarios.ScenarioConfig`.  :func:`~pgml.scenarios.read_dataset` still
  loads such a dataset and returns the stored config as a dict, with a warning.

The same version makes the ratings of an inverter control device totals; see "Changes in
0.5.1" in :doc:`/pgml/modeling/der-pv-storage`.

.. automodule:: pgml.scenarios
   :members:
   :show-inheritance:
