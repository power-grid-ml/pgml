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
- ``correlation`` and per-phase ``symmetry`` (other than ``"balanced"``) are not
  supported for harmonic specs.

IEC 61000-3-2 appliance current-emission limits
---------------------------------------------------

:mod:`pgml.scenarios.iec61000_3_2` is the DEFAULT reference for a device harmonic
current fingerprint (:class:`~pgml.scenarios.CoherentSpectrumConfig`'s
``harmonic_reference`` default) — it bounds what a device is PERMITTED TO INJECT into
the supply, unlike the DIN EN 50160 limits below, which bound the supply VOLTAGE
distortion and are not an appliance-emission model.  The data ships inside the package
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
  voltage), the per-device clamp :func:`~pgml.scenarios.sample_coherent_spectra` applies
  by default.

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

The active file is resolved in priority order: an explicit ``path`` argument, then the
``PGML_EN50160`` environment variable, then the packaged table.  Example::

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
per-order ``harmonic_reference`` limit.  The result is a ``[B, T]`` batch of harmonic
injections where each node keeps its characteristic fingerprint while varying
realistically over time.

**Reference and class.** ``harmonic_reference`` defaults to ``"iec61000-3-2"`` — each
device's fingerprint is clamped to its OWN IEC 61000-3-2 emission cap
(:func:`~pgml.scenarios.iec61000_3_2_device_caps`), keyed by ``emission_class``
(``"auto"`` resolves one per device from its ``consumer_type`` and nominal power).
Pass ``harmonic_reference="en50160"`` to instead shape the fingerprint by the DIN EN
50160 supply-voltage compatibility levels (background-distortion shape, not a device
emission model — kept for backward compatibility), or ``None`` for an unclamped
absolute-pu magnitude.

**Held-out fingerprint sets.** Reproducibility of a coherent dataset depends on both
``seed`` (the temporal Markov + AR(1) stream) and ``mode_bank_seed`` (the per-device
fingerprint bank). ``mode_bank_seed=None`` (default) draws the bank from the ``seed``
stream, byte-identical to a config with no ``mode_bank_seed`` set; give a DISTINCT
``mode_bank_seed`` to pin a different device-signature bank while every other setting
stays shared — the recipe for an unseen-fingerprint test split.

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
        # harmonic_reference="iec61000-3-2" (default) — per-device IEC current caps;
        # pass emission_class="auto" (default) or a concrete "A"/"B"/"C"/"D".
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

Time-varying fundamental (``LoadProfileConfig``)
---------------------------------------------------

By default the fundamental P/Q of a :class:`~pgml.scenarios.CoherentSpectrumConfig`
sequence is **constant** across its ``T`` steps — only the harmonic fingerprint varies.
Setting ``profile`` to a :class:`~pgml.scenarios.LoadProfileConfig` makes the fundamental
itself **time-varying**: every matched device's per-scenario base P/Q is multiplied by a
synthetic profile factor composed on four time scales (seasonal, weekly, daily,
short-term AR(1)), class-aware by ``consumer_type`` — a household evening peak, an
office business-hours plateau, an EV late-evening charging peak, a near-flat industrial
plateau, and a dedicated solar bell for ``"pv"`` that is zero at night with a
seasonally-widening daylight window. ``start_time`` (ISO 8601) is **required** with
``profile`` — the daily / weekly / seasonal phases need an absolute anchor::

    from pgml.scenarios import CoherentSpectrumConfig, LoadProfileConfig, run_scenarios

    cfg = CoherentSpectrumConfig(
        selector=Selector(component="load"),
        orders=[3, 5, 7, 11, 13],
        n_steps=48,               # two days at hourly resolution
        n_scenarios=32,
        step_size_s=3600.0,
        seed=42,
        profile=LoadProfileConfig(),          # defaults: daily_amplitude=1.0, ...
        start_time="2024-06-21T00:00:00",     # ISO 8601, anchors the daily/seasonal phase
    )
    result = run_scenarios(grid, cfg)
    # result.v  shape [32, 48, 6, N]  — the fundamental now moves step to step
    # result.sampled.samples["time_unix_s"]  shape [48]  (absolute epoch seconds)

Because the harmonic injection magnitude is defined **relative to** each device's own
fundamental current, a profile-scaled fundamental already scales the absolute harmonic
current at solve time — no extra coupling is needed. The profile draws on an RNG stream
distinct from the fingerprint bank, the Markov path, the AR(1) jitter, and the
operating-point cube, so ``profile=None`` (the default) is byte-identical to the
fingerprint-only behavior.

**Correlation model.** Two per-scenario shared latents make devices co-vary within a
scenario: a ``behavioral`` latent scales the daily-amplitude of every non-``pv`` device
together (a busy day lifts everyone's swing), and a ``cloudiness`` latent scales every
``pv`` device's output together (an overcast day dims all panels). Each device then draws
idiosyncratic per-scenario values on top — an overall level, an amplitude jitter, and a
daily phase offset — plus a per-step AR(1) short-term term.

**Sample record keys** added when ``profile`` is set (``name`` defaults to
``"harmonics"``, matching the fingerprint's own name):

- ``"<name>_profile_factor"`` — ``[B, n_dev, T]`` the realized multiplicative profile
  factor per profiled device per step (ML ground truth for profile attribution).
- ``"<name>_profile_device_ids"`` — ``[n_dev]`` the profiled device IDs.
- ``"time_unix_s"`` — ``[T]`` absolute epoch seconds (``start_time`` + ``k * step_size_s``);
  distinct from the always-present relative ``"time_s"``.

A per-scenario source ``u_ref_scale`` (from ``CoherentSpectrumConfig.parameters``, ``[B]``)
is automatically promoted to ``[B, 1]`` so the slack reference broadcasts against the
per-step ``[B, T]`` state — the scale itself stays constant over the sequence, only the
broadcast shape changes.

:func:`~pgml.scenarios.load_profile_factors` (draws the raw per-device, per-step factors
as a ``ProfileDraw`` — ``factor[B, n_dev, T]`` / ``device_ids[n_dev]`` / ``time_unix_s[T]``)
and :func:`~pgml.scenarios.apply_load_profiles` (lifts an operating point to the per-step
``[B, T]`` form) are the two functions :func:`~pgml.scenarios.sample_coherent_spectra`
calls internally when ``profile`` is set; call them directly to inspect the realized
profile factors before solving.

**Differentiability.** A profiled ``[B, T]`` operating-point batch solves and
differentiates like any other batch: the implicit-function-theorem backward pass builds
its state Jacobian over a single flattened scenario axis by collapsing the plan's
``[B, T]`` power to ``[B*T]`` internally, so gradients w.r.t. grid parameters flow through
a profiled sequence exactly as they do through a constant-fundamental one (verified
against a per-step-loop gradient and finite differences). A cartesian *states × operating
point* batch remains forward-only, as before.

.. note::

   The profile generator lives in ``pgml.scenarios.profiles``; only
   :func:`~pgml.scenarios.load_profile_factors` and
   :func:`~pgml.scenarios.apply_load_profiles` are re-exported at the
   :mod:`pgml.scenarios` package level (documented here alongside
   :class:`~pgml.scenarios.CoherentSpectrumConfig`, the surface that uses them).

Statistical device-class composition
----------------------------------------

The profile and fingerprint machinery above move an aggregated load's fundamental and
harmonic spectrum as two SEPARATE signals — a profile scales the fundamental, a
fingerprint wanders around a base spectrum — coupled only through the harmonic
convention (magnitude is a fraction of the fundamental current). Real aggregated
loads (a household, an office) are a mix of many devices whose OWN activity moves
both signals TOGETHER: a washing machine coming on draws more power AND changes the
spectrum in the same instant, and a state estimator that could learn to attribute an
observed spectrum to a device mix needs data where that joint structure is present.
:class:`~pgml.scenarios.CompositionConfig` (set on
:attr:`~pgml.scenarios.CoherentSpectrumConfig.composition`) builds exactly that: each
covered aggregated load becomes a SUM of statistical member devices, and the summed
current is what the solver injects.

**What it deliberately is NOT.** The device-class library
(:func:`~pgml.scenarios.default_device_classes`) is plausible statistical coverage,
not an appliance catalog — per-order magnitude ranges are LOOSELY IEC 61000-3-2-shaped
(odd-dominated, decreasing with order) rather than measured nameplate spectra, and a
class stands in for a whole family of similar devices (an "SMPS-electronics" class
covers laptop chargers, LED drivers, TVs alike), not a single named appliance. The goal
is diverse, physically-plausible training data with a KNOWN, recoverable ground truth —
not a faithful digital twin of any one device.

**The generation model** (per covered load, off the autograd tape, ``float64``, CPU,
seeded — see the module docstring of ``pgml.scenarios.composition`` for the full
derivation):

1. **Roster** (drawn once, persisted) — a set of member devices per
   :class:`~pgml.scenarios.ClassCount` in the matching
   :class:`~pgml.scenarios.ConsumerComposition` rule, each with a rated power, a sign,
   per-order rated harmonic magnitude/phase, a load-dependence exponent/slope, a mean
   loading, activity stickiness, and (multi-state classes) a small set of
   :class:`~pgml.scenarios.DeviceState` power/spectrum states.  With
   :attr:`~pgml.scenarios.CompositionConfig.scale_to_nominal` (default) the roster's
   share-weighted installed capacity is rescaled to the load's own ``p_nom_w``, so the
   composition's total nameplate stays meaningful across grids of different scale.
2. **Activity** ``a_d(t)`` — a diurnal availability rate (one of the profile daily
   shapes, keyed by :attr:`~pgml.scenarios.DeviceClassSpec.activity_preset`) scaled by
   a per-scenario shared latent (``behavioral`` for consumption classes, ``cloud`` for
   PV — the SAME correlation mechanism the profile generator uses).  A switching class
   (:attr:`~pgml.scenarios.DeviceClassSpec.discrete_activity` ``True``, the default)
   realizes this as an on/off Markov chain whose stationary occupancy tracks the rate;
   a continuously-modulated class (PV, a background base load) uses the rate directly
   as a fractional availability.
3. **Loading** ``lam_d(t)`` — an AR(1)-smoothed fluctuation around the member's mean
   loading (single-state classes) or the current :class:`~pgml.scenarios.DeviceState`'s
   ``power_fraction`` (multi-state classes: e.g. a heat-pump / white-goods class that
   jumps between a near-linear HEATING state and a harmonic-rich INVERTER state on its
   own Markov dwell).
4. **Contribution** — power ``P_d = a_d · lam_d · P_rated · sign`` and a harmonic
   current phasor whose magnitude and phase follow the SAME load-dependence laws as the
   composed spectrum sampling above: ``mag_h(lam) = mag_h_rated · lam ** gamma_h`` and
   ``ang_h(lam) = ang_h0 + s_h · (lam - 1)``, with ``gamma_h`` / ``s_h`` drawn per
   member per order from :attr:`~pgml.scenarios.DeviceClassSpec.gamma` /
   :attr:`~pgml.scenarios.DeviceClassSpec.phase_slope_deg`.  Per load, the members'
   powers and harmonic phasors are SUMMED; the aggregate magnitude is normalized to the
   solver's injection convention (a fraction of the aggregate fundamental current,
   capped at :attr:`~pgml.scenarios.CompositionConfig.max_injection_pu` — a physically
   real residual-THD blow-up near a net-zero fundamental, e.g. PV nearly cancelling
   load, not a numerical artifact) and the phase relative to the aggregate fundamental
   direction.  The aggregate power ``P_agg`` may go net-negative under enough PV — the
   Load then injects, exactly as a real net-metered feeder would.

**Config surface** (:mod:`pgml.scenarios`, all serializable pydantic models):

- :class:`~pgml.scenarios.DeviceClassSpec` — one statistical device class: ``name``,
  ``sign`` (``+1`` consuming, ``-1`` injecting), ``rated_power_w`` range,
  ``power_factor``, per-order ``harmonic_magnitude`` / ``harmonic_phase_deg`` ranges,
  ``gamma`` / ``phase_slope_deg`` load-dependence ranges, ``activity_preset`` +
  ``discrete_activity`` + ``on_off_dwell``, ``loading_min`` / ``loading_mean`` /
  ``loading_jitter`` / ``loading_rho``, and optional ``states``
  (:class:`~pgml.scenarios.DeviceState`: ``power_fraction``, ``spectrum_scale``,
  ``weight``) + ``state_dwell`` for a multi-state class.
- :class:`~pgml.scenarios.ClassCount` — how many instances of one class an aggregated
  load contains (``class_name``, ``count`` range, ``power_share`` for capacity
  rescaling).
- :class:`~pgml.scenarios.ConsumerComposition` — a composition rule: matches a load by
  ``load_ids`` (per-appliance override), else ``consumer_type``, else is the fallback
  for any load no other rule claims; lists its ``classes``
  (:class:`~pgml.scenarios.ClassCount` entries).
- :class:`~pgml.scenarios.CompositionConfig` — ``selector`` (default: every in-service
  load), ``classes`` (default :func:`~pgml.scenarios.default_device_classes`, six
  built-in classes — a harmonic-free linear base load, SMPS electronics, an EV charger,
  a PV inverter, a multi-state inverter drive, and a resistive heater),
  ``compositions`` (default :func:`~pgml.scenarios.default_compositions`, one rule per
  common ``consumer_type`` — ``"household"``, ``"office"``, ``"restaurant"``,
  ``"heat_pump"``, ``"ev_charging"``, ``"pv"``, plus a fallback), ``scale_to_nominal``,
  ``max_injection_pu``, ``behavioral_coupling`` / ``cloud_coupling`` (cross-device
  correlation strength), and ``roster_seed`` (optional, distinct from ``seed`` — a
  held-out device-composition bank, the same recipe as
  :attr:`~pgml.scenarios.CoherentSpectrumConfig.mode_bank_seed` above).

**Enabling it.** Set ``composition`` on a
:class:`~pgml.scenarios.CoherentSpectrumConfig` (``start_time`` is REQUIRED — the
activity model is diurnal and needs an absolute anchor, same as ``profile``)::

    from pgml.scenarios import (
        CoherentSpectrumConfig, CompositionConfig, Selector, run_scenarios
    )

    cfg = CoherentSpectrumConfig(
        selector=Selector(component="load"),
        orders=[3, 5, 7, 9, 11, 13],
        n_steps=96,                 # a day at 15-minute resolution
        n_scenarios=64,
        step_size_s=900.0,
        seed=42,
        composition=CompositionConfig(),         # defaults: the 6-class library
        start_time="2024-06-21T00:00:00",        # ISO 8601, anchors the activity model
    )
    result = run_scenarios(grid, cfg)
    # result.v  shape [64, 96, 7, N]

A load covered by the composition (matched by ``composition.selector`` AND claimed by a
:class:`~pgml.scenarios.ConsumerComposition` rule) draws its fundamental P/Q and its
harmonic injection ENTIRELY from the composition — it SUPERSEDES the mode-bank
fingerprint and any ``parameters`` / ``profile`` targeting that same load. A load
outside ``composition.selector``, or with no matching rule, is untouched and keeps the
ordinary fingerprint / profile behavior.  The mixed ``[B]`` (fingerprint-only devices)
and ``[B, T]`` (composed devices) operating point is unified to ``[B, T]`` so the whole
batch shares one leading step axis.

**Sample record keys** written to ``sampled.samples`` (``name`` defaults to
``"harmonics"``, matching :attr:`~pgml.scenarios.CoherentSpectrumConfig.name`) — the
per-class attribution ground truth, ordered along ``n_class`` as
``config.composition.class_names()``:

- ``"<name>_class_p_w"`` — ``[B, n_agg, n_class, T]`` signed per-class active-power
  contribution (the primary attribution label — which class drew how much power, when).
- ``"<name>_class_active"`` — ``[B, n_agg, n_class, T]`` (int64) active member count per
  class per step.
- ``"<name>_cap_binding"`` — ``[B, n_agg, n_ord, T]`` where
  ``max_injection_pu`` clipped the relative magnitude (a diagnostic on the residual-THD
  floor, not an error).
- ``"<name>_agg_ids"`` — ``[n_agg]`` the covered load ids, in aggregation order.
- ``"<name>_roster_p_rated"`` — ``[n_agg, n_class, max_count]`` the per-member rated
  powers (a zero-padded sidecar; ``max_count`` = the largest member count drawn for any
  one (load, class) pair).

:func:`~pgml.scenarios.sample_device_composition` and
:func:`~pgml.scenarios.resolve_composed_ids` are the underlying functions
:func:`~pgml.scenarios.sample_coherent_spectra` calls when ``composition`` is set — call
them directly to inspect the drawn roster and realized attribution before solving.  The
composition draws on roster + temporal RNG streams distinct from the fingerprint,
operating-point cube, and profile streams, so ``composition=None`` (the default)
reproduces the fingerprint-only output byte-for-byte.  ``pgl.data.CompositionLabels`` /
``pgl.data.DataSource.composition_labels()`` expose these same attribution samples on
the ML side — see :doc:`/pgl/api/data`.

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
:class:`~pgml.scenarios.SpectrumSweepConfig`, or a pre-built
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

When ``spec`` is a :class:`~pgml.scenarios.CoherentSpectrumConfig` or a
:class:`~pgml.scenarios.SpectrumSweepConfig` the runner automatically forces
``calculation="harmonic"`` and sets ``harmonic_orders`` to ``[1, *config.orders]``
if not provided, so the minimal invocations are::

    result = run_scenarios(grid, CoherentSpectrumConfig(...))
    # result.v  shape [B, T, H, N]

    result = run_scenarios(grid, SpectrumSweepConfig(...))
    # result.v  shape [B, H, N] (B = number of matched targets)

Convergence and failed scenarios
-----------------------------------

A batched run never raises on a single scenario that fails to converge — its
best-effort voltages are still returned so a large sweep yields data plus diagnosable
failures, not a total loss.
:attr:`~pgml.scenarios.ScenarioResult.converged` is ``True`` iff EVERY scenario
converged; :attr:`~pgml.scenarios.ScenarioResult.failed_states` lists the SCENARIO
indices (along the ``B`` axis of ``v``) that did not, and the solver logs the
residual and likely cause for each. For node-coherent (``[B, T, H, N]``) data, a
scenario counts as failed when ANY of its ``T`` steps failed — the per-step solver
failure index maps to its scenario via integer division by ``T``, so
``failed_states`` is always indexed the same way regardless of ``calculation`` or
whether :func:`~pgml.scenarios.run_scenarios` chunked the batch with ``chunk_size``.

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

Storage dispatch and SoC integration
--------------------------------------

:class:`~pgml.schemas.grid_schema.Storage` elements appear in the power-flow
snapshot as a signed ``(P, Q)`` injection (``p_nom_w > 0`` = discharging /
injecting).  What couples timesteps is the state of charge (SoC) and the
dispatch decision; these are resolved outside the per-snapshot solve into a
realized per-step active-power sequence the solver consumes.

:func:`~pgml.scenarios.integrate_soc`
    Realizes a requested power sequence ``[*batch, T]`` under SoC limits and
    an optional power rating.  Returns a
    :class:`~pgml.scenarios.StorageDispatchResult` with the realized power and
    (when a capacity is given) the SoC and energy trajectories.  Uses torch
    arithmetic so gradients flow through the realized setpoint value; no
    gradient flows through the dispatch decision itself.

:func:`~pgml.scenarios.dispatch_storage`
    Convenience wrapper: reads ``energy_capacity_wh``, ``soc``, ``soc_min`` /
    ``soc_max``, efficiencies, and ``p_rated_w`` from a
    :class:`~pgml.schemas.grid_schema.Storage` element and delegates to
    :func:`~pgml.scenarios.integrate_soc`.

:func:`~pgml.scenarios.storage_operating_point`
    Converts a ``{storage_id: realized_power_w}`` dict from a dispatch step
    into the ``operating_point`` format consumed by
    :func:`~pgml.solver.solve_power_flow` /
    :func:`~pgml.solver.solve_harmonic_flow`.

Example — one-day dispatch cycle::

    from pgml.scenarios import dispatch_storage, storage_operating_point

    # storage is a Storage element with a 10 kWh capacity, 50 % initial SoC
    result = dispatch_storage(
        storage,
        requested_power_w=[5000.0] * 4 + [-3000.0] * 4,   # charge/discharge
        dt_s=3600.0,
    )
    # result.realized_power_w  [8]  — clamped by SoC/rating
    # result.soc               [9]  — SoC at each step boundary (soc[0] = initial)

    # Build solver operating_point for step 2:
    op = storage_operating_point(
        {storage.id: float(result.realized_power_w[2])}
    )

.. automodule:: pgml.scenarios
   :members:
   :show-inheritance:
