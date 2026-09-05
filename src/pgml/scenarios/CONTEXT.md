# Interface ledger: scenarios (batched, reproducible sampling)

Config-driven generation of batched operating points for ML training data. A
`ScenarioConfig` + `seed` deterministically defines the batch — saving the config
reproduces the dataset (reproducibility is paramount). Reuses the already-batched
solver (`solve_power_flow` / `solve_harmonic_flow` broadcast a leading scenario dim;
verified `batched == loop-of-individual`).

## Public API (IMPLEMENTED — increment 1)
`from pgml.scenarios import ...`
- Distributions (closed-form `icdf(u)` for QMC; `u in [0,1]`): `Uniform(low,high)`,
  `Normal(loc,scale)`, `LogNormal(loc,scale)`, `LogUniform(low,high)`, `Constant(value)`.
  Discriminated union `Distribution` (field `kind`).
- `Selector(component="load"|"generator"|"source", ids=None, consumer_type=None)` ->
  `.resolve(grid) -> [ids]` (None+None = all of that kind; filters AND). `component="source"`
  targets the slack Source(s) (for the `u_ref` field; a source has no `consumer_type`).
- `ParameterSpec(name, selector, distribution, field="p"|"q"|"pq"|"u_ref", mode="scale"|"absolute",
  per="each"|"shared", correlation=None, symmetry="balanced"|"independent"|"small_imbalance",
  imbalance=0.0)`. `pq` varies P&Q by the same factor (scale only).
  - SOURCE-VOLTAGE field `field="u_ref"` (requires `selector.component="source"`, `mode="scale"`
    only, `symmetry="balanced"`, no harmonic options; `per`/`correlation` as usual): writes a
    per-source `operating_point[source_id] = {"u_ref_scale": Tensor[B]}` — a per-scenario
    multiplier the ideal-slack solve applies to the Source's `u_ref_v` (a BATCHED fundamental
    boundary; the network side stays operating-point-independent). Recorded in `samples` like
    any power draw. `is_source_voltage` flags it.
  - `correlation=Correlation(factor, rho)` couples matched components through a shared
    `LatentFactor` (single-factor Gaussian copula; rho=0 == `per="each"`, rho=1 ==
    `per="shared"`; supersedes `per`). Marginal distribution preserved. Composes with EVERY
    `symmetry`: under `independent` the coupling applies per PHASE draw (that symmetry has no
    component-level base), so a spec keeps its per-phase asymmetry and still co-moves.
    Correlation is what keeps an AGGREGATE varying — with rho=0 the mean over `N` matched
    components concentrates as `1/sqrt(N)`, so a few hundred independent loads leave the
    total demand nearly constant however wide the marginal is.
  - `symmetry` (per-phase, power fields only): `balanced` writes a scalar total
    (`p_w`/`q_var`, split equally downstream); `independent` draws each phase separately;
    `small_imbalance` = balanced base × (1 + small per-phase perturbation of fractional std
    `imbalance`). The latter two write per-phase `p_per_phase_w`/`q_per_phase_var`, which
    auto-promote the solve to ASYMMETRIC (`resolve_asymmetric` mode="auto"). `imbalance>0`
    required iff `small_imbalance`.
  - HARMONIC fields `field="h_mag"|"h_phase"` + `orders=[...]` (>=2): write a batched
    `harmonic_injection` instead of an operating point. `h_mag` magnitude = sampled value ×
    a per-order reference fraction (`harmonic_reference`, distribution in [0,1]) ×, or ×
    stored spectrum mag (`mode="scale"`), or absolute pu. `harmonic_reference`:
    `"iec61000-3-2"` = IEC 61000-3-2 appliance CURRENT-emission fraction (PER DEVICE, from
    nominal P + node L-N voltage + `emission_class`; the physically correct current
    fingerprint reference); `"en50160"` = DIN EN 50160 supply-VOLTAGE compatibility level
    (a background-distortion SHAPE, NOT an emission model — kept for compatibility);
    `None` = absolute pu. `emission_class="A"|"B"|"C"|"D"|"auto"` (valid only with the IEC
    reference; `"auto"` resolves per device from `consumer_type`+P). `h_phase` sets the
    phase (deg, `mode="absolute"`). Per-device injection is seeded from the stored
    `StaticSpectrum` so unspecified orders survive. Harmonic fields reject
    `correlation`/per-phase `symmetry`.
  - LOAD-DEPENDENT EMISSION fields `field="h_floor"|"h_floor_phase"|"h_slope"` (+ `orders`,
    `mode="absolute"` only; `h_floor` drawn from `[0, 1]`): drawn per (device, order) like
    the emission and folded into the device's realized (mag, phase) AFTER every spec has
    written, against the loading `lam` the device's OWN power draw realised (drawn active
    power over nameplate; per-phase draws average their phase ratios; `1` when no power
    spec varies the device; floored at `emission.LOADING_FLOOR` = 0.05). `h_floor` = the
    affine law's load-independent share `|A_h|/(|A_h|+|B_h|)` — magnitude × `|c(lam)|`,
    phase + `arg c(lam)` with `c = affine_emission_correction` (rated point unchanged, `0`
    = proportional bit-for-bit); `h_floor_phase` = `arg A_h − arg B_h` [deg]; `h_slope`
    adds `s_h·(lam−1)` [deg] (`phase_slope_shift`). `ParameterSpec.is_emission_law`;
    `EMISSION_LAW_FIELDS` / `HARMONIC_FIELDS` name the field sets. Realized columns:
    `"<spec>_mag"` / `"<spec>_phase"` are post-law; `"<spec>_loading"` `[B, n_dev]` is the
    loading the law read. `docs/pgml/modeling/harmonic-emission.md`.
  - `per="fixed"` (harmonic fields only): ONE draw per matched component held across every
    scenario of the batch (a device's signature), from a stream seeded by `config.seed` and
    the spec name (`zlib.crc32`), consuming NO cube column — every other draw is unchanged.
    `samples[<spec>]` still carries `[B, n_dev, n_ord]` (the draw broadcast). The preset's
    `emission_persistence="device"` sets it on every harmonic spec of loads and PV.
- `pgml.scenarios.emission` — the ONE emission-law definition both recipes apply:
  `affine_emission_correction(lam, floor, delta_deg) -> complex` (`z(lam)/(lam·z(1))`,
  `z = floor·e^{jδ} + (1−floor)·lam`), `phase_slope_shift(slope_deg, lam)`, `LOADING_FLOOR`.
  The composition path's `_emission_affine` delegates here.
- `LatentFactor(name)` — shared driver (one QMC dim). `Correlation(factor, rho∈[0,1])`.
- CALIBRATED SE RECIPE (`presets.py`) — the ONE source every state-estimation generator
  builds from (the pgl workflow's three tasks, the multi-grid corpus, `pgml.grids
  .se_benchmark_scenario_config`); a fix here reaches all of them.
  `se_random_scenario_config(grid, *, orders, n_samples, seed, method="sobol",
  load_scale=(0,1), load_correlation=0.5, imbalance=0.15, spectrum_fraction=(0,2),
  pv_scale=(0,1), pv_correlation=None, slack_voltage_std=0.0333,
  emission_floor=EMISSION_FLOOR=(0.43,0.73), emission_floor_phase_deg=(100,150),
  phase_slope_deg=(−25,25), emission_persistence="scenario"|"device") -> ScenarioConfig`
  (preset v3: the load-dependent emission
  law drawn per device and order for loads AND PV — `load_emission_{floor,floor_phase,
  slope}` / `pv_emission_*` specs; a `(0,0)` range emits no spec, all three `(0,0)` = the
  proportional v2 recipe bit-for-bit) and
  `se_coherent_scenario_config(grid, *, orders, n_scenarios, n_steps, seed,
  mode="composed"|"fingerprint", name, step_size_s=900, start_time=HIGH_ACTIVITY_START_TIME,
  n_modes=2, dwell=0.9, mode_bank_seed=None, fingerprint_fraction=(0,1), activity_scale=1.0,
  behavioral_coupling=0.3, cloud_coupling=0.5, composition=None, profile=None, + the same
  fundamental knobs) -> CoherentSpectrumConfig`. `orders` is the SOLVED set (order 1 is not
  injected; a fundamental-only set yields the operating-point specs alone, a coherent one
  raises); every injected order gets a load magnitude + phase draw and, on a PV grid, an
  inverter magnitude + phase draw, so ANY requested order set flows through. An order
  outside the IEC 61000-3-2 table raises. `SE_PRESET_VERSION`, `HIGH_ACTIVITY_START_TIME`
  ("2024-06-21T16:00:00"), `LOAD_PHASE_SPAN_DEG`, `PV_EMISSION_HIGH`, `PV_PHASE_SPAN_DEG`.
- EN 50160 per-order VOLTAGE limits: `en50160_limits() -> {order: max_pu}`,
  `en50160_limit(order)` (loads `pgml/data/standards/en50160.yaml`; `PGML_EN50160` env
  override). These are supply-voltage compatibility levels, NOT an appliance emission model.
  `en50160_provenance() -> {source, override, sha256}` identifies the ACTIVE table.
- IEC 61000-3-2 appliance CURRENT-emission limits (`iec61000_3_2.py`; packaged
  `pgml/data/standards/iec61000_3_2.yaml`, `PGML_IEC61000_3_2` env override). The default
  reference for device current fingerprints. `iec61000_3_2_limits(class=None) -> dict` (full
  table or one class's `{unit, limits, ...}`; Class B expanded to 1.5×A);
  `iec61000_3_2_fraction(order, *, emission_class, p_w, u_ln_v, power_factor=1.0) -> float`
  (limit → fraction of `I1 = p_w/(u_ln_v·pf)`; Class A/B amps/I1, Class C percent [h3 ×λ],
  Class D mA/W·p_w/1000/I1 [P cancels]; clamped ≤1.0; absent order → 0.0);
  `resolve_emission_class(consumer_type, p_w) -> "A"|"B"|"C"|"D"` (auto map: office(IT)≤600W
  → D else A; everything else incl. household/EV/PV → A; lighting → C, no enum member yet);
  `iec61000_3_2_device_caps(grid, ids, orders, *, emission_class="auto") -> {id:{order:frac}}`
  (per-device caps; PER-PHASE current = total P/phase count; off the autograd tape);
  `iec61000_3_2_provenance() -> {source, override, sha256}` (the ACTIVE table's identity).
- NODE-COHERENT harmonic "fingerprints" (temporal sequences):
  `CoherentSpectrumConfig(selector, orders, n_steps T, n_scenarios B, n_modes=2, seed,
  mag_distribution, harmonic_reference="iec61000-3-2", emission_class="auto",
  phase_distribution, jitter_mag, ar1_rho, dwell, step_size_s,
  resample_modes_per_scenario, mode_bank_seed=None)` ->
  `sample_coherent_spectra(grid, config) -> SampledScenarios`. Each device draws `n_modes`
  base spectra (fingerprint); over T steps it STICKS to a mode (Markov `dwell`) and WANDERS
  (AR(1) `ar1_rho` jitter), clamped to the per-order emission reference (IEC 61000-3-2
  PER-DEVICE cap by default; `en50160`/`None` optional). `mode_bank_seed` seeds ONLY the
  fingerprint (mode) bank: `None` draws it from the `seed` stream (byte-identical to today);
  an explicit value pins a DISTINCT bank (held-out unseen-fingerprint test set) while every
  other setting is shared. `harmonic_injection` is `[B,T]` per
  (device, order); `samples` records `<name>_mode [B,n_dev,T]` (attribution label),
  the REALIZED `<name>_mag`/`<name>_phase [B,n_dev,n_ord,T]` (audit columns, see below),
  `<name>_device_ids`, `time_s [T]`.
  - `parameters=[ParameterSpec,...]` + `factors=[LatentFactor,...]` add a FUNDAMENTAL
    operating-point variation on top of the fingerprint: the SAME `ScenarioConfig` machinery
    (Sobol cube / copula / per-phase symmetry / source `u_ref` scale), drawn ONCE PER SCENARIO
    (`[B]`, constant across the T steps — the solve broadcasts the `[B]` fundamental against
    the `[B,T]` injection). Harmonic fields (`h_mag`/`h_phase`) are REJECTED (the fingerprint
    owns harmonics). The `[B]` operating point + the raw `[B,...]` draws land in
    `operating_point` / `samples`. The op cube is seeded from a stream DISTINCT from the
    fingerprint RNG, so the realized `harmonic_injection` is byte-identical with/without
    `parameters` (reproducible reconstruction via `pgl.data.physics._reconstruct_sampled`);
    empty `parameters` (default) = the original fingerprint-only behavior (P/Q nominal,
    source at `u_ref_v`).
  - `profile=LoadProfileConfig(...)` + `start_time` (ISO 8601, REQUIRED with `profile`) make
    the fundamental P/Q TIME-VARYING over the T steps instead of `[B]`-constant: each
    profiled device's per-scenario base P/Q (from `parameters` if set, else nominal) is
    multiplied by a multi-scale synthetic factor `f_seasonal·f_weekly·f_daily·f_short`,
    class-aware by `consumer_type` (household/office/restaurant/ev/industrial presets +
    a `pv` solar bell that is ZERO at night with a seasonally-widening daylight window).
    The operating-point totals then gain the step axis (`[B,T]`, aligned with the `[B,T]`
    injection) → `run_scenarios` yields `[B,T,H,N]` with a moving fundamental. `samples`
    additionally records `<name>_profile_factor [B,n_dev,T]` (ground truth), `<name>_profile_device_ids
    [n_dev]`, and `time_unix_s [T]` (absolute epoch seconds; the relative `time_s` stays).
    Harmonic magnitudes are RELATIVE to the fundamental current, so the profile already
    scales the absolute harmonic current (no extra coupling). The profile draws on a stream
    DISTINCT from the fingerprint, Markov, jitter, and op-cube streams, so `profile=None`
    (default) is byte-identical to the fingerprint-only output. See `profiles.py` for the
    preset shapes + correlation model (a per-scenario shared `behavioral` latent scales all
    non-pv daily amplitudes; a shared `cloudiness` latent scales all pv output; per device an
    idiosyncratic level / amplitude / daily phase-offset draw + an AR(1) short-term term).
  - `load_profile_factors(grid, config) -> ProfileDraw(factor[B,n_dev,T], device_ids[n_dev],
    time_unix_s[T])` and `apply_load_profiles(grid, config, operating_point) ->
    (operating_point[B,T], samples)` are the profile generator + operating-point lift
    (`pgml.scenarios.profiles`); `sample_coherent_spectra` calls them when `profile` is set.
  - `composition=CompositionConfig(...)` + `start_time` (REQUIRED — the activity model is
    temporal) makes the covered aggregated loads a SUM of statistical member devices whose
    per-step activity drives BOTH the fundamental power AND the injected spectrum (a
    consistent load-to-spectrum mapping). It SUPERSEDES the mode-bank fingerprint + the
    fundamental for the loads it covers (those ids are dropped from the fingerprint device
    set; their operating point + injection come from the composition; any `parameters`/
    `profile` on them is superseded). Loads with no matching rule (or outside the
    composition selector) stay on the fingerprint. The mixed `[B]`/`[B,T]` operating point
    is unified to `[B,T]`. Config surface (`pgml.scenarios.config`): `DeviceState(name,
    power_fraction, spectrum_scale=1, weight=1)`; `DeviceClassSpec(name, sign=±1,
    rated_power_w=[lo,hi], power_factor=1, harmonic_magnitude={order:[lo,hi]} (FRACTION of
    the device's own fundamental current, loosely IEC 61000-3-2-shaped — NOT the standard's
    limits), harmonic_phase_deg={order:[lo,hi]}, gamma=[lo,hi] (mag∝lam**gamma, drawn per
    member per order), phase_slope_deg=[lo,hi] (ang=ang0+s·(lam−1)), activity_preset
    ("household"|"office"|"ev"|"restaurant"|"industrial"|"pv"|"flat"; reuses the profile
    daily shapes), discrete_activity=True (on/off Markov) | False (continuous rate, e.g. PV/
    base load), on_off_dwell=[lo,hi], loading_min, loading_mean=[lo,hi], loading_jitter,
    loading_rho, states=[DeviceState,...] (multi-state: heating vs inverter), state_dwell,
    emission_class=None ("A"/"B"/"C"/"D" | None = auto by per-phase power))`. Every drawn
    member harmonic ratio is CAPPED at the member's IEC 61000-3-2 emission fraction
    (evaluated at the effective scaled power) so composed aggregates stay inside the same
    physical envelope the randomized `h_mag` sampling references;
    `ClassCount(class_name, count=[min,max], power_share=1)`; `ConsumerComposition(
    consumer_type=None, load_ids=None, classes=[ClassCount,...])` (rule match: load_ids >
    consumer_type > fallback); `CompositionConfig(selector=None (all loads), classes=[...]
    (=default_device_classes()), compositions=[...] (=default_compositions()),
    scale_to_nominal=True (share-weighted installed capacity → load p_nom_w),
    max_injection_pu=3.0 (cap the residual-THD blow-up near a net-zero fundamental),
    behavioral_coupling=0.3, cloud_coupling=0.5, activity_scale=1.0 (multiplier on every
    member's diurnal availability RATE, clamped back to a probability — the class presets are
    per-device duty cycles, so a composed aggregate sits far below installed capacity and a
    sequence's FUNDAMENTAL barely moves; raise it to place the population in a loaded band),
    roster_seed=None (a held-out roster bank);
    `.class_names()`)`. `default_device_classes()` = 7 built-ins (base_linear,
    electronics_smps, ev_charger, pv_inverter, inverter_drive [multi-state],
    heat_pump_inverter, resistive_heating), balanced so the emission is INFORMATIVE about the
    drawn power (power share on the kW-scale nonlinear classes, EV load-dependence flat
    enough that its absolute emission tracks loading, spread power factors) and covering the
    odd orders through h19; `DEVICE_LIBRARY_VERSION` stamps the roster into dataset metadata.
    SILENT-ORDER GUARD: `CoherentSpectrumConfig` rejects a requested order no class in the
    roster emits at (and, without a composition, an order the `harmonic_reference` table does
    not list); `allow_silent_orders=(...)` declares deliberate silence.
    `composition_silent_orders(classes, orders) -> [order]` is the check itself.
  - `sample_device_composition(grid, config) -> CompositionDraw(operating_point[B,T],
    harmonic_injection {id:{order:(mag[B,T],phase[B,T])}}, samples, composed_ids)` and
    `resolve_composed_ids(grid, comp) -> [id]` (`pgml.scenarios.composition`);
    `sample_coherent_spectra` calls them when `composition` is set. The composition draws on
    roster + temporal streams DISTINCT from the fingerprint/op-cube/profile, so
    `composition=None` is byte-identical. Attribution `samples` (fixed shapes; `name` =
    `config.name`): `<name>_class_p_w [B,n_agg,n_class,T]` (signed per-class power),
    `<name>_class_active [B,n_agg,n_class,T]` int64 (active member count), `<name>_cap_binding
    [B,n_agg,n_ord,T]` (where the cap bound), `<name>_agg_ids [n_agg]`, `<name>_roster_p_rated
    [n_agg,n_class,max_count]` (per-member rated powers), plus the realized aggregate
    spectrum `<name>_composed_mag`/`<name>_composed_phase [B,n_agg,n_ord,T]` (audit columns,
    see below — post-cap, on the `<name>_agg_ids` axis). The `n_class` axis is ordered as
    `config.composition.classes` — names via `config.composition.class_names()` (they live in
    the config, not a sample tensor, since samples are tensor-only). `P_agg` may go
    net-negative under PV (the Load then injects).
- `ScenarioConfig(n_samples, seed=0, method="sobol"|"lhs"|"independent", parameters=[...],
  factors=[LatentFactor(...)])`. A spec's `correlation.factor` must name a declared factor.
- `sample(grid, config) -> SampledScenarios(operating_point, samples, n_samples, config)`
  where `operating_point = {id: {"p_w": Tensor[B], "q_var": Tensor[B]}}` (a Source id entry
  instead carries `{"u_ref_scale": Tensor[B]}`) and
  `samples = {param_name: Tensor[B, d]}` (raw realized values; ML input record).
- REALIZED-INJECTION AUDIT COLUMNS — ONE convention across all three generating paths, so a
  written dataset records WHAT WAS INJECTED and not only the draw behind it (a draw is a
  fraction of a per-device emission reference; a composed aggregate has no draw at all).
  A block is the pair `<key>_mag` / `<key>_phase` — magnitude in per unit of the device's
  own fundamental current, phase in degrees, i.e. exactly the `harmonic_injection` values
  the solver receives — over a device axis given by the block's id column:
  | path | `<key>` | shape | device axis |
  |---|---|---|---|
  | randomized `h_mag` spec (`sampler`) | `<spec.name>` | `[B, n_dev, n_ord]` | `<spec.name>_device_ids` |
  | coherent fingerprint (`harmonics`) | `<config.name>` | `[B, n_dev, n_ord, T]` | `<config.name>_device_ids` |
  | device composition (`composition`) | `<config.name>_composed` | `[B, n_agg, n_ord, T]` | `<config.name>_agg_ids` |
  The order axis follows the writing spec's / the config's `orders`. The raw draw stays
  under its own `<spec.name>` (it documents the DRAW; for a referenced spec the realized
  magnitude is `draw x reference`, and the reference is per device). `<name>_mode_base_mag`
  is the fingerprint's mode bank, NOT a realized injection. Purely additive parquet sample
  columns: `SCHEMA_VERSION` is unaffected (it versions the `pgml.schemas` contract) and a
  dataset written before them reads back unchanged. Consumer: `pgl.data.validate`
  (`i_h_emission_pct`, preferred over reconstructing `I(h)=Y(h)·V(h)`).
- CARTESIAN sweep (pgm-style, deterministic): `CartesianAxis(name, selector, values=[...],
  field, mode)`, `CartesianConfig(axes=[...])` -> `cartesian_sample(grid, config) ->
  SampledScenarios` (B = prod(len(axis.values)); each axis level applied to all matched
  comps; list id-selector axes for per-component sweeps).
- PERTURBATION sweep (one error per node): `Perturbation(name, field="p"|"q"|"pq",
  mode="scale"|"delta"|"set", value)`, `perturbation_sweep(grid, selector, perturbation) ->
  SampledScenarios` with `B = #targets` — scenario `j` perturbs ONLY target `j`'s operating
  point (diagonal; off-diagonal nominal). Records `ParameterPerturbation` ground truth in
  `SampledScenarios.perturbations`; `samples` has `<name>_target_id [B]` + `<name>_perturbed_<f> [B]`.
  Scope = P/Q injection errors; network-parameter (line/transformer) perturbation is deferred
  to the inverse/parameter-recovery phase (needs a branch selector + matrix-valued ground truth).
- `run_scenarios(grid, spec, *, calculation="power_flow"|"harmonic", harmonic_orders=None,
  slack="ideal", symmetry=None, dtype, device, chunk_size=None, output_device=None) ->
  ScenarioResult(v, index, sampled, frequencies_hz, converged, failed_states)`. `converged` is
  True iff EVERY scenario converged; `failed_states` lists the non-converged scenario indices.
  A batch NEVER raises on a failed scenario — its best-effort `v` is returned and the solver
  logs the failures (so a large sweep yields data + diagnosable failures). `chunk_size` streams
  the batch in slices of that many scenarios and concatenates (VRAM tiling for a batch whose
  dense `[B,H,N,N]` system would not fit); the result equals the whole solve within the solver
  tolerance and stays differentiable. Applies to the coherent `[B,T,H,N]` path too — the
  slice is along the SCENARIO axis `B` (each scenario's full `T`-step sequence solves together).
  `chunk_size` bounds the per-solve WORKSPACE, NOT the collected OUTPUT: the full `[B,...]` `v`
  still accumulates on `device`, so on a GPU a large `B` OOMs regardless of `chunk_size`. Set
  `output_device="cpu"` to move each chunk's result off the GPU as produced (VRAM bounded to
  one chunk; dataset written from host memory) — for non-differentiable data generation; leave
  `None` to keep `v` on the solve device for a differentiable GPU pipeline.
  `symmetry` forwards to the solver (None/"auto" lets per-phase samples promote to asymmetric).
  `spec` = `ScenarioConfig` | `CartesianConfig` | `CoherentSpectrumConfig` (forces harmonic,
  defaults `harmonic_orders=[1, *orders]`) | a pre-built `SampledScenarios`.
  `v` is `[B,N]` (power_flow), `[B,H,N]` (harmonic), or `[B,T,H,N]` (coherent; timestamps in
  `sampled.samples["time_s"]`).
- `SampledScenarios` also carries `harmonic_injection = {id: {order: (mag, phase)}}` (mag/phase
  `[B]` for spec-varied orders, `[B,T]` for coherent; orders seeded only from a device's stored
  `StaticSpectrum` and untouched by any spec stay plain float pairs — the solver broadcasts
  scalars), passed to `solve_harmonic_flow`. Two specs writing the same `(device, field, order)`
  (or the same power field of one component) raise `InputError` at resolve time.
- SPECTRUM-SWEEP (diagonal per-target harmonic-injection sweep): `SpectrumSweepConfig(name,
  selector, orders, magnitudes_pu, phases_deg)` — serializable config; classmethod
  `SpectrumSweepConfig.from_spectrum(selector, spectrum, *, name="injection")` builds it from
  a `{order: (mag_pu, phase_deg)}` dict (order 1 dropped). `spectrum_sweep(grid,
  selector_or_config, spectrum=None, *, name="injection") -> SampledScenarios` — scenario `i`
  injects the spectrum at target `i` ONLY (every other device silent); `B = #targets`.
  Harmonic analogue of `perturbation_sweep` — for "inject one spectrum at each node, measure
  how it spreads". `harmonic_injection` is the diagonal `{device_id: {order: (mag[B], phase[B])}}`;
  `samples["<name>_id"]` records the injected device id per scenario.
  `run_scenarios` also accepts `SpectrumSweepConfig` as `spec` (forces `calculation="harmonic"`,
  defaults `harmonic_orders=[1, *config.orders]`).
- PERSISTENCE (parquet training data): `write_dataset(result, path, *, layout="wide"|"long",
  compression="zstd", also_csv=False) -> Path`, `read_dataset(path) -> LoadedDataset(v, samples,
  frequencies_hz, node_ids, phase_codes, config, perturbations, meta)`. Writes a dataset DIR:
  `voltages.parquet` (long = tidy row per scenario×step×freq×node-phase; wide = compact
  array cols of `v_re`/`v_im` flattened over `[H*N]` per scenario×step), `samples.parquet`
  (B-leading sampled inputs as array cols, dtype-preserving), `meta.json` sidecar (config
  JSON + seed + frequencies + node/phase index + dims + ParameterPerturbation rows). Both
  layouts read back the IDENTICAL `v` (re-`torch.complex`-ed to the original shape/dtype);
  wide is the fast tensor cache, long the analysis/interchange table. `also_csv=True` writes
  a tidy `voltages.csv` (long layout, regardless of `layout`) alongside the parquet for manual
  inspection. Result I/O (detached).
  GENERATION PROVENANCE in `meta.json` — `config_hash` (`config_hash(config_or_json) -> str`,
  a 16-hex fingerprint of the serialized config: equal hash ⇒ same batch, so it is what a
  reuse gate compares), plus `generation_provenance()`: `provenance` (`pgml.provenance
  .code_provenance()` — commit, dirty flag, versions), `device_library_version`, and
  `standards` (the ACTIVE EN 50160 / IEC 61000-3-2 tables with `override` + content hash; a
  `PGML_*` env override silently changes every generated magnitude). `read_dataset` needs
  none of them — a dataset written before they existed simply lacks the keys.

## Storage dispatch / state of charge (`storage.py`)
- `integrate_soc(requested_power_w[*,T], dt_s, *, energy_capacity_wh, soc0, soc_min,
  soc_max, efficiency_charge, efficiency_discharge, p_rated_w, dtype, device) ->
  StorageDispatchResult(realized_power_w[*,T], soc[*,T+1], energy_wh[*,T+1])`. Realizes a
  requested signed power sequence (>0 discharge) under the SoC reserve/cap + power rating;
  OpenDSS energy equations (`E[t+1]=E[t]−P·dt/η_dis` discharge / `+|P|·η_chg·dt` charge).
  The dispatch RULE is the caller's (off-tape Python); the realized power + SoC recurrence
  are torch (gradient w.r.t. the setpoint VALUE, not the rule). No capacity => only the
  rating clamps (`soc`/`energy_wh` None). Batched over a leading scenario dim; loops over T.
- `dispatch_storage(storage, requested_power_w, dt_s, *, soc0=None)` — reads the
  `Storage` element's own state params. `storage_operating_point(power_by_id, q_by_id=None)`
  -> a solver `operating_point` dict for one step (feeds `solve_power_flow`/`solve_harmonic_flow`).
  See `docs/pgml/modeling/der-pv-storage.md` §4.4.

## Conventions
- Unit-cube layout `U[B,D]`: one column per declared factor, then per spec a BASE block
  (component-level draws: #matched comps for `each`/correlated, 1 for `shared`, 0 for
  `independent`) then a PER-PHASE block (`independent`: one draw/phase; `small_imbalance`:
  one perturbation/phase). `U` from Sobol (QMC, recommended) / LHS / independent; each
  column -> `distribution.icdf` (or the copula score path). Same config+seed -> identical
  output (deterministic). Correlation/per-phase noise are torch ops on `U` (autograd-safe,
  no `.item()`); the sampled tensors feed `operating_point`, gradients flow from there.
- `mode="scale"` multiplies the component's nominal `p_nom_w`/`q_nom_var`; `"absolute"`
  sets W/var directly. Unset of P or Q -> solver keeps the nominal for that one.

## Deferred (next increments — see memory `batching-scenarios-design`)
- Network-parameter perturbation sweep (line/transformer impedance errors, for parameter
  recovery) — extends `perturbation_sweep` with a branch-aware selector + matrix ground truth.
- Network-parameter & TOPOLOGY (switch-state) batching; MULTI-GRID batching.
- Beta / scipy-backed distributions (no closed-form icdf).
- **Long-term temporal-pattern simulation mode** — the FUNDAMENTAL P/Q recurrence is now
  covered by `CoherentSpectrumConfig.profile` (`LoadProfileConfig`): a diurnal / weekly /
  seasonal multi-scale generator over an absolute `start_time`, class-aware by
  `consumer_type` (incl. a `pv` solar bell). Remaining: the HARMONIC fingerprint itself is
  still short/medium-term only (Markov mode dwell + AR(1) jitter over `T`) — a discrete
  device on/off schedule / occupancy process that also switches the FINGERPRINT mode on the
  same diurnal clock (an EV charger's harmonic signature appearing only while it charges)
  would tie the harmonic attribution to the profile. Couples to the pgl temporal model's
  context length.
