# Interface ledger: scenarios (the batch contract)

A scenario batch is a set of per-component DELTAS on one grid: what a spec names varies,
every other component and field keeps the grid's nominal value. This subpackage owns that
contract end to end — how a batch is declared, drawn, solved and persisted — and nothing
about which variations a particular study should draw.

A construct belongs here if a user who has never heard of harmonic state estimation would
still reach for it when sweeping a grid. A device population, a calibrated emission range or
a training-task design does not; those live in the package that calibrates them, and reach
this one through the `ScenarioSpec` protocol.

Reuses the already-batched solver: `solve_power_flow` / `solve_harmonic_flow` broadcast a
leading scenario dim (verified `batched == loop-of-individual`).

## Public API (IMPLEMENTED)
`from pgml.scenarios import ...`

### The batch object
- `SampledScenarios(operating_point, samples, n_samples, config, harmonic_injection={},
  node_sources=[], perturbations=[], n_steps=1, shared_samples={})` — frozen dataclass.
  - `operating_point = {appliance_id: {...}}` — pass straight to the solver. Entries hold
    totals (`p_w` / `q_var`) and/or per-phase overrides (`p_per_phase_w` /
    `q_per_phase_var`, which auto-promote the solve to asymmetric); a Source id entry
    instead carries `{"u_ref_scale": Tensor}`.
  - `harmonic_injection = {id: {order: (mag_pu, phase_deg)}}` — passed to
    `solve_harmonic_flow`. Orders seeded only from a device's stored `StaticSpectrum` and
    untouched by any spec stay plain float pairs (the solver broadcasts scalars).
  - `node_sources` — realized `NodeHarmonicSource` entries (an upstream background or a
    node-level disturbance).
  - `n_steps` = `T`, the STEP axis. `1` (default) is a snapshot batch (`v` comes back
    `[B, N]` / `[B, H, N]`); `T > 1` declares a sequence batch (`[B, T, H, N]`). DECLARED,
    not inferred from a tensor rank: `[B, T, N]` and `[B, H, N]` have the same rank, so the
    persistence layer used to record one axis as the other.
  - `samples = {name: Tensor}` — per-scenario records, `[B, ...]` (the reproducible ML
    input record). `shared_samples` — records with NO leading scenario axis (a step-time
    vector, a device-id column, a per-device constant). The split is DECLARED because a
    record whose length happens to equal `B` is indistinguishable by shape.
  - `batch_shape` → `(B,)` or `(B, T)`. `all_samples` → both record dicts merged, the same
    view `read_dataset` returns.
  - `validate(grid=None)` — shape-only, run on construction: every operating-point,
    harmonic-injection and node-source batch axis must broadcast against `batch_shape`
    (leading `1` or `B`, step `1` or `T`), and no record may be declared both per-scenario
    and shared. With a `grid` it also resolves every written component id against that
    grid's in-service appliances. `run_scenarios` calls it with the grid it is about to
    solve, so a batch built against a different grid fails before the solver sees it.
  - RESERVED record keys a consumer may rely on: `time_s` `[T]` (relative step seconds),
    `time_unix_s` `[T]` (absolute epoch seconds), and per injection-writing spec
    `<key>_mag` / `<key>_phase` (the REALIZED injection, per unit of the device's own
    fundamental current and degrees) over the device axis `<key>_device_ids`.
- `batch_from_values(grid, *, n_samples, n_steps=1, p_w=None, q_var=None,
  p_per_phase_w=None, q_per_phase_var=None, u_ref_scale=None, harmonic_injection=None,
  node_sources=(), samples=None, shared_samples=None, config=None) -> SampledScenarios` —
  build a batch from tensors a caller already has (measured curves, an optimiser's iterate,
  a downstream generator's draw). Absent component or absent field = keep the nominal;
  there is no fill value to get wrong. Ids resolve against the grid, so a typo raises here
  instead of being silently ignored. Values pass through untouched, so gradients flow from
  them into the solve and their device/dtype are the solver's.
- `broadcast_operating_point(operating_point, b, t) -> dict` — lift a mixed `[B]` / `[B, T]`
  operating point to a uniform `[B, T]` (a `[B]` source `u_ref_scale` becomes `[B, 1]`);
  scalars and absent entries broadcast in the solver and are left untouched.

### Declaring a sampled batch (`config.py`)
- Distributions (closed-form `icdf(u)` for QMC; `u in [0,1]`): `Uniform(low,high)`,
  `Normal(loc,scale)`, `LogNormal(loc,scale)`, `LogUniform(low,high)`, `Constant(value)`.
  Discriminated union `Distribution` (field `kind`).
- `Selector(component="load"|"generator"|"source", ids=None, consumer_type=None)` →
  `.resolve(grid) -> [ids]` (None+None = all of that kind; filters AND). `component="source"`
  targets the slack Source(s) (for the `u_ref` field).
- `ParameterSpec(name, selector, distribution, field="p"|"q"|"pq"|"u_ref", mode="scale"|"absolute",
  per="each"|"shared"|"fixed"|"class", correlation=None,
  symmetry="balanced"|"independent"|"small_imbalance", imbalance=0.0, orders=None,
  harmonic_reference=None, emission_class="auto")`. `pq` varies P and Q by the same factor
  (scale only).
  - SOURCE-VOLTAGE `field="u_ref"` (requires `selector.component="source"`, `mode="scale"`,
    `symmetry="balanced"`, no harmonic options): writes a per-source
    `operating_point[source_id] = {"u_ref_scale": Tensor[B]}` — a per-scenario multiplier the
    ideal-slack solve applies to `u_ref_v` (a BATCHED fundamental boundary; the network side
    stays operating-point independent). `is_source_voltage` flags it.
  - `correlation=Correlation(factor, rho)` couples matched components through a shared
    `LatentFactor` (single-factor Gaussian copula; rho=0 == `per="each"`, rho=1 ==
    `per="shared"`; supersedes `per`). The marginal is preserved. Composes with EVERY
    `symmetry`: under `independent` the coupling applies per PHASE draw, so a spec keeps its
    per-phase asymmetry and still co-moves. Correlation is what keeps an AGGREGATE varying —
    with rho=0 the mean over `N` matched components concentrates as `1/sqrt(N)`, so a few
    hundred independent loads leave total demand nearly constant however wide the marginal.
  - `symmetry` (per-phase, power fields only): `balanced` writes a scalar total (split
    equally downstream); `independent` draws each phase separately; `small_imbalance` =
    balanced base × (1 + a small per-phase perturbation of fractional std `imbalance`). The
    latter two write per-phase overrides, which auto-promote the solve to ASYMMETRIC.
    `imbalance>0` required iff `small_imbalance`.
  - HARMONIC fields `field="h_mag"|"h_phase"` + `orders=[...]` (>=2): write a batched
    `harmonic_injection` instead of an operating point. `h_mag` magnitude = the sampled value
    × a per-order reference fraction (`harmonic_reference`, drawn in [0,1]), × the stored
    spectrum magnitude (`mode="scale"`), or absolute pu. `harmonic_reference`:
    `"iec61000-3-2"` = the IEC 61000-3-2 appliance CURRENT-emission fraction (PER DEVICE,
    from nominal P + node L-N voltage + `emission_class`; the physically correct current
    fingerprint reference); `"en50160"` = the DIN EN 50160 supply-VOLTAGE compatibility level
    (a background-distortion SHAPE, NOT an emission model; kept for compatibility); `None` =
    absolute pu. `emission_class="A"|"B"|"C"|"D"|"auto"` (valid only with the IEC reference).
    `h_phase` sets the phase (deg, `mode="absolute"`). Per-device injection is seeded from
    the stored `StaticSpectrum` so unspecified orders survive. Harmonic fields reject
    `correlation` and per-phase `symmetry`.
  - LOAD-DEPENDENT EMISSION fields `field="h_floor"|"h_floor_phase"|"h_slope"` (+ `orders`,
    `mode="absolute"` only; `h_floor` drawn from `[0, 1]`): drawn per (device, order) and
    folded into the device's realized (mag, phase) AFTER every spec has written, against the
    loading `lam` the device's OWN power draw realised (drawn active power over nameplate;
    per-phase draws average their phase ratios; `1` when no power spec varies the device;
    floored at `emission.LOADING_FLOOR` = 0.05). `h_floor` = the affine law's
    load-independent share `|A_h|/(|A_h|+|B_h|)` — magnitude × `|c(lam)|`, phase +
    `arg c(lam)` with `c = affine_emission_correction` (the rated point is unchanged, `0` =
    proportional bit-for-bit); `h_floor_phase` = `arg A_h − arg B_h` [deg]; `h_slope` adds
    `s_h·(lam−1)` [deg] (`phase_slope_shift`). `ParameterSpec.is_emission_law`;
    `EMISSION_LAW_FIELDS` / `HARMONIC_FIELDS` name the field sets. Realized columns:
    `"<spec>_mag"` / `"<spec>_phase"` are post-law; `"<spec>_loading"` `[B, n_dev]` is the
    loading the law read.
  - `per="fixed"` (harmonic fields only): ONE draw per matched component held across every
    scenario of the batch (a device's signature), from a stream seeded by `config.seed` and
    the spec name (`zlib.crc32`), consuming NO cube column — every other draw is unchanged.
    `samples[<spec>]` still carries `[B, n_dev, n_ord]` (the draw broadcast).
  - `per="class"` (harmonic fields only): ONE draw for every matched component, held across
    the batch and IDENTICAL for every seed (seeded by the spec name alone; `samples[<spec>]`
    is `[B, 1, n_ord]`) — a class constant.
- `ScenarioConfig(n_samples, seed=0, method="sobol"|"lhs"|"independent", parameters=[...],
  factors=[LatentFactor(...)], background=None)`. A spec's `correlation.factor` must name a
  declared factor. `.sample(grid)` makes it a `ScenarioSpec`.
- `CartesianAxis(name, selector, values=[...], field, mode)`, `CartesianConfig(axes=[...])` —
  the deterministic product sweep (pgm-style), `B = prod(len(axis.values))`; each axis level
  applies to all matched components (list id-selector axes for per-component sweeps).
  `.sample(grid)`.
- `BackgroundHarmonicConfig(magnitude_pu={order: pu}, phase_deg={}, node_id=None,
  source_power_va=20e6, drift_std=0.0, drift_phase_deg=0.0, drift_rho=0.99)` — a
  slowly-varying UPSTREAM harmonic background, the one way to express supply-side
  distortion. Realized as the Thevenin source of `docs/pgml/modeling/error-injection.md`,
  present in EVERY scenario: by default each in-service `Source` node receives an injection
  carrying the SAME realized spectrum (one upstream network state seen through every point
  of common coupling; an explicit `node_id` narrows it). `source_power_va` is constant across
  the batch, so `Y(h)` is unchanged scenario-to-scenario and only the Norton current varies.
  The level drifts along the STEP axis as an AR(1) shared by every order; a snapshot batch
  (`T == 1`) has no step to walk, so its scenarios draw independently from the drift's
  stationary distribution. Empty `magnitude_pu` (default) disables it entirely.
  Referenced from `ScenarioConfig.background`.
- `python -m pgml.scenarios.config --json-schema | --example` prints the contract or a valid
  example YAML.

### Drawing a batch (`sampler.py`)
- `sample(grid, config) -> SampledScenarios` and
  `cartesian_sample(grid, config) -> SampledScenarios`.
- `unit_samples(n, d, *, method="sobol", seed=0) -> Tensor[n, d]` — the shared unit-cube
  draw every path transforms through a distribution's `icdf` (float64, CPU, deterministic).
- `nominal_power(grid) -> {id: NominalPower(p_total, q_total, p_pp, q_pp, n)}` — the
  nameplate a `mode="scale"` operating point multiplies and the denominator the emission
  law's loading ratio reads. Tensor-valued nameplates pass through untouched, so a scaled
  operating point stays differentiable w.r.t. the rated power.

### Excitation primitives
- `perturbation_sweep(grid, selector, Perturbation(name, field="p"|"q"|"pq",
  mode="scale"|"delta"|"set", value)) -> SampledScenarios` with `B = #targets` — scenario
  `j` perturbs ONLY target `j` (diagonal; off-diagonal nominal). Records
  `ParameterPerturbation` ground truth in `.perturbations`; `samples` has
  `<name>_target_id [B]` + `<name>_perturbed_<f> [B]`. Scope = P/Q injection errors;
  network-parameter perturbation is deferred.
- `SpectrumSweepConfig(name, selector, orders, magnitudes_pu, phases_deg)` (+
  `.from_spectrum(selector, spectrum, *, name="injection")` from a
  `{order: (mag_pu, phase_deg)}` dict, order 1 dropped), `spectrum_sweep(grid,
  selector_or_config, spectrum=None, *, name="injection") -> SampledScenarios` — scenario
  `i` injects the spectrum at target `i` ONLY (every other device silent); `B = #targets`.
  The harmonic analogue of `perturbation_sweep`. `.harmonic_orders` is `[1, *orders]`, so
  `run_scenarios` switches to a harmonic calculation on its own.
- `NodeInjectionSweepConfig(node_ids, phases, orders, magnitudes_pu, phases_deg,
  source_power_va, kind="voltage")` (+ `.from_spectrum`), `run_node_injection_sweep(grid,
  config, *, slack="norton", dtype, device) -> ScenarioResult` — the per-node
  Thevenin/Norton disturbance source swept one node per scenario, `v [B, H, N]`.
- `build_background_sources(grid, config, shape, generator) -> [NodeHarmonicSource]` —
  realize a `BackgroundHarmonicConfig` at the given `(B, T)` batch shape. The entry point a
  generator with its own step axis uses; returns `[]` when no order is configured, so a
  caller can pass the result through unconditionally.

### The emission law and the standards tables
- `pgml.scenarios.emission` — the ONE emission-law definition:
  `affine_emission_correction(lam, floor, delta_deg) -> complex` (`z(lam)/(lam·z(1))`,
  `z = floor·e^{jδ} + (1−floor)·lam`), `phase_slope_shift(slope_deg, lam)`, `LOADING_FLOOR`.
- EN 50160 per-order VOLTAGE limits: `en50160_limits() -> {order: max_pu}`,
  `en50160_limit(order)` (loads `pgml/data/standards/en50160.yaml`; `PGML_EN50160` env
  override). These are supply-voltage compatibility levels, NOT an appliance emission model.
  `en50160_provenance() -> {source, override, sha256}` identifies the ACTIVE table.
- IEC 61000-3-2 appliance CURRENT-emission limits (`iec61000_3_2.py`; packaged
  `pgml/data/standards/iec61000_3_2.yaml`, `PGML_IEC61000_3_2` override). The default
  reference for device current fingerprints. `iec61000_3_2_limits(class=None) -> dict`
  (Class B expanded to 1.5×A); `iec61000_3_2_fraction(order, *, emission_class, p_w, u_ln_v,
  power_factor=1.0) -> float` (limit → fraction of `I1 = p_w/(u_ln_v·pf)`; Class A/B amps/I1,
  Class C percent [h3 ×λ], Class D mA/W·p_w/1000/I1 [P cancels]; clamped ≤1.0; absent order →
  0.0); `resolve_emission_class(consumer_type, p_w) -> "A"|"B"|"C"|"D"` (office(IT)≤600W → D
  else A; household/EV/PV → A; lighting → C); `iec61000_3_2_device_caps(grid, ids, orders, *,
  emission_class="auto") -> {id: {order: frac}}` (per-device caps; PER-PHASE current =
  total P / phase count; off the autograd tape); `iec61000_3_2_provenance()`.

### Running a batch (`run.py`)
- `class ScenarioSpec(Protocol)`: `sample(grid) -> SampledScenarios` plus an optional
  `harmonic_orders` hint. A non-empty hint declares that the spec only makes sense as a
  harmonic calculation, so `run_scenarios` switches `calculation` and takes the order set
  unless the caller named one. This is the seam a downstream generator plugs into; the
  serializable configs satisfy it themselves.
- `run_scenarios(grid, spec, *, calculation="power_flow"|"harmonic", harmonic_orders=None,
  slack="ideal", symmetry=None, dtype, device, chunk_size=None, output_device=None) ->
  ScenarioResult(v, index, sampled, frequencies_hz, converged, failed_states)`. `spec` is a
  `ScenarioSpec` or a pre-built `SampledScenarios`.
  - `v` is `[B, N]` (power flow), `[B, H, N]` (harmonic) or `[B, T, H, N]` (a sequence
    batch, `sampled.n_steps > 1`; timestamps in `sampled.all_samples["time_s"]`).
  - `converged` is True iff EVERY scenario converged; `failed_states` lists the
    non-converged SCENARIO indices (a sequence scenario counts as failed when ANY of its `T`
    steps failed). A batch NEVER raises on a failed scenario — its best-effort `v` is
    returned and the solver logs the failures, so a large sweep yields data plus diagnosable
    failures.
  - For `calculation="power_flow"` the operating-point-independent solve state (assembly,
    slack rows, factorization) is prepared ONCE via `prepare_power_flow` and reused across
    the batch and across every chunk. The harmonic path assembles per order inside
    `solve_harmonic_flow`.
  - `chunk_size` streams the batch in slices of that many SCENARIOS and concatenates (VRAM
    tiling for a batch whose dense `[B, H, N, N]` system would not fit); the result equals
    the whole solve within the solver tolerance and stays differentiable. Applies to the
    sequence path too — each scenario's full `T`-step sequence solves together.
    `chunk_size` bounds the per-solve WORKSPACE, NOT the collected OUTPUT: the full `[B, ...]`
    `v` still accumulates on `device`. Set `output_device="cpu"` to move each chunk's result
    off the GPU as produced (VRAM bounded to one chunk) — for non-differentiable data
    generation; leave `None` for a differentiable GPU pipeline.
  - `symmetry` forwards to the solver (None/"auto" lets per-phase samples promote to
    asymmetric).
- Two specs writing the same `(device, field, order)` — or the same power field of one
  component — raise `InputError` at resolve time: last-writer-wins would desync the
  recorded samples from the realized operating point.

### Persistence (`persistence.py`)
- `write_dataset(result, path, *, layout="wide"|"long", compression="zstd", also_csv=False,
  provenance=None) -> Path` — writes a dataset DIRECTORY: `voltages.parquet` (long = a tidy
  row per scenario×step×freq×node-phase; wide = compact array columns of `v_re`/`v_im`
  flattened over `[H*N]` per scenario×step), `samples.parquet` (the per-scenario records as
  dtype-preserving array columns), `meta.json`. Both layouts read back the IDENTICAL `v`;
  wide is the fast tensor cache, long the analysis/interchange table. `also_csv=True` adds a
  tidy `voltages.csv`. Result I/O, detached.
  - The on-disk layout comes from the DECLARATIONS (`n_samples`, `n_steps`, the result index,
    the frequency vector) and is only checked against the element count, so a sequence power
    flow is not recorded as a harmonic snapshot.
  - `provenance` is the caller's own generation stamps, recorded under `extra_provenance`:
    where a generator records what its config does not capture (the version of a calibrated
    recipe, of a device library) — those change the data without changing the config.
- `read_dataset(path, *, config_types=None) -> LoadedDataset(v, samples, frequencies_hz,
  node_ids, phase_codes, config, perturbations, meta, converged, failed_scenarios)` —
  layout-agnostic; `samples` merges the per-scenario columns and the shared records.
  `config_types` adds `{class_name: class}` entries over `SCENARIO_CONFIG_TYPES` for a config
  class defined elsewhere; reconstruction is explicit because reading a dataset must not
  import a module the file chose. Unknown type → the raw config dict.
  `failed_scenarios` identifies rows whose stored voltages are best-effort iterates, not
  solutions — a consumer building training data must drop (or explicitly keep) them.
- `SCENARIO_CONFIG_TYPES` — the config classes this package reconstructs by name.
- `config_hash(config_or_json) -> str` — a 16-hex fingerprint of the serialized config:
  equal hash ⇒ the same batch, so it is what a reuse gate compares. It covers the FIELDS
  only and carries no module or class name, so a config class moving between packages does
  not invalidate a recorded hash. `meta.json`'s `config_type` is the bare class name.
- `generation_provenance()` — `provenance` (`pgml.provenance.code_provenance()`: commit,
  dirty flag, versions) and `standards` (the ACTIVE EN 50160 / IEC 61000-3-2 tables with
  `override` + content hash; a `PGML_*` override silently changes every generated magnitude).
- `meta.json` additionally records the time axis as data (`n_steps`, `step_size_s`,
  `t0_unix_s`) and the config's owner (`config_module`, `config_class`), so a consumer does
  not parse a generator-specific config to find either. `read_dataset` requires none of
  these: a dataset written before a key existed simply lacks it.

## Conventions
- Unit-cube layout `U[B,D]`: one column per declared factor, then per spec a BASE block
  (component-level draws: #matched comps for `each`/correlated, 1 for `shared`, 0 for
  `independent`) then a PER-PHASE block (`independent`: one draw/phase; `small_imbalance`:
  one perturbation/phase). `U` from Sobol (QMC, recommended) / LHS / independent; each
  column → `distribution.icdf` (or the copula score path). Same config+seed → identical
  output. Correlation and per-phase noise are torch ops on `U` (autograd-safe, no
  `.item()`); the sampled tensors feed `operating_point`, and gradients flow from there.
- `mode="scale"` multiplies the component's nominal `p_nom_w`/`q_nom_var`; `"absolute"` sets
  W/var directly. An unset P or Q leaves the solver at the nominal for that one.
- REALIZED-INJECTION AUDIT COLUMNS — one convention across every generating path, so a
  written dataset records WHAT WAS INJECTED and not only the draw behind it (a draw is a
  fraction of a per-device emission reference; a composed aggregate has no draw at all). A
  block is the pair `<key>_mag` / `<key>_phase` — magnitude in per unit of the device's own
  fundamental current, phase in degrees, i.e. exactly the `harmonic_injection` values the
  solver receives — over a device axis given by the block's id column, which is a
  batch-shared record:

  | path | `<key>` | shape | device axis |
  |---|---|---|---|
  | randomized `h_mag` spec (`sampler`) | `<spec.name>` | `[B, n_dev, n_ord]` | `<spec.name>_device_ids` |
  | a sequence generator | its own `<name>` | `[B, n_dev, n_ord, T]` | `<name>_device_ids` |

  The order axis follows the writing spec's `orders`. The raw draw stays under its own
  `<spec.name>` (it documents the DRAW; for a referenced spec the realized magnitude is
  `draw × reference`, and the reference is per device). These are purely additive parquet
  sample columns: `SCHEMA_VERSION` is unaffected (it versions the `pgml.schemas` contract)
  and a dataset written before them reads back unchanged.

## Storage dispatch
Moved to `pgml.dispatch` (`integrate_soc`, `dispatch_storage`, `storage_operating_point`,
`StorageDispatchResult`): it is time coupling of a device, not scenario sampling, and it is
the only consumer of the `Storage` element's energy-state fields. See
`docs/pgml/modeling/der-pv-storage.md` §4.4.

## Deferred
- Network-parameter perturbation sweep (line/transformer impedance errors, for parameter
  recovery) — extends `perturbation_sweep` with a branch-aware selector and matrix-valued
  ground truth.
- Network-parameter and TOPOLOGY (switch-state) batching; MULTI-GRID batching.
- Beta / scipy-backed distributions (no closed-form `icdf`).
- A per-scenario time anchor: a sequence batch records ONE `[T]` `time_unix_s` for the whole
  batch, so a generator cannot stagger its scenarios over the day. A `[B, T]` `time_unix_s`
  is a contract change here and in every downstream time-feature consumer.
