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
- `Selector(component="load"|"generator", ids=None, consumer_type=None)` ->
  `.resolve(grid) -> [ids]` (None+None = all of that kind; filters AND).
- `ParameterSpec(name, selector, distribution, field="p"|"q"|"pq", mode="scale"|"absolute",
  per="each"|"shared", correlation=None, symmetry="balanced"|"independent"|"small_imbalance",
  imbalance=0.0)`. `pq` varies P&Q by the same factor (scale only).
  - `correlation=Correlation(factor, rho)` couples matched components through a shared
    `LatentFactor` (single-factor Gaussian copula; rho=0 == `per="each"`, rho=1 ==
    `per="shared"`; supersedes `per`). Marginal distribution preserved.
  - `symmetry` (per-phase, power fields only): `balanced` writes a scalar total
    (`p_w`/`q_var`, split equally downstream); `independent` draws each phase separately;
    `small_imbalance` = balanced base × (1 + small per-phase perturbation of fractional std
    `imbalance`). The latter two write per-phase `p_per_phase_w`/`q_per_phase_var`, which
    auto-promote the solve to ASYMMETRIC (`resolve_asymmetric` mode="auto"). `imbalance>0`
    required iff `small_imbalance`.
  - HARMONIC fields `field="h_mag"|"h_phase"` + `orders=[...]` (>=2): write a batched
    `harmonic_injection` instead of an operating point. `h_mag` magnitude = sampled value ×
    per-order EN 50160 limit (`harmonic_reference="en50160"`, distribution in [0,1]), ×
    stored spectrum mag (`mode="scale"`), or absolute pu. `h_phase` sets the phase (deg,
    `mode="absolute"`). Per-device injection is seeded from the stored `StaticSpectrum` so
    unspecified orders survive. Harmonic fields reject `correlation`/per-phase `symmetry`.
- `LatentFactor(name)` — shared driver (one QMC dim). `Correlation(factor, rho∈[0,1])`.
- EN 50160 per-order limits: `en50160_limits() -> {order: max_pu}`, `en50160_limit(order)`
  (loads the packaged `pgml/data/standards/en50160.yaml`; `PGML_EN50160` env override).
- NODE-COHERENT harmonic "fingerprints" (temporal sequences):
  `CoherentSpectrumConfig(selector, orders, n_steps T, n_scenarios B, n_modes=2, seed,
  mag_distribution, harmonic_reference="en50160", phase_distribution, jitter_mag, ar1_rho,
  dwell, step_size_s, resample_modes_per_scenario)` ->
  `sample_coherent_spectra(grid, config) -> SampledScenarios`. Each device draws `n_modes`
  base spectra (fingerprint); over T steps it STICKS to a mode (Markov `dwell`) and WANDERS
  (AR(1) `ar1_rho` jitter), clamped to EN 50160. `harmonic_injection` is `[B,T]` per
  (device, order); `samples` records `<name>_mode [B,n_dev,T]` (attribution label),
  `<name>_mag`/`<name>_phase [B,n_dev,n_ord,T]`, `<name>_device_ids`, `time_s [T]`.
- `ScenarioConfig(n_samples, seed=0, method="sobol"|"lhs"|"independent", parameters=[...],
  factors=[LatentFactor(...)])`. A spec's `correlation.factor` must name a declared factor.
- `sample(grid, config) -> SampledScenarios(operating_point, samples, n_samples, config)`
  where `operating_point = {id: {"p_w": Tensor[B], "q_var": Tensor[B]}}` and
  `samples = {param_name: Tensor[B, d]}` (raw realized values; ML input record).
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
  slack="ideal", symmetry=None, dtype, device, chunk_size=None) -> ScenarioResult(v, index,
  sampled, frequencies_hz, converged, failed_states)`. `converged` is True iff EVERY scenario
  converged; `failed_states` lists the non-converged scenario indices. A batch NEVER raises
  on a failed scenario — its best-effort `v` is returned and the solver logs the failures
  (so a large sweep yields data + diagnosable failures). `chunk_size` streams the batch in
  slices of that many scenarios and concatenates (VRAM tiling for a batch whose dense
  `[B,H,N,N]` system would not fit); the result equals the whole solve within the solver
  tolerance and stays differentiable. Not applied to the coherent `[B,T,H,N]` path.
  `symmetry` forwards to the solver (None/"auto" lets per-phase samples promote to asymmetric).
  `spec` = `ScenarioConfig` | `CartesianConfig` | `CoherentSpectrumConfig` (forces harmonic,
  defaults `harmonic_orders=[1, *orders]`) | a pre-built `SampledScenarios`.
  `v` is `[B,N]` (power_flow), `[B,H,N]` (harmonic), or `[B,T,H,N]` (coherent; timestamps in
  `sampled.samples["time_s"]`).
- `SampledScenarios` also carries `harmonic_injection = {id: {order: (mag, phase)}}` (mag/phase
  `[B]` for random specs, `[B,T]` for coherent), passed to `solve_harmonic_flow`.
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
- **Long-term temporal-pattern simulation mode** — `CoherentSpectrumConfig` models only
  short/medium-term dynamics (Markov mode dwell + AR(1) jitter over `T` steps). A long-horizon
  generator with DIURNAL / WEEKLY / SEASONAL device recurrence (e.g. an EV charger active daily,
  appliances on occupancy schedules) would let temporal state-estimation models (pgl) learn
  long-range patterns. Needs a device on/off schedule / occupancy process layered on the
  fingerprint, and longer sequences. Couples to the pgl temporal model's context length.
