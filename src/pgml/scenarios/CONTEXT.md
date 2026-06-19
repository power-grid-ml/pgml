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
  (loads `config/max_harmonic_values_din-en50160.yaml`; `PGML_EN50160` env override).
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
- `run_scenarios(grid, spec, *, calculation="power_flow"|"harmonic", harmonic_orders=None,
  slack="ideal", symmetry=None, dtype, device) -> ScenarioResult(v, index, sampled, frequencies_hz)`.
  `symmetry` forwards to the solver (None/"auto" lets per-phase samples promote to asymmetric).
  `spec` = `ScenarioConfig` | `CartesianConfig` | `CoherentSpectrumConfig` (forces harmonic,
  defaults `harmonic_orders=[1, *orders]`) | a pre-built `SampledScenarios`.
  `v` is `[B,N]` (power_flow), `[B,H,N]` (harmonic), or `[B,T,H,N]` (coherent; timestamps in
  `sampled.samples["time_s"]`).
- `SampledScenarios` also carries `harmonic_injection = {id: {order: (mag, phase)}}` (mag/phase
  `[B]` for random specs, `[B,T]` for coherent), passed to `solve_harmonic_flow`.

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
- Structured "one perturbation per node" sweep (use case: inject an error at each
  node, measure spread) — an enumeration over selector targets (not a cartesian of
  levels); needs a small dedicated builder.
- Network-parameter & TOPOLOGY (switch-state) batching; MULTI-GRID batching.
- Parquet persistence of (scenario, component, step, frequency) -> result_schema.
- Beta / scipy-backed distributions (no closed-form icdf).
