# pgml — status & open work

The "where am I / where do I pick up" doc for the pgml package. For the suite map and the
package layout see the root `CONTEXT.md` and `src/pgml/CONTEXT.md`; for how to work here see
`CLAUDE.md`; for the published human docs see `docs/pgml/`.

## Status — what works (validated)

Phases 0–4 of the roadmap are done bar the large-scale forks below. The subpackage
`CONTEXT.md` files hold the interfaces; `docs/pgml/` holds the concepts and modeling
decisions. One entry per capability:

- **Public API** — `pgml.simulate(grid, config) -> SolvedState` (eager voltages + lazy
  branch currents/flows/spectra/THD), `simulate_serializable`, `SimulationConfig`, the
  `pgml.errors` hierarchy (`docs/pgml/public-api.md`).
- **Provenance + calibrated SE recipes** — `pgml.provenance.code_provenance()` stamps
  commit/dirty/versions into every persisted artifact (dataset `meta.json`, corpus
  manifest, checkpoints, cluster run dirs); `pgml.scenarios.presets` is the single
  excitation recipe (IEC emission reference, phase diversity, PV h3–h19 spans, device
  library v3, and — preset v3 — the measured LOAD-DEPENDENT emission law: a complex affine
  `I_h(λ) = A_h + B_h·λ` floor at 43–73 % of the rated phasor, 100–150° to the
  proportional part, plus a ±25°/unit-loading phase slope, drawn per device and order in
  the randomized recipe exactly as the composed library draws it —
  `docs/pgml/modeling/harmonic-emission.md`) every SE dataset generator builds from.
  ⚠️ A Task-A dataset drawn before preset v3 has NO harmonic-to-fundamental coupling at all
  (proportional, linear and spline fits all at R² 0.43–0.44 on the bench topology), so an
  estimator trained on it cannot learn how a harmonic follows the fundamental; regenerate.
  ⚠️ Datasets/corpora generated
  BEFORE the presets (pre-2026-08-14: EN 50160-as-current-fractions fallback, silent
  h15–h19 band, midnight coherent window) are miscalibrated — regenerate before
  drawing conclusions from models trained on them; the missing `config_hash` /
  `device_library_version` keys in their `meta.json` identify them.
- **Load flow** — linear (const-Z) + nonlinear (const-P / full ZIP); current-injection
  fixed point AND Newton (matrix-free option); IFT gradients; `ConvergenceDiagnostics` +
  `loadability_limit` continuation (with a `capped` flag when no nose is found). Batched-
  robust: per-scenario `converged_mask`/`failed_states` instead of raising. Validated vs
  pandapower & OpenDSS on IEEE-33 / CIGRE LV.
- **Harmonic flow** — `solve_harmonic_flow`: nonlinear fundamental + linear per-harmonic,
  OpenDSS-exact spectrum injection, vector-group transformers (Dyn traps triplen),
  connection-aware per-phase injection. Integer orders only — non-integer orders raise
  (see the time-domain note under open work).
- **DER inverter control + storage** — `InverterControl` on `Generator`/`Storage`
  (constant PF, cosphi(P), Volt-VAr, Volt-Watt, combined; capability circle, C¹
  smoothing), differentiated through the IFT; `Storage` = signed injection, SoC/dispatch
  off-tape in `scenarios.storage`. Validated vs pandapower and OpenDSS InvControl
  (`docs/pgml/modeling/der-pv-storage.md`).
- **Geometry → impedance** — differentiable Carson/Deri (earth return + skin + Maxwell C),
  bit-exact vs OpenDSS incl. triplen; analytic sequence-based harmonic line models for R/X
  feeders (`docs/pgml/modeling/harmonic-line-model.md`).
- **Scenarios** — reproducible QMC/cartesian sampling, correlated/per-phase draws,
  IEC 61000-3-2 device current-emission spectra (default; EN 50160 stays a voltage-shaped
  option), perturbation/injection sweeps, parquet persistence with `converged` +
  `failed_scenarios` in the sidecar meta. Node-coherent sequences: optional multi-scale
  load profiles (per-step `[B,T]` operating points, absolute time axis) and the
  statistical device-class COMPOSITION (per-step device activity drives power AND spectrum
  jointly, per-class attribution labels). `batched == loop`, differentiable through the
  batch.
- **Scale / solver architecture** — factor-once-solve-many, the `InjectionPlan` fast path,
  prepared systems (`prepare_power_flow`), scenario chunk tiling, the CPU sparse (SuperLU)
  backend with backend-aware convergence floors, switch-state batching (`branch_states`
  admittance scaling, differentiable) with the optional Woodbury LOW-RANK update-solve
  (`branch_states_method="woodbury"` — one base factorization for the whole sweep,
  `pgml.solver.lowrank`), multi-grid disjoint-union batching
  (`pgml.multigrid.merge_grids`) plus its BLOCK-DIAGONAL factorization backend
  (`linear_solver="block"` + `MergedGrid.block_rows()` — factors each member's diagonal
  block, equal sizes stacked into one batched LU, so an ensemble costs `O(Σ n³)` instead
  of the union's `O((Σ N)³)`; the CUDA path, where dense is otherwise the only union
  option), and pre-solve connectivity checks
  (`ConnectivityError` / `on_disconnected="zero"`). Design + measurements:
  `docs/pgml/modeling/solver-performance.md`; benchmarks:
  `run/examples/pgml/benchmark_speed.py`, `run/examples/pgml/benchmark_sparse.py`,
  `run/examples/pgml/benchmark_woodbury.py`.
- **Convert** — pandapower / OpenDSS / power-grid-model → `Grid`, with per-terminal phase
  permutations, n_phases-aware neutrals, positive-sequence reduction (Z1 = Zself−Zmutual),
  pandapower `parallel` + line/trafo switches, ZIP load models, OpenDSS
  Capacitor/Reactor/Generator/PVSystem/Storage, and a dropped-element warning naming any
  unconverted kind. Coverage table: "Known modeling gaps" below.
- **Evaluation** — comparison plots + reference oracles (optional `oracles` extra),
  including the INDEPENDENT OpenDSS scenario oracle (`opendss_scenario_oracle`): full DSS
  export, snapshot + coherent dataset generation in `write_dataset` format with
  `meta["engine"]="opendss"`, matched + default assumption modes. Matched-mode parity is
  ~1e-8 pu on the feeder cases (~1e-6 on CIGRE LV 3-phase; the residual is the documented
  magnetizing-branch placement difference, `docs/pgml/modeling/transformer.md`).
- **Measurement instrumentation** — `MeasurementDevice` on the `Grid` (schema rev 0.0.3):
  node-anchored meters + CT channels, accuracy class, averaging intervals; inert metadata
  consumed by pgl and the acquisition service.
- **Infra** — PEP 621 packaging, `py.typed`, GitHub Actions (ruff + tests + strict docs),
  root `conftest` + markers (`gpu`/`opendss`/`slow`).

## How to run

- **Use it**: `import pgml; pgml.simulate(grid, pgml.SimulationConfig(...))` — see
  `docs/pgml/public-api.md` (and the suite's getting-started guide on the published site).
- **Tests**: `pixi run -e cpu pytest -q` (diff gate `tests/differentiability`, GPU gate
  `tests/gpu`). **Lint**: `pixi run -e cpu ruff check src tests run`. **Docs**: see `CLAUDE.md`.
- **Examples**: `run/examples/pgml/` (each self-documenting — see `run/examples/README.md`).

## Open work — where to start

WHAT / WHY / WHERE / HOW. "⚠️ decision" = confirm the approach with the maintainer before a
large rework (schema changes are orchestrator-only — ask first).

### A. Batching / scale → production GPU training-data generation  ⚠️ decision — main gap

**What.** Generate LARGE volumes of harmonic-flow training data on a GPU, reproducibly.
The sampling layer and the dense scale wins are done (see Status +
`docs/pgml/modeling/solver-performance.md`); what remains:

- **DONE — switch-state sweeps as a low-rank update.** A sweep over `S` switch states no
  longer costs `S` assemblies and factorizations: `branch_states_method="woodbury"`
  factors the sweep's BASE network once and reaches every state through a
  Sherman-Morrison-Woodbury update of that factorization (`pgml.solver.lowrank`,
  `O(N²k + k³)` per state with `k = Σ 2P` over the switched branches). Explicit opt-in;
  the default `"assemble"` path is unchanged. Measured on an i7-12700 (CPU, `auto`
  backend = SuperLU sparse, `S = 8` states, `run/examples/pgml/benchmark_woodbury.py`)
  for 1-4 switched 3-phase branches (`k = 6…24`): 3.2-3.5x at 600 rows, 4.3-6.8x at
  1200, ~5.8x at 2100, ~5.4x at 3000; still 1.7-4.4x at `k = 96`. The crossover is
  around `k ≈ N/3` (measured: 1.1x at `k = 192` and 0.2x at `k = 384` on 600 rows),
  beyond which assembling per state is cheaper. Voltages agree with the assemble path
  to ~1e-12 relative, and the memory profile changes from `[S, N, N]` to one base
  factorization plus `[S, k, k]`.
  The BASE omits every switched branch it can (opening one is a stable update, removing
  a near-ideal switch from the base is not — `pgml.solver.lowrank`).
- **GPU / memory micro-optimisations (benchmark-informed)**: a validated complex64
  data-generation fast path (generate at complex128, store complex64 — see the
  conditioning caveat under "Known modeling gaps"); retune the IFT backward's
  JVP-vs-dense threshold (`_IFT_DENSE_JAC_MAX_ELEMS`) on current GPU numbers;
  chunk-to-chunk warm starting for sorted/correlated scenario chunks;
  `torch.cuda.CUDAGraph` / `torch.compile` over the fixed-point iteration (static shapes
  per chunk — likely wins at small N).
- **Very large N / large unions — the designated GPU sparse-direct route: cuDSS via
  nvmath-python.** torch has no batched sparse direct solve, so today CUDA is always
  dense and a merged ensemble is bounded by union-sized dense memory. The intended
  backend is NVIDIA's cuDSS through nvmath-python's sparse direct-solver API: it
  supports complex matrices, batches uniformly over ONE sparsity pattern (exactly the
  scenario/ensemble case — the pattern is fixed, the values vary) and separates
  analysis / factorization / solve so the symbolic phase is paid once and reused across
  a whole run, matching the factor-once-solve-many design. It is Beta upstream — pin and
  re-validate before adopting. PREREQUISITE: COO/block-aware assembly, so the union's
  dense `[N, N]` `Y` is never materialised. The union-sized dense remainders today are
  (1) assembly itself (the stamps scatter into a dense accumulator), (2) the ideal-slack
  `Y_fs` coupling gather, and (3) the IFT backward's dense `[2N, 2N]` Jacobian (shared
  with Newton). `linear_solver="block"` already removes the union-sized FACTORIZATION for
  an ensemble; these three are what still bound it.
- **Beyond direct-factorization scale**: a GPU-resident Krylov path (block-Jacobi /
  additive-Schwarz preconditioning, the shape GPU power-flow solvers built on iterative
  methods take) is the fork for networks too large to factor at all — secondary at LV
  sizes, where direct factorization wins.
- Resolved as won't-do (measured): a batch-native Newton forward — the block-diagonal
  Jacobian build was 4× slower than the per-scenario path at B=64/N=180; bulk batches
  belong to the current-injection method.

**Where.** `src/pgml/solver/`, `src/pgml/scenarios/`, `tests/gpu/`.

### B. Harmonic state estimation — the `pgl` package

The ML layer is its own package (`pgl`, distribution `power-grid-learn`, its own
repository) consuming pgml's public API only. Design + status live there: `docs/pgl/index.md`,
`src/pgl/CONTEXT.md`, `src/pgl/STATUS.md` of the pgl repository.

### C. Frequency-dependent device models (harmonic load shunt + transformer curves)

**What.** `solve_harmonic_flow` uses the pure current-source injection model
(`include_load_shunt=False`, ≡ OpenDSS `NeglectLoadY=yes`); `include_load_shunt=True`
raises (the OpenDSS shunt split is unpinned). Transformers scale leakage reactance ∝ h
with constant R (no frequency-correction curve). The schema `HarmonicShuntModel` exists;
the transformer carries an (unconsumed) `resistance_frequency` + `harmonic_xr_constant`.
**How.** Implement the load Norton shunt from operating-point P,Q + the series/parallel
R-L split; wire the transformer curve fields into its stamp; validate the resonance vs
OpenDSS. **Where.** `solver/harmonic_flow.py`, `assembly/ybus`, `schemas` (ask first),
`docs/pgml/modeling/references/opendss/harmonics.md`, `tests/reference`.

### D. Smaller follow-ups (no decision needed)

- **Per-scenario staggered composition starts**: the composed coherent generator evaluates
  ONE absolute time window for every scenario (`composition.sample_device_composition`
  builds a single `[T]` hour/day axis from `config.start_time`), so a dataset's diurnal
  coverage is whatever the anchor hour provides — a midnight anchor leaves office/PV
  activity near zero for the whole dataset. Draw a per-scenario start offset (seeded,
  recorded in the samples sidecar) so the scenarios spread over the day; the per-step
  `time_unix_s` becomes `[B, T]` and the `pgl` time features must consume the per-sample
  axis. WHERE: `src/pgml/scenarios/composition.py`, `harmonics.py` (sidecar), and
  `pgl.time_features` in the pgl repository.
- **SolvedState mutation guard**: lazy accessors recompute from the referenced grid; the
  no-mutation-after-solve rule is currently a docstring contract only. Reuse the network
  fingerprint (the `PowerFlowSystem` guard mechanism) to detect post-solve grid mutation
  on lazy access and raise. WHERE: `src/pgml/simulation.py`.
- **Native OpenDSS end-to-end parity gate**: the parity test against OpenDSS's NATIVE
  harmonics mode on IEEE-33 (Carson geometry lines) and the 3-phase CIGRE LV (native Dyn
  transformers) lives with the paper's benchmark harness (the `pgml-paper` repository),
  because `pgml.evaluation.oracles.opendss_scenario_oracle` still refuses
  conductor-geometry lines. Teach the scenario oracle geometry lines (export the
  `LineGeometry` as an OpenDSS `LineGeometry`/`WireData` pair) and rebuild that gate on
  it here, self-contained. WHERE: `src/pgml/evaluation/oracles/opendss_scenario_oracle.py`,
  `tests/reference/`.
- **Legacy live oracle injection model**: `evaluation/oracles/opendss_oracle.py` stamps
  device harmonic injections on all host-node rows from phase-to-ground voltages and
  nameplate P/Q (documented in its docstring) — not connection-aware, unlike the solver
  and the scenario oracle. Fix it or deprecate those paths in favor of
  `opendss_scenario_oracle`. WHERE: `src/pgml/evaluation/oracles/`.
- **Convert (OpenDSS)**: not yet read: 3-winding units, `RegControl`, `XfmrCode`/
  frequency-correction curves, explicit non-solid neutral nodes, and load
  `Vminpu`/`Vmaxpu`/CVR (currently silently ignored — at minimum warn). WHERE:
  `src/pgml/convert/opendss/`.
- **Converter test gaps**: the OpenDSS SINGLE_PHASE_EQUIV positive-sequence reduction
  (`Z1 = Zself − Zmutual`, `_positive_sequence_scalar`) has no direct value-pinning test
  (only a shape check); pandapower line/trafo switches lack a minimal dedicated
  open-switch parity case (currently exercised only inside the large CIGRE MV /
  mv_oberrhein comparisons). WHERE: `tests/convert/`, `tests/reference/`.
- **DER (optional extensions)**: a true voltage-regulating PV bus (replace a terminal's
  power-balance row with `|V| − V_set`, free Q, smooth Q-limit —
  `docs/pgml/modeling/der-pv-storage.md` §4.5).
- **Storage dispatch (optional extensions)**: higher-level dispatch policies and a
  scenarios `Selector(component="storage")` to sample storage setpoints across a batch.
- **Composition (statistical device classes)**: per-phase member placement (members
  currently land on the load's total, split by the symmetry rule), a richer per-class
  reactive/power-factor shape, and coupling the fingerprint MODE to the same activity
  clock (an EV's harmonic mode appearing only while charging). WHERE:
  `src/pgml/scenarios/composition.py`.
- **Harmonic flow**: batch-dim mismatch guard (operating_point vs harmonic_injection);
  vectorize the device×order python loop in `harmonic_flow._harmonic_injections`.
- **Criticality on a batch**: the IFT-Jacobian criticality SVD is skipped for `b>1`
  (logged); a batched variant needs per-scenario operating-point slicing.
- **Per-node harmonic-source sweeps** (`run_node_injection_sweep`) loop one solve per
  node; batch the target-row index (`[B, P]` scatter) to lift the loop.
- **Transformer (assembly)**: non-solid neutral grounding (`GroundingImpedance`) and
  zigzag-zigzag pairings raise `ModelingError`; the `TransformerZeroSeq` VALUE override is
  not consumed (consume it to close the cross-tool Z0 gap).
- **Transformer (solver)**: a fully ungrounded secondary island has no absolute
  zero-sequence reference (line-to-line-correct, absolutely-undetermined voltages as
  load → 0); a reference injection / per-island pin would close it.
- **Geometry**: low-X R/X lines hit the GMR floor (flagged `synth_unphysical`); 2-phase
  lines are skipped by `synthesize_grid_geometry`; Carson `C` is correct but not bit-exact
  to OpenDSS's `capradius` (match it if a c≠0 feeder is added).
- **Continuation/Newton polish**: a true arc-length predictor-corrector; a GMRES
  preconditioner near the nose; batched continuation. Iwamoto's optimal multiplier
  (Iwamoto & Tamura 1981, IEEE Trans. PAS-100:1736): the complex power-flow residual is
  EXACTLY quadratic in `(V, conj(V))`, so the second-order Taylor term is exact and the
  optimal Newton step length has a closed form from one extra residual evaluation per
  iteration — an ill-conditioned/near-nose robustness upgrade (larger convergence
  region, no divergence overshoot), not a throughput lever; adopt it with the
  continuation work, on both the plain Newton solve and the corrector.
- **Typing / mypy gate** (incremental): targeted annotations on the public API + a gate on
  the non-duck-typed modules; don't fight the deliberate `Any` of the float/tensor duality.
- **Deferred (no priority) — time-domain simulation.** The steady-state frequency-domain
  scope is deliberate (integer harmonic orders only; non-integer orders raise).
  Interharmonics, flicker and transients belong to a future differentiable TIME-DOMAIN
  companion path over the same Grid contract, which would subsume interharmonic support.

## Known modeling gaps (physics NOT currently modeled — keep this list honest)

Each entry states what the simulator deliberately (or currently) does NOT capture, so
results are never read as more physical than they are. Details live in `docs/pgml/modeling/`.

- **Transformer, frequency dependence.** Leakage X ∝ h with CONSTANT winding resistance
  (curve fields unconsumed — item C). No saturation/inrush (steady-state tool). The
  magnetizing branch sits at the EXTERNAL HV terminal vs OpenDSS's internal "T" — ~1e-3 pu
  on a live comparison (`docs/pgml/modeling/transformer.md`).
- **Transformer, construction.** All winding pairings except zigzag-zigzag, at every clock
  of the pairing's parity; non-solid neutral grounding raises (fail loud); zigzag Z0 uses
  the positive-sequence leakage VALUE (`docs/pgml/modeling/transformer.md`).
- **Load harmonic behaviour.** Pure current-source injection (≡ `NeglectLoadY=yes`); the
  frequency-dependent load Norton shunt (damping near resonances) raises when requested
  (item C). Resonance magnitudes are conservative (undamped) at load-heavy buses.
- **Sources.** Zero-sequence source impedance = positive-sequence value (no converter
  reads `r0x0_max`/`z01_ratio`) — `docs/pgml/modeling/conventions.md` §6.
- **PV (voltage-regulating) buses.** There is no PV-bus appliance: a bus whose voltage
  MAGNITUDE is regulated with reactive power free (pandapower `net.gen`, OpenDSS
  `Generator model=3`) needs a mixed residual row pair `[P-balance; |V|² − V_set²]` in
  `solver/power_flow.py` plus a schema field to carry `V_set`. The pandapower converter can
  APPROXIMATE one on request (`gen_mode=GenMode.VOLT_VAR_APPROX`) with a steep Volt-VAr
  droop centred on `vm_pu`: the per-bus deviation from a live `runpp` falls as `1/slope`
  (case57: 4.3e-2 pu at slope 5 → 2.3e-4 pu at slope 2000), but outside the `1/slope`-wide
  band the droop's `dQ/d|V|` is exactly zero, so on a heavily loaded transmission grid the
  solve lands on the collapsed low-voltage branch: `case118` caps out around slope 5 (a few
  percent of voltage error) and `case39` converges SILENTLY onto that branch at every
  steepness (4.9e-1 pu off, all nine generators pinned at their reactive limit). Importing
  transmission benchmarks faithfully needs the residual-row fix (and `net.shunt`
  conversion, still missing).
  `docs/pgml/modeling/der-pv-storage.md` §4.5,
  `src/pgml/convert/pandapower/CONTEXT.md`.
- **ZIP loads sharing a bus with generation (cross-tool).** pandapower reduces a ZIP load's
  coefficients onto the BUS and applies them to that bus's NET injection
  (`_calc_pq_elements_and_add_on_ppc`), so a const-Z load and a generator on one bus cancel
  BEFORE the voltage-dependency is applied. pgml keeps the devices distinct, which is the
  physical model — a constant-impedance load and a constant-power inverter only cancel at
  nominal voltage. Measured on an LV bench where each inverter mirrors its bus's load:
  3.9e-3 pu at the affected buses (both tools agree to 5e-8 with the generation stopped). Not a converter defect; take it into account when a pandapower
  reference is used as ground truth for a grid with co-located ZIP load and generation.
- **Closed bus-bus switch impedance.** The pandapower converter reads `switch.z_ohm` as the
  switch RESISTANCE (`Switch.resistance_ohm`, `inductance_h = 0`); pandapower itself splits
  that value across R and X at its `switch_rx_ratio` (default 2, so X = z/√5). Physically a
  closed contact is resistive, but the difference is not negligible on a low-reactance cable
  network: on an LV cable feeder (line X ≈ 1.4 mΩ) a 1 mΩ switch contributed a 0.0125°
  node-angle divergence from pandapower until the reference run was given a purely resistive
  switch. Decide whether to reproduce pandapower's split when converting a net
  whose switches carry a non-zero `z_ohm`. WHERE: `src/pgml/convert/pandapower/converter.py`
  section 4 (bus-bus switches).
- **Line geometry (Carson/Deri).** No conductor temperature dependence, no sub-conductor
  bundling; transposition per the documented Deri assumptions.
- **Zero-sequence line impedance at harmonics — lumped R/L lines only.** Lines WITH
  `conductor_geometry` compute Z(h) from first principles (bit-exact vs OpenDSS at every
  order). Lines WITHOUT geometry embed the earth return AT f0; extrapolating Z0 to h·f0
  is an ASSUMPTION in every tool — pgml scales X∝h/R const (≡ OpenDSS `Rg=Xg=0`), OpenDSS
  defaults reconstruct Z0(f) via Deri, and the two can differ by up to ~0.3 pu at LV
  triplen voltages. A modeling-assumption divergence, not a solver error — supply
  conductor geometry when the zero-sequence earth return matters
  (`docs/pgml/modeling/harmonic-line-model.md`).
- **Missing zero-sequence line data** is invented with global overhead-line ratios
  (R0/R1=4, X0/X1=3, C0/C1=0.5 — `data/defaults.yaml`); weak for cables.
- **Spectra tables.** EN 50160 = supply-VOLTAGE compatibility levels (orders 26–49 a
  marked manual extension) — correct for source/background distortion, NOT device
  emission. Device current fingerprints default to IEC 61000-3-2 (per class A/B/C/D;
  `emission_class="auto"` maps consumer_type→class, an approximation — no lighting
  consumer_type yet for Class C).
- **complex64 on ill-conditioned grids.** Physical feeder Y reaches κ ~1e6–1e9 in SI, so
  complex64 ASSEMBLY+solve can lose most digits (measured). complex64 is the THROUGHPUT
  dtype; generate labels at complex128 and store complex64.
- **Scenario sampling runs on CPU** (`SobolEngine`), promoted to the solve device after —
  deliberate; the sampled tensors are tiny next to the solve.
- **Converter coverage.** Converted: pandapower `bus`/`line`/`load`/`asymmetric_load`/
  `trafo` (vector groups + taps)/bus-bus + line/trafo `switch` (+`parallel`)/`ext_grid`/
  `sgen`; OpenDSS Lines/2W-Transformers/Vsources/Loads (ZIP models)/Capacitor/Reactor/
  Generator/PVSystem/Storage; pgm `node`/`line`/`sym_load`/`asym_load`/`source`/
  `sym_gen`/`transformer`. NOT converted (a WARNING names any non-empty dropped kind):
  pandapower `gen` (PV bus), `shunt`, `trafo3w`, `impedance`, `ward`/`xward`, `dcline`,
  `storage`, `motor`, `asymmetric_sgen`; pgm `three_winding_transformer`, `shunt`,
  `asym_gen`, `link`, `transformer_tap_regulator`; OpenDSS items under D above.
  pandapower ideal phase-shifter taps raise.

## Conventions a contributor must respect (full list + the package map: root `CONTEXT.md`)

- `schemas/` is FROZEN (orchestrator-only); everything imports and conforms to it. Each
  subpackage's `CONTEXT.md` is its interface ledger — read before editing, update after a
  public-signature change.
- The two hard constraints: **differentiable** + **GPU-ready** (float64 `gradcheck` and the
  GPU device/dtype test must pass — see `CLAUDE.md`).
- Phase-domain, SI, store **L/C** not X/B; phasors as (real, imag); index by frequency;
  float/tensor duality (schema fields accept floats OR tensors, autograd flows through).
