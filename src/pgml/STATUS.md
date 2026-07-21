# pgml — status & open work

The "where am I / where do I pick up" doc for the pgml package. For the suite map and the
package layout see the root `CONTEXT.md` and `src/pgml/CONTEXT.md`; for how to work here see
`CLAUDE.md`; for the published human docs see `docs/pgml/`.

## Status — what works (validated)

Phases 0–3 done; phase 4 (batched sampling) done bar scale/topology. The subpackage
`CONTEXT.md` files and `docs/pgml/` hold the detail.

- **Public API** — `pgml.simulate(grid, config) -> SolvedState` (eager voltages + lazy
  branch currents/flows/spectra/THD), `simulate_serializable -> ResultBundle`,
  `SimulationConfig`, and the `pgml.errors` hierarchy. The front door; see
  `docs/pgml/public-api.md`.
- **Load flow** — linear (const-Z) + nonlinear (const-P / full ZIP), with **two solvers**:
  the current-injection fixed point AND **Newton** (`method="newton"`: linear const-Z warm
  start, converges near the loadability nose where the fixed point oscillates; dense or
  `linear_solver="matrix_free"`; batched per-scenario). IFT gradients; rich
  `ConvergenceDiagnostics`; and `loadability_limit` continuation (margin + critical bus +
  limiting load). Validated vs pandapower & OpenDSS on IEEE-33 / CIGRE LV. **Batched-robust:**
  dtype-aware convergence floor (complex64 settles, not max_iter); a batch returns best-effort
  data + per-scenario `converged_mask` / `failed_states` instead of raising on a failed element.
- **Harmonic flow** — `solve_harmonic_flow`: nonlinear fundamental + linear per-harmonic,
  OpenDSS-exact spectrum injection; phase-domain **vector-group transformer** (Dyn traps
  triplen). `assembly.branch_currents` derives KCL-exact terminal currents.
- **DER inverter control + storage** — `Generator`/`Storage` carry an optional
  `InverterControl` (constant PF, `cosphi(P)`, Volt-VAr `Q(V)`, Volt-Watt `P(V)`, combined),
  bounded by the `s_rated_va` capability circle, curves via the tensor-capable
  `Characteristic`. The voltage-dependent `(P,Q)` enters `device_current_injections` and is
  differentiated by the same IFT backward (gradcheck of `V*` w.r.t. the curve/rating passes);
  a `smoothing` soft-clamp keeps it C¹. Validated vs pandapower `CharacteristicControl` Q(V)
  (same equilibrium to ~1e-10 pu) and OpenDSS `InvControl` VOLTVAR/VOLTWATT. `Storage` is a
  signed bidirectional injection (>0 discharge); SoC integration + dispatch are resolved
  off-tape in `scenarios.storage` (`integrate_soc`/`dispatch_storage`). Stiff control loops
  use `method="newton"`. Design: `docs/pgml/modeling/der-pv-storage.md`.
- **Geometry → impedance** — differentiable Carson/Deri (earth return + skin + Maxwell C),
  **bit-exact vs OpenDSS** (incl. triplen, via feeding the same geometry to both engines);
  plus analytic `positive_sequence` / `sequence_aware` harmonic line models for R/X feeders.
- **Scenarios** — reproducible QMC/cartesian batched sampling, correlated / per-phase, EN50160
  harmonic spectra, per-node perturbation/injection sweeps, parquet persistence;
  `batched == loop`, differentiable through the batch.
- **Evaluation** — paper-ready + interactive comparison plots (refs-vs-ours); the reference
  oracles live in the OPTIONAL `pgml.evaluation.oracles` (`oracles` extra).
- **Measurement instrumentation** (schema rev 0.0.3) — `MeasurementDevice` on the `Grid`
  (node-anchored meter + CT channels on incident branches, accuracy class, averaging
  intervals, acquisition `connection` JSON), attached via `Grid.attach_measurement_devices`;
  inert metadata consumed by `pgl` (sensor placement, future noise model) and the external
  acquisition service. Multi-slack topology helpers (`slack_node_ids`, nearest-slack
  distances) landed with it.
- **Cross-tool conventions** pinned in `docs/pgml/modeling/conventions.md` (base voltage
  L-L/L-N, transformer TO/LV referral vs OpenDSS, earth return) — read before touching
  converters, the slack, or transformers.
- **Infra** — `pyproject.toml` (PEP 621, `py.typed`), GitHub Actions (ruff + tests + strict
  docs), root `conftest` + markers (`gpu`/`opendss`/`slow`). `ruff` + strict docs build clean.

## How to run

- **Use it**: `import pgml; pgml.simulate(grid, pgml.SimulationConfig(...))` — see
  `docs/getting-started/quickstart.md` and `docs/pgml/public-api.md` (entry-point table:
  `simulate` vs `solver.*` vs `scenarios.run_scenarios`).
- **Tests**: `pixi run -e cpu pytest -q` (diff gate `tests/differentiability`, GPU gate
  `tests/gpu`). **Lint**: `pixi run -e cpu ruff check src tests`. **Docs**: see `CLAUDE.md`.
- **Examples** (`examples/`, each self-documenting — see `examples/README.md`):
  `evaluate_ieee33.py`, `evaluate_harmonics_carson.py`, the two `scenario_*.py` studies,
  `current_injection_convergence.py`, `loadability_continuation.py`.

## Open work — where to start

- **IFT backward supports shared/derived parameter tensors.** RESOLVED — a single leaf may
  feed several Grid fields (``p_nom_w = p`` and ``q_nom_var = p * k``): the IFT captures the
  true autograd leaves (resolving each field through its derived-expression history) and
  differentiates the residual at those leaves under ``retain_graph=True``, so the first
  ``.backward()`` succeeds and the gradient is exact (no double count). Guaranteed by
  ``tests/differentiability/test_shared_param_tensor.py`` (finite-difference + gradcheck).

WHAT / WHY / WHERE / HOW. "⚠️ decision" = confirm the approach with the maintainer before a
large rework (schema changes are orchestrator-only — ask first).

### A. Batching / scale → production GPU training-data generation  ⚠️ decision — main gap

**What.** Generate LARGE volumes of harmonic-flow training data on a GPU, reproducibly. The
sampling layer is done; what remains is SCALE and two open batching forks (below).
**Why.** The dense `[B,H,N,N]` Y-bus + `torch.linalg.solve` blows up for large N × many
scenarios — the main fitness-for-purpose gap for the training-data goal.
**How.** Evaluate a sparse / block-diagonal batched solve, scenario-batch CHUNK tiling to
fit VRAM, streaming, mixed precision (complex64 data-gen / complex128 gradcheck). Tests:
GPU parity on a realistic feeder, `batched == loop` at scale, determinism, memory ceiling.
**Where.** `src/pgml/scenarios/`, `tests/gpu/`, `tests/scenarios/`.
**Benchmark.** `examples/benchmark_speed.py` times load/harmonic flow and both PF solvers
over a batch sweep on IEEE-33 vs CIGRE LV +PV, CPU and CUDA, in `complex64` + `complex128`.

**Dense scale wins — DONE** (GPUs are best at dense batched LU, so these came before any
sparse work):
1. ✅ **IFT backward block-diagonal**: past `_IFT_DENSE_JAC_MAX_ELEMS` the backward builds the
   `[B,2N,2N]` blocks column-by-column with `2N` batched JVPs (O(B)) instead of the
   `[B,2N,B,2N]` Jacobian — removes the gradient-path memory ceiling.
2. ✅ **Factor-once-solve-many** (`solver/harmonic.py` `lu_factor_system`/`solve_factored`):
   `Y_eff` is network-only and constant across the fixed-point iterations and scenarios, so
   one `lu_factor` is reused across iterations and the batch (~5–7× faster on CIGRE B=64).
   (Newton's `J` is op-dependent → no reuse; it stays the per-scenario path.)
3. ✅ **Scenario CHUNK tiling** (`run_scenarios(chunk_size=...)`): streams `B` in VRAM-sized
   slices and concatenates (grad-preserving) — including the node-coherent `[B,T,H,N]` path
   (the slice is along the scenario axis `B`). Any batch fits regardless of the dense
   `[B,H,N,N]` footprint.
4. ✅ **InjectionPlan fast path** (`assembly.build_injection_plan`/`injections_from_plan`):
   the V-independent operating-point resolution is computed once per solve and reused
   across every fixed-point / Newton / line-search / diagnostics evaluation (it dominated
   the CPU solve at ~57%). 2–3× end-to-end on CPU.
5. ✅ **Batch-shared `Y` as one GEMM** (`power_flow._apply_y`): residuals apply a
   scenario-shared `Y` as a single `[B,N]@[N,N]` GEMM instead of `B` broadcast GEMVs that
   re-read the matrix per scenario (bandwidth-bound at large N).
6. ✅ **Prepared system** (`solver.prepare_power_flow -> PowerFlowSystem`): assembly + slack
   rows + factorization + grid leaves computed once and reused across repeated solves;
   `run_scenarios` shares one system across its whole chunk loop.

**Pre-solve structural checks (convergence hygiene).** Every solve entry point now runs a
connectivity check first: rows with no galvanic path to an in-service source raise
`ConnectivityError` with the islands, the separating open/out-of-service branches, and the
concrete fixes (`pgml.topology.connectivity_report`); `on_disconnected="zero"` instead
solves the energized sub-grid and reports 0 V on dead rows (power-grid-model's "energized"
convention); `"ignore"` restores the historical behavior. This is the a-priori
non-convergence class the reference tools also guard structurally (pandapower's
connectivity check, pgm's energized flag) — the remaining causes (loadability, oscillating
fixed point) stay post-hoc via `ConvergenceDiagnostics.likely_cause`, the criticality SVD,
and `loadability_limit`.

**Topology / switch-state batching — DONE (admittance masking, options b + c).**
`branch_states: {branch_id: state}` on `assemble_ybus` / `assemble_network_ybus` /
`branch_currents` / `solve_power_flow` / `solve_harmonic_flow`: each listed branch is always
stamped and its primitive block is scaled by the state (0 = open, 1 = in service,
intermediate = continuous, differentiable — gradients flow through the IFT for
gradient-based topology search), OVERRIDING the static `in_service`/`closed` flags. A
`[*batch]` state solves every switch configuration in one batched call (one assembly,
per-scenario `Y`); it broadcasts against a batched `operating_point` (aligned or
cartesian). Per-scenario connectivity is checked up front (vectorized over the condensed
component graph) and raises `ConnectivityError` naming the failing scenarios. Genuinely
STRUCTURAL changes (adding branches absent from the superset) still need per-config
assembly — build the superset grid instead where possible.

**Multi-grid batching — DONE at the pgml layer (disjoint union, option a).**
`pgml.multigrid.merge_grids(grids) -> MergedGrid`: the union of an ensemble IS a valid
`Grid` (ids remapped per kind, member row slices contiguous), its assembly is the
block-diagonal `Y` for free, and the sparse backend factors it in ~O(Σ nnz) — no solver
changes. `MergedGrid` translates per-member `operating_point` / `branch_states` into
merged ids and `split()`s any solved `[..., N_total]` state back into per-member views
(differentiable; parameter tensors are SHARED with the members, so gradients reach the
original grids' leaves). Measured: 64 small feeders (3060 rows) solve in one prepared
call at 9× the per-grid loop; 12k grid-scenarios/s with batched operating points.
The pgl side (per-sample PyG `Data`/`Batch`, cross-grid normalization, multi-grid
training) is deliberately NOT part of this — pgl consumes the same pgml interfaces and
handles variable-size graphs natively via PyG when that milestone lands.

**Sparse solve — DONE (CPU scipy SuperLU behind `lu_factor_system(backend=...)`).**
`"auto"` (the default, also via `solve_power_flow(linear_solver=...)`) picks the sparse
factorization on CPU systems ≥ ~500 rows and the batched dense torch LU everywhere else;
CUDA stays dense (torch has no batched sparse direct solve — dense batched LU is what GPUs
are built for, so any GPU sparse path must first beat that baseline:
`examples/pgml/benchmark_sparse.py`). Differentiable through the adjoint `_SparseSolveFn`
(one conjugate-transposed solve + a batch-folded outer product; gradcheck-verified). A
singular factorization raises `ComputationError` pointing at `check_connectivity`.
Measured (i7-12700, c128): 4800 rows end-to-end nonlinear solve 4× dense; single-RHS
back-substitution 50×. Large multi-RHS batches split across up to 8 threads (SuperLU's
back-substitution releases the GIL and is deterministic under concurrent solves; 256 RHS
at 4800 rows 267 → 52 ms). Confirmed on an RTX A2000: at complex128, CPU-sparse beats
GPU-dense at every size ≥ 300 rows — the complex64 data-generation path is where the GPU
wins. The complex64 convergence floor is BACKEND-aware (`_rel_convergence_floor`):
SuperLU's single-precision back-substitution leaves ~2e-6 relative per-iterate noise
(vs ~1e-6 dense), so marginal scenarios of a large batch used to oscillate to max_iter
under the dense-calibrated floor (voltages floor-accurate, flag + runtime wrong); sparse
float32 floor = 4e-6, guarded by
`tests/reference/test_sparse_solver.py::test_sparse_complex64_batch_converges_at_backend_floor`. Open follow-ups: a sparse/matrix-free IFT backward + Newton Jacobian for very large
N (both are still dense `[2N, 2N]`); sparse-direct assembly (COO from the stamps,
skipping the dense `Y`) once grids exceed a few thousand rows.

**Open — GPU / memory micro-optimisations (benchmark-informed).** A validated complex64
data-generation fast path (generate at complex128, store complex64 — see the conditioning
caveat under "Known modeling gaps"); retune the IFT backward's JVP-vs-dense threshold
(`_IFT_DENSE_JAC_MAX_ELEMS`) on current GPU numbers; chunk-to-chunk warm starting for
sorted/correlated scenario chunks (OpenDSS-style); `torch.cuda.CUDAGraph` / `torch.compile`
over the fixed-point iteration (static shapes per chunk — likely wins at small N where
launch overhead dominates).

**Resolved as won't-do (measured): batch-native Newton forward.** A batched operating
point solves sequentially per scenario in Newton: the O(B²(2N)²) full-map Jacobian does
not fit, and building the block diagonal column-by-column (2N JVP sweeps per step)
measured 4× slower than the per-scenario vectorized Jacobian at B=64/N=180. Kept from the
attempt: the shared `_batched_state_jacobian` helper, the per-element line search, and the
batch-broadcasting const-Z fold. Bulk batches belong to the current-injection method.

### B. Harmonic state estimation — the `pgl` package

The ML layer is its own package, `pgl` (clean API border, separate deps, own agents). It
consumes pgml's public API only — topology `assembly.node_phase_index` + `pgml.topology`,
the forward `pgml.simulate`/`solver.*` (gradients flow params→V), and training data
`scenarios.run_scenarios`/`read_dataset`/`write_dataset` (measurement model = masked subset of
the state). The PyG `Data`/`Batch` builder is a `pgl` concern. Design + status:
`docs/pgl/index.md`, `src/pgl/CONTEXT.md`, `src/pgl/STATUS.md`.

### C. Frequency-dependent device models (harmonic load shunt + transformer curves) — extend

**What.** `solve_harmonic_flow` uses the pure current-source injection model
(`include_load_shunt=False`, ≡ OpenDSS `NeglectLoadY=yes`); `include_load_shunt=True` raises
(the OpenDSS shunt split is unpinned). Transformers scale leakage reactance ∝ h with constant
R (no frequency-correction curve). The schema `HarmonicShuntModel` already exists; the
transformer carries an (unconsumed) `resistance_frequency` + `harmonic_xr_constant`.
**How.** Implement the load Norton shunt from operating-point P,Q + the series/parallel R-L
split; wire the transformer's `resistance_frequency`/`harmonic_xr_constant` into its stamp;
validate the resonance vs OpenDSS. Reuse the `FrequencyParam`/`CurveParam` machinery.
**Where.** `solver/harmonic_flow.py`, `assembly/ybus._transformer_block_groups`, `schemas`
(ask first), `docs/pgml/modeling/references/opendss/harmonics.md`, `tests/reference`.

### D. Smaller follow-ups (no decision needed)

- **DER control (optional extensions)**: a true voltage-regulating PV bus (replace a
  terminal's power-balance row with `|V| − V_set`, free Q, smooth Q-limit —
  `docs/pgml/modeling/der-pv-storage.md` §4.5); the harmonic Norton load shunt that lets a
  grid-following inverter present its output impedance at harmonics is item C above.
- **Storage dispatch (optional extensions)**: higher-level dispatch policies and a scenarios
  `Selector(component="storage")` to sample storage setpoints across a batch.
- **Typing / mypy gate** (incremental): `py.typed` ships, but no mypy gate. Add targeted
  annotations on the public API + a gate on the non-duck-typed modules (errors, simulation,
  solver signatures, config). Don't fight the deliberate `Any` of the float/tensor duality.
- **Harmonic flow**: batch-dim mismatch guard (operating_point vs harmonic_injection);
  vectorize the device×order python loop in `harmonic_flow._harmonic_injections`.
- **Criticality on a batch**: the single-grid IFT-Jacobian criticality SVD is skipped for a
  batched solve (`b>1`, logged). A per-element batched criticality would need per-scenario
  operating-point slicing (and a `criticality` knob on `solve_harmonic_flow`).
- **Convert**: The OpenDSS converter emits two-winding `Transformer` elements (solidly
  grounded wye or delta windings; clock from `LeadLag` plus cyclic winding-bus rotation,
  reaching every clock except the polarity-flip groups {2, 6, 10}); not yet read:
  3-winding units, `RegControl` regulators, `XfmrCode`/frequency-correction curves, and
  an explicit non-zero (floating/impedance-grounded) neutral node — see
  `src/pgml/convert/opendss/CONTEXT.md`.
- **Transformer (assembly)**: non-solid neutral grounding (`GroundingImpedance`) and
  zigzag-zigzag pairings raise `ModelingError`; the `TransformerZeroSeq` VALUE override
  is not consumed (the zero-sequence path follows the winding topology, its value equals
  the positive-sequence leakage — the tools disagree here: pgm hardcodes grounded-zigzag
  Z0 = 0.1·Z1, pandapower reads vk0; consume the override to close the gap).
- **Transformer (solver)**: a delta-only or otherwise fully ungrounded secondary island
  has no absolute zero-sequence reference; the nonlinear const-P solve returns
  line-to-line-correct but absolutely-undetermined voltages there as load → 0 (found via
  the YNd5 OpenDSS oracle, which anchors the comparison with a grounded-wye load). A
  reference-injection (or per-island angle/zero-seq pin) would close it.
- **Geometry**: low-X R/X lines hit the GMR floor (flagged `synth_unphysical`; still matches
  OpenDSS on the same geometry); 2-phase lines are skipped by `synthesize_grid_geometry`.
- **Capacitance**: Carson `C` is physically correct but not bit-exact to OpenDSS's
  `capradius` (irrelevant for c=0 feeders) — match it if a c≠0 feeder is added.
- **Continuation/Newton polish**: a true arc-length predictor-corrector; a preconditioner for
  the matrix-free GMRES near the nose; batched continuation (currently single-grid).
- **Deferred (no priority)**: appliance-state harmonic mixture — a node fingerprint as a sum
  of per-appliance state spectra (state→spectrum library keyed by `consumer_type`).

## Known modeling gaps (physics NOT currently modeled — keep this list honest)

Consolidated during the 2026-07 architecture/correctness review. Each entry states what
the simulator deliberately (or currently) does NOT capture, so results are never read as
more physical than they are. Items already tracked as open work above are referenced.

- **Transformer, frequency dependence.** Leakage reactance scales ∝ h with CONSTANT
  winding resistance — no frequency-correction curve; the schema's
  `resistance_frequency` / `harmonic_xr_constant` fields are not yet consumed (item C
  above). No saturation / no inrush (steady-state tool). The magnetizing/core-loss
  branch IS modeled (`y_m` on the HV diagonal) — but as a shunt at the EXTERNAL HV
  terminal, whereas OpenDSS places it inside its leakage "T" model; for a typical
  ~0.5 % magnetizing current the difference is ~1e-3 pu on a live-solve comparison
  (documented in `docs/pgml/modeling/transformer.md`).
- **Transformer, construction.** Every winding pairing of {wye, grounded wye, delta,
  zigzag, grounded zigzag} except zigzag-zigzag is modeled, at every clock of the
  pairing's parity (`docs/pgml/modeling/transformer.md`). Non-solid neutral grounding
  (`GroundingImpedance`) raises `ModelingError` (deliberate: fail loud, never
  approximate silently). The zigzag zero-sequence self path uses the positive-sequence
  leakage VALUE (`TransformerZeroSeq` not consumed) — real units have Z0 < Z1 there.
- **Load harmonic behaviour.** Loads inject harmonics as PURE current sources
  (`include_load_shunt=False`, ≡ OpenDSS `NeglectLoadY=yes`); the frequency-dependent
  load Norton shunt (damping near resonances!) is unimplemented and RAISES when
  requested (item C above). Harmonic resonance magnitudes are therefore conservative
  (undamped) at load-heavy buses.
- **Sources.** Zero-sequence source impedance is taken equal to the positive-sequence
  value (no converter reads `r0x0_max` / `R0/X0` / `z01_ratio`) — see
  `docs/pgml/modeling/conventions.md` §6. Affects asymmetric fault-like states, not the
  balanced fundamental.
- **Line geometry (Carson/Deri).** No conductor temperature dependence (`Rdc` is a
  constant), no sub-conductor bundling (HV construction), transposition/balance per the
  documented Deri assumptions. Carson shunt `C` is physically correct but not bit-exact
  to OpenDSS's `capradius` convention (irrelevant for c=0 feeders). The
  `geometry.sequence.two_conductor_*` helpers are diagnostic-only (not differentiable).
- **Zero-sequence line impedance at harmonics — LUMPED R/L lines only.** pgml has BOTH
  line models: lines WITH `conductor_geometry` compute Z(h) from first principles
  (Carson/Deri incl. frequency-dependent earth return — bit-exact vs OpenDSS at every
  order, 50–750 Hz), and that path is the authoritative answer to the earth-return
  question. This entry concerns lines WITHOUT geometry (lumped R/L matrices, e.g. from
  pandapower): their stored values embed the earth return AT f0, so extrapolating Z0 to
  h·f0 is an ASSUMPTION in every tool. OpenDSS reconstructs Z0(f) through its `Rg`/`Xg`
  parameters (Deri from the base-frequency values; can yield negative X0 — CIGRE LV
  cable at 150 Hz: 0.131−0.086j Ω vs plain-scaled 0.023+0.026j Ω), for matrix- and
  sequence-specified lines alike. pgml's plain phase model scales X∝h, R const
  (identical to OpenDSS with `Rg=Xg=0`, the metallic-return convention; native triplen
  |V3| parity then ~1e-6 pu — `tests/reference/test_native_harmonics_opendss.py`);
  `geometry.sequence.zero_sequence_harmonic_z` instead adds a monotone Carson
  earth-resistance growth — deliberately NOT OpenDSS's Deri reconstruction (which is
  non-monotone and can be non-physical for cables). With OpenDSS defaults on
  sym-component lines the two extrapolations differ by up to ~0.3 pu at LV triplen
  voltages — a modeling-assumption divergence, not a solver error; supply conductor
  geometry when the zero-sequence earth return matters.
- **EN 50160 table.** Orders 1–25 are the standard's (amended A2:2019) values; orders
  26–49 are a manual flat extension (marked in `data/standards/en50160.yaml`).
- **Scenario sampling.** All pre-solve sampling executes on CPU (`SobolEngine` is
  CPU-only); tensors are promoted to the solve device afterwards. Deliberate — the
  sampled tensors are tiny next to the `[B,H,N,N]` solve.
- **complex64 accuracy on ill-conditioned grids.** A physical feeder's `Y` reaches
  condition numbers of 1e6–1e9 in SI units (CIGRE LV fundamental ~3.5e8), so a
  complex64 ASSEMBLY + direct solve can lose most of its digits (κ·eps₆₄ ≈ O(1)) —
  measured, not theoretical. complex64 remains the THROUGHPUT dtype; for datasets on
  such grids, generate at complex128 and store as complex64 (the persisted values
  then carry only the final rounding). `pgl.physics.NetworkSolver` assembles/solves
  its decode operators at complex128 internally for exactly this reason.
- **Per-node harmonic-source sweeps** (`scenarios.run_node_injection_sweep`) loop one
  solve per node: the solver cannot yet stamp a different target row per batch element.
  Batch the target-row index (`[B, P]` scatter) to lift the loop.
- **Converter coverage.** Converted: pandapower `bus`/`line`/`load`/`asymmetric_load`/
  `trafo` (vector groups incl. zigzag + tap changer)/bus-bus `switch`/`ext_grid`/
  `sgen`; pgm `node`/`line`/`sym_load`/`asym_load`/`source`/`sym_gen`/`transformer`
  (winding types incl. zigzag, clock, taps). NOT converted (a WARNING names any
  non-empty dropped kind): pandapower `gen` (PV bus — pgml has no voltage-regulating
  bus yet), `shunt`, `trafo3w`, `impedance`, `ward`/`xward`, `dcline`, `storage`,
  `motor`, `asymmetric_sgen`; pgm `three_winding_transformer`, `shunt`, `asym_gen`,
  `link`, `transformer_tap_regulator`. pandapower ideal phase-shifter taps
  (`tap_step_degree`, `tap_phase_shifter`) raise.

## Conventions a contributor must respect (full list + the package map: root `CONTEXT.md`)

- `schemas/` is FROZEN (orchestrator-only); everything imports and conforms to it. Each
  subpackage's `CONTEXT.md` is its interface ledger — read before editing, update after a
  public-signature change.
- The two hard constraints: **differentiable** + **GPU-ready** (float64 `gradcheck` and the
  GPU device/dtype test must pass — see `CLAUDE.md`).
- Phase-domain, SI, store **L/C** not X/B; phasors as (real, imag); index by frequency;
  float/tensor duality (schema fields accept floats OR tensors, autograd flows through).
