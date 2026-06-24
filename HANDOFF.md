# pgml — orientation & open work

The single "where am I / where do I pick up" doc. New here?
- **Users** → `README.md` (install + quickstart) and the published docs in `docs/`
  (Read-the-Docs; start at `docs/public-api.md`).
- **Agents** → `CONTEXT.md` (architecture, the package map, and the conventions you MUST
  follow), then the `CONTEXT.md` of the package you touch, then the code. Big-picture
  rationale: `references/ARCHITECTURE.md`. How to *work* here: `CLAUDE.md`.

## What this library is
A single **PyTorch** library that (1) loads/generates power grids, (2) simulates
**harmonic power quality in steady state** (harmonic power flow, orders 1–50+), and (3)
supports **ML on the simulated data** (graph state estimation). The defining requirement is
end-to-end **differentiability + GPU-readiness**: gradients flow from grid parameters (down
to line geometry) through Y-bus assembly and the complex solve to the outputs — so the same
code is a forward simulator, a differentiable physics engine for ML, and an
inverse/parameter-recovery tool. (Why all-PyTorch: harmonic flow decouples per harmonic
into a LINEAR complex solve `Y(h)·V(h)=I(h)` whose adjoint is cheap, so gradients flow
without differentiating iterations.)

## The suite (multi-package monorepo)
`pgml` is the **base** package; the project is now a monorepo of one-way-dependent packages
(one distribution, `pip install -e ".[learn]"`). `pgml` imports none of the others.

| package | role | status |
|---|---|---|
| **pgml** | power-grid-machine-learning — differentiable, GPU-ready harmonic power flow (the gradient engine) | active (this doc) |
| **pgl** | power-grid-learn — harmonic state-estimation models + training (DNN/GNN/Graphormer, masking curriculum) | scaffolded — `src/pgl/HANDOFF.md`, `references/pgl/README.md` |
| **pgg** | power-grid-generation — differentiable synthetic grid generation | scaffold only — `references/pgg/README.md` |
| **pgd** *(future)* | dashboard: visualize grids, simulation, training/ML process | idea — not started |
| **pghub** *(future)* | hub: overview of existing grids (load from sources, backing database) | idea — not started |

`pgl`/`pgg` import `pgml`'s PUBLIC API only (`references/pgml/README.md` lists it); they
never touch `pgml` internals or edit `pgml/schemas/`. Deployment: the clusters have conda
only (no pixi) and run CUDA 13.0 / 13.2 — torch comes from conda, the packages pip-install on
top (`deploy/environment.yml`); dev/quick-test on an RTX A2000, large-batch on the cluster.

## Status — what works (validated)
Phases 0–3 done. The per-package `CONTEXT.md` and `README.md` hold the detail.
- **Public API** — `pgml.simulate(grid, config) -> SolvedState` (eager voltages + lazy
  branch currents/flows/spectra/THD), `simulate_serializable -> ResultBundle`,
  `SimulationConfig`, and the `pgml.errors` hierarchy. The front door; see `docs/public-api.md`.
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
  use `method="newton"`. Design: `references/der_pv_storage_modeling.md`.
- **Geometry → impedance** — differentiable Carson/Deri (earth return + skin + Maxwell C),
  **bit-exact vs OpenDSS** (incl. triplen, via feeding the same geometry to both engines);
  plus analytic `positive_sequence` / `sequence_aware` harmonic line models for R/X feeders.
- **Scenarios** — reproducible QMC/cartesian batched sampling, correlated / per-phase, EN50160
  harmonic spectra, per-node perturbation/injection sweeps, parquet persistence;
  `batched == loop`, differentiable through the batch.
- **Evaluation** — paper-ready + interactive comparison plots (refs-vs-ours); the reference
  oracles live in the OPTIONAL `pgml.evaluation.oracles` (`oracles` extra).
- **Cross-tool conventions** pinned in `references/conventions.md` (base voltage L-L/L-N,
  transformer TO/LV referral vs OpenDSS, earth return) — read before touching converters,
  the slack, or transformers.
- **Infra** — `pyproject.toml` (PEP 621, `py.typed`), GitHub Actions (ruff + tests + strict
  docs), root `conftest` + markers (`gpu`/`opendss`/`slow`). ~531 tests pass, 32
  gpu/opendss-skipped; `ruff` + strict docs build clean.

## How to run
- **Use it**: `import pgml; pgml.simulate(grid, pgml.SimulationConfig(...))` — README
  quickstart + `docs/public-api.md` (entry-point table: `simulate` vs `solver.*` vs
  `scenarios.run_scenarios`).
- **Tests**: `pixi run -e cpu pytest -q` (diff gate `tests/differentiability`, GPU gate
  `tests/gpu`). **Lint**: `pixi run -e cpu ruff check src tests`. **Docs**: see `CLAUDE.md`.
- **Examples** (`examples/`, each self-documenting — see `examples/README.md`):
  `evaluate_ieee33.py`, `evaluate_harmonics_carson.py`, the two `scenario_*.py` studies,
  `current_injection_convergence.py`, `loadability_continuation.py`.

## Open work — where to start
WHAT / WHY / WHERE / HOW. "⚠️ decision" = confirm the approach with the maintainer before a
large rework (schema changes are orchestrator-only — ask first).

### A. Batching / scale → production GPU training-data generation  ⚠️ decision — main gap
**What.** Generate LARGE volumes of harmonic-flow training data on a GPU, reproducibly. The
sampling layer is done; what remains is SCALE and the two open forks in
`src/pgml/scenarios/ROADMAP.md`: (1) topology / switch-state batching (varying in-service
branches changes `Y` sparsity — masking vs per-config assembly); (2) multi-grid batching
(distinct grids in one call — PyG disjoint-union / block-diagonal `Y` vs padded-dense).
**Why.** The dense `[B,H,N,N]` Y-bus + `torch.linalg.solve` blows up for large N × many
scenarios — the main fitness-for-purpose gap for the training-data goal.
**How.** Evaluate a sparse / block-diagonal batched solve, scenario-batch CHUNK tiling to
fit VRAM, streaming, mixed precision (complex64 data-gen / complex128 gradcheck). Tests:
GPU parity on a realistic feeder, `batched == loop` at scale, determinism, memory ceiling.
**Where.** `src/pgml/scenarios/`, `tests/gpu/`, `tests/scenarios/`; read `scenarios/ROADMAP.md`.
**Benchmark.** `examples/benchmark_speed.py` times load/harmonic flow and both PF solvers
over a batch sweep on IEEE-33 vs CIGRE LV +PV, CPU and CUDA, in `complex64` + `complex128`
(writes `results_<device>.json` per device + merges them; the dense `[B,H,N,N]` per-scenario
cost on CIGRE is the GPU-amortization motivation). **Resolved batched-solve robustness**
(was the data-gen blockers): `complex64` now converges at the dtype's resolvable floor
(`||ΔV|| < max(tol, floor·||V||)`, `floor≈1e-6` for float32) instead of spinning; Newton
accepts a batched `operating_point` (solved sequentially per scenario, shared IFT backward);
a batch with infeasible scenarios returns best-effort data + `failed_states` (no raise),
and the single-grid criticality SVD is skipped for batches.
**Dense scale wins — DONE (GPUs are best at dense batched LU, so these come before any
sparse work):**
1. ✅ **IFT backward block-diagonal**: the backward no longer always materializes the
   `[B,2N,B,2N]` Jacobian — past `_IFT_DENSE_JAC_MAX_ELEMS` it builds the `[B,2N,2N]` blocks
   column-by-column with `2N` batched JVPs (O(B), correct for every batch source; small B
   keeps the fast vectorized path). Removes the gradient-path memory ceiling.
2. ✅ **Factor-once-solve-many** (`solver/harmonic.py` `lu_factor_system`/`solve_factored`):
   `Y_eff` is network-only and constant across the fixed-point iterations (const-P/ZIP loads
   live on the RHS as `I_device(V)`, never in `Y`), and `Y(h)` is scenario-independent — so
   one `lu_factor` is reused across iterations and the batch. ~5–7× faster on CIGRE B=64.
   (Newton's `J` IS op-dependent → no reuse; it stays the per-scenario path.)
3. ✅ **Scenario CHUNK tiling** (`run_scenarios(chunk_size=...)`): streams `B` in
   VRAM-sized slices and concatenates (grad-preserving; == whole within tol). So any batch
   fits regardless of the dense `[B,H,N,N]` footprint.
Batched Newton is O(B) sequential (fine for the hard-grid / near-nose case; current injection
is the vectorized bulk path).
**Deferred — sparse solve (only past ~thousands of buses).** A power-flow `Y` is ~O(N) nnz
and distribution feeders are radial, so a sparse direct factorization (KLU/SuiteSparse, as in
pandapower / power-grid-model / OpenDSS; GPU: cuSPARSE/cuDSS) is ~O(N) vs dense O(N³). But at
the small N this library validates on (hundreds of rows), dense-on-GPU is faster (sparse is
irregular + hard to batch) and the VRAM wall is hit on `B`, not `N`. So sparse is premature
here — revisit ONLY if target grids exceed a few thousand buses. The dense wins 1–3 cover the
distribution-feeder-ML use case.

### B. Harmonic state estimation — now its OWN package `pgl` (power-grid-learn)
**Moved out of pgml.** The ML layer is the `pgl` package (clean API border, separate deps,
own agents). Design + decisions: `references/pgl/README.md`; open work: `src/pgl/HANDOFF.md`;
interface ledger: `src/pgl/CONTEXT.md`. Implemented foundations: `pgl.config`/`encoding`/
`masking`; stubs to build: `pgl.{data,normalization,loss,models,train,metrics}`. Agent:
**`pgl-ml-engineer`**.
**pgml pieces pgl consumes** (the public API it builds on): topology
`assembly.node_phase_index` + `evaluation.topology`; the physics residual `equations/`
(`Y(h)V−I`); the forward `pgml.simulate`/`solver.*` (gradients flow params→V); training data
`scenarios.run_scenarios`/`read_dataset`/`write_dataset`; measurement model = masked subset
of the state. The PyG `Data`/`Batch` builder is a `pgl` concern (`pgl.data.build_graph`),
built on `node_phase_index` + `evaluation.topology` (which provide networkx topology, not
PyG). Deps present via the `learn` extra: `torch-geometric`, `lightning`, `mlflow`.

### C. Frequency-dependent device models (harmonic load shunt + transformer curves) — extend
**What.** `solve_harmonic_flow` uses the pure current-source injection model
(`include_load_shunt=False`, ≡ OpenDSS `NeglectLoadY=yes`); `include_load_shunt=True` raises
(the OpenDSS shunt split is unpinned). Transformers scale leakage reactance ∝ h with constant
R (no frequency-correction curve). The schema `HarmonicShuntModel` already exists; the
transformer carries an (unconsumed) `resistance_frequency` + `harmonic_xr_constant`.
**How.** Implement the load Norton shunt from operating-point P,Q + the series/parallel R-L
split; wire the transformer's `resistance_frequency`/`harmonic_xr_constant` into its stamp;
validate the resonance vs OpenDSS. Reuse the `FrequencyParam`/`CurveParam` machinery (a
`SusceptanceFrequencyModel` mirror is the natural schema addition). **Where.**
`solver/harmonic_flow.py`, `assembly/ybus._transformer_block_groups`, `schemas` (ask first),
`references/opendss/harmonics.md`, `tests/reference`.

### D. Smaller follow-ups (no decision needed)
- **DER control (optional extensions)**: a true voltage-regulating PV bus (replace a
  terminal's power-balance row with `|V| − V_set`, free Q, smooth Q-limit —
  `references/der_pv_storage_modeling.md` §4.5); the harmonic Norton load shunt that lets a
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
  operating-point slicing (and a `criticality` knob on `solve_harmonic_flow`) — add if a
  batched loadability margin is wanted; for now re-run one scenario or use `loadability_limit`.
- **Convert**: `convert/pandapower/` lacks a `CONTEXT.md` (others have one); the OpenDSS
  converter does not yet emit `Transformer` elements (DSS→pgml transformer parsing).
- **Transformer (assembly)**: non-solid neutral grounding (`GroundingImpedance`), zigzag
  windings, and clocks other than Dyn1/Dyn11 raise `ModelingError` — add when needed.
- **Geometry**: low-X R/X lines hit the GMR floor (flagged `synth_unphysical`; still matches
  OpenDSS on the same geometry); 2-phase lines are skipped by `synthesize_grid_geometry`.
- **Capacitance**: Carson `C` is physically correct but not bit-exact to OpenDSS's
  `capradius` (irrelevant for c=0 feeders) — match it if a c≠0 feeder is added.
- **Continuation/Newton polish**: a true arc-length predictor-corrector (the ramp+bisect
  locates `λ*` but doesn't traverse past the nose); a preconditioner for the matrix-free
  GMRES near the nose; batched continuation (currently single-grid).
- **Deferred (no priority)**: appliance-state harmonic mixture — a node fingerprint as a sum
  of per-appliance state spectra (state→spectrum library keyed by `consumer_type`).

## Conventions a new agent must respect (full list + the package map: `CONTEXT.md`)
- `schemas/` is FROZEN (orchestrator-only); everything imports and conforms to it. Each
  package's `CONTEXT.md` is its interface ledger — read before editing, update after a
  public-signature change.
- The two hard constraints: **differentiable** + **GPU-ready** (float64 `gradcheck` and the
  GPU device/dtype test must pass — see `CLAUDE.md` / `CONTEXT.md`).
- Phase-domain, SI, store **L/C** not X/B; phasors as (real, imag); index by frequency;
  float/tensor duality (schema fields accept floats OR tensors, autograd flows through).
