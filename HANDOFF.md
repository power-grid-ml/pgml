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

## Status — what works (validated)
Phases 0–3 done. The per-package `CONTEXT.md` and `README.md` hold the detail.
- **Public API** — `pgml.simulate(grid, config) -> SolvedState` (eager voltages + lazy
  branch currents/flows/spectra/THD), `simulate_serializable -> ResultBundle`,
  `SimulationConfig`, and the `pgml.errors` hierarchy. The front door; see `docs/public-api.md`.
- **Load flow** — linear (const-Z) + nonlinear (const-P / full ZIP), with **two solvers**:
  the current-injection fixed point AND **Newton** (`method="newton"`: linear const-Z warm
  start, converges near the loadability nose where the fixed point oscillates; dense or
  `linear_solver="matrix_free"`). IFT gradients; rich `ConvergenceDiagnostics`; and
  `loadability_limit` continuation (margin + critical bus + limiting load). Validated vs
  pandapower & OpenDSS on IEEE-33 / CIGRE LV.
- **Harmonic flow** — `solve_harmonic_flow`: nonlinear fundamental + linear per-harmonic,
  OpenDSS-exact spectrum injection; phase-domain **vector-group transformer** (Dyn traps
  triplen). `assembly.branch_currents` derives KCL-exact terminal currents.
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
  docs), root `conftest` + markers (`gpu`/`opendss`/`slow`). ~464 tests pass, 29
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
**Benchmark + limits surfaced.** `examples/benchmark_speed.py` times load/harmonic flow and
both PF solvers over a batch sweep on IEEE-33 vs CIGRE LV +PV, CPU and CUDA (writes
`results_<device>.json` per device + merges them; the dense `[B,H,N,N]` per-scenario cost on
CIGRE is the GPU-amortization motivation). Two blockers it confirmed: (1) **mixed precision
is not yet viable** — the fixed-point tol sits below the float32 rounding floor (~1e-5 V on a
230 V base), so `complex64` does not converge a large batch; the data-gen path needs either a
relative/dtype-aware convergence test or a non-iterative harmonic solve. (2) **Newton does
not batch** — `solve_power_flow(method="newton")` with a batched `operating_point` fails (its
const-Z warm start `_linear_const_z_init` can't stack per-device batched P/Q); current
injection is the only batchable PF path.

### B. PyG harmonic state estimation (the ML layer) — new `src/pgml/ml/`, physics-guided
**What.** Harmonic state estimation from few measurements, trained on the generated data,
using the equations as physics guidance (the maintainer will brief the SE method).
**Where the existing pieces are.** Topology: `schemas/grid_schema.py`, layout
`assembly.node_phase_index`, graph helpers `evaluation/topology.py` → PyG `Data`/`Batch`.
Physics loss: the `equations/` residual registry (`Y(h)V−I`). Forward model: `pgml.simulate`
/ `solver.*` / `assembly` (gradients flow params→V; `SolvedState` exposes V / branch currents
/ spectra). Training data: `scenarios.run_scenarios -> ScenarioResult`; measurement model =
masked subset of `result_schema`. Deps present: `torch-geometric`, `lightning`, `mlflow`.

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

### D. DER / PV inverter control  ⚠️ decision (schema + solver)
**What.** A PV system is at most a fixed P (or P/Q) injection — no inverter control (Volt-VAr
`Q(V)`, Volt-Watt `P(V)`, constant-PF, MPPT). **Why.** The control law sets the operating
point → it changes the fundamental voltages, the harmonic injection derived from them, AND
the loadability limit. **How.** A control model on `Generator` (or a dedicated DER appliance);
the `Q(V)`/`P(V)` droop can reuse the `CurveParam` machinery. The V-dependent injection enters
`I_device(V)` — the IFT still applies (the curve enters the residual + its Jacobian). Validate
vs pandapower controllers / OpenDSS `InvControl`. **Where.** `schemas` (ask first),
`solver/power_flow.py`, `scenarios/`.

### E. Storage element + dispatch/control  ⚠️ decision (schema + solver)
**What.** No storage component exists. A battery is a bidirectional P (and Q) injection with
a state-of-charge constraint and a dispatch law. **Why.** Central to modern LV/MV studies and
to time-series scenario generation. **How.** Start with static-dispatch storage (a signed
injection with ratings + a fixed P,Q setpoint), then SoC-aware time-series dispatch in
`scenarios`. **Decision:** extend `Generator` vs a new `Storage` component. **Where.**
`schemas` (ask first), `solver/power_flow.py`, `scenarios/`.

### F. Smaller follow-ups (no decision needed)
- **Typing / mypy gate** (incremental): `py.typed` ships, but no mypy gate. Add targeted
  annotations on the public API + a gate on the non-duck-typed modules (errors, simulation,
  solver signatures, config). Don't fight the deliberate `Any` of the float/tensor duality.
- **Harmonic flow**: batch-dim mismatch guard (operating_point vs harmonic_injection);
  vectorize the device×order python loop in `harmonic_flow._harmonic_injections`.
- **Criticality on batched non-convergence (bug)**: when a batched fundamental solve does not
  converge, `criticality="auto"` runs `_jacobian_criticality`, which then crashes on the
  batched 2-D participation vector (`int(r)` where `r` is a list). Because `solve_harmonic_flow`
  exposes no `criticality` kwarg, a batch with ANY non-converging scenario crashes instead of
  reporting non-convergence — a robustness hole for large-batch data-gen. Fix the batched
  indexing in `_jacobian_criticality` (reduce to the worst element) and/or thread a
  `criticality` knob through `solve_harmonic_flow`. **Where.** `solver/power_flow.py`.
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
