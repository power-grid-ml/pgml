# pgml — open TODOs

Ordered by theme, not strictly priority. Each item says WHAT, WHY, WHERE (files), and
HOW to approach. "Decision needed" items should be confirmed with the maintainer before
a large rework. See `HANDOFF.md` for the orientation and `CONTEXT.md` for the map.

## Recently shipped (see HANDOFF.md "Current status" + git history)
- Phase-domain **vector-group transformer** (Dyn traps zero-sequence; fixes triplen) +
  **3-phase Carson geometry synthesis** — 3-phase harmonics bit-exact vs a live OpenDSS
  Dyn transformer. Decision record `references/opendss/transformer.md`.
- **Public API** (`pgml.simulate` → `SolvedState`, `simulate_serializable` → `ResultBundle`,
  `SimulationConfig`) + **exception hierarchy** (`pgml.errors`, REST `http_status` hints,
  used consistently across the codebase) + **`pgml.assembly.branch_currents`**.
- **Packaging** (`pyproject.toml`, `py.typed`, README), **CI** (`.github/workflows/ci.yml`),
  **test infra** (root `conftest`, pytest markers `gpu`/`opendss`/`slow`).
- **Optional oracle subpackage** `pgml.evaluation.oracles` (`oracles` extra) + **light
  branch-stamp registry** in `assembly/ybus.py` (the extension point for new branch kinds).
- **Scenarios**: correlated/per-phase-symmetry sampling, EN50160 harmonic-spectrum sampling,
  per-node perturbation/injection sweeps, parquet persistence (ROADMAP items 3–6 — done).

---

## 1. Batching / scale → production GPU training-data generation  ⚠️ design decisions
**Goal A.** Generate LARGE volumes of harmonic-flow training data on a GPU, reproducibly.
The sampling layer is done (`scenarios/` QMC + cartesian + correlated + harmonic + sweeps +
parquet persistence; `batched == loop`, differentiable through the batch). What remains is
SCALE and the two batching forks still open in `src/pgml/scenarios/ROADMAP.md` (read it):
- **Topology / switch-state batching** (ROADMAP §1) — vary in-service branches across the
  batch (changes `Y` sparsity, not expressible as an operating-point override). Decision:
  switch open/close only (admittance masking, one assembly) vs structural add/remove
  (per-config assembly).
- **Multi-grid batching** (ROADMAP §2) — solve distinct grids in one call (PyG disjoint
  union / block-diagonal `Y` vs padded-dense). Interacts with sparse `Y`.
- **Scale / performance (architecture-review item E).** The dense `[B,H,N,N]` Y-bus +
  `torch.linalg.solve` blows up for large N × many scenarios. Evaluate a sparse /
  block-diagonal batched solve, scenario-batch CHUNK tiling to fit VRAM, streaming, and
  mixed precision (complex64 for data-gen, complex128 for gradcheck only). This is the
  main fitness-for-purpose gap for the training-data goal.
- Tests: GPU performance/parity on a realistic feeder; `batched == loop` at scale;
  determinism (config+seed → identical bytes); memory ceiling per chunk.
- Where: `src/pgml/scenarios/`, `tests/gpu/`, `tests/scenarios/`.

**Deferred (no priority) — appliance-state harmonic mixture.** A "Model C" replacing a
node's fingerprint with a sum of PER-APPLIANCE state spectra (small state→spectrum library
keyed by `consumer_type`; states evolve over time; node injection = current-domain sum).
Capture only; revisit after the modes+jitter sampler.

---

## 2. PyG harmonic state-estimation (the ML layer)  — new agent, physics-guided
**Goal B.** Harmonic state estimation from few measurements, trained on the generated data,
using the equations as physics guidance. (Old GNN relics removed; start fresh.) Maintainer
will brief the dedicated agent on the SE method; this is the handoff to the EXISTING pieces:
- Topology: `schemas/grid_schema.py`; node-phase layout `assembly.node_phase_index`; graph
  helpers `evaluation/topology.py` (`grid_graph`, `branch_edges`) → PyG `Data`/`Batch`
  (disjoint union; see ROADMAP §2).
- Physics loss: `equations/` residual registry; the `Y(h)V−I` residual is the natural
  harmonic-SE physics loss.
- Differentiable forward model: `pgml.simulate` / `solver.solve_*` / `assembly` — gradients
  flow params→V; `SolvedState` exposes V, branch currents, spectra as tracked tensors.
- Training data: `scenarios.run_scenarios(...)` → `ScenarioResult`; measurement model =
  masked subset of `result_schema` (`NodeResult`/`BranchResult`/`InjectionResult`).
- Deps present: `torch-geometric`, `lightning`, `mlflow`. Build in new `src/pgml/ml/`.

---

## 3. Load convergence — rich diagnostics, no silent fallback  — later agent
**What.** OpenDSS const-P loads silently switch to constant impedance outside
`[Vminpu, Vmaxpu]` to "nearly always converge". We do NOT want a silent model swap.
`pgml.simulate` now RAISES `ConvergenceError` (carrying `iterations`/`residual`) by default
when the nonlinear fixed point does not converge (`strict=False` to opt out) — the basic
"don't hide it" behaviour exists. STILL TODO: the RICH diagnostics — which nodes/phases are
out of a sane voltage band, per-node residual, iteration history, worst offenders, likely
cause — and a gradient/continuation HOMOTOPY (ramp load S by λ∈[0,1] from a converged
const-Z point) to locate the divergence point differentiably and report the breaking λ.
**Where.** `solver/power_flow.py` (extend `PowerFlowResult` with a structured diagnostics
object; populate `ConvergenceError.diagnostics`). Add a deliberately non-convergent heavy
const-P grid test asserting the diagnostics.

---

## 4. Harmonic load shunt + transformer frequency-correction curves  — extend
The transformer VECTOR-GROUP modelling is DONE (item under "Recently shipped"). What
remains here is the FREQUENCY-DEPENDENT device models:
**Current choice (documented in `solver/CONTEXT.md`, `references/opendss/harmonics.md`).**
`solve_harmonic_flow` uses the PURE current-source injection model
(`include_load_shunt=False`, ≡ OpenDSS `NeglectLoadY=yes`): each load/gen injects a harmonic
current from its spectrum, NO frequency-dependent shunt; `include_load_shunt=True` raises
`ModelingError` (the exact OpenDSS shunt split is unpinned). Transformers scale leakage
reactance ∝ h with constant R (no frequency-correction curve).
**Extend.** Optional frequency-dependent load shunt (`HarmonicShuntModel` in the schema) and
transformer frequency-correction curves; pick a model, validate the resonance against
OpenDSS, make it a configurable differentiable law (the `FrequencyParam` curve/analytic
machinery already exists). **Where.** `solver/harmonic_flow.py`, `schemas`
(`HarmonicShuntModel`, transformer), `references/opendss/harmonics.md`, `tests/reference`.

---

## 5. Typing / static checks (architecture-review item G)  — incremental
`py.typed` ships (consumers get hints), but there is no mypy gate. The tensor/float duality
deliberately uses `Any` (the schema is framework-free) — don't fight that. Add targeted type
annotations on the public API + a mypy gate on the non-duck-typed modules (errors, simulation,
solver signatures, config). Lower priority; do incrementally.

---

## 6. Smaller follow-ups (no decision needed)
- Harmonic flow: add a batch-dim mismatch guard (operating_point vs harmonic_injection) and
  vectorize the device×order python loop in `harmonic_flow._harmonic_injections`.
- Capacitance: Carson `C` is physically correct but not bit-exact to OpenDSS's `capradius`
  convention (irrelevant for c=0 feeders) — match it if a c≠0 feeder is added.
- Transformer: non-solid neutral grounding (`GroundingImpedance`), zigzag windings, and
  vector-group clocks other than Dyn1/Dyn11 (need a cyclic phase-permutation incidence) are
  not modelled — `ModelingError` is raised for them.
- Geometry: low-X R/X lines still hit the GMR floor (flagged `synth_unphysical`, still
  matches OpenDSS on the same geometry); 2-phase lines are skipped by `synthesize_grid_geometry`.
- Convert: `convert/pandapower/` lacks a `CONTEXT.md` (others have one); the OpenDSS converter
  does not yet emit `Transformer` elements (DSS→pgml transformer parsing).
- Newton-Raphson power-flow method (currently current-injection fixed point only).
