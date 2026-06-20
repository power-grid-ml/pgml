# pgml — open TODOs

Ordered by theme, not strictly priority. Each item says WHAT, WHY, WHERE (files), and
HOW to approach. "Decision needed" items should be confirmed with the maintainer before
a large rework. See `HANDOFF.md` for the orientation and `CONTEXT.md` for the map.

---

## 1. Batch operations → production (GPU training-data generation)  ⚠️ design decisions
**Goal A.** Generate LARGE volumes of harmonic-flow training data on a GPU,
reproducibly. Increment 1 exists (`scenarios/` QMC + cartesian, batched solve == loop).
**Open design decisions are written up in `src/pgml/scenarios/ROADMAP.md`** (read it):
topology / switch-state batching, multi-grid batching (PyG disjoint union vs padded),
correlated / "shared-by-physics" sampling + per-phase symmetry, harmonic-spectrum
distribution sampling (EN50160), per-node structured perturbation sweep, parquet
persistence, beta/scipy distributions.

**Deferred (no priority) — appliance-state harmonic mixture.** Beyond the planned
node-coherent harmonic sampler (per-node base modes + Markov/AR(1) step dynamics),
a more physical "Model C" can later replace a node's fingerprint with a sum of
PER-APPLIANCE state spectra: each appliance has a small state→spectrum library
(e.g. washing-machine heating vs spinning), states evolve over time, and the node
injection is the current-domain sum. Needs an archetype spectrum library keyed by
`consumer_type`. Capture only; revisit after the modes+jitter sampler ships.

**Production / performance work still to scope.**
- GPU throughput: batched `torch.linalg.solve` memory vs batch size; pick CHUNK SIZES
  (scenario batch tiling) to fit VRAM; stream chunks; mixed precision (complex64 for
  data gen, complex128 for gradcheck only).
- Sparse Y at scale (dense `[B,H,N,N]` blows up for large N×many scenarios) — evaluate
  sparse batched solve / block-diagonal multi-grid.
- Persistence: map `ScenarioResult` (B×freq×node×phase) to the columnar `result_schema`
  → parquet (polars/pyarrow/duckdb already deps); always write the serialized
  `ScenarioConfig`+seed alongside (reproducibility).
- Tests: GPU performance/parity on a realistic feeder; `batched == loop` at scale;
  determinism (config+seed → identical bytes); memory ceiling per chunk.
- Where: `src/pgml/scenarios/`, new persistence module, `tests/gpu/`, `tests/scenarios/`.

---

## 2. PyG harmonic state-estimation (the ML layer)  — new agent, physics-guided
**Goal B.** Re-implement the ML (state estimation) on the NEW grid structure +
equations: **harmonic state estimation from few measurements, trained on the generated
data, using the equations as physics guidance.** (The old root `CONTEXT.md`/`README.md`
GNN relics were removed; start fresh.) The maintainer will brief the dedicated agent on
the SE method; this entry is the handoff to the EXISTING pieces it must build on.

**Where the building blocks are.**
- Grid description / topology: `src/pgml/schemas/grid_schema.py` (`Grid`, `Node`,
  branches, appliances); node-phase row layout `assembly.node_phase_index`; branch graph
  helpers in `evaluation/topology.py` (`grid_graph`, `branch_edges`) → adapt to a PyG
  `Data`/`Batch` (disjoint union; see multi-grid item in ROADMAP).
- Physics equations (for physics-guided loss): `src/pgml/equations/` — residual-form
  `0 = a − b` registry; evaluate residuals on predicted state for a physics penalty.
- The differentiable forward model (physics engine): `assembly.assemble_ybus` /
  `assemble_network_ybus` + `solver.solve_power_flow` / `solver.solve_harmonic_flow`.
  Gradients flow params→V, so the solver is usable as a differentiable layer; the Y-bus
  residual `Y(h)V−I` is the natural physics loss for harmonic SE.
- Training data: `scenarios.run_scenarios(...)` → `ScenarioResult(v, index, sampled, …)`.
- Results contract / measurement model: `src/pgml/schemas/result_schema.py`
  (NodeResult v_re/v_im, BranchResult i_from/i_to, InjectionResult) — measurements are a
  masked subset of these; phasors as (real, imag) deliberately (no angle-wrap targets).
- Deps already present: `torch-geometric`, `lightning`, `mlflow`.
- Where to build: new `src/pgml/ml/` (or `pgml/estimation/`) package + `CONTEXT.md`;
  PyG dataset adapter from `ScenarioResult`; physics-loss using `equations` + Y-bus.

---

## 3. Load convergence — NO silent fallback, rich diagnostics  — later agent
**What.** OpenDSS const-P loads silently switch to constant impedance outside
`[Vminpu, Vmaxpu]` to "nearly always converge"
(https://opendss.epri.com/LoadModelsThatNearlyAlwaysConver.html). We do NOT want a
silent model swap. Instead, when the nonlinear fixed point / Newton does not converge,
emit MAXIMUM detail: which nodes/phases are out of a sane voltage band, the per-node
residual, the iteration history, the worst offenders, and the likely cause (e.g.
under-voltage collapse at a heavy const-P load).
**Approach to explore.** A gradient/continuation-based homotopy (ramp load from a
converged const-Z point to full const-P, or scale S by λ∈[0,1]) to locate the divergence
point differentiably, rather than a hard fallback; report the λ at which it breaks.
**Where.** `solver/power_flow.py` (the current-injection fixed point + IFT;
`PowerFlowResult` already carries `converged`, `iterations`, `residual` — extend with a
structured diagnostics object). Add tests with a deliberately non-convergent heavy
const-P grid asserting the diagnostics, not a fallback.

---

## 4. Harmonic load / transformer frequency models — document + extend
**What.** OpenDSS harmonics modeling
(https://opendss.epri.com/HarmonicsLoadModeling.html) uses user-defined frequency
correction curves; transformer/load models with different curves give different resonant
frequencies and harmonic magnitudes.
**Our current choice (document this prominently).** `solver.solve_harmonic_flow` uses the
PURE current-source injection model (`include_load_shunt=False`, equivalent to OpenDSS
`NeglectLoadY=yes`): each load/gen injects a harmonic current from its spectrum and adds
NO frequency-dependent shunt; `include_load_shunt=True` raises `NotImplementedError` (the
exact OpenDSS shunt split is unpinned). Lines DO get full frequency dependence (Carson).
Transformers currently scale reactance ∝ h with constant R (no frequency-correction
curve). State WHY (simplest defensible harmonic model; avoids unpinned conventions) in
`solver/CONTEXT.md` and `references/opendss/harmonics.md`.
**Extend (TODO).** Optional frequency-dependent load shunt (`HarmonicShuntModel` in the
schema) and transformer frequency-correction curves; pick a model, validate the resonance
against OpenDSS, and make it a configurable, differentiable law (the `FrequencyParam`
machinery in `equations`/schema already supports curve/analytic forms).
**Where.** `solver/harmonic_flow.py`, `schemas` (`HarmonicShuntModel`, transformer),
`references/opendss/harmonics.md`, `tests/reference`.

**Triplen / zero-sequence discrepancy — vector-group transformer DONE (2026-06-20).**
The transformer now uses a phase-domain VECTOR-GROUP winding-incidence stamp
(`assembly/_transformer.py`, `Y = Nᵀ Y_winding N`): a Dyn delta winding correctly BLOCKS
the zero sequence, so triplen harmonics no longer pass transparently from LV to MV. The
nominal ratio + 30° clock now come from `u_rated` + connections (`tap` is off-nominal
only); default group is config `transformer.vector_group` (Dyn11). Decision record:
`references/opendss/transformer.md`. Closes part (a) of the original gap.
PART (b) DONE (2026-06-20): 3-phase Carson GEOMETRY synthesis
(`geometry.synthesize_three_phase_geometry`; `synthesize_grid_geometry` now dispatches
3-phase lines) builds an equilateral 3-conductor geometry reproducing each line's Z1 + X0
at f0 (R0 follows from earth physics). Feeding the SAME geometry to pgml and OpenDSS makes
the 3-phase harmonic comparison apples-to-apples Carson on every order incl. triplen
(closes the `sequence_aware` Z0 vs OpenDSS Carson gap). Config seed
`line.conductor.phase_spacing_m`. A true live-OpenDSS Dyn transformer oracle (real
`Transformer` element) validates the vector group directly. Supported vector groups:
Dyn1/Dyn11, in-phase wye-wye / delta-delta; non-solid neutral grounding + zigzag + other
clocks are still open. Remaining geometry caveats: low-X lines still hit the GMR floor
(non-physical flag, matches OpenDSS on the same geometry); 2-phase lines skipped.

---

## 5. Smaller follow-ups (no decision needed)
- Harmonic flow: add a batch-dim mismatch guard (operating_point vs harmonic_injection)
  and vectorize the device×order python loop in `harmonic_flow._harmonic_injections`.
- Capacitance: Carson `C` is physically correct but not bit-exact to OpenDSS's
  `capradius` convention (irrelevant for c=0 feeders) — match it if a c≠0 feeder is added.
- Live-OpenDSS harmonic-voltage oracle test (full `Solve mode=harmonics`) once the
  load harmonic shunt model (item 4) lands.
- DRY: extract a shared series+shunt pi-block + scatter helper for
  `_stamp_line_groups` / `_stamp_geometry_lines` in `assembly/ybus.py`.
- Convert: `convert/pandapower/` lacks a `CONTEXT.md` (others have one).
- Newton-Raphson power-flow method (currently current-injection fixed point only).
