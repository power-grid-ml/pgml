# Batching & scenarios — deferred decisions and options

This file expands the "Deferred" section of `scenarios/CONTEXT.md` into concrete
design options for the batching increments that are **postponed pending user input**.
Tracked as DEFERRED tasks in the session task list. The goal throughout is
generating ML training data in controlled, reproducible distributions
(config + seed reproduces the dataset).

**Status (2026-06-22).** Implemented: distribution-based per-component QMC sampling
(`ScenarioConfig`) + cartesian sweeps (`CartesianConfig`); **§3** correlated /
per-phase-symmetry sampling; **§4** EN50160 harmonic-spectrum sampling; **§5** parquet
persistence (`write_dataset`); **§6** per-node perturbation / injection sweeps. STILL OPEN:
**§1** topology / switch-state batching and **§2** multi-grid batching — the two batching
forks below — plus the SPARSE / chunked batched solve for scale (HANDOFF.md open work,
"Batching / scale"). See `CONTEXT.md`.

---

## 1. Topology / switch-state batching  — OPEN (decision needed)

**Problem.** Vary which branches/switches are in service (and, more generally,
slightly different topologies) across the batch — use case 3. Today a batch shares
ONE `grid`; only its operating point varies. Topology changes the sparsity of
`Y(h)`, so it cannot be expressed as an `operating_point` override.

**Options.**
- **(a) Per-config assembly, stacked solve.** Enumerate the K distinct topologies,
  assemble `Y` for each, then solve all scenarios with a per-scenario gather into
  the K matrices. Simple and exact; cost = K assemblies. Best when K is small
  (a handful of switch configurations) — which matches the stated use case.
- **(b) Admittance masking (single assembly).** Assemble the *superset* `Y` once
  and zero out open branches per scenario via a `[B, n_branch]` 0/1 mask applied at
  stamp time (mask multiplies each branch's primitive stamp). One assembly, fully
  vectorized, differentiable w.r.t. the surviving branches. Open switch = exact
  zero; a closed ideal switch needs the usual node-merge or a large-but-finite
  admittance. Cannot add branches absent from the superset.
- **(c) Differentiable continuous switch state.** Mask in [0,1] instead of {0,1}
  (a "soft" switch) — enables gradient-based topology search later (roadmap phase 5
  inverse problems). Superset of (b).

**Recommendation to discuss.** (b)/(c) for switch states (one assembly, vectorized,
differentiable — fits the ML-data and inverse goals); (a) for genuinely structural
topology changes. Decision needed: how varied are the topologies (switch open/close
only, or branch add/remove)? That picks (b) vs (a).

---

## 2. Multi-grid batching  — OPEN (decision needed)

**Problem.** Solve several *distinct* grids (different node counts) in one batched
call — for training across many feeders.

**Options.**
- **(a) Disjoint union (block-diagonal `Y`).** Concatenate node indices; `Y` is
  block-diagonal `[ΣN, ΣN]`; one solve. Natural fit for PyTorch-Geometric (its
  `Batch` is exactly a disjoint union), so the ML layer consumes it directly.
  Sparse `Y` strongly preferred at scale.
- **(b) Padded + masked dense.** Pad every grid to `N_max`, stack to `[G, N_max,
  N_max]`, mask padded rows. Trivially batched dense solve; wastes work when sizes
  vary a lot.
- **(c) Group-by-size.** Bucket grids of equal `N`, dense-batch each bucket. No
  waste, several solves. Good when a few distinct sizes dominate.

**Recommendation to discuss.** (a) for the GNN pipeline (disjoint union = PyG's
native batch; reuse one autograd tape end to end), with sparse `Y`. Decision needed:
does the training pipeline want one PyG `Batch` (→ (a)) or fixed-size dense tensors
(→ (b))? This also interacts with the (currently dense) `torch.linalg.solve`; sparse
batched solve may need a different backend.

---

## 3. Correlated / shared-by-physics sampling + per-phase symmetry  ✅ DONE
Shipped: latent-factor correlation (option (a), per-spec `correlation`/`rho`) + per-phase
`symmetry` (`balanced`/`independent`/`small_imbalance`). See `sampler.py`,
`tests/scenarios/test_correlation_symmetry.py`. Original design notes kept below.

**Problem.** `per="each"` (independent) and `per="shared"` (identical) are the two
extremes. Real fleets are *correlated* (all PV rise/fall together but not
identically). Also per-phase loads are often near-symmetric with small imbalance.

**Options.**
- **(a) Latent-factor model.** Draw a shared latent `z ~ D_shared` and per-device
  `e_i ~ D_idio`; value `= f(z, e_i)` with a mixing weight `rho` (rho=1 → shared,
  rho=0 → each). One extra QMC dimension for `z`. Clean, interpretable, covers the
  whole correlation spectrum with a single knob.
- **(b) Explicit correlation matrix.** Gaussian copula with a user `[d, d]`
  correlation matrix, then push through marginal `icdf`s. Most general; heavier
  config; user must supply/estimate the matrix.
- **(c) Grouped `per="group"`.** Add a `group` key to `ParameterSpec`; one shared
  draw per group (extends the existing `shared`). Coarse but trivial.
- **Per-phase symmetry.** Orthogonal knob on a spec: `symmetry="balanced"` (one draw
  applied to all phases) | `"independent"` | `"small_imbalance"` (balanced + small
  per-phase perturbation). Implementable on top of (a).

**Recommendation to discuss.** (a) latent-factor with a `rho` knob — one new config
field, covers the realistic middle ground, QMC-friendly. Confirm the API shape
(per-spec `correlation: {factor, rho}`?) before building.

---

## 4. Harmonic-spectrum distribution sampling  ✅ DONE
Shipped: option (a) — `ParameterSpec` gained `field="h_mag"`/`"h_phase"` + `orders` and writes
a batched `harmonic_injection`; EN50160 preset bounds (`en50160.py`); plus a node-coherent
harmonic sampler. See `harmonics.py`, `tests/scenarios/test_harmonic_sampling.py`.

**Problem.** Vary per-device harmonic injection magnitude/phase across the batch
(use case 2 for spectra), EN50160-bounded. The solver hook already exists:
`solve_harmonic_flow(harmonic_injection={id: {order: (mag_pu, phase_deg)}})`, and it
accepts batched tensors.

**Options.**
- **(a) Extend `ParameterSpec` with a harmonic target.** New `field` values like
  `"h_mag"`/`"h_phase"` plus an `order` (or list of orders); sampler writes a batched
  `harmonic_injection` dict instead of an `operating_point`. Reuses all the
  distribution/QMC/selector machinery. Most consistent with what exists.
- **(b) Separate `SpectrumSpec`.** A dedicated spec type targeting `(device, order,
  mag|phase)` with EN50160 default bounds baked in. Clearer for spectra-heavy
  configs; some duplication.
- **EN50160 bounds.** Provide per-order default uniform/truncated-normal ranges
  (the standard's compatibility levels) as a ready-made distribution preset.

**Recommendation to discuss.** (a) extend `ParameterSpec` (one machinery), with an
EN50160 preset distribution. Confirm whether spectra and power profiles should be
sampled jointly in one `ScenarioConfig` (shared QMC cube — better space-filling) or
in separate configs.

---

## 5. Parquet persistence  ✅ DONE
Shipped: `write_dataset(result, dir, layout="wide"|"long", also_csv=...)` (both the tidy
`result_schema` layout and a compact wide layout) with the serialized `ScenarioConfig`+seed
written alongside for reproducibility. See `persistence.py`,
`tests/scenarios/test_persistence.py`.

**Problem.** Persist a `ScenarioResult` (`v` `[B, (H), N]` + sampled inputs) to disk
as ML training data, reproducibly, in the columnar `result_schema` layout.

**Options.**
- **(a) Long/tidy table** (one row per scenario × frequency × node × phase) via
  `result_schema`'s columnar materialisation → `polars`/`pyarrow` parquet. Canonical,
  joins cleanly to grid/scenario by id; larger on disk.
- **(b) Tensor/wide parquet** (arrays per column, e.g. `v_re[node]`) — compact, fast
  to reload into tensors; less self-describing.
- **Provenance.** Always write the serialized `ScenarioConfig` + seed alongside (the
  experiment record) so the dataset is reproducible from the file.

**Recommendation to discuss.** (a) for interchange/analysis, (b) as a fast training
cache; both keyed by `result_set_id`. Decide the primary consumer (DuckDB/polars
analysis → (a); direct tensor reload in the training loop → (b)).

---

## 6. Per-node structured perturbation sweep (use case 1)  ✅ DONE
Shipped: option (a) — `perturbation_sweep(grid, selector, perturbation)` and
`spectrum_sweep` / node-injection sweeps (one perturbed/injected node per scenario, all else
nominal), with `ParameterPerturbation` ground-truth records. See `perturbation.py`,
`tests/scenarios/test_perturbation.py`.

**Problem.** "Inject a specific error once at each node, measure the spread." This is
an *enumeration over which node is perturbed*, not a cartesian of levels: batch of
size `N` (one perturbed node per scenario), all else nominal.

**Options.**
- **(a) Dedicated builder** `perturbation_sweep(grid, selector, perturbation)` →
  `SampledScenarios` with `B = #targets`, each scenario perturbing exactly one
  target. Small, explicit; composes with `run_scenarios`.
- **(b) Express as a structured `CartesianConfig`** with an identity/diagonal
  selection — awkward (cartesian is a product, this is a diagonal).

**Recommendation to discuss.** (a) a small dedicated builder; cheap to add once the
perturbation API (absolute Δ vs scale, which field) is confirmed.
