# solver

Batched, differentiable, device-agnostic solves of the nodal system `Y(f) V(f) = I(f)`:
the linear per-frequency solve, the nonlinear fundamental power flow, and the harmonic
power flow built from the two. Consumes the node-phase layout and stamps of `assembly/`.

Signatures: docstrings / API_INDEX.md. Design narrative and measurements: the published
pages `differentiability.md` and `modeling/solver-performance.md`.

## Codemap

- `harmonic.py` — the linear algebra. `solve_harmonic` (direct solve, Norton or ideal
  slack), `lu_factor_system` / `FactoredSystem` / `solve_factored` (factor once, solve
  many; backends dense, sparse SuperLU, block diagonal), `estimate_condition`,
  `solve_anchored` / `AnchoredSystem` (measurement-anchored least squares),
  `back_substitute` / `ideal_slack_rhs` / `scatter_slack_solution` (the three building
  blocks every factored solve uses), `lu_factor_values` (internal `(values, pattern)`
  entry of the sparse backend), `_select_backend` (the one backend decision).
- `_dense_lapack.py` — every dense CPU `lu_factor` / `lu_solve` / `solve` on a stack of
  more than one matrix; guards a multi-threaded batched-LU defect of some LAPACK builds
  and reads LAPACK `info`.
- `equilibration.py` — diagonal scaling around factorizations (`EQUILIBRATION_MODES`,
  `resolve_equilibration`, `equilibrate_matrix`, `equilibrated_lu_factor` for the real
  Newton Jacobian and the IFT adjoint).
- `lowrank.py` — Sherman-Morrison-Woodbury update-solve on a `FactoredSystem`
  (`low_rank_update`, `solve_factored_updated`, `branch_state_terms`,
  `LowRankOperator`, `LowRankUpdate`).
- `power_flow.py` — `solve_power_flow`, `prepare_power_flow` / `PowerFlowSystem`,
  `PowerFlowResult`, `ConvergenceDiagnostics`, `VoltageRegulationResult`,
  `loadability_limit` / `LoadabilityResult`, `check_connectivity`,
  `check_branch_impedances`, the IFT autograd function `_IFTPowerFlow`, convergence state
  (`_PuConvergence`, `_BatchIterationState`), the Woodbury sweep base
  (`_woodbury_base_states`).
- `_pv_bus.py` — voltage-regulating generator terminals (`PVTerminals`,
  `collect_pv_terminals`): the substituted residual rows and PV-to-PQ switching.
- `harmonic_flow.py` — `solve_harmonic_flow`, `HarmonicFlowResult`,
  `assemble_harmonic_system`, `assemble_harmonic_ybus`, `harmonic_injections`,
  `NodeHarmonicSource`, `prepare_harmonic_flow`; per-order chunked solve
  (`_solve_harmonic_orders`), structural route (`_structural_route`), device-shunt
  low-rank terms (`_harmonic_shunt_lowrank_terms`), fused-branch shunt current
  (`_harmonic_shunt_currents`).
- `_harmonic_cache.py` — `HarmonicFlowSystem`, the lazy, bounded harmonic preparation.

## Differentiability

- The forward iterations (fixed point, Newton, PV switching, Woodbury, prepared reuse) run
  detached. Gradients of the nonlinear solve come only from `_IFTPowerFlow`: implicit
  function theorem at the converged `V*`, real coordinates `x = [Re V; Im V]` (the power
  flow is not holomorphic in `V`), one adjoint solve `Jᵀλ = grad_x` and one residual vjp.
  Iterations are never unrolled. This `.detach()` is the sanctioned one.
- The backward differentiates at the TRUE autograd leaves (`_collect_leaves` walks
  `grad_fn` to `AccumulateGrad`), never at derived Grid-field intermediates; otherwise a
  leaf feeding several fields is double counted.
- INVARIANT (do not swap): the `dR/dθ` vjp uses the differentiable residual
  (`make_residual_complex`, re-resolves from the parameter leaves); the iteration, state
  Jacobian and diagnostics use the detached plan residual (`make_fast_residual_complex`).
  The detached one in the vjp silently zeroes parameter gradients.
- Every forward reuse (prepared system, Woodbury, cached harmonic factors) is
  forward-only: the backward rebuilds its system differentiably from the leaves, so
  gradients are identical between reuse and no reuse.
- The linear solves differentiate by the linear-solve adjoint on every backend
  (`λ = A⁻ᴴ grad`, `grad_A = −λ xᴴ`; sparse via `trans="H"`, structural route w.r.t. the
  `nnz` entries only). Mixed precision differentiates the exact adjoint, not the
  refinement steps.
- Equilibration scales are rounded to powers of two, so scaling adds no rounding and
  `round`'s zero derivative keeps the scales off the gradient.
- The IFT node deep-copies the resolved modeling defaults and restores them for the
  backward reassembly, so a gradient keeps the forward's model after a preset exits or
  `PGML_DEFAULTS` / `defaults.reload` changes.
- `AnchoredSystem` refuses an operator on the tape (its images are constants);
  `solve_anchored` keeps `Y` and `op` on the tape.

## Conventions

- Shapes: `Y` `[*batch, H, N, N]`, `I` / `V` `[*batch, H, N]`; factor and RHS batches
  broadcast right-aligned, and singleton factor axes fold into the multiple-RHS columns
  (one factor per `(B, H)` serves all `T` steps of a `[B, T, H, N]` RHS).
- Slack, fundamental: `slack="ideal"` pins source rows to `u_ref∠u_angle` through the
  Schur partition (pandapower / power-grid-model); `slack="norton"` keeps the source
  folded as a Norton shunt (OpenDSS Vsource). At harmonic orders the source is ALWAYS a
  Norton shunt held at zero harmonic EMF regardless of `slack`; upstream distortion is a
  voltage-kind `NodeHarmonicSource` supplied per solve, never grid data.
- `operating_point[source_id]["u_ref_scale"]` scales the ideal slack per scenario; the
  network side stays operating-point independent, so a prepared factorization is reused.
- Harmonic phase convention (shared with OpenDSS):
  `|I_h| = (mag_h/mag_1)|I1|`, `arg I_h = ang_h + h·(arg I1 − ang_1)`, per ELEMENT, with
  `I1_elem = sign·conj(S_eff)/conj(V_term)` (load convention, `V_term = M·V` through the
  connection incidence). The device shunt and the harmonic injection read ONE per-element
  power (`_effective_element_power`), so they never describe different operating points.
- `harmonic_injection` override values: a list/tuple is ALWAYS per element (length
  `n_elem`), a scalar or bare tensor is ALWAYS broadcast. No trailing-dimension sniffing:
  a scenario batch of length `n_elem` can never be misread.
- `NodeHarmonicSource` acts at orders `h > 1` only (the fundamental is untouched);
  `kind="voltage"` stamps a real shunt `S_sc / V_LN²` plus its Norton current,
  `kind="current"` the current only.
- Convergence is per element and per unit, both criteria must hold: the largest nodal
  apparent-power mismatch on `s_base_va` (pandapower's `tolerance_mva` quantity, so
  iteration counts compare across tools) and the largest per-row voltage update on the
  row's L-N rated voltage. Each threshold is capped by a precision floor that then governs
  (and is logged). A scenario batch is a batch of independent problems: each scenario
  exits and is held on its own criteria; a stalled scenario inside the floor band counts
  as converged at the floor, above it as failed.
- A batched solve never raises on a failed element: best-effort `V` per element,
  `converged_mask` / `failed_states` and an ERROR log. Harmonic orders add
  `harmonic_finite` to the verdict.
- Diagnostics (`ConvergenceDiagnostics`) are autograd-free and cost no extra solve;
  `criticality` analyses the hardest scenario of a batch, is dense and is skipped above
  `2N = 4000`, never raises (`{"skipped": ...}`).
- Results are always reported in the grid's FULL row layout (`.index` is the full
  `NodePhaseIndex`), whatever reduction the solve used.

## Backends and numerics

- `linear_solver="auto"`: SuperLU sparse on CPU from `_SPARSE_MIN_ROWS` rows, batched
  dense torch LU below and ALWAYS on CUDA. `"block"` (with `block_rows`, one partition of
  the rows) is explicit opt-in, never chosen by `"auto"`; the partition is taken on trust.
  Newton supports `"dense"` and `"matrix_free"` (GMRES on finite-difference `J·v`).
  Backends are never mixed silently and tensors never move devices to reach one.
- The structural sparse route: where the backend is sparse (CPU, `precision="full"`) the
  sparse backend gathers the `nnz` stamped positions of `pgml.assembly.ybus_structure`
  and forms no dense matrix; harmonic orders are assembled directly as structural
  entries (`PatternAccumulator`). It reproduces the dense-assembled route at equal
  scenario chunking. `harmonic._select_backend` is the one backend decision both the
  structural route and `lu_factor_system` take; keep it single. Dense-route cases
  (dense/block backends, CUDA, mixed precision, the Woodbury device-shunt update, every
  caller that returns `Y`) are commented at their sites.
- Equilibration (`solver.equilibration.mode`, symmetric van der Sluis by default) wraps
  every power-flow and harmonic factorization, the Newton Jacobian and the IFT adjoint,
  and is invisible in results. `FactoredSystem.y_mat` is the matrix AS FACTORED (scaled);
  `estimate_condition` describes that matrix. `solve_anchored` / `AnchoredSystem` apply
  no equilibration.
- `precision="mixed"` factors complex64 and refines against complex128 residuals with a
  FIXED step count (branch-free, no sync); it requires a complex128 working dtype and
  raises otherwise. The nonlinear loop then runs in residual-correction form.
- The current-injection loop forms the nodal residual by the identity
  `F(V_new) = I_dev(V_new) − I_dev(V_old)` on free rows and confirms the true residual
  only in the exiting iteration; `precision="mixed"` keeps the explicit residual (it is
  the next RHS).
- Newton is the hard-grid solver: a batched operating point is solved sequentially per
  scenario (a batch-native Newton was measured slower and rejected); bulk batches use the
  current-injection fixed point. A grid with a voltage-regulating generator is always
  solved by Newton (with a warning when `current_injection` was asked).
- PV terminals: the reactive power is eliminated analytically, so the state stays
  `[Re V; Im V]` and the IFT Jacobian keeps its shape; PV-to-PQ switching is off-tape,
  the residual at the resolved active set is on-tape. Regulation is fundamental-only; at
  `h > 1` such a machine is an ordinary Norton current source.
- `_dense_lapack`: a call is checked against a probe RHS with distinct entries, a failure
  latches a per-process repair (per-matrix factorization, then one thread) and a call
  still wrong raises `ComputationError`. Single matrices, CUDA and the sparse backend are
  untouched. The thread count is process-global. A negative LAPACK `info` raises; a
  positive one (exact zero pivot) keeps torch's error.
- Woodbury: the update never inverts `C`, so an open state (`s = 0`) is exact and a branch
  at its base state reproduces the base solve. The sweep base omits every switched branch
  it can while connected (`_woodbury_base_states`): removing a near-ideal branch from the
  base spends digits by cancellation. Do not simplify it to a fixed base of 1.
  `branch_states_method="woodbury"` is explicit opt-in (the win depends on `k/N`),
  forward-only, `current_injection` only. The harmonic device-shunt path selects its
  low-rank update automatically under `3k < N`, applied to the touched-row count before
  `C` is built, with a backward-error fallback to direct factorization.

## Zero-impedance branches (exact bus fusion)

- The fusion map (`pgml.assembly.fusion_map`, policy `branch.zero_impedance`) is resolved
  ONCE per solve and threaded into assembly, slack rows, the injection plan, warm start,
  Woodbury base, every harmonic order and diagnostics. The solve runs on the reduced
  rows; `PowerFlowResult.v` / `HarmonicFlowResult.v` are prolonged back by a gather and
  both results carry `fusion` (`None` when nothing fused). Persisted layouts and
  `layout_fingerprint` never change.
- Two in-service sources on one fused row are deduplicated when their references agree
  and refused otherwise; two regulating generators on one fused row are refused. A fused
  group reports one diagnostics row under its representative node id. `block_rows` maps
  through the fusion and stays a partition.
- The harmonic device shunt composes with fusion through `index_add_` accumulation
  (`Pᵀ Y P`); `assemble_harmonic_system` takes `v1` in the full layout and samples the
  representative row. A fused branch's harmonic current needs the device-side current
  `harmonic_injections(...) − Y_shunt(h)·V(h)`; `SolvedState.branch_currents()`
  subtracts `_harmonic_shunt_currents`.
- `check_branch_impedances` with the solve's map refuses only what neither a stamp nor a
  fused row expresses: a zero-impedance transformer, a zero-series branch with a shunt,
  and a zero-impedance branch listed in `branch_states` (fusion and admittance scaling
  are mutually exclusive by construction). It reads values under `no_grad`.

## Caches and their validity

- `PowerFlowSystem` caches assembly, slack rows, factorization, grid-leaf walk, fusion
  map, `row_abs_scale` (the mismatch floor's per-row scale), `full_index`, `v_base`, its
  `precision`, `equilibration` and `modeling_defaults`. A consuming solve must request the
  same precision, equilibration, `branch_states_method`, fusion and modeling defaults,
  else it is rejected; parameter overrides and branch states are snapshot-checked.
- `HarmonicFlowSystem`: changed grid / defaults / overrides / states / frequencies /
  layout rebuild network assembly; the evaluated matrix and numerical options govern
  exact factor reuse, compared BY VALUE (structural route: entries plus pattern). Device
  admittances and RHS are always recomputed. Matrix gradients bypass the numerical cache;
  RHS-only gradients may reuse factors. No cached autograd graph, no global
  sparse-symbolic cache, bounded by entry count (not bytes), not thread-safe (one
  instance per worker). A per-scenario `Y(h)` is not retained unless
  `cache_batched_factors=True`.
- The IFT adjoint factorization is cached on the autograd node under
  `solver.ift.adjoint_factor_cache_mb`; the Jacobian build (whole batch, chunk, or
  column-wise) is chosen by `solver.ift.jacobian_budget_mb`, and a chunk of one is always
  affordable, so the budget never forces the column build.
- Harmonic scenario chunks fit `solver.harmonic.system_budget_mb`; chunking changes the
  RHS in its last bit (vectorized vs scalar complex multiply), so results across chunk
  sizes agree to rounding, not exactly.

## Contract surface used by other packages

`solve_power_flow`, `prepare_power_flow`, `PowerFlowResult` (`v`, `index`, `converged_mask`,
`diagnostics`, `regulation`, `fusion`), `solve_harmonic_flow`, `HarmonicFlowResult`,
`assemble_harmonic_system` (the physics-consistency residual `Y(h)·V − I(h)`),
`assemble_harmonic_ybus`, `harmonic_injections`, `NodeHarmonicSource`,
`HarmonicFlowSystem`, `solve_anchored` / `AnchoredSystem` (injection-decode anchoring and
WLS state estimation), `lu_factor_system` / `solve_factored` / `FactoredSystem` /
`estimate_condition`, `check_connectivity`, the defaults keys under `solver.*`.

## Tests

Gates: `tests/differentiability` (float64 gradcheck of every path: `test_solver_gradcheck`,
`test_power_flow_gradcheck`, `test_harmonic_*_gradcheck`, `test_woodbury_gradcheck`,
`test_mixed_precision_gradcheck`, `test_sparse_solver_gradcheck`, `test_anchored_gradcheck`)
and `tests/gpu` (device/dtype parity). Oracles and structure: `tests/reference`
(`test_harmonic_flow`, `test_opendss_load_shunt`, `test_sparse_solver`,
`test_structural_harmonic_route`, `test_prepared_system`, `test_prepared_harmonics`),
`tests/topology`, `tests/asymmetric`. A const-impedance ZIP solve must reproduce the linear
`assemble_ybus` + `solve_harmonic` result.
