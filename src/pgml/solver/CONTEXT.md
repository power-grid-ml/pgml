# Interface ledger: solver  (FROZEN rev 1)

Complex linear solve of the per-frequency nodal system `Y(f) V(f) = I(f)`, batched,
differentiable, GPU. Consumes the compact node-phase layout from `assembly/`.

## Public API (IMPLEMENTED — final signature)
Module: `pgml.solver` (`from pgml.solver import solve_harmonic`).
- `solve_harmonic(y_bus, i_inj, *, fixed_rows=None, v_fixed=None, precision="full") -> v`
  - `y_bus`: complex `[*batch, H, N, N]`  (or `[N, N]` unbatched)
  - `i_inj`: complex `[*batch, H, N]`     (or `[N]` unbatched)
  - returns `v`: complex `[*batch, H, N]` (or `[N]` when BOTH inputs were unbatched
    2-D/1-D)
  - Broadcasts `i_inj` against `y_bus` over all leading/batch dims and over H
    (explicit `broadcast_shapes` + `broadcast_to`).
  - `fixed_rows`: int64 1-D row indices to hold fixed (ideal slack); requires
    `v_fixed`. `v_fixed`: complex, broadcastable to `[*batch, H, len(fixed_rows)]`
    (e.g. `[len(fixed_rows)]` constant across H/batch).
  - Gradients flow w.r.t. `y_bus`, `i_inj`, `v_fixed`. Dense `torch.linalg.solve`.
  - `precision`: `"full"` (default, solve at `y_bus`'s dtype) or `"mixed"` (factor a
    complex64 copy, refine against complex128 residuals — needs a complex128 `y_bus`;
    routed through `lu_factor_system`/`solve_factored`, see MIXED PRECISION below).
- `solve_anchored(y_bus, i_inj, *, row_weight=None, row_target=None, op=None,
  op_weight=None, op_target=None, fixed_rows=None, v_fixed=None) -> v` — MEASUREMENT-ANCHORED
  (over-determined) network solve: `min_V ‖Y·V−I‖² + Σ w_r|V_r−t_r|² + Σ w_k|(op·V)_k−t_k|²`
  s.t. `V[fixed_rows]=v_fixed`. `row_*` softly anchor node values ([*batch,N] real weight ≥0,
  complex target) — e.g. measured bus voltages; `op` is a grid-constant complex `[K,N]` linear
  operator (e.g. the branch-current map) whose functional `op·V` is anchored by `op_weight`
  [*batch,K] / `op_target` [*batch,K] — e.g. measured branch currents. No anchors ⇒ identical
  to `solve_harmonic`.
  - `y_bus` is a SINGLE shared operator `[N,N]` (loop externally over any H/topology axis); the
    solve inverts `Y` ONCE for the whole batch via the reduced correction
    `V = V₀ + Y⁻¹r`, `V₀ = Y⁻¹I`: the anchors form a correction system `G·r = …` with
    `G = I + Σ w·(A Y⁻¹)^H(A Y⁻¹)` — Hermitian PD, eigenvalues ≥ 1, Cholesky-factored per
    right-hand side. The identity floor keeps the factorization stable and the physics block
    avoids normal-equations κ(Y)²; κ(G) itself still grows with `w·σmax(Y⁻¹)²`, so scale
    anchor weights relative to `Y` (a typical singular value — the convention a downstream
    consumer's auto-scale follows). Anchor weights are cast to the real dtype paired with
    `y_bus` (complex64/128 both supported). Returns `[*batch, N]`.
  - Gradients flow w.r.t. `y_bus`, `i_inj`, the targets and `op`. Built for injection-decode
    anchoring in a downstream state-estimation solver; reusable for a classical WLS
    state-estimation solve too.
- `AnchoredSystem(y_bus, *, op=None, fixed_rows=None)` — FACTOR-ONCE state of the anchored
  solve of ONE shared operator: precomputes what `solve_anchored` rebuilds per call (the
  free-block inverse image `Z = Y_ff⁻¹`, the slack coupling, `op_free·Z`).
  `.solve(i_inj, *, row_weight=None, row_target=None, op_weight=None, op_target=None,
  v_fixed=None) -> v` answers each batch through the push-through identity on the ANCHORED
  rows: with `B = √W·A·Z` `[R,F]` the correction is `r = −Bᴴ(I_R + BBᴴ)⁻¹d` — a Cholesky of
  the `[R,R]` capacitance instead of the dense `[F,F]` build + factorization, `O(R²F+R³)`
  per call for `R` anchored channels. Algebraically identical to `solve_anchored` on the
  same inputs (same identity-floored objective; rounding differs at machine precision);
  per-sample heterogeneous anchor patterns are padded with zero-weight rows (exact).
  Construction REFUSES an operator on the autograd tape (the cached images are constants —
  use `solve_anchored` for a learned `Y`); gradients flow through `.solve` w.r.t. the RHS,
  the targets, the weights and `v_fixed`. `.nbytes()` (cache accounting) / `.to(device)`.
  Built for a downstream operator cache used by training runs that solve the same network
  tens of thousands of times.

## Slack / reference handling (both modes, both differentiable)
1. **Norton (default, `fixed_rows=None`)**: sources are already stamped as a shunt
   `Y_s` + current `I_s` by `assembly/`, so `Y` is non-singular; just
   `v = torch.linalg.solve(Y, I)`. The slack bus voltage equals `u_ref` only up to
   the drop across `Z_s` (this matches OpenDSS Vsource behaviour).
2. **Ideal slack (`fixed_rows` int64 + `v_fixed` complex given)**: hold
   `v[..., fixed_rows] = v_fixed` exactly via a partitioned (Schur) solve
   `v_free = Y_ff^-1 (I_free - Y_fs v_fixed)`; reassemble full `v`. Use this to
   match pandapower / pgm ideal-slack results. Implement with gather/index_select
   (no in-place on tracked tensors) so gradients flow w.r.t. `Y`, `I`, `v_fixed`.

## Rules
- DIFFERENTIABLE + GPU (CLAUDE.md). No `.detach()/.item()/.numpy()`, no in-place on
  tracked tensors, no Python control flow on tensor values. Dense first
  (`torch.linalg.solve`); sparse only if size later demands it.
- Identical results CPU vs CUDA within tol; honor input dtype (complex128 available
  for gradcheck).
- Self-check: gradcheck (complex128) of `v` w.r.t. `Y` and `I` on a tiny system,
  for both slack modes; finite-difference spot check on at least one entry.

## Implementation notes (DONE)
- Ideal-slack reassembly uses `index_select` (gather Y_ff/Y_fs/I_free) + an
  out-of-place `scatter` into a fresh zero vector (no in-place on tracked tensors);
  free rows are the boolean complement of `fixed_rows`.
- Tests: `tests/differentiability/test_solver_gradcheck.py` (gradcheck both modes,
  batched, + finite-difference spot check). The end-to-end grid path (assemble +
  solve) gradcheck is `tests/differentiability/test_gradcheck.py`.

# =====================================================================
# NONLINEAR fundamental power flow (FROZEN — IMPLEMENTED)
# =====================================================================
`solve_harmonic` stays as the LINEAR per-frequency solve (used by the const-Z path
and, later, by each harmonic). Add the nonlinear fundamental solver:

- `solve_power_flow(grid, *, slack="ideal", method="current_injection",
     tol=None, tol_update_pu=None, s_base_va=None, max_iter=100,
     dtype=torch.complex128, precision="full", device=None,
     operating_point=None, param_overrides=None, symmetry=None,
     criticality="auto") -> PowerFlowResult`
  - `symmetry`: `None`/`"auto"`/`"symmetric"`/`"asymmetric"` (`None`
    -> config `calculation.symmetry`). Resolved ONCE here (`resolve_asymmetric`),
    logged ONCE (`log_modeling_summary`), and threaded as the resolved string into
    every `device_current_injections` call of the iteration (which resolves silently,
    no per-iteration logging). `symmetric` ignores per-phase data (equal split);
    loads/gens are folded connection-aware (WYE/DELTA/neutral). Existing kwargs/
    defaults unchanged. The forward warm start is PHASE-AWARE (balanced rotation;
    neutral rows ~0 V) so DELTA / WYE-neutral const-P currents do not hit 0/0. Its
    MAGNITUDE is per-row: each fixed (source,phase) row seeded with its own
    `|v_fixed|`, other rows with a balanced default = the source's Phase-A reference
    magnitude (not row 0 — robust to a non-A first phase / mixed source magnitudes).
    Entire warm start under `no_grad`; never changes the converged fixed point.
  - Solves the const-P / ZIP fundamental power flow at f0 = `grid.base_frequency_hz`.
  - `PowerFlowResult` (frozen dataclass): `v` complex `[*batch, N]` (DIFFERENTIABLE),
    `index: NodePhaseIndex`, `iterations: int`, `residual: Tensor` (the achieved PRIMARY
    criterion: the largest per-unit apparent-power mismatch), `converged: bool`
    (ALL scenarios), `diagnostics: ConvergenceDiagnostics`, `converged_mask: Tensor|None`
    (`[*batch]` bool, `None` unbatched), `failed_states: tuple[int,...]` (flat indices of
    non-converged scenarios). A BATCHED solve NEVER raises on a failed element — every
    element's best-effort `V` is returned, the failures are listed here AND logged as an
    error (count, indices, worst residual, likely cause).
  - CONVERGENCE is PER element and PER UNIT — both criteria must hold (`_PuConvergence`):
    - PRIMARY `tol` (= `solver.convergence.mismatch_pu`, default 1e-8 pu): the largest
      nodal apparent-power mismatch `max_free |V_i conj(F_i)| / s_base_va` with
      `F = Y_eff V + I_device(V) - I_slack`. This is pandapower's `tolerance_mva`
      quantity on a 1 MVA base and the same order as power-grid-model's
      `error_tolerance`, so ITERATION COUNTS ARE COMPARABLE across the three tools.
    - SECONDARY `tol_update_pu` (= `solver.convergence.update_pu`, default 1e-8 pu): the
      largest per-row voltage update `max_rows |ΔV| / V_LN(node)`.
    - `s_base_va` (= `solver.convergence.s_base_va`, default 1e6) is the power base.
      `None` on any of the three resolves the documented default.
    Per-row normalisation makes each measure independent of the voltage level AND of the
    row count, so a multi-voltage grid and a merged ensemble are judged exactly like a
    single feeder. Each threshold is capped by the PRECISION FLOOR: the voltage update by
    `_rel_convergence_floor` (16 eps at float64; 1e-6 float32 dense / 1.2e-5 float32
    sparse SuperLU - measured plateaus), the mismatch per row by
    `_mismatch_floor_rel * sum_j|Y_ij||V_j|` (the cancellation scale of `Y V`; a 20 kV
    node behind a milliohm impedance bottoms out near 1e-9 pu at complex128), both
    multiplied by a low-rank update's measured `amplification`. A tolerance below the
    floor logs a WARNING naming the floor, and the floor governs - the solve converges
    instead of running to `max_iter` at a converged voltage.
  - `precision`: `"full"` (default) factors at `dtype`; `"mixed"` factors at complex64
    and keeps the iteration, the residual and the convergence test at complex128 (which
    it requires). The fixed point then runs in RESIDUAL-CORRECTION form
    `V_{k+1} = V_k - A_s^-1 F(V_k)` - algebraically the same fixed point, so an inexact
    `A_s^-1` changes only the contraction rate and the converged voltage keeps
    complex128 accuracy; Newton solves its direction in single precision (inexact
    Newton). Measured below 1e-9 pu against a complex128 reference on IEEE-33, CIGRE LV
    three-phase and a 3600-row feeder. A plain complex64 solve (`dtype=complex64`,
    `precision="full"`) logs a ONE-TIME warning naming the estimated condition number
    when it exceeds `solver.precision.complex64_cond_warn`.
  - `ConvergenceDiagnostics` (autograd-free, computed at `V*`): `converged`, `iterations`,
    `mismatch_max_pu` (PRIMARY criterion) + `update_max_pu` (SECONDARY) in per unit, with
    the SI values under explicit unit names `mismatch_max_va`, `mismatch_max_a`
    (= max `|F_c|` over free rows [A]) and `update_norm_v` (= `||ΔV||` 2-norm [V]), plus
    `s_base_va`, `residual_history` (the per-iteration per-unit update),
    `worst_nodes` (per-row `mismatch_pu`/`mismatch_va`/`mismatch_a`/`v_pu`),
    `out_of_band_nodes` (pu on each node's L-N base), `voltage_band_pu`, `likely_cause`
    (heuristic: converged / diverged / oscillating / overload / near-singular), and
    `criticality`. The cheap state diagnostics are ALWAYS populated and cost no extra
    solve (the forward's own residual is reused); `simulate(strict=True)` passes
    `diagnostics.as_dict()` into the raised `ConvergenceError`.
  - `criticality` kwarg (`"auto"`/`"always"`/`"never"`): runs the IFT-Jacobian analysis
    — the SAME real `[2N,2N]` `J = dR/dV` the IFT backward builds, then `svdvals(J)` +
    the right singular vector of the smallest σ for the critical-bus participation.
    `"auto"` = only on non-convergence; `"always"` = also on a converged solve (a
    voltage-collapse MARGIN: σ_min shrinks toward the nose). Rigorous AT a solution; at a
    DIVERGED iterate it is only a local linearization (flagged `evaluated_at`, never
    claims `near_singular`) — a definitive loadability limit needs the (future)
    homotopy/continuation. Dense; skipped above `2N=4000`. SINGLE-GRID only: SKIPPED for a
    batched solve (`b>1`, logged) — re-run one scenario, or use `loadability_limit`. A
    batch of ONE is still one grid: the singleton axis a batched residual closure leaves
    on the Jacobian is folded away, and any other non-square Jacobian shape is reported
    as a `{"skipped": ...}` dict (a diagnostic never raises).
  - `method="current_injection"` (default) forward = FIXED POINT: with
    `Y_net = assemble_network_ybus` (+ source Norton if `slack="norton"`), iterate
    `V_{k+1} = solve_harmonic(Y_eff, I_slack − device_current_injections(grid,V_k),
    slack...)` until `||V_{k+1}-V_k|| < tol` or `max_iter`, under `torch.no_grad()`.
  - `method="newton"` forward = NEWTON on the real residual `R(x)=0` (`x=[Re V; Im V]`):
    per step solve `J·Δx = −R` with `J = dR/dx` (the SAME real `[2N,2N]` Jacobian the IFT
    backward builds, via `torch.autograd.functional.jacobian`), backtracking line search
    on `‖R‖∞`, converge on `‖Δx‖ < tol`. WARM START = the LINEAR const-Z solution
    (`_linear_const_z_init` → one `assemble_ybus` solve; OpenDSS-style). Quadratic, and
    converges where the fixed point oscillates (near the loadability nose — see
    `run/examples/current_injection_convergence.py`). Same `PowerFlowResult` + diagnostics +
    IFT gradients (gradcheck-verified). `linear_solver="dense"` (per-element `[2N,2N]`
    Jacobian + direct solve; no `[B,2N,B,2N]` blowup) or `"matrix_free"` (Jacobian-free
    Newton-Krylov: GMRES on finite-difference `J·v`, `O(N)` memory for large grids).
    A BATCHED `operating_point` is solved SEQUENTIALLY per scenario (Newton's const-Z
    warm start + per-element Jacobian are single-grid, and the residual closes over the
    batched op) and the per-scenario `V*` are stacked; the SHARED IFT backward (full op,
    batch-aligned) then attaches batched gradients — so differentiability is unchanged.
    Newton is the hard-grid / near-nose solver; for bulk batches use current-injection.
  - VOLTAGE-REGULATING TERMINALS (PV buses): a `Generator` carrying a
    `VoltageRegulation` block (setpoint `v_set_pu` in per unit of the host node's rated
    voltage, optional `q_min_var`/`q_max_var` totals, `regulated` =
    positive-sequence | per-phase) has its terminal's REACTIVE power-balance row
    replaced by `(|V_reg|² − V_set²)/(2·V_set)` and its ACTIVE row by
    `Re(conj(V)·F_c)/v0` (both scaled so the active row stays in amperes and the
    setpoint row matches the ideal-slack pin rows). The reactive power is eliminated
    ANALYTICALLY (it enters only the imaginary part of the power-form row), so the
    state stays `[Re V; Im V]`, the `[2N,2N]` IFT Jacobian / adjoint is unchanged, and
    `Q` is recovered at `V*` as `Q_e = Q_pinned − Im(conj(V_r)·F_c,r)` summed over the
    unit's elements. For a 3-phase positive-sequence unit the other two imaginary rows
    carry the equal-reactive-split conditions `Im(g_e) − Im(g_0) = 0`, which are
    reactive-power-free, so the elimination stays exact. Implementation:
    `solver/_pv_bus.py` (`PVTerminals`, `collect_pv_terminals`).
    - `enforce_q_limits: Optional[bool] = None` (kwarg; `None` → config
      `appliance.generator.enforce_q_limits`, default TRUE — pandapower `runpp`'s own
      default is False): limits are enforced by PV-to-PQ SWITCHING, one complete solve
      per round at a FIXED active set, with a hysteresis band
      (`appliance.generator.q_limit_hysteresis_pu` / `_rel`, max rounds
      `q_limit_switch_rounds_max`). The switching decision is off-tape; the residual at
      the resolved active set is on-tape, so `dV/dv_set` (regulating) and
      `dV/dq_limit` (pinned) are exact. A non-converged round stops the loop (a
      decision read off an unsettled iterate would switch on noise).
    - METHOD: a grid with a PV terminal is always solved by Newton (the
      current-injection fixed point has no setpoint to iterate on); `method=
      "current_injection"` logs a WARNING and switches. Newton gets a SECOND warm
      start in that case — the balanced nominal profile with every regulated row AT its
      setpoint (`_pv_nominal_init`) — ordered against the const-Z seed by whether that
      seed collapses below `_SEED_COLLAPSE_PU = 0.5` of nominal; a start that fails is
      followed by the other, and the restart is logged (`_newton_from_starts`).
    - RESULT: `PowerFlowResult.regulation: Optional[VoltageRegulationResult]` —
      `q_var {gen_id: [*batch] var}` (solved total, autograd-free like the
      diagnostics), `regulating {gen_id: [*batch] bool}`, `switch_rounds`,
      `enforce_q_limits`. The convergence diagnostics report the ACTIVE component of
      the mismatch at a regulating row (its raw current mismatch is the reactive
      current the machine supplies).
    - BATCHED: `operating_point[gen_id]["v_set_pu"]` is a per-scenario setpoint (float
      or `[*batch]` tensor, same per-unit base); limits and the active set batch with
      it, so different scenarios may pin different units. A `v_set_pu` on an appliance
      that is not an in-service regulating generator raises; a `q_var` override on one
      that is warns and is ignored.
    - REFUSED (ModelingError, at the solve): DELTA connection, a WYE terminal returning
      through its node's neutral row (use `return_path='ground'`), a positive-sequence
      setpoint on a 2-phase terminal, and a regulating generator on an in-service
      Source's node.
    - HARMONICS: regulation is a fundamental-frequency concept; at orders h>1 the
      machine stays the Norton current source it is today (see `solve_harmonic_flow`).
  - `loadability_limit(grid, *, slack, lambda_max, lambda_step, bisect_tol, tol,
    tol_update_pu, s_base_va, max_iter, top_k, ramp="all") -> LoadabilityResult`:
    λ-RAMP loadability. Scales the injections by `λ`
    (`R(V,λ)=Y_eff·V+λ·I_dev(V)−I_slack`) from a feasible base, Newton-correcting at each
    step and bisecting onto the first λ the corrector cannot solve. `breaking_lambda` is
    therefore the largest λ at which the Newton corrector CONVERGES, a LOWER BOUND on the
    P-V nose (a plain corrector fails before the singularity; measured ~4 % below the
    closed-form nose of a two-bus feeder) — a step-and-bisect on feasibility, NOT an
    arc-length predictor-corrector, and the Jacobian figures describe the last converged
    point. `ramp="all"` (default) scales every injecting device (loads AND
    generators/storage together); `ramp="load"` scales loads only and holds generation at
    nameplate — the textbook continuation ramp (on a two-bus feeder with a load at
    `0.8 P*` and generation at `0.3 P*` the two give 2.0 vs 1.625, both closed-form).
    `LoadabilityResult.ramp` records the choice. At the breaking λ the Jacobian's RIGHT
    singular vector = the voltage-collapse mode (`critical_nodes`, where it breaks) and
    the LEFT singular vector projected on each device's current = `limiting_loads` (which
    injection most reduces the margin). `breaking_lambda<1` ⇒ the nameplate loading does
    not solve. Single grid; detached.
  - Backward = IMPLICIT FUNCTION THEOREM at the converged `V*` (do NOT unroll
    iterations): one adjoint linear solve with the transposed power-flow Jacobian.
    Implement as a `torch.autograd.Function` whose backward solves `J^T λ = grad_V`
    and forms parameter grads via a vjp of the residual `F(V*,θ)` (autograd on a
    single residual evaluation at `V*`). Gradients must flow to network params,
    device powers (P/Q), and slack voltage. **Use REAL (re/im split) coordinates**
    for the residual/Jacobian/adjoint (the power flow is non-holomorphic in V).
  - `slack`: `"ideal"` (fix source-node V = `u_ref∠u_angle` via the Schur path in
    `solve_harmonic`; matches pandapower/pgm) or `"norton"` (source folded; OpenDSS).
  - `operating_point` may carry a per-source `{source_id: {"u_ref_scale": Tensor[*b]}}`
    entry (alongside the usual load/gen `p_w`/`q_var`/per-phase keys): a per-scenario
    multiplier the IDEAL slack applies to the Source's `u_ref_v` (magnitude scaled, angle
    kept) → a BATCHED `v_fixed` `[*b, S]`. The network side stays operating-point-independent
    (a `prepare_power_flow` system reuses its cached factorization; only the slack VALUE is
    recomputed when a scale is present), so a source-voltage scenario sweep threads the slack
    through the operating point, NOT the grid object. Differentiable w.r.t. the scale leaf via
    the IFT (it is collected like any operating-point leaf); the warm start uses the un-scaled
    reference (a ballpark seed). Written by `pgml.scenarios` `ParameterSpec(field="u_ref")`.
  - Batched over leading/scenario dims (the fixed point solves the batch in one
    `solve_harmonic`; Newton loops the per-element Jacobian, so it is best for a single
    hard grid rather than a large batch).

## Required validation links
- A const-impedance ZIP run of `solve_power_flow` must reproduce the linear
  `assemble_ybus` + `solve_harmonic` result EXACTLY (same linear system).
- A const-power run must match pandapower/pgm with their DEFAULT const-power loads
  (the real PF) on IEEE33 (and CIGRE LV) — owned by the reference agents. DONE:
  IEEE33 ~3e-9 pu, CIGRE LV ~1e-7 pu.
- gradcheck (float64) of `v` w.r.t. line R/L and a load's P/Q through the IFT path.

# =====================================================================
# HARMONIC power flow (IMPLEMENTED — harmonic_flow.py)
# =====================================================================
Reuses `assemble_network_ybus` (builds Y at any `h·f0`) + `solve_harmonic` (batched
per-frequency linear solve). The OpenDSS conventions are pinned in
`docs/pgml/modeling/references/opendss/harmonics.md` (READ IT — esp. the spectrum phase convention,
verified empirically). New orchestration:

- `solve_harmonic_flow(grid, harmonic_orders, *, slack="ideal", method="current_injection",
     operating_point=None, harmonic_injection=None, node_sources=None,
     load_shunt=None, tol=None, tol_update_pu=None, s_base_va=None,
     max_iter=100, dtype=torch.complex128, precision="full",
     device=None, symmetry=None, on_disconnected="raise", branch_states=None,
     param_overrides=None, enforce_q_limits=None) -> HarmonicFlowResult`
  - `tol` / `tol_update_pu` / `s_base_va` are the PER-UNIT convergence settings of the
    nonlinear fundamental (see `solve_power_flow`); the harmonic orders are direct linear
    solves with no iteration and therefore no convergence criterion of their own.
  - `enforce_q_limits` likewise reaches the fundamental only: regulation is a
    fundamental-frequency concept, so at h>1 a regulating generator is the same Norton
    current source as any other.
  - `precision` applies to the fundamental AND to every per-order solve (where
    `"mixed"` is the classic iterative refinement of `lu_factor_system`).
  - `param_overrides` is the SAME parameter-substitution hook `solve_power_flow` and the
    assemblers take, reaching the fundamental solve, every order's `Y(h)`, the source
    stamp and the device powers behind each harmonic injection, so a gradient w.r.t. a
    substituted parameter flows into every order (gradchecked). `assemble_harmonic_system`
    and `assemble_harmonic_ybus` take it too.
  - `method` is forwarded to the fundamental `solve_power_flow`; use `"newton"` for a
    controlled DER (Volt-VAr/Volt-Watt loops oscillate under the current-injection fixed
    point). A controlled Generator/Storage's harmonic injection scales from its
    CONTROL-RESOLVED fundamental current (consistent with the control-aware fundamental
    solve), not the nominal — see `assembly/_control.py` + `docs/pgml/modeling/der-pv-storage.md`.
  - `symmetry`: `None`/`"auto"`/`"symmetric"`/`"asymmetric"`. Resolved
    ONCE here; threaded into the fundamental `solve_power_flow` (which emits the single
    modeling-summary log) and into the harmonic-injection power resolution
    (`resolve_operating_power(..., asymmetric=...)`).
  - CONNECTION-AWARE / PER-PHASE HARMONIC INJECTION: the earlier node-level
    DELTA / 4-wire NotImplementedError guard is LIFTED. `_harmonic_injections` now
    mirrors `device_current_injections`: each injecting Load/Generator has a terminal
    incidence `M [n_elem, n_used]` (`pgml.assembly._incidence`; WYE-ground `M=I`,
    WYE-neutral `[I|-1]`, DELTA-3 circulant). The per-ELEMENT fundamental current is
    `I1_elem = sign*conj(S0_elem)/conj(V_term)` with `V_term = M @ V_used` (TERMINAL
    voltage: WYE phase row, WYE-N `V_phase - V_N`, DELTA L-L), and per element/order
    `|I_h^e|=(mag_h^e/mag_1^e)|I1_elem|`, `arg=ang_h^e + h*(arg(I1_elem)-ang_1^e)`.
    The NODAL injection is `-(M^T @ i_h_elem)` scattered into `used_rows` (out-of-place
    complex `index_add`). WYE-to-ground reduces EXACTLY to the earlier node-level per-phase
    form (identical values: `tests/reference/test_harmonic_flow.py` +
    `test_carson_harmonics_feeders.py`).
    Spectrum coefficients are PER ELEMENT, from three sources (override > schema):
    device `spectrum` (StaticSpectrum, same on all elements), `spectrum_per_phase`
    (element k <- `phases[k]`; a phase/element with no entry injects 0; for DELTA-3 the
    key is the branch's starting phase `phases[k]`), and the runtime `harmonic_injection`
    override (see below).
  - `harmonic_orders`: iterable of orders (e.g. `[1,5,7]`; order 1 = fundamental).
  - `HarmonicFlowResult` (frozen dataclass): `v` complex `[*batch, H, N]` (V per
    order; **order 1 = the nonlinear `solve_power_flow` solution**, other orders =
    the linear per-harmonic solve), `frequencies_hz [H]`, `index`, `pf`
    (the fundamental `PowerFlowResult`). Convergence properties `converged` /
    `converged_mask` / `failed_states` re-expose `pf`'s (the harmonics are direct solves).
  - DIFFERENTIABLE end to end (network params, load P/Q, AND harmonic injections)
    and BATCHED over scenario dims, same conventions as `solve_power_flow`.
  - A per-scenario `operating_point` (`[B]`) combined with a DEEPER-batched
    `harmonic_injection` (a node-coherent `[B, T]` sequence over a `[B]` fundamental) is
    supported: v1 stays `[B, N]` (in step with the same-batch op that forms each device's
    fundamental current), and the per-device fundamental current is broadcast across the
    injection's extra step axis; only the order-1 slice returned to the caller has its batch
    rank lifted to the injection's so it stacks against the `[B, T, N]` harmonic slices. A
    no-op for the snapshot (`[B]`/`[B]`) and nominal (empty-op) cases — byte-identical.

- `assemble_harmonic_system(grid, harmonic_orders, v1, *, operating_point=None,
     harmonic_injection=None, node_sources=None, load_shunt=None, symmetry=None,
     dtype=torch.complex128, device=None, branch_states=None,
     param_overrides=None) -> (Y, I, index)`
  - Exposes the per-harmonic LINEAR system `Y(h) V(h) = I(h)` for orders `h > 1` —
    EXACTLY the `(yh, ih)` `solve_harmonic_flow` builds (`assemble_network_ybus` +
    source Norton stamp + the device harmonic shunt for `Y`; `_harmonic_injections` +
    `node_sources` for `I`), so
    `solve_harmonic(Y, I)` reproduces the harmonic slices. The harmonic network is
    LINEAR, hence `r(V) = einsum('...hij,...hj->...hi', Y, V) − I` is the
    physics-consistency residual (`≈ 0` at the true `V`); a downstream consumer can form it
    without re-deriving the assembly. `solve_harmonic_flow`
    CALLS this (single source of the harmonic assembly — no duplication).
  - `harmonic_orders`: orders `h > 1` only (order 1 is the nonlinear fundamental —
    passing 1 raises `InputError`). `v1`: converged fundamental node voltage
    `[*batch, N]` complex (typically `solve_power_flow(grid, ...).v`), aligned to
    `index`. `operating_point` / `harmonic_injection` / `node_sources` / `load_shunt` /
    `symmetry`: same meaning/format as `solve_harmonic_flow` (resolve `symmetry`
    upstream and pass the canonical string to reproduce a `solve_harmonic_flow` run
    exactly).
  - Returns `Y` complex `[Hh, N, N]` (or `[*batch, Hh, N, N]` if a batched device shunt
    or a BATCHED voltage `node_source` promotes it), `I` complex `[*batch, Hh, N]`, and the
    `NodePhaseIndex`. `device=None` -> `v1.device`; honours `dtype`/`device`.
  - DIFFERENTIABLE (grad to grid params, `v1`, and the injections; no
    `.item()/.detach()/.numpy()`, no in-place on tracked tensors) + GPU + batched.
    `v1` enters `I(h)` via each device's fundamental terminal current
    `I1_elem = sign·conj(S0_elem)/conj(V_term)` (and `E_h` for voltage `node_sources`).
  - Self-check: `tests/differentiability/test_harmonic_system_residual_gradcheck.py`
    (float64 gradcheck of `r = Y·V − I` w.r.t. `V` and a line `R`; `r ≈ 0` at the true
    `V = solve_harmonic(Y, I)`).

### Steps (per the OpenDSS model)
1. Fundamental: `pf = solve_power_flow(grid, slack=slack, operating_point=...)`.
   Compute each device's FUNDAMENTAL current phasor `I1` (per phase) from the
   converged `pf.v` — needs a PER-DEVICE current (add a helper / option to
   `device_current_injections` to return per-device, not just the nodal sum).
2. Per harmonic `h>1` (batched over all orders at once):
   - `Y(h) = assemble_network_ybus(grid, [h·f0]) + source Norton shunt Y_s(h)
     + each device's harmonic shunt Y_load(h)` (`load_shunt`, default
     `appliance.harmonic_shunt.model` = `"opendss"`; `"none"` = the OpenDSS
     `NeglectLoadY` pure current-source model, `"motor"` = the blocked-rotor series
     branch). Per element: `Y_eq = conj(S_eff)/V_rated²` at the realised fundamental
     power, split into `(1−s)Re(Y_eq) + j(1−s)Im(Y_eq)/h` and
     `1/(Re(Z_s) + j·h·Im(Z_s))` with `Z_s = 1/(s·Y_eq)`, stamped
     `Mᵗ diag(y_elem) M` through the SAME incidence the injection uses
     (`pgml.assembly._load_shunt`, `_stamp_harmonic_load_shunt`). A per-device
     `HarmonicShuntModel` overrides the model; a per-scenario operating point makes
     `Y(h)` `[*batch, Hh, N, N]`.
   - `I(h)` = sum of per-device harmonic injections using the verified convention
     `|I_h|=(mag_h/mag_1)|I1|`, `arg(I_h)=ang_h + h·(arg(I1) − ang_1)` from each
     device's `Spectrum` (or the `harmonic_injection` override).
   - `V(h) = solve_harmonic(Y(h), I(h))` in NORTON mode (no ideal slack at
     harmonics: the source contributes ONLY its Norton shunt `Y_s(h)`, i.e. its own
     harmonic EMF is zero). Upstream / background distortion is an OPERATING-POINT
     quantity, not grid data: supply it per solve as a `NodeHarmonicSource` at the
     source's node (`kind="voltage"` = a Thevenin EMF behind that same shunt), which
     `pgml.scenarios`' `BackgroundHarmonicConfig` realizes reproducibly
     (`build_background_sources`). A `Source` has no `spectrum` field.
3. Stack order 1 (from PF) + harmonics into `v [*batch, H, N]`.

### `harmonic_injection` override (scenario-ready, tensor-friendly, per-element)
A per-device override of the stored spectrum, carrying tensors so scenarios can vary
harmonic injections DIFFERENTIABLY (the stored `Spectrum`/`HarmonicComponent` are
plain floats; do NOT hard-bind to them). FORMAT:
`{appliance_id: {order:int -> (magnitude_pu, phase_deg)}}`. Each `magnitude_pu` /
`phase_deg` value follows an UNAMBIGUOUS, type-driven convention (NO trailing-dim
sniffing — a bare length-`n_elem` tensor is NOT treated as per-element, so a SCENARIO
batch of length `n_elem` can never be silently misread):
- a python `list`/`tuple` is ALWAYS PER-ELEMENT — it MUST have length `n_elem`
  (aligned to the device's elements: WYE phase `k` / DELTA branch `k`); each entry may
  itself be a python float or a 0-d / `[*batch]` tensor (per-entry grad preserved via
  `torch.stack`). A list/tuple of any OTHER length raises `ValueError`. E.g.
  `{30:{1:(1.0,0.0),5:([m_a,m_b,m_c],0.0)}}`.
- a SCALAR or bare TENSOR (python float, 0-d tensor, or a `[*batch]` tensor carrying
  ONLY leading SCENARIO batch dims, NO element axis) is BROADCAST identically to every
  element (the backward-compatible inc-1 path: `{2:{1:(1.0,0.0),5:(m5,0.0)}}`).
The override wins over the stored `spectrum` / `spectrum_per_phase`. Analogous to
`operating_point` for P/Q. Implemented in `_device_element_spectra` + `_element_coeff`
(broadcast via unsqueeze+expand on the element axis). Guarded 0/0: an element with no
order-1 coefficient (mag1==0) injects 0 (torch.where, gradient-finite on live
elements); the per-element `conj(vt)` divide is likewise masked for a dead/zero
terminal (vt==0 -> 0, gradient-safe) so a gradcheck perturbation cannot poison it.

### `node_sources` — per-node harmonic "error" source (full physics: `docs/pgml/modeling/error-injection.md`)
`node_sources: Optional[Sequence[NodeHarmonicSource]] = None` — a disturbance at ANY
node (NOT tied to a load), applied ONLY at orders `h>1` so the fundamental PF is
preserved EXACTLY (no reactor needed; pgml solves each harmonic as its own linear
system). `None` (default) is BYTE-IDENTICAL to today. A list of sources superposes.

`NodeHarmonicSource` (frozen dataclass, exported from `pgml.solver`):
`NodeHarmonicSource(node_id: int, phases: Optional[tuple[Phase,...]] = None,
spectrum: dict[int, tuple[mag_pu, phase_deg]] = {}, source_power_va: float = 0.0,
kind: Literal["voltage","current"] = "voltage")`. `phases=None` -> all of the node's
phases. `spectrum` order 1 = reference (may be omitted -> `mag_1=1.0`, `ang_1=0.0`).
`source_power_va` (S_sc / MVAsc) and every spectrum coefficient may be a python float
OR a 0-d / `[*batch]` tensor (differentiable, scenario-batchable).

Math, per order `h>1` per source at the node-phase rows (`index.rows_for_terminal`),
in `_apply_node_sources`:
- `V_base = phase_voltage_magnitude(node.u_rated_v, len(node.phases))` (L-N base).
- `Y_s = source_power_va / V_base**2` (REAL, frequency-flat / resistive, x1r1≈0).
- `E_h = (mag_h/mag_1)*|V1| * exp(j*(rad(ang_h) + h*(angle(V1) - rad(ang_1))))`, V1 =
  converged fundamental at the row — SAME phase convention as `_harmonic_injections`.
- `I_N = E_h * Y_s`. `kind="voltage"` (Thévenin): add `Y_s` to the diagonal
  `Y(h)[...,row,row]` AND `I_N` to `I(h)[...,row]`. `kind="current"` (Norton): add
  `I_N` to `I(h)[...,row]` only.

DIFF + GPU + BATCHED: grads flow to `source_power_va`, the spectrum, and (via V1) grid
params. Adds are OUT-OF-PLACE (complex `index_add` on the current; `index_add` on a
flattened diagonal of a FRESH zero matrix for `Y`, then `Y + diag`). A batched
voltage-source `Y_s` promotes `Y(h)` to `[*batch,H,N,N]` (broadcast then add) — built
autograd/GPU-safely. Guarded `mag_1==0` (torch.where -> contributes nothing). Stiff
voltage source (large S_sc) -> `V_node(h) -> E_h`; weak -> near-zero. OpenDSS oracle:
`kind="current"` ↔ ISource, `kind="voltage"` ↔ VSource (MVAsc1=S_sc, x1r1≈0); the
OpenDSS 50-Hz-cancellation reactor is NOT needed here (h>1-only injection).
Tests: `tests/reference/test_node_harmonic_source.py` (byte-identical None, fundamental
preserved, stiff->E_h, current independent of network, voltage-divider law, multiple
sources, phase-subset), `tests/differentiability/test_node_source_gradcheck.py`
(float64 gradcheck of V w.r.t. `source_power_va` voltage+current, spectrum mag, line
R/L with source, BATCHED S_sc), `tests/gpu/test_node_source_parity.py` (CPU-vs-CUDA,
scalar + batched-S_sc-promoted Y).

## Implementation notes (DONE)
- `slack="norton"` matches OpenDSS Vsource (use for OpenDSS parity); `slack="ideal"`
  matches pandapower/pgm at the fundamental. At harmonics the source is ALWAYS a
  Norton shunt held at 0 V (regardless of `slack`).
- Per-element fundamental current `I1_elem = sign*conj(S0_elem)/conj(V_term)` computed
  inline in `_harmonic_injections` from `pf.v` via the incidence `M` (no change to
  `device_current_injections`; reuses `group_appliances`/`build_incidence`/`used_rows`).
- `harmonic_injection` magnitudes/phases may carry a leading scenario batch dim
  (batched harmonic injection works; full scenario batching is the next phase).
- The device harmonic shunt and the harmonic current injection read ONE per-element
  power (`_effective_element_power`: control-resolved / ZIP-scaled at the converged
  fundamental terminal voltage), so they cannot describe different operating points.
  `assemble_harmonic_ybus` has no fundamental solution and therefore evaluates the
  shunt at the RATED terminal voltage (exact for a constant-power device; a WARNING
  names a voltage-dependent one), unless `v1` is passed.
- A scenario-dependent `Y(h)` (device shunt or batched voltage node source) is padded
  to the harmonic injection's batch rank (`_align_y_batch_rank`), so one matrix per
  scenario serves every step of a node-coherent sequence.

### Validation (DONE)
`tests/reference/test_harmonic_flow.py`: independent numpy oracle (exact, ~1e-9, with
and without the device shunt and at both splits), fundamental==PF, OpenDSS ballpark,
shapes/orders. `tests/differentiability/test_harmonic_flow_gradcheck.py`: gradcheck of
`V(h)` w.r.t. line R/L, load P/Q, and injection magnitude (incl. batched).
`tests/reference/test_opendss_load_shunt.py`: the device shunt against a live OpenDSS
engine — the element admittance equals the DSS `Load`'s own `YPrim` to 4.7e-16 relative
(1-phase WYE, 3-phase WYE, 3-phase DELTA; `%SeriesRL` 0/50/100 and the motor branch),
and harmonic bus voltages agree to 1.6e-12 pu of nominal on IEEE-33 (6.8e-12 with a
370 kvar bank resonating at order 6.9), 1.3e-9 on the Carson-geometry feeder and 4.5e-9
on the three-phase CIGRE LV benchmark.
`tests/differentiability/test_harmonic_load_shunt_gradcheck.py` + `tests/gpu/
test_load_shunt_parity.py`: the shunt's gradient and device/dtype parity.

Per-phase / connection-aware harmonic injection: `tests/asymmetric/test_harmonic_per_phase.py`
(WYE `spectrum_per_phase` A-only, DELTA L-L terminal voltage + `M^T` scatter vs numpy
oracle, DELTA `spectrum_per_phase` branch-k<-phases[k] mapping + missing-branch-injects-0
vs numpy oracle, WYE-N Kirchhoff return into the N row, device-spectrum WYE-ground
regression, scalar-broadcast == device-spectrum and per-element override differs);
`tests/asymmetric/test_harmonic_connection_guard.py` now asserts DELTA / WYE-N SOLVE
(guard lifted); `tests/differentiability/test_harmonic_per_phase_gradcheck.py` float64
gradcheck of `V(h)` w.r.t. a per-element injection magnitude (a python list of leaf
tensors) on WYE AND DELTA, plus the `mag1==0` dead-element guard (finite + zero grad);
`tests/gpu/test_asymmetric_parity.py` CPU-vs-CUDA `solve_harmonic_flow` parity for a
DELTA-spectrum and a WYE `spectrum_per_phase` grid (skips without CUDA).

# =====================================================================
# Solve-performance & structural-check surface (rev 2)
# =====================================================================
Extensions shipped together; every entry is differentiable + GPU-ready unless
stated, and validated by `tests/topology`, `tests/reference/test_sparse_solver.py`,
`tests/reference/test_prepared_system.py`, and the differentiability suite.

- `check_connectivity(grid) -> None` — raises `pgml.errors.ConnectivityError`
  when any (node, phase) row has no galvanic path to an in-service Source
  (open switch / out-of-service branch / no source), naming islands and fixes.
  Runs by DEFAULT at every solve entry (`on_disconnected="raise"`); `"zero"`
  solves `pgml.topology.energized_subgrid` and scatters back 0 V on dead rows
  (full row layout kept); `"ignore"` skips. Report: `pgml.topology.connectivity_report`.
  `pgml.simulate(..., on_disconnected=…)` threads the same choice into BOTH
  calculations (power flow and harmonic) as a call-level EXECUTION kwarg.
- `solve_power_flow(..., linear_solver="auto"|"dense"|"sparse"|"block"|"matrix_free")`
  — for `current_injection` this selects the `Y_eff` factorization backend
  (`harmonic.lu_factor_system(backend=...)`): `"auto"` = scipy SuperLU sparse on
  CPU ≥ ~500 rows (`_SPARSE_MIN_ROWS`), dense batched torch LU otherwise and
  ALWAYS on CUDA. For `newton`: `"dense"` (auto) / `"matrix_free"`; `"sparse"` and
  `"block"` raise. The sparse backend is differentiable via the linear-solve adjoint
  (`_SparseSolveFn`: one trans='H' solve + batch-folded `-λ·conj(V)ᵀ`).
- `solve_power_flow(..., linear_solver="block", block_rows=[rows_0, …])` /
  `prepare_power_flow(..., linear_solver="block", block_rows=…)` /
  `lu_factor_system(..., backend="block", block_rows=…)` — BLOCK-DIAGONAL
  factorization for an ensemble of independent grids
  (`pgml.multigrid.MergedGrid.block_rows()` supplies the partition). `block_rows`
  is one int64 row-index tensor per block, together covering the rows EXACTLY once
  (validated; a non-partition raises). Each block's diagonal sub-matrix is gathered
  straight from `Y` and blocks of EQUAL size are stacked into one batched
  `torch.linalg.lu_factor` (`_BlockLU`, bucket-per-distinct-size — an internal
  detail), so an ensemble costs `O(Σ n³)` / `O(Σ n²)` memory instead of the union's
  `O((Σ N)³)` / `O((Σ N)²)`, and `FactoredSystem.lu` stays `None`. The solve
  gathers each bucket's rows out of the RHS, folds the whole scenario batch into
  its multiple-RHS axis (`_lu_solve_shared`) and scatters back; the only Python
  loop is over buckets. Ideal slack is honored exactly as elsewhere: a block's free
  rows are its rows minus its fixed rows, the free-free `[F,F]` block is never
  materialised, and the `Y_fs` slack coupling stays a dense gather + matvec.
  Differentiable (pure torch — the `lu_solve` backward answers the adjoint with the
  same factors) and GPU-ready; supports `[N,N]`, `[H,N,N]` and per-scenario `Y`
  (the extra factorization batch multiplies the bucket axis). `"auto"` NEVER selects
  it — it is an explicit opt-in because the row partition is taken on trust
  (admittance outside the listed blocks is ignored), and on CPU the sparse union
  backend exploits the same structure and remains the better choice. Unsupported
  with `on_disconnected="zero"` (dropping dead rows re-indexes the system).
  `pgml.simulate(..., linear_solver=…, block_rows=…)` threads it as a call-level
  EXECUTION kwarg (never a `SimulationConfig` field), `calculation="power_flow"` only.
- `solve_power_flow(..., branch_states={branch_id: state})` (also on
  `solve_harmonic_flow`, `assemble_harmonic_system`, `assemble_harmonic_ybus`,
  and the assemblers) — topology / switch-state batching by admittance masking:
  state ∈ [0,1] (float / 0-d / `[*batch]` tensor) scales the branch's primitive
  stamp and OVERRIDES `in_service`/`closed`; batched states solve every switch
  configuration in one call and broadcast against a batched `operating_point`;
  continuous states are IFT-differentiable topology parameters. Per-scenario
  connectivity is pre-checked (vectorized condensed-graph propagation).
- `solve_power_flow(..., branch_states_method="assemble"|"woodbury")` (also on
  `prepare_power_flow`) — HOW a switch-state sweep reaches each state's linear
  system. `"assemble"` (default, unchanged behaviour) assembles + factors every
  state: `O(S·N³)` and an `[S,N,N]` matrix. `"woodbury"` assembles + factors the
  BASE network ONCE and reaches each state through a low-rank update of that
  factorization (`pgml.solver.lowrank`, below): `O(N²k + k³)` per state with
  `k = Σ 2P` over the switched branches. Explicit opt-in only — `"auto"` does not
  exist, because the win depends on `k/N`. Requires `branch_states` and
  `method="current_injection"`; a `system=` must have been prepared with the SAME
  method. FORWARD-only: the IFT backward rebuilds the per-state admittance
  differentiably, so gradients (including w.r.t. the state values) are identical.
  Measured on an i7-12700 (CPU, `run/examples/pgml/benchmark_woodbury.py`, S=8
  states, `auto` backend = SuperLU sparse) for 1-4 switched 3-phase branches
  (k=6…24): 3.2-3.5x at 600 rows, 4.3-6.8x at 1200, ~5.8x at 2100, ~5.4x at 3000;
  1.7-4.4x still at k=96. Crossover ≈ `k ≈ N/3` (600 rows: 1.1x at k=192, 0.2x at
  k=384). Voltages agree with the assemble path to ~1e-12 relative.
- `prepare_power_flow(grid, *, slack, dtype, precision, device, param_overrides,
  branch_states, branch_states_method, linear_solver, block_rows) -> PowerFlowSystem` +
  `solve_power_flow(..., system=...)` — assembly + slack rows + factorization +
  grid-leaf walk once, reused across repeated solves (the `run_scenarios` chunk
  loop shares one system). Forward-only reuse: the IFT backward always rebuilds
  differentiably, so gradients are unchanged. With
  `branch_states_method="woodbury"` the cached `y_eff` is a matrix-free
  `LowRankOperator` and `factorization` a `LowRankUpdate`. The system records its
  `precision`; a consuming solve must request the same one (a mixed-precision system
  caches single-precision factors, which only the residual-correction iteration reads).

- `harmonic.back_substitute(fac, rhs)` / `ideal_slack_rhs(fac, i_inj, v_fixed)` /
  `scatter_slack_solution(fac, v_free, v_fixed)` — the three building blocks a
  factored solve is made of (backend dispatch, the ideal-slack RHS correction, the
  free+slack reassembly). `solve_factored` and the low-rank update-solve below share
  them, so the two can never drift.

## Low-rank update-solve — `pgml.solver.lowrank` (module-level public surface)
Sherman-Morrison-Woodbury solve of `(A + U C Vᴴ) x = b` on top of a
`FactoredSystem`. Used by the switch-state sweep above; designed to be reused by
any "solve a MUTATED grid from the parent's factorization" consumer.

- `low_rank_update(fac, u, c, *, v=None) -> LowRankUpdate` — precompute
  `W = A⁻¹U` (`k` back-substitutions of the base factorization) and the LU of the
  capacitance matrix `I + C VᴴW` (batched over `c`'s leading state dims), reused by
  every later solve. `u`/`v` are complex `[N,k]` in the FULL row space (reduced to
  the free rows internally under ideal slack); `c` is complex `[*states,k,k]`.
- `solve_factored_updated(system, i_inj, *, u=None, c=None, v=None, v_fixed=None)
  -> Tensor` — the updated solve; `system` is a prepared `LowRankUpdate` or a bare
  `FactoredSystem` plus `u`/`c`. Identical contract to `solve_factored`
  (`[*batch,N]` in / out, both slack modes, batched right-hand sides); the update's
  effect on the slack coupling `ΔY_fs = U_f C V_sᴴ` is applied to the RHS, so a
  switched branch incident to a slack node is exact. All three factorization
  backends (dense / sparse / block) work — the base back-substitution is shared with
  `solve_factored`.
- `branch_state_terms(grid, index, branch_states, frequency_hz, *, dtype, device,
  param_overrides, base_states=1.0) -> (u, c)` — builds `U` (a `[N,k]` row selector)
  and `C` (block-diagonal `(s_b − base_b)·block_b`) from
  `assembly.branch_stamp_blocks`. `base_states` is a float or `{branch_id: state}`.
- `LowRankOperator(base, u, c, v)` — the updated matrix as a matrix-free operator
  (`.correction(x)` = `U C Vᴴ x`, `.shape/.dtype/.device`); `_apply_y` applies it so
  residuals and diagnostics never materialise the `[S,N,N]` per-state admittance.
- NUMERICS (why the base matters): the identity is used in the arrangement
  `A⁻¹ − A⁻¹U (I + C VᴴA⁻¹U)⁻¹ C VᴴA⁻¹`, which never inverts `C` — so `s = 0` (an
  OPEN switch) is exact and `C = 0` (a branch at its base state) reproduces the base
  solve bit-for-bit. The update AMPLIFIES the base solution's rounding by
  `‖(I+CZ)⁻¹CZ‖` (`Z = VᴴA⁻¹U`): O(1) when it ADDS admittance, but ~`|y·z_thevenin|`
  when it REMOVES a near-ideal switch (whose voltage drop is lost to cancellation) —
  a 1e-4 Ω switch on an ohm-scale feeder already costs 4 digits. Therefore the sweep
  BASE omits every switched branch it can (`_woodbury_base_states` opens each in turn
  while `check_connectivity` passes; bridges stay in), and `low_rank_update` warns
  when the measured amplification exceeds 1e6.
- Internal fast paths (no API): `assembly.build_injection_plan` /
  `injections_from_plan` resolve the operating point once per solve (the
  V-independent tensors) and make every iteration pure tensor ops; residuals
  apply a batch-shared `Y` as one GEMM (`_apply_y`).
- INVARIANT (do not swap): the IFT backward's `dR/dθ` vjp must use the
  DIFFERENTIABLE residual (`make_residual_complex`, re-resolves from the
  parameter leaves each eval); the iteration / state-Jacobian / diagnostics
  paths use the detached plan residual (`make_fast_residual_complex`). Using
  the detached one for `dR/dθ` silently zeroes parameter gradients; using the
  differentiable one in the loop rebuilds python resolution per iteration.


# =====================================================================
# MIXED PRECISION (complex64 factors, complex128 accuracy)
# =====================================================================
`lu_factor_system(y_bus, *, fixed_rows=None, backend="auto", block_rows=None,
precision="full", refine_steps=None) -> FactoredSystem`

- `precision="mixed"` factors a complex64 copy of the system and keeps the
  full-precision matrix (`FactoredSystem.y_mat`, or the per-bucket blocks of the block
  backend) for residuals. It REQUIRES a complex128 `y_bus` — refining against residuals
  of the same precision buys nothing, so a complex64 working dtype raises instead of
  silently doing nothing (`resolve_precision`).
- `back_substitute` then runs `refine_steps` iterative-refinement corrections
  (`solver.precision.refine_steps`, default 2): `x <- x + A_s^-1 (b - A x)` with the
  residual formed at the working dtype. The step count is FIXED — no data-dependent exit,
  so the routine is branch-free on GPU and never synchronises. The error contracts by
  about `cond(A)·eps_single` per step, so two steps reach the double-precision floor for
  `cond` up to ~1e5 and the nonlinear outer iteration (itself a residual correction)
  continues from there.
- Differentiability: the refined solve is wrapped in `_MixedPrecisionSolveFn`, whose
  backward is the EXACT linear-solve adjoint (`λ = A^-H grad`, `grad_b = λ`,
  `grad_A = -λ x^H`) solved by the same refined solve. Differentiating the refinement
  steps instead would push the gradient through their single-precision rounding, which a
  float64 `gradcheck` measures. The block backend keeps only its diagonal blocks, so a
  mixed-precision block factorization of a matrix that requires grad RAISES.
- All three backends work: dense (`torch.linalg.lu_factor/lu_solve` with `adjoint=`),
  scipy SuperLU (`trans="H"`), block-diagonal (per-bucket `lu_solve`, `_BlockLU.apply`
  for the residual matvec).
- `estimate_condition(fac, *, iters=5) -> float`: 1-norm condition estimate from the
  cached factorization (Hager's power method; a lower bound, `nan` for the block backend
  which holds no single matrix). Used for the one-time complex64 warning; measured
  against `torch.linalg.cond` within a factor of 1.7 on real feeders.
- MEASURED (CPU, i7-12700, 2026-09-11): factoring at complex64 is 1.9x faster dense and
  back-substitution 2.9-6x faster, so end-to-end `mixed` is 1.5-1.9x faster than
  complex128 on the DENSE path (132-3600 rows) at complex128 accuracy. On the CPU
  SuperLU SPARSE path single precision does NOT speed the factorization up (measured
  slower at 3600 rows), so `mixed` is not a win there — it is a dense/CUDA lever.

# =====================================================================
# PRE-SOLVE MODELING GATE
# =====================================================================
`check_branch_impedances(grid) -> None` — raises `pgml.errors.ModelingError` naming every
branch whose SERIES IMPEDANCE IS EXACTLY ZERO (a bus coupler or jumper modelled as a
zero-impedance line, a zero-length line, a closed switch with the schema's zero R/L
default, a zero-impedance generic branch or transformer). Such a branch has no primitive
admittance — the nodal formulation inverts the series impedance — and without the gate it
surfaced as a raw `torch._C._LinAlgError` naming an internal batch index. The message
names the branch ids and the two ways out: the documented near-ideal series resistance
`branch.near_ideal_series_resistance_ohm` (1e-4 Ohm, what the pandapower converter
substitutes for a bus-bus switch), or merging the two nodes. Every solve entry point runs
it (`solve_power_flow`, `prepare_power_flow`, `solve_harmonic_flow`,
`loadability_limit`); a prepared system carries the result. Values are read under
`no_grad` (structural, never on the tape) and a `conductor_geometry` line is skipped (the
geometry path always yields a finite impedance).

# =====================================================================
# IFT backward cost (open work, design note)
# =====================================================================
The backward builds the real state Jacobian `J = dR/dx` `[B, 2N, 2N]` by autograd
(`_batched_state_jacobian`) and solves `J^T λ = grad_x` densely. Measured cost: ~8 forward
solves at batch 12 on IEEE-33, and 200-730 s on a 1176-row grid at batch 16-64 on the CPU
sparse path against an 85 ms forward. The forward already holds a factorization of the
COMPLEX `Y_eff`, and `J` is the real embedding of `Y_eff + dI_device/dV` — the device term
is BLOCK-DIAGONAL per node-phase row (each device current depends only on its own terminal
voltage). So the adjoint can reuse the forward's factorization plus a low-rank / diagonal
correction instead of a dense autograd Jacobian:

1. Form the per-row device derivative analytically (it is the ZIP/control law's
   derivative, already differentiable) as a sparse diagonal-block operator `D`.
2. Apply the adjoint as `J^T λ = grad` with `J = [[Re, -Im], [Im, Re]]` of `Y_eff + D`,
   solved by the existing factorization of `Y_eff` plus a Woodbury/Neumann correction for
   `D`, or by GMRES preconditioned with that factorization (the matrix-free Newton path
   already has the GMRES machinery).
3. Keep the `dR/dθ` vjp exactly as it is — it is one residual evaluation and cheap.

Effort estimate: 3-5 days including gradcheck parity against the dense path on every
device model (ZIP, inverter control, DELTA/WYE-N incidences) and a batched benchmark; the
risk is the control-law derivative, which today comes free from autograd.
