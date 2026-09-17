# Interface ledger: solver  (FROZEN rev 1)

Complex linear solve of the per-frequency nodal system `Y(f) V(f) = I(f)`, batched,
differentiable, GPU. Consumes the compact node-phase layout from `assembly/`.

## Public API (IMPLEMENTED — final signature)
Module: `pgml.solver` (`from pgml.solver import solve_harmonic`).
- `solve_harmonic(y_bus, i_inj, *, fixed_rows=None, v_fixed=None, precision="full",
  equilibrate=None) -> v`
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
  - `equilibrate`: `None` (the documented default `solver.equilibration.mode`) /
    `"symmetric"` / `"off"` / a bool — the diagonal equilibration applied
    around the factorization (see EQUILIBRATION below). Invisible in the result.
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
     criticality="auto", equilibrate=None) -> PowerFlowResult`
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
  - PER-SCENARIO EXIT AND THE STALL WATCH (`_BatchIterationState`): a scenario batch is a
    batch of INDEPENDENT problems, so each scenario exits on its own criteria and is then
    HELD at the iterate that met them (`torch.where` on the `[*b]` mask, one `all()` per
    iteration - the synchronisation the loop already had). A batched solve therefore
    returns per scenario what the unbatched solve of that scenario returns, instead of
    requiring every scenario to satisfy the criteria in the SAME iteration - which a
    single-precision batch of thousands practically never does, because a float32 fixed
    point does not settle: it reaches an exact fixed point of the ROUNDED map (update
    exactly 0) or a limit cycle whose amplitude is one back-substitution's rounding
    (measured 4e-7 to 6e-4 pu on distribution feeders, up to 50x the calibrated floor, and
    varying from run to run with the reduction order of a multithreaded matvec).
    A scenario that stops making progress - no improvement of `stall_decay` (0.9) in
    EITHER criterion for `stall_patience` (4) consecutive iterations; either, because a
    line-search Newton step can raise the voltage update while the mismatch falls by a
    decade - is then resolved instead of iterated to `max_iter`: inside the floor's BAND
    (`stall_tolerance_factor`, 100x the floor, on both criteria) it counts as CONVERGED AT
    THE PRECISION FLOOR (`converged_mask` true, `diagnostics.floor_governed`, one WARNING
    naming the level and the `precision="mixed"` recipe); above the band it is a FAILURE
    reported by its own `likely_cause` naming both plateaus, and the solve stops as soon as
    every scenario is converged or stalled. Side effect: an infeasible scenario now fails
    in ~8 iterations instead of at the cap.
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
    (heuristic: converged / converged at the precision floor / stalled / diverged /
    oscillating / overload / near-singular), the PRECISION-FLOOR report
    (`floor_governed`, `n_floor_governed`, `n_stalled`, `update_floor_pu`,
    `mismatch_floor_pu`, `stall_update_pu`, `stall_mismatch_pu` - the thresholds that
    actually governed and what the stalled scenarios reached), and `criticality`. The cheap state diagnostics are ALWAYS populated and cost no extra
    solve (the forward's own residual is reused); `simulate(strict=True)` passes
    `diagnostics.as_dict()` into the raised `ConvergenceError`.
  - `criticality` kwarg (`"auto"`/`"always"`/`"never"`): runs the IFT-Jacobian analysis
    — the SAME real `[2N,2N]` `J = dR/dV` the IFT backward builds, then `svdvals(J)` +
    the right singular vector of the smallest σ for the critical-bus participation.
    `"auto"` = only on non-convergence; `"always"` = also on a converged solve (a
    voltage-collapse MARGIN: σ_min shrinks toward the nose). Rigorous AT a solution; at a
    DIVERGED iterate it is only a local linearization (flagged `evaluated_at`, never
    claims `near_singular`) — a definitive loadability limit needs the (future)
    homotopy/continuation. Dense; skipped above `2N=4000`. On a scenario BATCH it analyses
    the HARDEST scenario (largest nodal mismatch) and names it as `criticality["batch"]`:
    that scenario's own admittance, slack current, injection powers and setpoints are
    index-selected, which is what makes the Jacobian single-grid at all (the residual
    otherwise keeps the whole batch's powers). Its figures equal the ones that scenario
    produces solved alone (pinned by a test). The build goes through the same budgeted path
    as the gradient, so the diagnostic of a failed solve is not its most memory-hungry part.
    A residual that still carries a batch, or any non-square Jacobian shape, is reported as
    a `{"skipped": ...}` dict (a diagnostic never raises and never analyses a matrix that is
    not the one it claims).
  - `method="current_injection"` (default) forward = FIXED POINT: with
    `Y_net = assemble_network_ybus` (+ source Norton if `slack="norton"`), iterate
    `V_{k+1} = solve_harmonic(Y_eff, I_slack − device_current_injections(grid,V_k),
    slack...)` until both per-unit criteria hold or `max_iter`, under `torch.no_grad()`.
    The per-unit MISMATCH criterion needs the nodal residual `F(V) = Y V + I_dev − I_slack`
    every iteration, and forming it with a matrix-vector product cost 0.2 to 0.8 times a
    dense back-substitution and up to 14 times a SPARSE one (measured, 1176 rows,
    complex128, CPU) — i.e. it was the dominant per-iteration cost on the sparse path.
    It is not needed: the back-substitution has just enforced `(Y V_new)_free = I_free`,
    so on every free row (the only rows the criteria measure)
    `F(V_new) = I_dev(V_new) − I_dev(V_old)` EXACTLY, which the iteration already holds.
    The identity is used for the per-iteration test and CONFIRMED against the true nodal
    residual in the iteration that would exit (and once more if the loop hits `max_iter`),
    so the reported mismatch and the convergence decision are still the nodal ones: one
    matrix-vector product per SOLVE instead of one per iteration. `precision="mixed"`
    keeps the explicit residual — there it IS the next right-hand side.
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
    The dense Newton direction factors the Jacobian EQUILIBRATED (the real state Jacobian
    of an SI-unit system mixes admittance rows with voltage rows, which matters most for
    the single-precision direction of `precision="mixed"`).
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
      `q_var {gen_id: [*batch] var}` (solved total, DIFFERENTIABLE whenever the solve
      tracks gradients: the nodal residual is evaluated on-tape at the IFT-attached
      voltages, one extra differentiable assembly), `regulating {gen_id: [*batch]
      bool}`, `switch_rounds`, `enforce_q_limits`, `settled [*batch] bool` and
      `unsettled_generators` (ids). When the round cap ends the switching, the kept
      active set contradicts its own limit check, so the scenarios concerned are
      reported as NOT converged (`converged`, `converged_mask`, `failed_states`,
      `diagnostics.likely_cause`) and the WARNING names the generators. The convergence diagnostics report the ACTIVE component of
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
    tol_update_pu, s_base_va, max_iter, top_k, ramp=None, equilibrate=None)
    -> LoadabilityResult`:
    λ-RAMP loadability. Scales the injections by `λ`
    (`R(V,λ)=Y_eff·V+λ·I_dev(V)−I_slack`) from a feasible base, Newton-correcting at each
    step and bisecting onto the first λ the corrector cannot solve. `breaking_lambda` is
    therefore the largest λ at which the Newton corrector CONVERGES, a LOWER BOUND on the
    P-V nose (a plain corrector fails before the singularity; measured ~4 % below the
    closed-form nose of a two-bus feeder) — a step-and-bisect on feasibility, NOT an
    arc-length predictor-corrector, and the Jacobian figures describe the last converged
    point. `ramp="load"` (the default, `solver.loadability.ramp`) scales loads only and
    holds generation at nameplate — the textbook continuation ramp; `ramp="all"` scales
    every injecting device, loads AND generators/storage together (on a two-bus feeder with
    a load at `0.8 P*` and generation at `0.3 P*` the two give 1.625 vs 2.0, both
    closed-form). `LoadabilityResult.ramp` records which was measured.
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
     load_shunt=None, load_shunt_basis=None, tol=None, tol_update_pu=None, s_base_va=None,
     max_iter=100, dtype=torch.complex128, precision="full",
     device=None, symmetry=None, on_disconnected="raise", branch_states=None,
     branch_states_method="assemble", param_overrides=None, enforce_q_limits=None,
     linear_solver="auto", block_rows=None, criticality="auto", equilibrate=None)
     -> HarmonicFlowResult`
  - `tol` / `tol_update_pu` / `s_base_va` are the PER-UNIT convergence settings of the
    nonlinear fundamental (see `solve_power_flow`); the harmonic orders are direct linear
    solves with no iteration and therefore no convergence criterion of their own.
  - `enforce_q_limits` likewise reaches the fundamental only: regulation is a
    fundamental-frequency concept, so at h>1 a regulating generator is the same Norton
    current source as any other.
  - `precision` applies to the fundamental AND to every per-order solve (where
    `"mixed"` is the classic iterative refinement of `lu_factor_system`).
  - `linear_solver` / `block_rows` / `equilibrate` likewise select the backend and the
    equilibration of the fundamental factorization AND of every order's own
    (`"matrix_free"` is a Newton option of the fundamental solve and leaves the orders on
    the automatic backend). `criticality` and `branch_states_method` are forwarded to the
    fundamental solve. For a flat scenario batch whose operating-point load shunt touches
    fewer than one third of the rows under a conservative automatic heuristic, the harmonic
    orders factor the shunt-free network once per order and apply their exact
    connection-aware shunt blocks through Woodbury. A
    dimensionless backward-error guard falls back to the assembled per-state factorization;
    high-rank, topology-batched, and deeper-injection cases use that direct path immediately.
  - `on_disconnected` is executed ONCE here for the whole study (the inner fundamental
    solve is told to skip the repeat through an explicit resolved policy, not a hidden
    `"ignore"`); with `branch_states` the per-scenario check runs inside that solve.
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
    (the fundamental `PowerFlowResult`), `harmonic_finite` bool `[*batch]` (every
    solved voltage of the scenario is finite; computed without a host sync). The
    harmonics are direct solves and a batched LU does not raise on a singular `Y(h)`, so
    `converged` / `converged_mask` / `failed_states` combine `pf`'s verdict with
    `harmonic_finite` (a deeper injection batch is reduced onto the fundamental's batch
    shape), and a non-finite scenario is logged at ERROR with its index (one sync at the
    end of the solve).
  - DIFFERENTIABLE end to end (network params, load P/Q, AND harmonic injections)
    and BATCHED over scenario dims, same conventions as `solve_power_flow`.
  - A per-scenario `operating_point` (`[B]`) combined with a DEEPER-batched
    `harmonic_injection` (a node-coherent `[B, T]` sequence over a `[B]` fundamental) is
    supported: v1 stays `[B, N]` (in step with the same-batch op that forms each device's
    fundamental current), and the per-device fundamental current is broadcast across the
    injection's extra step axis; only the order-1 slice returned to the caller has its batch
    rank lifted to the injection's so it stacks against the `[B, T, N]` harmonic slices. A
    no-op for the snapshot (`[B]`/`[B]`) and nominal (empty-op) cases — byte-identical.

- `harmonic_injections(grid, v1, harmonic_orders, *, operating_point=None,
     harmonic_injection=None, node_sources=None, symmetry=None, dtype=torch.complex128,
     device=None, param_overrides=None, index=None) -> Tensor` complex `[*batch, Hh, N]`
  — the RHS half of `assemble_harmonic_system` without assembling `Y(h)`: every injecting
  device's connection-aware harmonic current from its converged fundamental terminal
  current and its spectrum, plus the Norton current of every CURRENT-kind
  `NodeHarmonicSource` (a VOLTAGE-kind one is refused — it adds a shunt admittance as
  well, so it belongs to the system assembly). `index` defaults to the grid's full layout
  (the layout `v1` is in); pass a reduced one to get the injection summed onto fused rows.
  The nodal-injection input a fused branch's current recovery needs
  (`pgml.assembly.branch_currents(..., i_inj=…)`).
- `assemble_harmonic_system(grid, harmonic_orders, v1, *, operating_point=None,
     harmonic_injection=None, node_sources=None, load_shunt=None, load_shunt_basis=None,
     symmetry=None,
     dtype=torch.complex128, device=None, branch_states=None, param_overrides=None,
     fusion=None) -> (Y, I, index)`
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
     `Y(h)` `[*batch, Hh, N, N]`, which `load_shunt_basis` governs (below). A GENERATION device carries no shunt under the shipped
     `appliance.harmonic_shunt.generation_model = none` (the expression's conductance is
     negative for an injecting device), with one WARNING per solve naming how many were
     left as pure current sources.
   - `load_shunt_basis` decides WHICH power and terminal voltage that `Y_eq` is built
     from, and with it whether `Y(h)` is shared across a scenario batch:
     `"operating_point"` (the shipped default `appliance.harmonic_shunt.basis`) uses this
     scenario's power at the solved fundamental terminal voltage — what OpenDSS's own
     `YPrim` does with its Load's specified kW/kvar — so `Y(h)` is `[B, Hh, N, N]` and a
     batch costs `B·Hh` factorizations; `"nameplate"` uses the device's stored P, Q at its
     rated terminal voltage, so `Y(h)` stays `[Hh, N, N]` and one factorization per order
     serves the whole batch. Measured on a 294-row Kerber feeder at 13 orders, CPU,
     complex128, sparse backend, batch 256: 66-73 against 1513-2008 studies/s (21-31x),
     with the no-shunt bound at 1830-1992; on three-phase CIGRE LV (132 rows, dense)
     282-322 against 4038-4076 studies/s (13-14x), bound 3606-4336. The price is a model error wherever a scenario's loading
     differs from nameplate — the shunt is then the nameplate load's, measured against a
     live OpenDSS carrying the scenario's own kW (IEEE-33, orders 3…25, pu of nominal):
     2.1e-4 at 0.5x and 6.7e-4 at 1.5x loading, and 4.1e-3 / 1.0e-2 with a 370 kvar bank
     that puts a parallel resonance at order 6.9, against 6e-13 … 3e-11 for the
     operating-point basis. For a constant-power device the two bases are IDENTICAL at
     nameplate loading (`Y_eq` reads the RATED voltage on both).
   - The `"operating_point"` basis assembles and factors the batch in SCENARIO CHUNKS that
     fit `solver.harmonic.system_budget_mb` (default 1 GiB, charged the matrix plus its
     factorization): `_solve_harmonic_orders` / `_harmonic_chunk`. Without it, 1024
     scenarios of a 294-row grid at 13 orders ask for an 18 GB matrix. Inside a chunk
     there is no Python loop over orders or scenarios — one `lu_factor_system` call
     factors every `(scenario, order)` system of the chunk, so the batched dense/CUDA path
     is one call and the sparse path is one SuperLU per system — and a batched `Y(h)` now
     honours `linear_solver` (it went through the dense direct solve before). Gradients
     flow through the concatenation (`tests/differentiability/
     test_harmonic_load_shunt_gradcheck.py` pins chunked == whole-batch gradients to
     `rtol=1e-12`). A deeper-than-flat scenario batch or a batched `node_source` keeps the
     whole-batch path.
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
on the three-phase CIGRE LV benchmark, and 6.2e-11 on a feeder whose ideal switch FUSES
two shunted, injecting loads onto one reduced row (the residual is the reference's own
closed-`Switch` impedance: 4.0e-13 with that 1e-06 Ohm switch stamped on both sides).
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
  branch_states, branch_states_method, linear_solver, block_rows, equilibrate)
  -> PowerFlowSystem` (which also carries the STRUCTURAL quantities a reusing solve would
  otherwise rebuild per call: `row_abs_scale`, the `Σ_j |Y_ij| V_base,j` the mismatch
  criterion's per-row floor is built from — a property of the network and the rated
  voltages, so the prepared path pays no full `|Y|` pass (7.6 ms on a 1176-row feeder at
  complex128, CPU) per solve; `full_index`, the grid's own N-row layout the result is
  reported on; and `v_base`, the line-to-neutral per-unit base of every row) +
  `solve_power_flow(..., system=...)` — assembly + slack rows + factorization +
  grid-leaf walk once, reused across repeated solves (the `run_scenarios` chunk
  loop shares one system). Forward-only reuse: the IFT backward always rebuilds
  differentiably, so gradients are unchanged. With
  `branch_states_method="woodbury"` the cached `y_eff` is a matrix-free
  `LowRankOperator` and `factorization` a `LowRankUpdate`. The system records its
  `precision` and its `equilibration`; a consuming solve must request the same ones (a
  mixed-precision system caches single-precision factors, which only the
  residual-correction iteration reads; an equilibrated system caches the SCALED matrix plus
  the scales that undo it).

- `harmonic.back_substitute(fac, rhs)` / `ideal_slack_rhs(fac, i_inj, v_fixed)` /
  `scatter_slack_solution(fac, v_free, v_fixed)` — the three building blocks a
  factored solve is made of (backend dispatch, the ideal-slack RHS correction, the
  free+slack reassembly). `solve_factored` and the low-rank update-solve below share
  them, so the two can never drift. Factor and RHS batch shapes follow PyTorch's
  right-aligned broadcasting. Extra RHS axes and axes where the factor batch is
  singleton share the same factorization and are folded into its multiple-RHS columns;
  for example `Y=[B,1,H,N,N]` and `I=[B,T,H,N]` use one factor per `(B,H)` across all
  `T` steps without tiling the LU. Dense, sparse, block, mixed-precision, and adjoint
  paths use the same axis split.

## Low-rank update-solve — `pgml.solver.lowrank` (module-level public surface)
Sherman-Morrison-Woodbury solve of `(A + U C Vᴴ) x = b` on top of a
`FactoredSystem`. Used by the switch-state sweep above; designed to be reused by
any "solve a MUTATED grid from the parent's factorization" consumer.

`harmonic_flow._harmonic_shunt_lowrank_terms` also uses this surface internally for
scenario-dependent operating-point shunts. Its selector `U` contains the unique rows touched
by the modeled devices and its compact `C` is assembled from the same WYE/DELTA/neutral
blocks as the exact matrix. Explicit Generator/Storage harmonic impedances belong to the
scenario-independent base. The automatic path uses the conservative selection rule
`3k < N`; the actual performance crossover depends on the factorization backend and hardware.

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
# EQUILIBRATION — `pgml.solver.equilibration` (module-level public surface)
# =====================================================================
The power-flow and harmonic factorization paths use the EQUILIBRATED matrix
`Â = D_r A D_c` by default; the right-hand side is scaled by `D_r` and the solution by
`D_c`, so nothing outside the factorization sees it. Disable it per call with
`equilibrate="off"`. `solve_anchored` and `AnchoredSystem` do not yet apply this setting.

- `EQUILIBRATION_MODES = ("off", "symmetric")`
- `resolve_equilibration(equilibrate) -> str` — `None` -> the documented default, `True` /
  `False` -> that default / `"off"`, a name validated against the modes.
- `equilibration_scales(a, *, mode, power_of_two=None) -> (d_row, d_col)` — `[*batch, m]`
  real scales in the real dtype paired with `a`'s (so a complex64 matrix stays complex64);
  `(None, None)` for `"off"`. `"symmetric"` returns one tensor twice.
- `equilibrate_matrix(a, *, mode, power_of_two=None) -> (a_hat, d_row, d_col)`,
  `scale_matrix(a, d_row, d_col) -> a_hat`. A two-sided scaling writes ONE full matrix,
  not two: with no gradient being recorded — every forward solve, which is where the cost
  shows — the column scaling runs in place on the tensor the row scaling just produced
  (measured on a 1176-row feeder, complex128, CPU: 6.7 ms against 13.9 ms, where that
  matrix's SuperLU factorization is 7.0 ms and its dense LU 29.9 ms). With autograd active
  both multiplications stay out-of-place and on the tape.
- `equilibrated_lu_factor(a, *, mode, power_of_two=None, factor_dtype=None)
  -> EquilibratedLU` with `.solve(rhs, *, adjoint=False) -> x` — factor once, solve many
  for a REAL dense system that is not a network admittance: the Newton state Jacobian and
  the implicit-function adjoint. The adjoint form swaps the two scales
  (`x = D_r Â^-H D_c b`).

Mode `"symmetric"` is van der Sluis `d_i = |A_ii|^-1/2` applied as the congruence
`D A D` (keeps symmetry and sparsity, reads only the diagonal, within `sqrt(n)` of the best
condition number any diagonal scaling reaches).

Scale factors are rounded to POWERS OF TWO (`solver.equilibration.power_of_two`), so the
scaled matrix is exact in binary floating point: the equilibration adds no rounding error
of its own, and `round`'s zero derivative keeps the scale off the gradient while the
scaling multiplications stay on the tape — gradients w.r.t. the matrix and the right-hand
side are unchanged (measured identical to the last printed digit; float64 gradcheck passes
on the dense and sparse backends).

`FactoredSystem` carries `equilibration`, `scale_row`, `scale_col`, and its `y_mat` is the
matrix AS FACTORED (scaled) — which is what the mixed-precision residual and
`estimate_condition` must use. The Woodbury path needs no change: it reads `A^-1 U`
through `back_substitute`, which answers the SI system.

MEASURED (CPU, i7-12700, complex128, 1-norm estimate / exact 2-norm, 2026-09-11):

| system | rows | off | symmetric |
|---|---|---|---|
| IEEE-33 fundamental `Y_ff` | 32 | 2.8e3 / 1.7e3 | 1.4e3 / 7.7e2 |
| IEEE-33 `Y(13)` | 33 | 8.0e8 / 5.7e8 | 1.5e3 / 7.7e2 |
| CIGRE LV 3-phase fundamental | 129 | 5.5e4 / 4.6e4 | 2.1e3 / 1.0e3 |
| `mv_oberrhein` `Y(13)` | 179 | 7.2e9 / 6.2e9 | 1.7e5 / 7.2e4 |
| Kerber `Y(13)` | 294 | 5.0e7 / 4.2e7 | 3.4e4 / 1.8e4 |
| ladder rung fundamental | 2016 | 1.4e5 / 3.1e4 | 1.3e4 / 3.9e3 |
| ladder rung `Y(13)` | 10044 | 7.5e6 / — | 4.0e5 / — |

The exception is a LOW-VOLTAGE-only harmonic system (CIGRE LV at order 13: 1.9e7 ->
2.4e7), where the conditioning is not a scaling artefact and the symmetric scaling is
neutral to slightly worse.

What it buys is ROBUSTNESS, not forward accuracy: LU with partial pivoting is
backward-stable, so `cond·eps` is a pessimistic forward bound and the complex64 error is
unchanged to within a factor 2 on small grids. On a 2016-row system it is the difference
between converging and not — plain complex64 runs to the 100-iteration cap unscaled and
converges in 9 iterations equilibrated (max |dV| 4.1e-5 -> 1.8e-5 pu).

# =====================================================================
# MIXED PRECISION (complex64 factors, complex128 accuracy)
# =====================================================================
`lu_factor_system(y_bus, *, fixed_rows=None, backend="auto", block_rows=None,
precision="full", refine_steps=None, equilibrate=None) -> FactoredSystem`

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
- `estimate_condition(fac, *, iters=5, per_matrix=False) -> float | Tensor`: 1-norm
  condition estimate from the cached factorization (Hager's power method; a lower bound,
  `nan` for the block backend which holds no single matrix, `inf` for a factorization
  that back-substitutes to non-finite values). A batched factorization is estimated per
  matrix in one batched pass; the default returns the worst case as a float (one host
  sync), `per_matrix=True` the `[*fb]` tensor without a sync. Used for the one-time complex64 warning; measured
  against `torch.linalg.cond` within a factor of 1.7 on real feeders. It describes the
  matrix AS FACTORED, i.e. the EQUILIBRATED one unless `equilibrate="off"` — that is the
  conditioning the factorization actually sees, and it is what the precision decision
  needs; factor with `equilibrate="off"` for the condition number of the matrix as
  assembled.
- MEASURED (CPU, i7-12700, 2026-09-11): factoring at complex64 is 1.9x faster dense and
  back-substitution 2.9-6x faster, so end-to-end `mixed` is 1.5-1.9x faster than
  complex128 on the DENSE path (132-3600 rows) at complex128 accuracy. On the CPU
  SuperLU SPARSE path single precision does NOT speed the factorization up (measured
  slower at 3600 rows), so `mixed` is not a win there — it is a dense/CUDA lever.

# =====================================================================
# ZERO-IMPEDANCE BRANCHES: EXACT BUS FUSION + THE PRE-SOLVE GATE
# =====================================================================
A branch whose series impedance is EXACTLY zero (a closed switch with the schema's zero
R/L default, a bus coupler or jumper modelled as a zero-impedance line, a zero-length
line, a zero-impedance generic branch) has no primitive admittance — the nodal formulation
inverts the series impedance. It IS representable: it is an ideal conductor, and every
solve entry point now collapses its terminal node-phase rows into ONE row of the solved
system (`pgml.assembly.fusion_map`, `assembly/CONTEXT.md` rev 3) instead of refusing it.

- The map is resolved ONCE per solve from the documented policy `branch.zero_impedance`
  (`fuse` default / `error`) and threaded into the assembly, the slack rows, the injection
  plan, the warm start, the Woodbury base, the per-order harmonic systems and the
  diagnostics. The row layout of the SOLVE is the reduced one; the row layout of the
  RESULT is the grid's own — `PowerFlowResult.v` / `HarmonicFlowResult.v` are prolonged
  back (a gather, so the IFT gradient reaches the reduced state through its scatter-add
  adjoint) and `.index` is the full `NodePhaseIndex` as before. Both results carry the
  map in a new `fusion` field (`None` when nothing fused); `PowerFlowSystem` caches it and
  a consuming solve must match it.
- `layout_fingerprint` / persisted tensors are UNAFFECTED: fusion never changes the full
  row layout a result or a dataset is written in.
- Two terminals the solve has to pin separately may not share a fused row: two in-service
  Sources are deduplicated when their references AGREE (logged) and refused when they do
  not, and two voltage-REGULATING generators on one fused row are refused by name.
- A fused group's per-row diagnostics (`worst_nodes`, `out_of_band_nodes`) report ONE row
  per group, under the group's representative node id — the group is one electrical node.
- `block_rows` (a merged ensemble's partition) is mapped through the fusion, which keeps
  it a partition: a fused group never spans two member grids.
- The HARMONIC DEVICE SHUNT composes with fusion through the same reduced index: each
  element admittance is scattered with `scatter_blocks_into`, whose `index_add_`
  accumulates duplicate targets, so two devices whose nodes are fused add their shunts
  into ONE reduced row — which is what `Pᵀ Y P` says. `assemble_harmonic_system` and
  `assemble_harmonic_ybus` therefore take the caller's `v1` in the grid's FULL layout and
  read the representative row of each fused group (`FusionMap.sample`, exact because the
  group shares one voltage) before the shunt is built. Validated against a live OpenDSS
  solve of the same circuit with a closed `Switch` element
  (`tests/reference/test_opendss_load_shunt.py`, 6.2e-11 pu of nominal, the reference
  switch's own near-ideal drop, against a 4.0e-13 pu floor when pgml keeps that switch
  stamped).
- The CURRENT through a fused branch at a harmonic order needs the device shunt too: the
  defect `i_inj − Y_network_without_fused · V` is formed from the passive network, so the
  nodal injection handed to `pgml.assembly.branch_currents` must be the DEVICE-side
  current `harmonic_injections(...) − Y_shunt(h)·V(h)` (the shunt sits inside `Y(h)`).
  `_harmonic_shunt_currents(grid, v1, vh, orders, *, operating_point, load_shunt,
  symmetry, dtype, device, param_overrides, index) -> [*batch, Hh, N]` returns that
  shunt term, and `pgml.simulation.SolvedState.branch_currents()` subtracts it.

`check_branch_impedances(grid, *, fusion=None, param_overrides=None, zero_branches=None)
-> None` — the
pre-solve gate, which answers the STRUCTURAL question "is every in-service branch
stampable?" when called with no map (unchanged behaviour, and the message now names exact
bus fusion as a third way out next to the documented near-ideal series resistance
`branch.near_ideal_series_resistance_ohm` and merging the two nodes). Every solve entry
point passes the map it resolved, so what remains is what neither a stamp nor a fused row
can express: a zero-impedance TRANSFORMER (its ratio and vector group relate the terminals
by more than equality), a zero-series branch that still carries a shunt admittance
(fusing would drop that shunt), and a zero-impedance branch listed in `branch_states` (a
swept branch is reached by scaling a stamped admittance; the two mechanisms are mutually
exclusive by construction). `param_overrides` makes the check read the EFFECTIVE impedance
the stamps will use, and `zero_branches` passes in the
`pgml.assembly.zero_impedance_branches` list the caller already holds, so a solve walks
the branch list ONCE for the map and the gate together instead of once per consumer.
Values are read under `no_grad` (structural, never on the tape) and a
`conductor_geometry` line is skipped (the geometry path always yields a finite impedance).

# =====================================================================
# IFT backward: the Jacobian build, the adjoint solve, and what is still open
# =====================================================================
The backward is a VECTOR-Jacobian product, so it needs one adjoint solution and never the
Jacobian of the solve. It builds the real block-diagonal state Jacobian `J = dR/dx`
`[B, 2N, 2N]` (`_batched_state_jacobian`), factors it ONCE equilibrated, solves
`J^T λ = grad_x`, and forms `grad_θ = -(dR/dθ)^T λ` with one residual vjp.

- BUILD, chosen by `solver.ift.jacobian_budget_mb` (default 1 GiB) against the measured
  peak `chunk² · 2N³ · itemsize` (`_vectorized_jacobian_peak_bytes`): the whole batch in
  one vectorized call, a CHUNK of the batch per call, or column-by-column with `2N` batched
  JVPs. The quadratic term is the allocator's own request to the byte (a 294-row grid at
  chunk 4 asks for 13 011 038 208 = 16·2·294³·16; a 1176-row grid at chunk 4 for
  832 706 445 312), because what vmap replicates per output row is the BROADCAST admittance
  a batch shares. A chunk of ONE has no broadcast and is always affordable, so the budget
  never forces the slow column build on the gradient path; a zero budget selects it
  explicitly. Chunking needs the residual of a batch SLICE, which is why
  `assembly.ybus.select_plan_batch` and `PVTerminals.select` exist.
- ADJOINT, cached: the factorization is kept on the autograd node while it stays under
  `solver.ift.adjoint_factor_cache_mb` (default 256 MiB), so every FURTHER product of the
  same solve — a full output Jacobian row by row, or any second backward under
  `retain_graph` — is one back-substitution instead of another build.
- MEASURED (CPU, i7-12700, complex128, one process per point, 2026-09-11), backward time
  and the process' resident high-water mark, before -> after the budget:
  IEEE-33 at batch 64, 1775 ms / 4856 MiB -> 773 ms / 1018 MiB; Kerber (294 rows) at batch
  4 and 16, out of memory (13.0 GB and 208 GB requests) -> 146 ms and 453 ms; a 1176-row
  Kerber ensemble at batch 4, out of memory (832.7 GB) -> 2.26 s. The repeated product is
  6 to 120x cheaper than the first one wherever the factors fit the cache (at 1176 rows and
  batch 16 they do not — 708 MiB — and it is rebuilt, by design).

Still open: the adjoint of `Y_eff + dI_device/dV` could reuse the FORWARD's factorization
of the complex `Y_eff` instead of building `J` by autograd at all (the device term is
BLOCK-DIAGONAL per node-phase row, since each device current depends only on its own
terminal voltage):

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

## Modeling contexts and convergence-scale reuse

`PowerFlowSystem.modeling_defaults` stores the preparation defaults; a consuming solve
rejects different defaults instead of reusing a factorization from another physical
model. The IFT autograd node deep-copies the complete resolved defaults mapping and
restores that stable snapshot for backward reassembly. The gradient therefore retains the
forward model after a preset exits and after `PGML_DEFAULTS` / `defaults.reload` changes
the process-wide source.

The precision-floor row scale is shared across Newton retries, batch members and PV/PQ
rounds. A conservative Cauchy-Schwarz bound skips its exact computation when the floor
cannot govern. CPU non-gradient singleton matrices read magnitudes from CSR nonzeros;
batched, gradient-carrying and accelerator matrices retain the dense expression.
