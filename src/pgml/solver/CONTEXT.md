# Interface ledger: solver  (FROZEN rev 1 — orchestrator-pinned)

Complex linear solve of the per-frequency nodal system `Y(f) V(f) = I(f)`, batched,
differentiable, GPU. Consumes the compact node-phase layout from `assembly/`.

## Public API (IMPLEMENTED — final signature)
Module: `pgml.solver` (`from pgml.solver import solve_harmonic`).
- `solve_harmonic(y_bus, i_inj, *, fixed_rows=None, v_fixed=None) -> v`
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
    anchor weights relative to `Y` (a typical singular value — the pgl consumer's auto-scale).
    Anchor weights are cast to the real dtype paired with `y_bus` (complex64/128 both
    supported). Returns `[*batch, N]`.
  - Gradients flow w.r.t. `y_bus`, `i_inj`, the targets and `op`. Consumers: the pgl
    injection-decode anchoring (`pgl.physics.NetworkSolver`); reusable for a classical WLS
    state-estimation solve.

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
# Phase-2: NONLINEAR fundamental power flow (FROZEN — to implement)
# =====================================================================
`solve_harmonic` stays as the LINEAR per-frequency solve (used by the const-Z path
and, later, by each harmonic). Add the nonlinear fundamental solver:

- `solve_power_flow(grid, *, slack="ideal", method="current_injection",
     tol=1e-8, max_iter=100, dtype=torch.complex128, device=None,
     operating_point=None, param_overrides=None, symmetry=None,
     criticality="auto") -> PowerFlowResult`
  - `symmetry` (Increment 1): `None`/`"auto"`/`"symmetric"`/`"asymmetric"` (`None`
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
    `index: NodePhaseIndex`, `iterations: int`, `residual: Tensor`, `converged: bool`
    (ALL scenarios), `diagnostics: ConvergenceDiagnostics`, `converged_mask: Tensor|None`
    (`[*batch]` bool, `None` unbatched), `failed_states: tuple[int,...]` (flat indices of
    non-converged scenarios). A BATCHED solve NEVER raises on a failed element — every
    element's best-effort `V` is returned, the failures are listed here AND logged as an
    error (count, indices, worst residual, likely cause).
  - CONVERGENCE is PER element on `||ΔV|| < max(tol, floor·||V||)` where `floor` is the
    dtype's resolvable relative precision (`0` for float64 → unchanged `||ΔV|| < tol`;
    `~1e-6` for float32, since `tol` below the rounding floor is unreachable). A `tol`
    below the float32 floor logs a one-time WARNING and the floor governs.
  - `ConvergenceDiagnostics` (autograd-free, computed at `V*`): `converged`, `iterations`,
    `update_norm` (final `||ΔV||`), `power_mismatch_max` (max `|F_c|` over free rows [A]),
    `residual_history`, `worst_nodes`, `out_of_band_nodes` (pu on each node's L-N base),
    `voltage_band_pu`, `likely_cause` (heuristic: converged / diverged / oscillating /
    overload / near-singular), and `criticality`. The cheap state diagnostics are ALWAYS
    populated; `simulate(strict=True)` passes `diagnostics.as_dict()` into the raised
    `ConvergenceError`.
  - `criticality` kwarg (`"auto"`/`"always"`/`"never"`): runs the IFT-Jacobian analysis
    — the SAME real `[2N,2N]` `J = dR/dV` the IFT backward builds, then `svdvals(J)` +
    the right singular vector of the smallest σ for the critical-bus participation.
    `"auto"` = only on non-convergence; `"always"` = also on a converged solve (a
    voltage-collapse MARGIN: σ_min shrinks toward the nose). Rigorous AT a solution; at a
    DIVERGED iterate it is only a local linearization (flagged `evaluated_at`, never
    claims `near_singular`) — a definitive loadability limit needs the (future)
    homotopy/continuation. Dense; skipped above `2N=4000`. SINGLE-GRID only: SKIPPED for a
    batched solve (`b>1`, logged) — re-run one scenario, or use `loadability_limit`.
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
  - `loadability_limit(grid, *, slack, lambda_max, lambda_step, ...) -> LoadabilityResult`:
    CONTINUATION power flow. Ramps the load by `λ` (`R(V,λ)=Y_eff·V+λ·I_dev(V)−I_slack`)
    from a feasible base, Newton-correcting + bisecting onto the breaking `λ*` (the P-V
    nose). At `λ*` the singular Jacobian's RIGHT singular vector = the voltage-collapse
    mode (`critical_nodes`, where it breaks) and the LEFT singular vector projected on each
    load's current = `limiting_loads` (which injection most reduces the margin).
    `breaking_lambda<1` ⇒ the nameplate load is infeasible. Single grid; detached.
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
# Phase-3: HARMONIC power flow (IMPLEMENTED — harmonic_flow.py)
# =====================================================================
Reuses `assemble_network_ybus` (builds Y at any `h·f0`) + `solve_harmonic` (batched
per-frequency linear solve). The OpenDSS conventions are pinned in
`docs/pgml/modeling/references/opendss/harmonics.md` (READ IT — esp. the spectrum phase convention,
verified empirically). New orchestration:

- `solve_harmonic_flow(grid, harmonic_orders, *, slack="ideal", method="current_injection",
     operating_point=None, harmonic_injection=None, node_sources=None,
     include_load_shunt=False, tol=1e-10, max_iter=100, dtype=torch.complex128,
     device=None, symmetry=None) -> HarmonicFlowResult`
  - `method` is forwarded to the fundamental `solve_power_flow`; use `"newton"` for a
    controlled DER (Volt-VAr/Volt-Watt loops oscillate under the current-injection fixed
    point). A controlled Generator/Storage's harmonic injection scales from its
    CONTROL-RESOLVED fundamental current (consistent with the control-aware fundamental
    solve), not the nominal — see `assembly/_control.py` + `docs/pgml/modeling/der-pv-storage.md`.
  - `symmetry` (Increment 1): `None`/`"auto"`/`"symmetric"`/`"asymmetric"`. Resolved
    ONCE here; threaded into the fundamental `solve_power_flow` (which emits the single
    modeling-summary log) and into the harmonic-injection power resolution
    (`resolve_operating_power(..., asymmetric=...)`).
  - Increment 2 (CONNECTION-AWARE / PER-PHASE HARMONIC INJECTION): the inc-1
    DELTA / 4-wire NotImplementedError guard is LIFTED. `_harmonic_injections` now
    mirrors `device_current_injections`: each injecting Load/Generator has a terminal
    incidence `M [n_elem, n_used]` (`pgml.assembly._incidence`; WYE-ground `M=I`,
    WYE-neutral `[I|-1]`, DELTA-3 circulant). The per-ELEMENT fundamental current is
    `I1_elem = sign*conj(S0_elem)/conj(V_term)` with `V_term = M @ V_used` (TERMINAL
    voltage: WYE phase row, WYE-N `V_phase - V_N`, DELTA L-L), and per element/order
    `|I_h^e|=(mag_h^e/mag_1^e)|I1_elem|`, `arg=ang_h^e + h*(arg(I1_elem)-ang_1^e)`.
    The NODAL injection is `-(M^T @ i_h_elem)` scattered into `used_rows` (out-of-place
    complex `index_add`). WYE-to-ground reduces EXACTLY to the pre-inc-2 per-phase form
    (bit-exact: `tests/reference/test_harmonic_flow.py` + `test_carson_harmonics_feeders.py`).
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
     harmonic_injection=None, node_sources=None, symmetry=None,
     dtype=torch.complex128, device=None) -> (Y, I, index)`
  - Exposes the per-harmonic LINEAR system `Y(h) V(h) = I(h)` for orders `h > 1` —
    EXACTLY the `(yh, ih)` `solve_harmonic_flow` builds (`assemble_network_ybus` +
    source Norton stamp for `Y`; `_harmonic_injections` + `node_sources` for `I`), so
    `solve_harmonic(Y, I)` reproduces the harmonic slices. The harmonic network is
    LINEAR, hence `r(V) = einsum('...hij,...hj->...hi', Y, V) − I` is the
    physics-consistency residual (`≈ 0` at the true `V`); the downstream consistency
    package (pgl) forms it without re-deriving the assembly. `solve_harmonic_flow`
    CALLS this (single source of the harmonic assembly — no duplication).
  - `harmonic_orders`: orders `h > 1` only (order 1 is the nonlinear fundamental —
    passing 1 raises `InputError`). `v1`: converged fundamental node voltage
    `[*batch, N]` complex (typically `solve_power_flow(grid, ...).v`), aligned to
    `index`. `operating_point` / `harmonic_injection` / `node_sources` / `symmetry`:
    same meaning/format as `solve_harmonic_flow` (resolve `symmetry` upstream and pass
    the canonical string to reproduce a `solve_harmonic_flow` run exactly).
  - Returns `Y` complex `[Hh, N, N]` (or `[*batch, Hh, N, N]` if a BATCHED voltage
    `node_source` promotes it), `I` complex `[*batch, Hh, N]`, and the
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
   - `Y(h) = assemble_network_ybus(grid, [h·f0]) + source Norton shunt Y_s(h)`
     (+ load Norton shunt `Y_load(h)` from `HarmonicShuntModel` when
     `include_load_shunt=True`; `False` = OpenDSS `NeglectLoadY` pure-source model).
   - `I(h)` = sum of per-device harmonic injections using the verified convention
     `|I_h|=(mag_h/mag_1)|I1|`, `arg(I_h)=ang_h + h·(arg(I1) − ang_1)` from each
     device's `Spectrum` (or the `harmonic_injection` override).
   - `V(h) = solve_harmonic(Y(h), I(h))` in NORTON mode (no ideal slack at
     harmonics: the source is a Norton shunt held at 0 harmonic voltage unless it
     has its own spectrum).
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
- DEFERRED: `include_load_shunt=True` (load Norton shunt at harmonics) raises
  `NotImplementedError` — the OpenDSS shunt split is unpinned. EXACT OpenDSS
  per-order VOLTAGE parity also needs the Carson earth-return line model (the
  harmonic line impedance differs ~2.5%/h; see `docs/pgml/modeling/references/opendss/harmonics.md`),
  which is the POSTPONED geometry path. The INJECTION convention IS OpenDSS-exact.

### Validation (DONE)
`tests/reference/test_harmonic_flow.py`: independent numpy oracle (exact, ~1e-9),
fundamental==PF, OpenDSS ballpark (fundamental exact, harmonics within 4% — Carson
gap), shapes/orders. `tests/differentiability/test_harmonic_flow_gradcheck.py`:
gradcheck of `V(h)` w.r.t. line R/L, load P/Q, and injection magnitude (incl.
batched). The live-OpenDSS per-order comparison (with Carson + load shunt) is for
the reference-integrator agent (OpenDSS) when those models land.

Increment 2 (per-phase / connection-aware): `tests/asymmetric/test_harmonic_per_phase.py`
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
- `prepare_power_flow(grid, *, slack, dtype, device, param_overrides,
  branch_states, linear_solver, block_rows) -> PowerFlowSystem` +
  `solve_power_flow(..., system=...)` — assembly + slack rows + factorization +
  grid-leaf walk once, reused across repeated solves (the `run_scenarios` chunk
  loop shares one system). Forward-only reuse: the IFT backward always rebuilds
  differentiably, so gradients are unchanged.
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
