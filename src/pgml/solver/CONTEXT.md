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
     operating_point=None, param_overrides=None) -> PowerFlowResult`
  - Solves the const-P / ZIP fundamental power flow at f0 = `grid.base_frequency_hz`.
  - `PowerFlowResult` (frozen dataclass): `v` complex `[*batch, N]` (DIFFERENTIABLE),
    `index: NodePhaseIndex`, `iterations: int`, `residual: Tensor`, `converged: bool`.
  - Forward = current-injection FIXED POINT: with `Y_net = assemble_network_ybus`
    (+ folded constant-admittance device part + source Norton if `slack="norton"`),
    iterate `V_{k+1} = solve_harmonic(Y_eff, device_current_injections(grid,V_k)+I_slack,
    slack...)` until `||V_{k+1}-V_k|| < tol` or `max_iter`. Run the iterations under
    `torch.no_grad()`.
  - Backward = IMPLICIT FUNCTION THEOREM at the converged `V*` (do NOT unroll
    iterations): one adjoint linear solve with the transposed power-flow Jacobian.
    Implement as a `torch.autograd.Function` whose backward solves `J^T λ = grad_V`
    and forms parameter grads via a vjp of the residual `F(V*,θ)` (autograd on a
    single residual evaluation at `V*`). Gradients must flow to network params,
    device powers (P/Q), and slack voltage. **Use REAL (re/im split) coordinates**
    for the residual/Jacobian/adjoint (the power flow is non-holomorphic in V).
  - `slack`: `"ideal"` (fix source-node V = `u_ref∠u_angle` via the Schur path in
    `solve_harmonic`; matches pandapower/pgm) or `"norton"` (source folded; OpenDSS).
  - Batched over leading/scenario dims; `method="newton"` is a later add (autograd
    Jacobian via `torch.func.jacrev`), interface unchanged.

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
`references/opendss/harmonics.md` (READ IT — esp. the spectrum phase convention,
verified empirically). New orchestration:

- `solve_harmonic_flow(grid, harmonic_orders, *, slack="ideal", operating_point=None,
     harmonic_injection=None, include_load_shunt=False, tol=1e-10, max_iter=100,
     dtype=torch.complex128, device=None) -> HarmonicFlowResult`
  - `harmonic_orders`: iterable of orders (e.g. `[1,5,7]`; order 1 = fundamental).
  - `HarmonicFlowResult` (frozen dataclass): `v` complex `[*batch, H, N]` (V per
    order; **order 1 = the nonlinear `solve_power_flow` solution**, other orders =
    the linear per-harmonic solve), `frequencies_hz [H]`, `index`, `pf`
    (the fundamental `PowerFlowResult`).
  - DIFFERENTIABLE end to end (network params, load P/Q, AND harmonic injections)
    and BATCHED over scenario dims, same conventions as `solve_power_flow`.

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

### `harmonic_injection` override (scenario-ready, tensor-friendly)
A per-device override of the stored Spectrum, carrying tensors so scenarios can vary
harmonic injections DIFFERENTIABLY (the stored `Spectrum`/`HarmonicComponent` are
plain floats; do NOT hard-bind to them). FINAL format:
`{appliance_id: {order:int -> (magnitude_pu, phase_deg)}}`, where each value is a
python float OR a tensor (0-d, or a leading SCENARIO-batched tensor). Overrides the
stored Spectrum. Analogous to `operating_point` for P/Q.

## Implementation notes (DONE)
- `slack="norton"` matches OpenDSS Vsource (use for OpenDSS parity); `slack="ideal"`
  matches pandapower/pgm at the fundamental. At harmonics the source is ALWAYS a
  Norton shunt held at 0 V (regardless of `slack`).
- Per-device fundamental current `I1 = sign*conj(S0)/conj(Vt)` computed inline in
  `_harmonic_injections` from `pf.v` (no change to `device_current_injections`).
- `harmonic_injection` magnitudes/phases may carry a leading scenario batch dim
  (batched harmonic injection works; full scenario batching is the next phase).
- DEFERRED: `include_load_shunt=True` (load Norton shunt at harmonics) raises
  `NotImplementedError` — the OpenDSS shunt split is unpinned. EXACT OpenDSS
  per-order VOLTAGE parity also needs the Carson earth-return line model (the
  harmonic line impedance differs ~2.5%/h; see `references/opendss/harmonics.md`),
  which is the POSTPONED geometry path. The INJECTION convention IS OpenDSS-exact.

### Validation (DONE)
`tests/reference/test_harmonic_flow.py`: independent numpy oracle (exact, ~1e-9),
fundamental==PF, OpenDSS ballpark (fundamental exact, harmonics within 4% — Carson
gap), shapes/orders. `tests/differentiability/test_harmonic_flow_gradcheck.py`:
gradcheck of `V(h)` w.r.t. line R/L, load P/Q, and injection magnitude (incl.
batched). The live-OpenDSS per-order comparison (with Carson + load shunt) is for
the opendss-reference agent when those models land.
