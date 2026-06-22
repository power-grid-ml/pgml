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
    `index: NodePhaseIndex`, `iterations: int`, `residual: Tensor`, `converged: bool`,
    `diagnostics: ConvergenceDiagnostics`.
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
    homotopy/continuation. Dense; skipped above `2N=4000`.
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
    `examples/current_injection_convergence.py`). Same `PowerFlowResult` + diagnostics +
    IFT gradients (gradcheck-verified). `linear_solver="dense"` (per-element `[2N,2N]`
    Jacobian + direct solve; no `[B,2N,B,2N]` blowup) or `"matrix_free"` (Jacobian-free
    Newton-Krylov: GMRES on finite-difference `J·v`, `O(N)` memory for large grids).
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
`references/opendss/harmonics.md` (READ IT — esp. the spectrum phase convention,
verified empirically). New orchestration:

- `solve_harmonic_flow(grid, harmonic_orders, *, slack="ideal", operating_point=None,
     harmonic_injection=None, node_sources=None, include_load_shunt=False, tol=1e-10,
     max_iter=100, dtype=torch.complex128, device=None, symmetry=None) -> HarmonicFlowResult`
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

### `node_sources` — per-node harmonic "error" source (full physics: `references/error_injection.md`)
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
  harmonic line impedance differs ~2.5%/h; see `references/opendss/harmonics.md`),
  which is the POSTPONED geometry path. The INJECTION convention IS OpenDSS-exact.

### Validation (DONE)
`tests/reference/test_harmonic_flow.py`: independent numpy oracle (exact, ~1e-9),
fundamental==PF, OpenDSS ballpark (fundamental exact, harmonics within 4% — Carson
gap), shapes/orders. `tests/differentiability/test_harmonic_flow_gradcheck.py`:
gradcheck of `V(h)` w.r.t. line R/L, load P/Q, and injection magnitude (incl.
batched). The live-OpenDSS per-order comparison (with Carson + load shunt) is for
the opendss-reference agent when those models land.

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
