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
