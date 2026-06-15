# Interface ledger: solver

Complex linear solve of the per-harmonic nodal system, batched, differentiable, GPU.

Intended public API (FREEZE before implementation; record final form here):
- [ ] `solve_harmonic(y_bus, i_inj) -> v`
      y_bus: complex `[*batch, H, N, N]`, i_inj: complex `[*batch, H, N]`,
      returns v: complex `[*batch, H, N]`. Implemented via batched
      `torch.linalg.solve`; broadcasts over all leading dims; unbatched [N,N]/[N] works.
- [ ] slack/reference handling documented (fixed source nodes via row replacement
      or Schur reduction) and kept differentiable.
Rules: no `.detach()`/`.item()`; dense first (sparse later if size demands);
double precision available for gradcheck; results identical CPU vs CUDA.
