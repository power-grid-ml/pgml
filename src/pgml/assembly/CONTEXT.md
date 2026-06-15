# Interface ledger: assembly (Y-bus)

Builds the complex nodal admittance tensor from a (materialised) Grid, per phase,
per harmonic, batched, differentiable, device/dtype-honoring.

Intended public API (FREEZE before implementation; record final form here):
- [ ] `assemble_ybus(grid, frequencies_hz, *, dtype=torch.complex128, device=None,
        batch=None) -> YBus`
      returns complex Y of shape `[*batch, H, N, N]` (H=len(frequencies), N=node*phase
      slots in a fixed (A,B,C,N) layout with a mask), plus the node/phase index map.
- [ ] `node_index_map(grid) -> mapping (node_id, phase) -> matrix row`
Rules:
- Unbatched inputs work without reshaping: a single grid, single frequency -> [N,N].
- Pure scatter-add of per-branch primitive stamps; no Python loop over branches
  (vectorize via index_add / sparse coo). Every entry differentiable w.r.t. R,L,G,C,
  length, tap, source impedance.
- Stamp: series y, shunt y/2 each end; transformer generalized stamp with complex
  tap (magnitude + shift). Frequency scaling X=2*pi*h*f0*L, B=2*pi*h*f0*C.
