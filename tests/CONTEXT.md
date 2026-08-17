# Testing conventions (read before writing or running tests)

Layout:
- `tests/reference/`        cross-library oracle comparisons (Y-bus + results).
- `tests/differentiability/` gradcheck and finite-difference vs autograd.
- `tests/gpu/`              device/dtype parity (skipped if no CUDA).

Differentiability gate (MANDATORY for assembly + solver):
- Use float64/complex128 and `torch.autograd.gradcheck` on small systems.
- Check grads of node voltages w.r.t. line R/L/C, transformer R/L/tap, source Z.
- A finite-difference spot check backs up gradcheck on at least one parameter.

GPU gate:
- Build a small grid, assemble + solve on CPU, then on CUDA (if available); assert
  results match within tol and dtype/device are honored. Mark cuda tests to skip
  cleanly when unavailable.

Oracle comparisons (load-flow milestone):
- Y-bus vs OpenDSS `Export Y` / `SystemY` (align by YNodeOrder; reconcile SI vs
  the reference's units/pu; tol e.g. rtol=1e-6 on matched entries).
- Node voltages / branch flows vs pandapower `res_bus` / `res_line` (convert pp
  engineering units to SI; align by id_map; tol e.g. atol on pu voltage 1e-4).
- power-grid-model as a second results oracle (asym power flow).
Always align per-phase ordering between our (A,B,C,N) layout and the reference.
