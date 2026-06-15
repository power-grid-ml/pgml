# Interface ledger: equations

Residual-form (`0 = a - b`) registry, SymPy-backed, generating numpy AND torch
(differentiable) evaluators and LaTeX docs. Normalization is an EVALUATION-TIME
option (None / by-symbol / relative), never part of the stored equation.

Public API (record signatures here as implemented):
- [ ] `Equation` (id, description, residual_expr, symbol->schema-field map, tags)
- [ ] `registry.evaluate(eq_id, values, normalize_by=None) -> tensor`
- [ ] `registry.torch_fn(eq_id) -> Callable[..., Tensor]`  (autograd-safe)
- [ ] `registry.solve_for(eq_id, target_symbol)`  (closed form when invertible)

Load-flow-relevant equations to register first: pi-form series/shunt admittance
from R,L,G,C at frequency h; Y-bus diagonal/off-diagonal stamp; per-harmonic
reactance/susceptance scaling.
