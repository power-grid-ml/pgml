# Interface ledger: equations  (FROZEN rev 1 — orchestrator-pinned)

Residual-form (`0 = a - b`) registry, SymPy-backed, generating an autograd-safe
torch evaluator (and a numpy one) plus LaTeX docs. Normalization is an
EVALUATION-TIME option (None / by-symbol / relative), never part of the stored
equation.

## Scope split (READ THIS FIRST)
The registry holds **scalar / elementwise** physical laws only. Anything requiring
a matrix inverse or a scatter (e.g. the n×n series admittance `Z(f)^-1`, the Y-bus
stamp) lives in `assembly/`, which *composes* these laws and does the tensor
algebra in torch. Do not try to express matrix inversion as a SymPy expression.

## Public API (IMPLEMENTED — final signatures)
Module: `pgml.equations` (`from pgml.equations import Equation, SymbolMeta, registry`).
Importing the package registers the M1 laws (`laws.py`) on the `registry` singleton.

- `Equation` (frozen dataclass) — `registry.py`:
  - `id: str`, `description: str`
  - `residual: sympy.Expr`  (the expression equal to 0, i.e. `a - b`)
  - `symbols: dict[str, SymbolMeta]`  (`SymbolMeta(unit: str = "", schema_field: str | None = None, description: str = "")`)
  - `tags: tuple[str, ...]`
  - property `free_symbol_names -> tuple[str, ...]` (sorted; the evaluator arg order)
- `registry.register(eq: Equation) -> None`  (raises `KeyError` on duplicate id)
- `registry.get(eq_id: str) -> Equation`  (raises `KeyError` if absent)
- `registry.ids() -> tuple[str, ...]`  (sorted; convenience)
- `registry.evaluate(eq_id, values: dict[str, Tensor | float], *, normalize_by: str | None = None) -> Tensor`
  residual on torch tensors; broadcasts; autograd-safe.
  `normalize_by`: `None` | `"<symbol>"` (divide by that symbol) | `"relative"`
  (divide by `max(|a|,|b|,eps)` recovered from the `a-b` Add split).
- `registry.torch_fn(eq_id) -> Callable[..., Tensor]`  (kwargs = symbol names; autograd-safe;
  exposes `.symbol_names`)
- `registry.numpy_fn(eq_id) -> Callable`  (numpy backend; kwargs = symbol names)
- `registry.solve_for(eq_id, target_symbol) -> Callable[..., Tensor]`  (closed form via
  `sympy.solve`, first branch lambdified to autograd-safe torch; kwargs = remaining
  symbols; raises `ValueError` if not invertible)
- `registry.latex(eq_id) -> str`

## Laws registered for the load-flow milestone (M1) — DONE
Residual `a - b`; torch evaluators autograd-safe (`torch.pi`, complex via `sympy.I`),
broadcast over any shape, CPU/CUDA-agnostic:
- `reactance_from_inductance`:  `X - 2*pi*f*L`
- `susceptance_from_capacitance`:  `B - 2*pi*f*C`
- `series_admittance_scalar`:  `y - 1/(R + j*X)`
- `shunt_admittance_scalar`:  `y - (G + j*B)`
- `skin_effect_multiplier`:  `R_eff - R0*m`
- `seq_to_phase_self`:  `Z_self - (Z0 + 2*Z1)/3`
- `seq_to_phase_mutual`:  `Z_mutual - (Z0 - Z1)/3`
(`f` is the absolute frequency in Hz; harmonic order h = f/f0 is a caller concern.
The matrix series admittance `Z(f)^-1` and Y-bus stamps live in `assembly/`.)

## Rules
- Differentiable + GPU per CLAUDE.md. No `.item()/.detach()/.numpy()`, no in-place
  on tracked tensors, no Python control flow on tensor values.
- The torch evaluator must run unchanged on CPU and CUDA and honor input dtype.
- A `gradcheck` (float64/complex128) on each registered law is the self-check.
