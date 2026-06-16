"""Residual-form equation registry (SymPy-backed, autograd-safe torch evaluators).

The registry holds **scalar / elementwise** physical laws in residual form
``0 = a - b`` (``residual = a - b``). Each :class:`Equation` carries a SymPy
expression; from it we lambdify an autograd-safe torch evaluator and a numpy
evaluator, and we can emit LaTeX.

Anything that needs a matrix inverse or a scatter (the n x n series admittance
``Z(f)^-1``, the Y-bus stamp) is **not** a registry concern; that lives in
``pgml.assembly`` which composes these scalar laws and does the tensor algebra in
torch. Never try to express a matrix inverse as a SymPy expression.

Differentiability + GPU (see CLAUDE.md): the torch evaluators use ``torch.*`` ops
only, run unchanged on CPU and CUDA, honor the input tensor dtype/device, and
contain no ``.item()/.detach()/.numpy()``, no in-place ops on tracked tensors, and
no Python control flow on tensor values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import sympy
import torch
from torch import Tensor

Number = Union[Tensor, float, complex]

# Modules used by sympy.lambdify for the torch backend. ``1j`` in an expression
# becomes ``I`` (sympy.I); lambdify with the torch module maps standard functions
# to torch and the imaginary unit to a python complex, which torch promotes
# correctly when combined with a (complex) tensor.
_TORCH_MODULE = [
    {
        "pi": torch.pi,
        "sqrt": torch.sqrt,
        "exp": torch.exp,
        "Abs": torch.abs,
        "conjugate": torch.conj,
        "re": torch.real,
        "im": torch.imag,
    },
    torch,
]


@dataclass(frozen=True)
class SymbolMeta:
    """Metadata for one SymPy symbol used by an :class:`Equation`.

    - ``unit``: SI short unit string (e.g. ``"Ohm"``, ``"H"``, ``"Hz"``).
    - ``schema_field``: optional dotted path into the grid schema this symbol maps
      to (provenance only; not consumed by the evaluator).
    - ``description``: human-readable description.
    """

    unit: str = ""
    schema_field: Optional[str] = None
    description: str = ""


@dataclass(frozen=True)
class Equation:
    """A single residual-form physical law.

    Attributes
    ----------
    id:
        Unique registry id.
    description:
        Human-readable description.
    residual:
        SymPy expression equal to 0 (i.e. ``a - b``).
    symbols:
        Mapping ``sympy symbol name -> SymbolMeta``.
    tags:
        Free-form tags for grouping/filtering.
    """

    id: str
    description: str
    residual: sympy.Expr
    symbols: dict[str, SymbolMeta] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    @property
    def free_symbol_names(self) -> tuple[str, ...]:
        """Deterministic, alphabetically sorted free-symbol names of ``residual``."""
        return tuple(sorted(s.name for s in self.residual.free_symbols))


def _lambdify(expr: sympy.Expr, symbol_names: tuple[str, ...], modules) -> Callable:
    """Lambdify ``expr`` over the given ordered symbol names for a backend."""
    syms = [sympy.Symbol(name) for name in symbol_names]
    return sympy.lambdify(syms, expr, modules=modules)


class _Registry:
    """In-memory equation registry. A module-level singleton ``registry`` is used."""

    def __init__(self) -> None:
        self._eqs: dict[str, Equation] = {}
        # Lambdified evaluator caches keyed by equation id.
        self._torch_cache: dict[str, tuple[tuple[str, ...], Callable]] = {}
        self._numpy_cache: dict[str, tuple[tuple[str, ...], Callable]] = {}

    # -- registration / lookup ------------------------------------------------
    def register(self, eq: Equation) -> None:
        """Register ``eq``; raises ``KeyError`` on duplicate id."""
        if eq.id in self._eqs:
            raise KeyError(f"Equation id '{eq.id}' already registered.")
        self._eqs[eq.id] = eq

    def get(self, eq_id: str) -> Equation:
        """Return the registered :class:`Equation` for ``eq_id``."""
        try:
            return self._eqs[eq_id]
        except KeyError as exc:
            raise KeyError(f"No equation registered with id '{eq_id}'.") from exc

    def ids(self) -> tuple[str, ...]:
        """All registered equation ids (sorted)."""
        return tuple(sorted(self._eqs))

    # -- evaluators -----------------------------------------------------------
    def _torch_evaluator(self, eq_id: str) -> tuple[tuple[str, ...], Callable]:
        cached = self._torch_cache.get(eq_id)
        if cached is None:
            eq = self.get(eq_id)
            names = eq.free_symbol_names
            fn = _lambdify(eq.residual, names, _TORCH_MODULE)
            cached = (names, fn)
            self._torch_cache[eq_id] = cached
        return cached

    def _numpy_evaluator(self, eq_id: str) -> tuple[tuple[str, ...], Callable]:
        cached = self._numpy_cache.get(eq_id)
        if cached is None:
            eq = self.get(eq_id)
            names = eq.free_symbol_names
            fn = _lambdify(eq.residual, names, "numpy")
            cached = (names, fn)
            self._numpy_cache[eq_id] = cached
        return cached

    def torch_fn(self, eq_id: str) -> Callable[..., Tensor]:
        """Return an autograd-safe torch callable for the residual.

        The returned callable takes the equation's free symbols as keyword
        arguments (tensors or python scalars), broadcasts them, and returns the
        residual tensor. Missing required symbols raise ``TypeError``.
        """
        names, fn = self._torch_evaluator(eq_id)

        def _call(**values: Number) -> Tensor:
            missing = [n for n in names if n not in values]
            if missing:
                raise TypeError(
                    f"Equation '{eq_id}' missing required symbol(s): {missing}; "
                    f"expected {list(names)}."
                )
            args = [values[n] for n in names]
            return fn(*args)

        _call.__name__ = f"torch_fn[{eq_id}]"
        _call.symbol_names = names  # type: ignore[attr-defined]
        return _call

    def numpy_fn(self, eq_id: str) -> Callable:
        """Return a numpy callable for the residual (keyword args = symbol names)."""
        names, fn = self._numpy_evaluator(eq_id)

        def _call(**values):
            missing = [n for n in names if n not in values]
            if missing:
                raise TypeError(
                    f"Equation '{eq_id}' missing required symbol(s): {missing}; "
                    f"expected {list(names)}."
                )
            args = [values[n] for n in names]
            return fn(*args)

        _call.__name__ = f"numpy_fn[{eq_id}]"
        _call.symbol_names = names  # type: ignore[attr-defined]
        return _call

    def evaluate(
        self,
        eq_id: str,
        values: dict[str, Number],
        *,
        normalize_by: Optional[str] = None,
    ) -> Tensor:
        """Evaluate the residual on torch tensors; broadcasts; autograd-safe.

        Parameters
        ----------
        eq_id:
            Registry id of the equation.
        values:
            Mapping ``symbol name -> tensor | python scalar``.
        normalize_by:
            ``None`` -> raw residual.
            ``"<symbol>"`` -> residual divided by ``values["<symbol>"]``.
            ``"relative"`` -> residual divided by the magnitude of the larger side
            of the ``a - b`` split (with a tiny epsilon), giving a relative
            residual. This requires the residual to be a difference ``a - b``; for
            a generic expression it falls back to dividing by the residual's own
            scale, which is degenerate, so callers should prefer ``"<symbol>"``.
        """
        fn = self.torch_fn(eq_id)
        residual = fn(**values)
        if normalize_by is None:
            return residual
        if normalize_by == "relative":
            return _normalize_relative(self.get(eq_id), values, residual)
        if normalize_by in values:
            return residual / _as_tensor_like(values[normalize_by], residual)
        raise KeyError(
            f"normalize_by='{normalize_by}' is neither 'relative', None, nor a "
            f"provided symbol of equation '{eq_id}'."
        )

    # -- symbolic solve -------------------------------------------------------
    def solve_for(self, eq_id: str, target_symbol: str) -> Callable[..., Tensor]:
        """Return a closed-form torch callable for ``target_symbol``.

        SymPy solves ``residual == 0`` for ``target_symbol``; the first solution
        branch is lambdified to an autograd-safe torch callable whose keyword
        arguments are the remaining free symbols. Raises ``ValueError`` if SymPy
        cannot invert the relation.
        """
        eq = self.get(eq_id)
        # Resolve the target from the residual's OWN free symbols by name, so that
        # symbol assumptions (e.g. Symbol("X", real=True)) match. A bare
        # sympy.Symbol(name) carries no assumptions and would not compare equal.
        matches = [s for s in eq.residual.free_symbols if s.name == target_symbol]
        if not matches:
            raise ValueError(
                f"Symbol '{target_symbol}' does not appear in equation '{eq_id}'."
            )
        target = matches[0]
        solutions = sympy.solve(sympy.Eq(eq.residual, 0), target, dict=False)
        if not solutions:
            raise ValueError(
                f"SymPy could not solve equation '{eq_id}' for '{target_symbol}'."
            )
        sol = solutions[0]
        names = tuple(sorted(s.name for s in sol.free_symbols))
        fn = _lambdify(sol, names, _TORCH_MODULE)

        def _call(**values: Number) -> Tensor:
            missing = [n for n in names if n not in values]
            if missing:
                raise TypeError(
                    f"solve_for('{eq_id}', '{target_symbol}') missing symbol(s): "
                    f"{missing}; expected {list(names)}."
                )
            args = [values[n] for n in names]
            return fn(*args)

        _call.__name__ = f"solve_for[{eq_id}->{target_symbol}]"
        _call.symbol_names = names  # type: ignore[attr-defined]
        return _call

    def latex(self, eq_id: str) -> str:
        """Return a LaTeX string of the residual equation ``residual = 0``."""
        eq = self.get(eq_id)
        return sympy.latex(sympy.Eq(eq.residual, 0))


def _as_tensor_like(value: Number, ref: Tensor) -> Tensor:
    """Return ``value`` as a tensor broadcastable with ``ref`` (no copy of a tensor)."""
    if isinstance(value, Tensor):
        return value
    return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)


def _normalize_relative(
    eq: Equation, values: dict[str, Number], residual: Tensor
) -> Tensor:
    """Relative normalization: residual / max(|a|, |b|, eps) using the a-b split.

    ``residual`` is ``a - b``; we recover ``a`` and ``b`` from the SymPy ``Add``
    structure (the terms with negative sign form ``-b``). Evaluated with the same
    autograd-safe torch backend.
    """
    expr = eq.residual
    # Split into positive (a) and negative (b) parts of the Add.
    a_terms: list[sympy.Expr] = []
    b_terms: list[sympy.Expr] = []
    add = sympy.Add.make_args(sympy.expand(expr))
    for term in add:
        coeff = term.as_coeff_Mul()[0]
        if coeff.is_negative:
            b_terms.append(-term)
        else:
            a_terms.append(term)
    a_expr = sympy.Add(*a_terms) if a_terms else sympy.Integer(0)
    b_expr = sympy.Add(*b_terms) if b_terms else sympy.Integer(0)

    names_a = tuple(sorted(s.name for s in a_expr.free_symbols))
    names_b = tuple(sorted(s.name for s in b_expr.free_symbols))
    fn_a = _lambdify(a_expr, names_a, _TORCH_MODULE)
    fn_b = _lambdify(b_expr, names_b, _TORCH_MODULE)
    a_val = fn_a(*[values[n] for n in names_a])
    b_val = fn_b(*[values[n] for n in names_b])
    a_t = _as_tensor_like(a_val, residual)
    b_t = _as_tensor_like(b_val, residual)
    scale = torch.maximum(torch.abs(a_t), torch.abs(b_t))
    eps = torch.finfo(scale.dtype if scale.is_floating_point() else torch.float64).eps
    return residual / (scale + eps)


# Module-level singleton registry.
registry = _Registry()


__all__ = ["Equation", "SymbolMeta", "registry"]
