"""pgml.equations — residual-form scalar physical-law registry.

Importing this package registers the M1 load-flow laws (see ``laws.py``) on the
module-level ``registry`` singleton. Public surface:

- ``Equation``, ``SymbolMeta``: the frozen dataclasses.
- ``registry``: ``register / get / evaluate / torch_fn / numpy_fn / solve_for /
  latex`` (autograd-safe torch evaluators).
"""

from __future__ import annotations

from .registry import Equation, SymbolMeta, registry

# Side effect: register the M1 laws on import.
from . import laws  # noqa: F401  (import for registration side effect)

__all__ = ["Equation", "SymbolMeta", "registry"]
