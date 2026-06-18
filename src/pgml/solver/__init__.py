"""pgml.solver — complex batched differentiable solve of Y(f) V(f) = I(f).

Public surface (see ``solver/CONTEXT.md`` for the frozen contract):

- ``solve_harmonic(y_bus, i_inj, *, fixed_rows=None, v_fixed=None) -> v``
  Norton mode (default) and ideal-slack Schur-partition mode, both differentiable.
- ``solve_power_flow(grid, *, slack, method, tol, max_iter, dtype, device,
  operating_point, param_overrides, symmetry=None) -> PowerFlowResult``
  Nonlinear const-P / ZIP fundamental power flow: current-injection fixed point
  forward, implicit-function-theorem backward (real-coordinate adjoint).
  ``symmetry`` selects per-phase vs balanced load modeling (``None`` -> config).
- ``solve_harmonic_flow(grid, harmonic_orders, *, slack, operating_point,
  harmonic_injection, include_load_shunt, tol, max_iter, dtype, device,
  symmetry=None) -> HarmonicFlowResult``
  Fundamental + per-harmonic flow; ``symmetry`` is resolved once and threaded
  into the fundamental solve and all harmonic injection steps.
"""

from __future__ import annotations

from .harmonic import solve_harmonic
from .harmonic_flow import HarmonicFlowResult, solve_harmonic_flow
from .power_flow import PowerFlowResult, solve_power_flow

# Canonical __module__ for public re-exports (avoids autodoc duplicate warnings).
PowerFlowResult.__module__ = __name__
HarmonicFlowResult.__module__ = __name__

__all__ = [
    "solve_harmonic",
    "solve_power_flow",
    "PowerFlowResult",
    "solve_harmonic_flow",
    "HarmonicFlowResult",
]
