"""pgml.solver — complex batched differentiable solve of Y(f) V(f) = I(f).

Public surface (see ``solver/CONTEXT.md`` for the frozen contract):

- ``solve_harmonic(y_bus, i_inj, *, fixed_rows=None, v_fixed=None) -> v``
  Norton mode (default) and ideal-slack Schur-partition mode, both differentiable.
- ``solve_power_flow(grid, *, slack, method, tol, max_iter, dtype, device,
  operating_point, param_overrides, symmetry=None) -> PowerFlowResult``
  Nonlinear const-P / ZIP fundamental power flow: current-injection fixed point
  forward, implicit-function-theorem backward (real-coordinate adjoint).
  ``symmetry`` selects per-phase vs balanced load modeling (``None`` -> config).
- ``solve_harmonic_flow(grid, harmonic_orders, *, slack, method,
  operating_point, harmonic_injection, node_sources=None, include_load_shunt,
  tol, max_iter, dtype, device, symmetry=None) -> HarmonicFlowResult``
  Fundamental + per-harmonic flow; ``method`` selects the fundamental-frequency
  solver (``"current_injection"`` or ``"newton"`` for stiff inverter control
  loops). ``symmetry`` is resolved once and threaded into the fundamental solve
  and all harmonic injection steps. Optional ``node_sources`` (a list of
  :class:`NodeHarmonicSource`) injects per-node Thévenin/Norton harmonic
  disturbances at orders ``h > 1`` only.
- ``NodeHarmonicSource(node_id, phases=None, spectrum={}, source_power_va=0.0,
  kind="voltage") -> NodeHarmonicSource``
  Frozen dataclass describing a per-node harmonic "error" source
  (Thévenin voltage or Norton current) injected at orders ``h > 1``.
  Physics: ``docs/pgml/modeling/error-injection.md``.
- ``assemble_harmonic_system(grid, harmonic_orders, v1, *, operating_point,
  harmonic_injection, node_sources, symmetry, dtype, device) -> (Y, I, index)``
  The assembled per-harmonic LINEAR system ``Y(h) V(h) = I(h)`` for orders
  ``h > 1`` — exactly the ``(Y, I)`` :func:`solve_harmonic_flow` solves, so
  ``r(V) = Y(h)·V − I(h)`` is the physics-consistency residual (``≈ 0`` at the
  true ``V``). The fundamental ``v1`` enters ``I(h)`` via each device's
  fundamental terminal current.
"""

from __future__ import annotations

from .harmonic import solve_harmonic
from .harmonic_flow import (
    HarmonicFlowResult,
    NodeHarmonicSource,
    assemble_harmonic_system,
    solve_harmonic_flow,
)
from .power_flow import (
    ConvergenceDiagnostics,
    LoadabilityResult,
    PowerFlowResult,
    loadability_limit,
    solve_power_flow,
)

# Canonical __module__ for public re-exports (avoids autodoc duplicate warnings).
PowerFlowResult.__module__ = __name__
ConvergenceDiagnostics.__module__ = __name__
LoadabilityResult.__module__ = __name__
HarmonicFlowResult.__module__ = __name__
NodeHarmonicSource.__module__ = __name__

__all__ = [
    "solve_harmonic",
    "solve_power_flow",
    "PowerFlowResult",
    "ConvergenceDiagnostics",
    "loadability_limit",
    "LoadabilityResult",
    "solve_harmonic_flow",
    "assemble_harmonic_system",
    "HarmonicFlowResult",
    "NodeHarmonicSource",
]
