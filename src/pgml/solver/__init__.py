"""pgml.solver — complex batched differentiable solve of Y(f) V(f) = I(f).

Public surface (see ``solver/CONTEXT.md`` for the frozen contract):

- ``solve_harmonic(y_bus, i_inj, *, fixed_rows=None, v_fixed=None) -> v``
  Norton mode (default) and ideal-slack Schur-partition mode, both differentiable.
- ``solve_power_flow(grid, *, slack, method, tol, max_iter, dtype, device,
  operating_point, param_overrides, symmetry=None, criticality="auto",
  linear_solver="auto", on_disconnected="raise", branch_states=None,
  system=None) -> PowerFlowResult``
  Nonlinear const-P / ZIP fundamental power flow: current-injection fixed point
  forward, implicit-function-theorem backward (real-coordinate adjoint).
  ``symmetry`` selects per-phase vs balanced load modeling (``None`` -> config).
  ``linear_solver`` picks the inner factorization backend (sparse SuperLU on
  large CPU systems by default). ``on_disconnected`` controls the pre-solve
  connectivity check (``"raise"``/``"zero"``/``"ignore"``). ``branch_states``
  batches switch/topology configurations by differentiable admittance masking.
  ``system`` reuses a :class:`PowerFlowSystem` from :func:`prepare_power_flow`
  across repeated solves of the same grid.
- ``solve_harmonic_flow(grid, harmonic_orders, *, slack, method,
  operating_point, harmonic_injection, node_sources=None, include_load_shunt,
  tol, max_iter, dtype, device, symmetry=None, on_disconnected="raise",
  branch_states=None) -> HarmonicFlowResult``
  Fundamental + per-harmonic flow; ``method`` selects the fundamental-frequency
  solver (``"current_injection"`` or ``"newton"`` for stiff inverter control
  loops). ``symmetry`` is resolved once and threaded into the fundamental solve
  and all harmonic injection steps. Optional ``node_sources`` (a list of
  :class:`NodeHarmonicSource`) injects per-node Thévenin/Norton harmonic
  disturbances at orders ``h > 1`` only. ``on_disconnected``/``branch_states``
  match :func:`solve_power_flow`.
- ``check_connectivity(grid) -> None``
  Raises :class:`~pgml.errors.ConnectivityError` when part of the grid has no
  galvanic path to an in-service source; the pre-solve gate every entry point
  above runs by default.
- ``prepare_power_flow(grid, *, slack, dtype, device, param_overrides,
  branch_states, linear_solver="auto") -> PowerFlowSystem``
  Assembles and factors the operating-point-independent part of a nonlinear
  solve once, for reuse across repeated :func:`solve_power_flow` calls on the
  same grid (e.g. a chunked scenario batch).
- ``NodeHarmonicSource(node_id, phases=None, spectrum={}, source_power_va=0.0,
  kind="voltage") -> NodeHarmonicSource``
  Frozen dataclass describing a per-node harmonic "error" source
  (Thévenin voltage or Norton current) injected at orders ``h > 1``.
  Physics: ``docs/pgml/modeling/error-injection.md``.
- ``assemble_harmonic_system(grid, harmonic_orders, v1, *, operating_point,
  harmonic_injection, node_sources, symmetry, branch_states, dtype, device)
  -> (Y, I, index)``
  The assembled per-harmonic LINEAR system ``Y(h) V(h) = I(h)`` for orders
  ``h > 1`` — exactly the ``(Y, I)`` :func:`solve_harmonic_flow` solves, so
  ``r(V) = Y(h)·V − I(h)`` is the physics-consistency residual (``≈ 0`` at the
  true ``V``). The fundamental ``v1`` enters ``I(h)`` via each device's
  fundamental terminal current.
- ``assemble_harmonic_ybus(grid, harmonic_orders, *, dtype, device,
  branch_states=None) -> (Y, index)``
  ``Y(h)`` only, without the injection right-hand side — the self-consistency
  building block for a learned injection decoder.
"""

from __future__ import annotations

from .harmonic import solve_harmonic
from .harmonic_flow import (
    HarmonicFlowResult,
    NodeHarmonicSource,
    assemble_harmonic_system,
    assemble_harmonic_ybus,
    solve_harmonic_flow,
)
from .power_flow import (
    ConvergenceDiagnostics,
    LoadabilityResult,
    PowerFlowResult,
    PowerFlowSystem,
    check_connectivity,
    loadability_limit,
    prepare_power_flow,
    solve_power_flow,
)

# Canonical __module__ for public re-exports (avoids autodoc duplicate warnings).
PowerFlowResult.__module__ = __name__
PowerFlowSystem.__module__ = __name__
ConvergenceDiagnostics.__module__ = __name__
LoadabilityResult.__module__ = __name__
HarmonicFlowResult.__module__ = __name__
NodeHarmonicSource.__module__ = __name__

__all__ = [
    "check_connectivity",
    "prepare_power_flow",
    "PowerFlowSystem",
    "solve_harmonic",
    "solve_power_flow",
    "PowerFlowResult",
    "ConvergenceDiagnostics",
    "loadability_limit",
    "LoadabilityResult",
    "solve_harmonic_flow",
    "assemble_harmonic_system",
    "assemble_harmonic_ybus",
    "HarmonicFlowResult",
    "NodeHarmonicSource",
]
