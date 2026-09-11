"""pgml.solver — complex batched differentiable solve of Y(f) V(f) = I(f).

Public surface (see ``solver/CONTEXT.md`` for the frozen contract):

- ``solve_harmonic(y_bus, i_inj, *, fixed_rows=None, v_fixed=None) -> v``
  Norton mode (default) and ideal-slack Schur-partition mode, both differentiable.
- ``solve_power_flow(grid, *, slack, method, tol, max_iter, dtype, device,
  operating_point, param_overrides, symmetry=None, criticality="auto",
  linear_solver="auto", block_rows=None, on_disconnected="raise",
  branch_states=None, branch_states_method="assemble", system=None,
  enforce_q_limits=None) -> PowerFlowResult``
  Nonlinear const-P / ZIP fundamental power flow: current-injection fixed point
  forward, implicit-function-theorem backward (real-coordinate adjoint).
  ``symmetry`` selects per-phase vs balanced load modeling (``None`` -> config).
  ``linear_solver`` picks the inner factorization backend (sparse SuperLU on
  large CPU systems by default); ``linear_solver="block"`` with ``block_rows``
  factors a BLOCK-DIAGONAL system (an ensemble of independent grids, see
  :meth:`pgml.multigrid.MergedGrid.block_rows`) one sub-grid at a time — the CUDA
  path for a many-grid ensemble. ``on_disconnected`` controls the pre-solve
  connectivity check (``"raise"``/``"zero"``/``"ignore"``). ``branch_states``
  batches switch/topology configurations by differentiable admittance masking, and
  ``branch_states_method="woodbury"`` (opt-in) solves that whole sweep from ONE base
  factorization through a Sherman-Morrison-Woodbury low-rank update
  (:mod:`pgml.solver.lowrank`) instead of assembling every state.
  ``system`` reuses a :class:`PowerFlowSystem` from :func:`prepare_power_flow`
  across repeated solves of the same grid. A grid with voltage-regulating
  generators (PV terminals) is solved by Newton with the regulated row pair
  substituted, its reactive limits enforced by PV-to-PQ switching
  (``enforce_q_limits``, default from :mod:`pgml.defaults`), and the solved
  reactive powers reported in ``PowerFlowResult.regulation``
  (:class:`VoltageRegulationResult`).
- ``solve_harmonic_flow(grid, harmonic_orders, *, slack, method,
  operating_point, harmonic_injection, node_sources=None, load_shunt=None,
  tol, max_iter, dtype, device, symmetry=None, on_disconnected="raise",
  branch_states=None) -> HarmonicFlowResult``
  Fundamental + per-harmonic flow; ``method`` selects the fundamental-frequency
  solver (``"current_injection"`` or ``"newton"`` for stiff inverter control
  loops). ``symmetry`` is resolved once and threaded into the fundamental solve
  and all harmonic injection steps. ``load_shunt`` selects the harmonic device
  Norton shunt (``"opendss"`` / ``"motor"`` / ``"none"``; ``None`` = the
  documented modeling default). Optional ``node_sources`` (a list of
  :class:`NodeHarmonicSource`) injects per-node Thévenin/Norton harmonic
  disturbances at orders ``h > 1`` only. ``on_disconnected``/``branch_states``
  match :func:`solve_power_flow`.
- ``check_connectivity(grid) -> None``
  Raises :class:`~pgml.errors.ConnectivityError` when part of the grid has no
  galvanic path to an in-service source; the pre-solve gate every entry point
  above runs by default.
- ``check_branch_impedances(grid) -> None``
  Raises :class:`~pgml.errors.ModelingError` naming any branch with ZERO series
  impedance (a bus coupler or jumper modelled as a zero-impedance line, a
  zero-length line, an ideal closed switch): such a branch has no primitive
  admittance. The second pre-solve gate, run by every entry point above.
- ``prepare_power_flow(grid, *, slack, dtype, device, param_overrides,
  branch_states, branch_states_method="assemble", linear_solver="auto",
  block_rows=None) -> PowerFlowSystem``
  Assembles and factors the operating-point-independent part of a nonlinear
  solve once, for reuse across repeated :func:`solve_power_flow` calls on the
  same grid (e.g. a chunked scenario batch). A consuming solve must request the
  same ``branch_states_method``.
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

from .harmonic import AnchoredSystem, solve_anchored, solve_harmonic
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
    VoltageRegulationResult,
    check_branch_impedances,
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
VoltageRegulationResult.__module__ = __name__
HarmonicFlowResult.__module__ = __name__
NodeHarmonicSource.__module__ = __name__

__all__ = [
    "check_branch_impedances",
    "check_connectivity",
    "prepare_power_flow",
    "PowerFlowSystem",
    "solve_harmonic",
    "solve_anchored",
    "AnchoredSystem",
    "solve_power_flow",
    "PowerFlowResult",
    "ConvergenceDiagnostics",
    "VoltageRegulationResult",
    "loadability_limit",
    "LoadabilityResult",
    "solve_harmonic_flow",
    "assemble_harmonic_system",
    "assemble_harmonic_ybus",
    "HarmonicFlowResult",
    "NodeHarmonicSource",
]
