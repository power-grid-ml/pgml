"""pgml.solver — complex batched differentiable solve of Y(f) V(f) = I(f).

Public surface (see ``solver/CONTEXT.md`` for the frozen contract):

- ``solve_harmonic(y_bus, i_inj, *, fixed_rows=None, v_fixed=None) -> v``
  Norton mode (default) and ideal-slack Schur-partition mode, both differentiable.
- ``lu_factor_system(y_bus, *, fixed_rows=None, backend="auto", block_rows=None,
  precision="full", refine_steps=None, equilibrate=None) -> FactoredSystem`` and
  ``solve_factored(fac, i_inj, *, v_fixed=None) -> v``
  The factor-once, solve-many form of :func:`solve_harmonic`: one factorization
  (dense, sparse SuperLU or block diagonal; full or mixed precision; diagonally
  equilibrated by default) answers any number of right-hand sides, differentiably.
  ``estimate_condition(fac, *, iters=5, per_matrix=False)`` estimates the 1-norm
  condition number of what was factored, per matrix of a batched factorization.
- ``solve_power_flow(grid, *, slack, method, tol, tol_update_pu, s_base_va, max_iter,
  dtype, precision="full", device, operating_point, param_overrides, symmetry=None,
  criticality="auto", linear_solver="auto", block_rows=None,
  on_disconnected="raise", branch_states=None, branch_states_method="assemble",
  system=None, enforce_q_limits=None, equilibrate=None) -> PowerFlowResult``
  Nonlinear const-P / ZIP fundamental power flow: current-injection fixed point
  forward, implicit-function-theorem backward (real-coordinate adjoint).
  Convergence is judged per unit on two criteria that must both hold: ``tol``, the
  nodal apparent-power mismatch on the base ``s_base_va``, and ``tol_update_pu``, the
  per-row voltage update. ``precision="mixed"`` factors in single precision and
  refines against double-precision residuals; ``equilibrate`` selects the diagonal
  scaling around every factorization (:mod:`pgml.solver.equilibration`).
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
  reactive powers (differentiable) reported in ``PowerFlowResult.regulation``
  (:class:`VoltageRegulationResult`), together with whether the switching settled.
- ``solve_harmonic_flow(grid, harmonic_orders, *, slack, method,
  operating_point, harmonic_injection, node_sources=None, load_shunt=None,
  load_shunt_basis=None, tol, tol_update_pu, s_base_va, max_iter, dtype,
  precision="full", device, symmetry=None, on_disconnected="raise",
  branch_states=None, branch_states_method="assemble", param_overrides=None,
  enforce_q_limits=None, linear_solver="auto", block_rows=None, criticality="auto",
  equilibrate=None, system=None) -> HarmonicFlowResult``
  Fundamental + per-harmonic flow; ``method`` selects the fundamental-frequency
  solver (``"current_injection"`` or ``"newton"`` for stiff inverter control
  loops). ``symmetry`` is resolved once and threaded into the fundamental solve
  and all harmonic injection steps. ``load_shunt`` selects the harmonic device
  Norton shunt (``"opendss"`` / ``"motor"`` / ``"none"``; ``None`` = the
  documented modeling default) and ``load_shunt_basis`` whether it is built from
  the solved operating point or from the nameplate. A harmonic order whose solution
  is not finite is reported as not converged. Optional ``node_sources`` (a list of
  :class:`NodeHarmonicSource`) injects per-node Thévenin/Norton harmonic
  disturbances at orders ``h > 1`` only. ``on_disconnected``/``branch_states``
  match :func:`solve_power_flow`.
- ``HarmonicFlowSystem(*, cache_batched_factors=False)`` and
  ``prepare_harmonic_flow(grid, harmonic_orders, **solve_kwargs)`` provide explicit
  repeated-call preparation. Pass it as ``system`` to the harmonic solve; current
  input values and evaluated admittances govern reuse, and changes automatically
  rebuild the affected entries. Matrix gradients bypass harmonic numerical caches.
  A per-scenario ``Y(h)`` (an ``operating_point`` device shunt over a batch) is not
  retained unless ``cache_batched_factors=True``, which serves a replay of the same
  batch and keeps a copy of the whole ``[B, H, N, N]`` system beside its factors.
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
- ``harmonic_injections(grid, v1, harmonic_orders, *, operating_point,
  harmonic_injection, node_sources, symmetry, dtype, device, param_overrides,
  index=None) -> Tensor``
  ``I(h)`` only, the injection right-hand side of the same system, on the grid's
  full (or a given) row layout.
"""

from __future__ import annotations

from .harmonic import (
    AnchoredSystem,
    FactoredSystem,
    estimate_condition,
    lu_factor_system,
    solve_anchored,
    solve_factored,
    solve_harmonic,
)
from .harmonic_flow import (
    HarmonicFlowResult,
    HarmonicFlowSystem,
    NodeHarmonicSource,
    assemble_harmonic_system,
    assemble_harmonic_ybus,
    harmonic_injections,
    prepare_harmonic_flow,
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
HarmonicFlowSystem.__module__ = __name__
NodeHarmonicSource.__module__ = __name__
FactoredSystem.__module__ = __name__

__all__ = [
    "check_branch_impedances",
    "check_connectivity",
    "prepare_power_flow",
    "PowerFlowSystem",
    "solve_harmonic",
    "lu_factor_system",
    "solve_factored",
    "FactoredSystem",
    "estimate_condition",
    "solve_anchored",
    "AnchoredSystem",
    "solve_power_flow",
    "PowerFlowResult",
    "ConvergenceDiagnostics",
    "VoltageRegulationResult",
    "loadability_limit",
    "LoadabilityResult",
    "solve_harmonic_flow",
    "prepare_harmonic_flow",
    "HarmonicFlowSystem",
    "assemble_harmonic_system",
    "assemble_harmonic_ybus",
    "harmonic_injections",
    "HarmonicFlowResult",
    "NodeHarmonicSource",
]
