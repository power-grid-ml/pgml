"""pgml.assembly — differentiable, batched, per-frequency Y-bus assembly.

Public surface (see ``assembly/CONTEXT.md`` for the frozen contract):

- ``node_phase_index(grid) -> NodePhaseIndex`` — the compact (node, phase) layout.
- ``assemble_ybus(grid, frequencies_hz, *, dtype, device, operating_point,
  param_overrides=None, symmetry=None, branch_states=None) -> YBus`` — LINEAR
  const-Z ``Y`` ``[*batch, H, N, N]``.  ``symmetry`` selects per-phase vs
  balanced calculation (``None`` -> config ``calculation.symmetry``).
  ``branch_states`` masks/batches branch admittance by a differentiable
  switch/topology state (overrides ``in_service``/``closed``).
- ``assemble_network_ybus(grid, frequencies_hz, *, dtype, device,
  param_overrides=None, branch_states=None) -> YBus`` — PASSIVE-network
  ``Y_net`` (no loads, no source Norton); the nonlinear power-flow assembler.
- ``branch_currents(grid, v, frequencies_hz, index, *, dtype, device,
  param_overrides=None, branch_states=None) -> list[BranchCurrent]`` — per-branch
  terminal currents from solved node voltages (``branch_states`` must match
  the assembly the voltages were solved with).
- ``build_injections(grid, frequencies_hz, index, *, dtype, device,
  operating_point, param_overrides=None) -> Tensor`` — Norton ``I`` ``[*batch, H, N]``.
- ``device_current_injections(grid, v, index, frequencies_hz, *, dtype, device,
  operating_point=None, param_overrides=None, symmetry=None) -> Tensor`` — ZIP
  voltage-dependent device current ``[*batch, H, N]``.  ``symmetry`` matches
  the value passed to the outer assembler so both remain consistent.
- ``build_injection_plan(grid, index, frequencies_hz, *, dtype, device,
  operating_point=None, param_overrides=None, symmetry=None) -> InjectionPlan``
  / ``injections_from_plan(plan, v) -> Tensor`` — the two halves of
  ``device_current_injections``: the V-independent resolution (once per solve)
  and the pure-tensor evaluation (once per iteration). The nonlinear solvers
  reuse one plan across all their iterations.
"""

from __future__ import annotations

from .index import NodePhaseIndex, base_voltage_per_row, node_phase_index
from .ybus import (
    BranchCurrent,
    InjectionPlan,
    YBus,
    assemble_network_ybus,
    assemble_ybus,
    branch_currents,
    build_injection_plan,
    build_injections,
    device_current_injections,
    injections_from_plan,
)

# Set canonical __module__ so autodoc registers symbols under the public
# package path rather than the private sub-module, avoiding "duplicate object
# description" warnings when viewcode and autodoc both traverse the codebase.
NodePhaseIndex.__module__ = __name__
YBus.__module__ = __name__
BranchCurrent.__module__ = __name__
InjectionPlan.__module__ = __name__

__all__ = [
    "NodePhaseIndex",
    "node_phase_index",
    "base_voltage_per_row",
    "YBus",
    "BranchCurrent",
    "InjectionPlan",
    "assemble_ybus",
    "assemble_network_ybus",
    "branch_currents",
    "build_injection_plan",
    "build_injections",
    "device_current_injections",
    "injections_from_plan",
]
