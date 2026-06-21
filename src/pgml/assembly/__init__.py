"""pgml.assembly — differentiable, batched, per-frequency Y-bus assembly.

Public surface (see ``assembly/CONTEXT.md`` for the frozen contract):

- ``node_phase_index(grid) -> NodePhaseIndex`` — the compact (node, phase) layout.
- ``assemble_ybus(grid, frequencies_hz, *, dtype, device, operating_point,
  param_overrides=None, symmetry=None) -> YBus`` — LINEAR const-Z ``Y``
  ``[*batch, H, N, N]``.  ``symmetry`` selects per-phase vs balanced calculation
  (``None`` -> config ``calculation.symmetry``).
- ``assemble_network_ybus(grid, frequencies_hz, *, dtype, device,
  param_overrides=None) -> YBus`` — PASSIVE-network ``Y_net`` (no loads, no source
  Norton); the nonlinear power-flow assembler.
- ``build_injections(grid, frequencies_hz, index, *, dtype, device,
  operating_point, param_overrides=None) -> Tensor`` — Norton ``I`` ``[*batch, H, N]``.
- ``device_current_injections(grid, v, index, frequencies_hz, *, dtype, device,
  operating_point=None, param_overrides=None, symmetry=None) -> Tensor`` — ZIP
  voltage-dependent device current ``[*batch, H, N]``.  ``symmetry`` matches
  the value passed to the outer assembler so both remain consistent.
"""

from __future__ import annotations

from .index import NodePhaseIndex, node_phase_index
from .ybus import (
    BranchCurrent,
    YBus,
    assemble_network_ybus,
    assemble_ybus,
    branch_currents,
    build_injections,
    device_current_injections,
)

# Set canonical __module__ so autodoc registers symbols under the public
# package path rather than the private sub-module, avoiding "duplicate object
# description" warnings when viewcode and autodoc both traverse the codebase.
NodePhaseIndex.__module__ = __name__
YBus.__module__ = __name__
BranchCurrent.__module__ = __name__

__all__ = [
    "NodePhaseIndex",
    "node_phase_index",
    "YBus",
    "BranchCurrent",
    "assemble_ybus",
    "assemble_network_ybus",
    "branch_currents",
    "build_injections",
    "device_current_injections",
]
