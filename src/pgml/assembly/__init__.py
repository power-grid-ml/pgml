"""pgml.assembly — differentiable, batched, per-frequency Y-bus assembly.

Public surface (see ``assembly/CONTEXT.md`` for the frozen contract):

- ``node_phase_index(grid) -> NodePhaseIndex`` — the compact (node, phase) layout.
- ``assemble_ybus(grid, frequencies_hz, *, dtype, device, operating_point,
  param_overrides=None) -> YBus`` — complex ``Y`` ``[*batch, H, N, N]``.
- ``build_injections(grid, frequencies_hz, index, *, dtype, device,
  operating_point, param_overrides=None) -> Tensor`` — Norton ``I`` ``[*batch, H, N]``.
"""

from __future__ import annotations

from .index import NodePhaseIndex, node_phase_index
from .ybus import YBus, assemble_ybus, build_injections

__all__ = [
    "NodePhaseIndex",
    "node_phase_index",
    "YBus",
    "assemble_ybus",
    "build_injections",
]
