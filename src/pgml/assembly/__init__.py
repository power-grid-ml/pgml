"""pgml.assembly — differentiable, batched, per-frequency Y-bus assembly.

Public surface (see ``assembly/CONTEXT.md`` for the frozen contract):

- ``node_phase_index(grid) -> NodePhaseIndex`` — the compact (node, phase) layout.
- ``assemble_ybus(grid, frequencies_hz, *, dtype, device, operating_point,
  param_overrides=None) -> YBus`` — LINEAR const-Z ``Y`` ``[*batch, H, N, N]``.
- ``assemble_network_ybus(grid, frequencies_hz, *, dtype, device,
  param_overrides=None) -> YBus`` — PASSIVE-network ``Y_net`` (no loads, no source
  Norton); the nonlinear power-flow assembler.
- ``build_injections(grid, frequencies_hz, index, *, dtype, device,
  operating_point, param_overrides=None) -> Tensor`` — Norton ``I`` ``[*batch, H, N]``.
- ``device_current_injections(grid, v, index, frequencies_hz, *, dtype, device,
  operating_point=None, param_overrides=None) -> Tensor`` — ZIP voltage-dependent
  device current ``[*batch, H, N]``.
"""

from __future__ import annotations

from .index import NodePhaseIndex, node_phase_index
from .ybus import (
    YBus,
    assemble_network_ybus,
    assemble_ybus,
    build_injections,
    device_current_injections,
)

__all__ = [
    "NodePhaseIndex",
    "node_phase_index",
    "YBus",
    "assemble_ybus",
    "assemble_network_ybus",
    "build_injections",
    "device_current_injections",
]
