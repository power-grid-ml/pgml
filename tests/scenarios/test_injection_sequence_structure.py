"""Sequence structure of a three-phase device's harmonic injection.

Harmonic ``h`` on phase ``b`` is the phase-``a`` waveform delayed by a third of a cycle,
so the per-phase angles of an injected order follow the sequence of that order: triplen
orders are zero-sequence, h5 is negative-sequence and h7 positive-sequence. A scenario
batch states one (magnitude, phase) pair per device and order; the solver has to expand
it across the device's phases with that structure.
"""

from __future__ import annotations

import math

import pytest
import torch

ORDERS = [3, 5, 7]


# =============================================================================
# The solver keeps the physical sequence structure across a device's phases
# =============================================================================
def test_a_three_phase_device_injects_sequence_consistent_harmonics(grid_3ph):
    """Harmonic ``h`` on phase ``b`` is the phase-``a`` waveform delayed by a third of a
    cycle, i.e. rotated by ``-h * 120 deg``: h3 is zero-sequence (all phases in phase),
    h5 negative-sequence, h7 positive-sequence. The solver derives the per-phase angle as
    ``ang_h + h * arg(I_1,phase)``, which is exactly that structure."""
    from pgml.assembly import node_phase_index
    from pgml.schemas.grid_schema import Load
    from pgml.solver.harmonic_flow import _harmonic_injections

    load = next(
        a for a in grid_3ph.appliances if isinstance(a, Load) and len(a.phases) == 3
    )
    index = node_phase_index(grid_3ph)
    node = next(n for n in grid_3ph.nodes if n.id == load.node)
    v_ln = float(node.u_rated_v) / math.sqrt(3.0)
    v1 = torch.zeros(index.size, dtype=torch.complex128)
    for n in grid_3ph.nodes:
        for k, p in enumerate(n.phases):
            v1[int(index.row(n.id, p))] = v_ln * torch.exp(
                torch.tensor(-1j * 2 * math.pi * k / 3, dtype=torch.complex128)
            )
    inj = {load.id: {h: (0.1, 0.0) for h in ORDERS}}
    out = _harmonic_injections(
        grid_3ph,
        v1,
        index,
        ORDERS,
        {},
        inj,
        torch.complex128,
        torch.float64,
        torch.device("cpu"),
    )  # [Hh, N]
    rows = [int(index.row(load.node, p)) for p in load.phases]
    for k, h in enumerate(ORDERS):
        ang = torch.rad2deg(torch.angle(out[k, rows]))
        d_ba = float((ang[1] - ang[0] + 180.0) % 360.0 - 180.0)
        d_ca = float((ang[2] - ang[0] + 180.0) % 360.0 - 180.0)
        expected = {3: (0.0, 0.0), 5: (120.0, -120.0), 7: (-120.0, 120.0)}[h]
        assert d_ba == pytest.approx(expected[0], abs=1e-6), h
        assert d_ca == pytest.approx(expected[1], abs=1e-6), h
        # equal magnitudes on the three phases of a balanced device
        assert torch.allclose(out[k, rows].abs(), out[k, rows[0]].abs().expand(3))
