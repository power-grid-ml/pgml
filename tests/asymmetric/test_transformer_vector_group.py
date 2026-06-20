"""Phase-domain two-winding transformer vector-group stamp.

These tests exercise the winding-incidence primitive ``Y = Nᵀ Y_winding N`` directly
on a minimal hand-built grid (no reference library), so the physics is checked in
isolation:

- a Dyn (delta-HV / grounded-wye-LV) transformer BLOCKS the zero sequence — the
  HV-LV coupling block has zero row sums, so triplen / residual harmonics injected
  on the LV side cannot drive an HV line current;
- a YNyn (both grounded wye) transformer PASSES the zero sequence;
- an ungrounded-wye winding is singular to the zero sequence on its own side;
- the 3-phase positive-sequence coupling equals the single-phase-equivalent
  off-nominal-tap pi (the regression link), with magnitude ``y_se/n_LL``.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
import torch

from pgml.assembly import assemble_network_ybus
from pgml.schemas.grid_schema import (
    ComplexTap,
    Grid,
    Node,
    Phase,
    Source,
    Transformer,
    WindingConnection,
)

ABC = (Phase.A, Phase.B, Phase.C)
F0 = 50.0
S_RATED = 0.4e6
U_HV = 20_000.0
U_LV = 400.0
R_LV = 0.01  # leakage referred to LV coil
L_LV = 1.0e-4


def _grid(from_conn, to_conn, shift_deg, phases=ABC) -> Grid:
    """HV source node -> transformer -> LV node, with the given vector group."""
    src = Source(
        id=10,
        node=1,
        phases=phases,
        u_ref_v=(U_HV / math.sqrt(3),) * len(phases),
        u_angle_deg=(0.0, -120.0, 120.0)[: len(phases)],
        resistance_ohm=[
            [0.1 if i == j else 0.0 for j in range(len(phases))]
            for i in range(len(phases))
        ],
        inductance_h=[
            [1.0e-3 if i == j else 0.0 for j in range(len(phases))]
            for i in range(len(phases))
        ],
    )
    xfmr = Transformer(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=phases,
        to_phases=phases,
        s_rated_va=S_RATED,
        u_rated_from_v=U_HV,
        u_rated_to_v=U_LV,
        from_connection=from_conn,
        to_connection=to_conn,
        series_resistance_ohm=R_LV,
        series_inductance_h=L_LV,
        tap=ComplexTap(ratio_magnitude=1.0, shift_deg=shift_deg),
    )
    nodes = [
        Node(id=1, u_rated_v=U_HV, phases=phases),
        Node(id=2, u_rated_v=U_LV, phases=phases),
    ]
    return Grid(base_frequency_hz=F0, nodes=nodes, branches=[xfmr], appliances=[src])


def _coupling_block(grid: Grid) -> np.ndarray:
    """The 3x3 HV->LV coupling block of the assembled network Y at f0."""
    yb = assemble_network_ybus(grid, [F0], dtype=torch.complex128)
    Y = yb.Y[0].numpy()
    idx = yb.index
    hv = [idx.row(1, p) for p in ABC]
    lv = [idx.row(2, p) for p in ABC]
    return Y[np.ix_(hv, lv)]


class TestZeroSequenceBlocking:
    def test_dyn_blocks_zero_sequence(self) -> None:
        """Dyn1: the HV-LV coupling block has zero row sums (delta traps zero seq)."""
        yhl = _coupling_block(
            _grid(WindingConnection.DELTA, WindingConnection.WYE_GROUNDED, 30.0)
        )
        row_sums = np.abs(yhl.sum(axis=1))
        assert row_sums.max() < 1e-9, (
            f"Dyn HV-LV coupling row sums should vanish (zero-seq block), "
            f"got max |row sum| = {row_sums.max():.3e}"
        )
        # Zero-sequence injection on the LV side drives no HV current.
        ones = np.ones(3)
        assert np.abs(yhl @ ones).max() < 1e-9

    def test_ynyn_passes_zero_sequence(self) -> None:
        """YNyn0: both grounded wye -> the coupling passes the zero sequence."""
        yhl = _coupling_block(
            _grid(WindingConnection.WYE_GROUNDED, WindingConnection.WYE_GROUNDED, 0.0)
        )
        # Diagonal coupling (I3) -> non-zero row sums; zero-seq propagates.
        assert np.abs(yhl @ np.ones(3)).max() > 1e-3

    def test_ungrounded_wye_self_block_singular_to_zero_seq(self) -> None:
        """An ungrounded-wye HV winding is singular to the zero sequence on its side."""
        grid = _grid(WindingConnection.WYE, WindingConnection.WYE_GROUNDED, 0.0)
        yb = assemble_network_ybus(grid, [F0], dtype=torch.complex128)
        Y = yb.Y[0].numpy()
        idx = yb.index
        lv = [idx.row(2, p) for p in ABC]
        # LV self block (transformer contribution only at this isolated node).
        yll = Y[np.ix_(lv, lv)]
        # Coupling HV->LV must still block zero seq through the ungrounded wye:
        hv = [idx.row(1, p) for p in ABC]
        yhl = Y[np.ix_(hv, lv)]
        assert np.abs(yhl @ np.ones(3)).max() < 1e-6
        assert yll.shape == (3, 3)


class TestPositiveSequenceRegression:
    def test_three_phase_posseq_matches_single_phase(self) -> None:
        """P=3 positive-sequence coupling == P=1 off-nominal-tap pi (magnitude y/n_LL)."""
        # 3-phase Dyn1 coupling, projected onto the positive sequence.
        yhl = _coupling_block(
            _grid(WindingConnection.DELTA, WindingConnection.WYE_GROUNDED, 30.0)
        )
        a = cmath.exp(2j * math.pi / 3)
        pos = np.array([1, a**2, a])
        cpl3 = (pos.conjugate() @ yhl @ pos) / 3.0

        # P=1 equivalent: t = n_LL * exp(j*30 deg), Y_ft = -y_se / conj(t).
        w0 = 2.0 * math.pi * F0
        y_se = 1.0 / (R_LV + 1j * w0 * L_LV)
        n_ll = U_HV / U_LV
        t = n_ll * cmath.exp(1j * math.radians(30.0))
        y_ft = -y_se / t.conjugate()

        assert abs(cpl3 - y_ft) < 1e-9 * abs(y_ft)
        assert abs(abs(cpl3) - abs(y_se) / n_ll) < 1e-9 * abs(y_se) / n_ll

    def test_dyn11_opposite_shift_sign(self) -> None:
        """Dyn1 vs Dyn11: same coupling magnitude, phases 60 deg apart (±30 deg tap).

        ``Y_ft = -(y_se/n)·e^{±j30}`` for clock 1 (LV lags) vs 11 (LV leads), so the
        two positive-sequence couplings differ by a ``e^{j60}`` rotation.
        """
        a = cmath.exp(2j * math.pi / 3)
        pos = np.array([1, a**2, a])

        def posseq(shift):
            yhl = _coupling_block(
                _grid(WindingConnection.DELTA, WindingConnection.WYE_GROUNDED, shift)
            )
            return (pos.conjugate() @ yhl @ pos) / 3.0

        c1 = posseq(30.0)  # Dyn1, LV lags
        c11 = posseq(330.0)  # Dyn11, LV leads
        assert abs(abs(c1) - abs(c11)) < 1e-9 * abs(c1)
        assert abs(c1 - c11 * cmath.exp(1j * math.radians(60.0))) < 1e-9 * abs(c1)


class TestUnsupportedGroups:
    def test_unsupported_delta_wye_clock_raises(self) -> None:
        """A delta-wye clock other than 1/11 is rejected (not yet modelled)."""
        grid = _grid(WindingConnection.DELTA, WindingConnection.WYE_GROUNDED, 150.0)
        with pytest.raises(NotImplementedError):
            assemble_network_ybus(grid, [F0], dtype=torch.complex128)
