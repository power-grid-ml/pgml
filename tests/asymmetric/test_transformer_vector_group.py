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
  off-nominal-tap pi (the regression link), with magnitude ``y_se/n_LL``;
- an explicit ``zero_sequence`` leakage VALUE changes only the zero-sequence
  eigenvalue of the primitive, never the positive-/negative-sequence one, and
  ``Z0 == Z1`` reproduces the scalar stamp.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
import torch

from pgml.assembly import assemble_network_ybus
from pgml.errors import ModelingError
from pgml.schemas.grid_schema import (
    ComplexTap,
    Grid,
    Node,
    Phase,
    Source,
    Transformer,
    TransformerZeroSeq,
    WindingConnection,
)

ABC = (Phase.A, Phase.B, Phase.C)
F0 = 50.0
S_RATED = 0.4e6
U_HV = 20_000.0
U_LV = 400.0
R_LV = 0.01  # leakage referred to LV coil
L_LV = 1.0e-4


def _grid(from_conn, to_conn, shift_deg, phases=ABC, zero_sequence=None) -> Grid:
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
        zero_sequence=zero_sequence,
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


class TestSinglePhaseEquivalentSequenceShift:
    """The single-phase equivalent shifts each order by ITS sequence's angle.

    Reference: the three-phase winding-incidence stamp, which is sequence-correct by
    construction. Its HV/LV 2x2 seen by a positive-sequence set must equal the
    single-phase equivalent at orders ``3k+1``, and the one seen by a negative-sequence
    set at orders ``3k+2`` (the shift flips sign).
    """

    @staticmethod
    def _sequence_two_port(grid3: Grid, order: int, seq: np.ndarray) -> np.ndarray:
        yb = assemble_network_ybus(grid3, [order * F0], dtype=torch.complex128)
        y = yb.Y[0].numpy()
        hv = [yb.index.row(1, p) for p in ABC]
        lv = [yb.index.row(2, p) for p in ABC]
        out = np.zeros((2, 2), dtype=complex)
        for i, ri in enumerate((hv, lv)):
            for j, rj in enumerate((hv, lv)):
                out[i, j] = (seq.conjugate() @ y[np.ix_(ri, rj)] @ seq) / 3.0
        return out

    @staticmethod
    def _equivalent_two_port(grid1: Grid, order: int) -> np.ndarray:
        yb = assemble_network_ybus(grid1, [order * F0], dtype=torch.complex128)
        y = yb.Y[0].numpy()
        rows = [yb.index.row(1, Phase.A), yb.index.row(2, Phase.A)]
        return y[np.ix_(rows, rows)]

    @pytest.mark.parametrize("shift", [30.0, 330.0, 150.0])
    @pytest.mark.parametrize("order", [1, 2, 4, 5, 7, 11, 13])
    def test_matches_the_three_phase_stamp_per_sequence(self, shift, order) -> None:
        conns = (WindingConnection.DELTA, WindingConnection.WYE_GROUNDED)
        grid3 = _grid(*conns, shift)
        grid1 = _grid(*conns, shift, phases=(Phase.A,))
        a = cmath.exp(2j * math.pi / 3)
        seq = np.array([1, a**2, a]) if order % 3 == 1 else np.array([1, a, a**2])
        ref = self._sequence_two_port(grid3, order, seq)
        got = self._equivalent_two_port(grid1, order)
        assert np.abs(got - ref).max() < 1e-9 * np.abs(ref).max()

    def test_negative_sequence_order_conjugates_the_rotation(self) -> None:
        """h = 5 against h = 7 on a Dyn11: the transfer terms rotate by -/+ 30 deg."""
        grid1 = _grid(
            WindingConnection.DELTA,
            WindingConnection.WYE_GROUNDED,
            330.0,
            phases=(Phase.A,),
        )
        for order, sign in ((5, -1.0), (7, 1.0), (6.5, 1.0)):
            y = self._equivalent_two_port(grid1, order)
            y_se = 1.0 / (R_LV + 1j * 2.0 * math.pi * order * F0 * L_LV)
            t = (U_HV / U_LV) * cmath.exp(1j * sign * math.radians(330.0))
            assert abs(y[0, 1] + y_se / t.conjugate()) < 1e-12 * abs(y[0, 1])
            assert abs(y[1, 0] + y_se / t) < 1e-12 * abs(y[1, 0])

    def test_triplen_order_on_a_blocking_pairing_is_reported_once(self, caplog) -> None:
        import logging

        from pgml.assembly import ybus as ybus_module

        grid1 = _grid(
            WindingConnection.DELTA,
            WindingConnection.WYE_GROUNDED,
            330.0,
            phases=(Phase.A,),
        )
        ybus_module._TRIPLEN_EQUIVALENT_NOTICE_LOGGED = False
        try:
            with caplog.at_level(logging.WARNING, logger="pgml"):
                self._equivalent_two_port(grid1, 5)
                assert not [r for r in caplog.records if "triplen" in r.getMessage()]
                self._equivalent_two_port(grid1, 3)
                self._equivalent_two_port(grid1, 9)
            hits = [r for r in caplog.records if "triplen" in r.getMessage()]
            assert len(hits) == 1
        finally:
            ybus_module._TRIPLEN_EQUIVALENT_NOTICE_LOGGED = False


class TestClockSix:
    """Clock 6 (Yy6 / Dd6) is a 180° group: reversed LV winding polarity.

    The HV↔LV coupling block must be the exact NEGATION of the clock-0 group's
    (LV phasors inverted), while both self blocks are unchanged
    (``(−N)ᵀ·Y·(−N) = Nᵀ·Y·N``).
    """

    @pytest.mark.parametrize(
        "conn",
        [WindingConnection.WYE_GROUNDED, WindingConnection.DELTA],
        ids=["Yy", "Dd"],
    )
    def test_clock6_coupling_is_negated_clock0(self, conn) -> None:
        c0 = _coupling_block(_grid(conn, conn, 0.0))
        c6 = _coupling_block(_grid(conn, conn, 180.0))
        assert np.abs(c0).max() > 1e-6  # non-vacuous
        assert np.abs(c0 + c6).max() < 1e-12 * np.abs(c0).max(), (
            "clock-6 coupling must invert the clock-0 coupling"
        )

    @pytest.mark.parametrize(
        "conn",
        [WindingConnection.WYE_GROUNDED, WindingConnection.DELTA],
        ids=["Yy", "Dd"],
    )
    def test_clock6_self_blocks_unchanged(self, conn) -> None:
        def blocks(shift):
            yb = assemble_network_ybus(
                _grid(conn, conn, shift), [F0], dtype=torch.complex128
            )
            Y = yb.Y[0].numpy()
            idx = yb.index
            hv = [idx.row(1, p) for p in ABC]
            lv = [idx.row(2, p) for p in ABC]
            return Y[np.ix_(hv, hv)], Y[np.ix_(lv, lv)]

        hv0, lv0 = blocks(0.0)
        hv6, lv6 = blocks(180.0)
        assert np.abs(hv0 - hv6).max() < 1e-12 * np.abs(hv0).max()
        assert np.abs(lv0 - lv6).max() < 1e-12 * np.abs(lv0).max()

    def test_yy6_posseq_matches_single_phase_rotation(self) -> None:
        """The Yy6 positive-sequence coupling equals the 180°-rotated Yy0 one,
        consistent with the single-phase-equivalent path's ``rot = e^{jπ}``."""
        a = cmath.exp(2j * math.pi / 3)
        pos = np.array([1, a**2, a])

        def posseq(shift):
            yhl = _coupling_block(
                _grid(
                    WindingConnection.WYE_GROUNDED,
                    WindingConnection.WYE_GROUNDED,
                    shift,
                )
            )
            return (pos.conjugate() @ yhl @ pos) / 3.0

        c0 = posseq(0.0)
        c6 = posseq(180.0)
        assert abs(c6 - c0 * cmath.exp(1j * math.pi)) < 1e-12 * abs(c0)


class TestUnsupportedGroups:
    def test_parity_inconsistent_clock_raises(self) -> None:
        """A clock whose parity contradicts the winding pairing is rejected."""
        grid = _grid(WindingConnection.DELTA, WindingConnection.WYE_GROUNDED, 60.0)
        with pytest.raises(NotImplementedError, match="clock 2 is inconsistent"):
            assemble_network_ybus(grid, [F0], dtype=torch.complex128)

    def test_dyn5_coupling_carries_150_deg(self) -> None:
        """Dyn5 (the German LV std type clock) assembles with a 150° coupling.

        The positive-sequence coupling is ``−y_se·e^{jθ}/n``; referencing its
        phase against the leakage admittance ``y_se`` isolates the clock angle.
        """
        grid = _grid(WindingConnection.DELTA, WindingConnection.WYE_GROUNDED, 150.0)
        coupling = _coupling_block(grid)
        v_pos = np.array(
            [1.0, cmath.exp(-2j * math.pi / 3), cmath.exp(2j * math.pi / 3)]
        )
        lam = complex(v_pos.conj() @ coupling @ v_pos / 3.0)
        y_se = 1.0 / complex(R_LV, 2.0 * math.pi * F0 * L_LV)
        shift = math.degrees(cmath.phase(-lam / y_se)) % 360.0
        assert shift == pytest.approx(150.0, abs=1e-9)

    def test_finite_grounding_impedance_raises(self) -> None:
        """A nonzero neutral grounding impedance is rejected, not ignored."""
        from pgml.schemas.grid_schema import GroundingImpedance, Transformer

        grid = _grid(
            WindingConnection.WYE_GROUNDED, WindingConnection.WYE_GROUNDED, 0.0
        )
        trafo = next(b for b in grid.branches if isinstance(b, Transformer))
        trafo.to_grounding = GroundingImpedance(r_ohm=5.0)
        with pytest.raises(NotImplementedError, match="grounding"):
            assemble_network_ybus(grid, [F0], dtype=torch.complex128)

    def test_zero_grounding_impedance_is_solid(self) -> None:
        """An explicit r=x=0 grounding equals the solid default and assembles."""
        from pgml.schemas.grid_schema import GroundingImpedance, Transformer

        grid = _grid(
            WindingConnection.WYE_GROUNDED, WindingConnection.WYE_GROUNDED, 0.0
        )
        trafo = next(b for b in grid.branches if isinstance(b, Transformer))
        trafo.to_grounding = GroundingImpedance(r_ohm=0.0, x_ohm=0.0)
        assemble_network_ybus(grid, [F0], dtype=torch.complex128)


# ---------------------------------------------------------------------------
# zero-sequence leakage VALUE (Transformer.zero_sequence)
# ---------------------------------------------------------------------------
X_LV = L_LV * 2.0 * math.pi * F0  # positive-sequence leakage reactance at f0


def _sequence_eigenvalues(block: np.ndarray) -> np.ndarray:
    """``[lambda_0, lambda_1, lambda_2]`` of a 3x3 symmetric-circulant block."""
    a = np.exp(2j * np.pi / 3.0)
    t = np.array([[1, 1, 1], [1, a, a**2], [1, a**2, a]]) / np.sqrt(3.0)
    return np.diag(t.conj().T @ block @ t)


def _transformer_blocks(grid: Grid):
    """``(Y_HH, Y_HL, Y_LL)`` 3x3 blocks of the assembled network Y at f0."""
    yb = assemble_network_ybus(grid, [F0], dtype=torch.complex128)
    y = yb.Y[0].numpy()
    idx = yb.index
    hv = [idx.row(1, p) for p in ABC]
    lv = [idx.row(2, p) for p in ABC]
    return y[np.ix_(hv, hv)], y[np.ix_(hv, lv)], y[np.ix_(lv, lv)]


class TestZeroSequenceLeakageValue:
    """``Transformer.zero_sequence`` sets the VALUE on the topology-derived path."""

    def test_z0_equal_z1_reproduces_the_scalar_stamp(self) -> None:
        """An explicit Z0 == Z1 is numerically the scalar (per-phase uniform) stamp."""
        scalar = _transformer_blocks(
            _grid(WindingConnection.WYE_GROUNDED, WindingConnection.WYE_GROUNDED, 0.0)
        )
        matrix = _transformer_blocks(
            _grid(
                WindingConnection.WYE_GROUNDED,
                WindingConnection.WYE_GROUNDED,
                0.0,
                zero_sequence=TransformerZeroSeq(r0_ohm=R_LV, x0_ohm=X_LV),
            )
        )
        for a, b in zip(scalar, matrix):
            scale = max(np.abs(a).max(), 1.0)
            assert np.abs(a - b).max() < 1e-12 * scale

    @pytest.mark.parametrize("factor", [0.4, 2.5])
    def test_ynyn_sequence_eigenvalues(self, factor: float) -> None:
        """YNyn: seq-0 eigenvalue is 1/Z0, seq-1/2 stay 1/Z1, in every block."""
        z1 = complex(R_LV, X_LV)
        z0 = z1 * factor
        grid = _grid(
            WindingConnection.WYE_GROUNDED,
            WindingConnection.WYE_GROUNDED,
            0.0,
            zero_sequence=TransformerZeroSeq(
                r0_ohm=R_LV * factor, x0_ohm=X_LV * factor
            ),
        )
        y_hh, y_hl, y_ll = _transformer_blocks(grid)
        tau = U_HV / U_LV  # both windings are wye -> coil ratio == line ratio
        lam_ll = _sequence_eigenvalues(y_ll)
        lam_hh = _sequence_eigenvalues(y_hh) * tau**2
        lam_hl = _sequence_eigenvalues(y_hl) * tau
        for lam, sign in ((lam_ll, 1.0), (lam_hh, 1.0), (lam_hl, -1.0)):
            assert lam[0] == pytest.approx(sign / z0, rel=1e-10)
            assert lam[1] == pytest.approx(sign / z1, rel=1e-10)
            assert lam[2] == pytest.approx(sign / z1, rel=1e-10)

    def test_dyn_zero_sequence_value_only_reaches_the_lv_self_block(self) -> None:
        """Dyn: the delta still blocks the zero sequence; Z0 sets the LV shunt value."""
        factor = 0.4
        grid = _grid(
            WindingConnection.DELTA,
            WindingConnection.WYE_GROUNDED,
            30.0,
            zero_sequence=TransformerZeroSeq(
                r0_ohm=R_LV * factor, x0_ohm=X_LV * factor
            ),
        )
        y_hh, y_hl, y_ll = _transformer_blocks(grid)
        # The delta winding keeps the coupling zero-sequence-free whatever Z0 is.
        assert np.abs(y_hl @ np.ones(3)).max() < 1e-9
        assert np.abs(y_hh @ np.ones(3)).max() < 1e-9
        # The LV side sees 1/Z0 in the zero sequence (its path to ground).
        z0 = complex(R_LV, X_LV) * factor
        assert _sequence_eigenvalues(y_ll)[0] == pytest.approx(1.0 / z0, rel=1e-10)

    def test_grounded_zigzag_zero_sequence_is_the_override_value(self) -> None:
        """A grounding zigzag's low Z0 is now expressible (its own side's shunt)."""
        factor = 0.05  # a grounding transformer: Z0 << Z1 by design
        grid = _grid(
            WindingConnection.ZIGZAG_GROUNDED,
            WindingConnection.DELTA,
            0.0,  # Zd: two shifting windings -> an even clock
            zero_sequence=TransformerZeroSeq(
                r0_ohm=R_LV * factor, x0_ohm=X_LV * factor
            ),
        )
        y_hh, y_hl, _ = _transformer_blocks(grid)
        z0 = complex(R_LV, X_LV) * factor
        # The zigzag's own side presents 1/Z0 to the zero sequence, referred through
        # its coil ratio (a zigzag coil is rated line-to-neutral, the delta coil
        # line-to-line).
        tau = (U_HV / math.sqrt(3.0)) / U_LV
        assert _sequence_eigenvalues(y_hh)[0] * tau**2 == pytest.approx(
            1.0 / z0, rel=1e-10
        )
        # ... and cannot TRANSFER it to the other winding.
        assert np.abs(y_hl @ np.ones(3)).max() < 1e-9

    def test_default_ratios_are_unity(self) -> None:
        """The shipped default is Z0 = Z1, so a grid without the override is scalar."""
        from pgml.assembly._transformer import (
            is_sequence_aware,
            zero_sequence_leakage_ratios,
        )

        assert zero_sequence_leakage_ratios() == (1.0, 1.0)
        grid = _grid(
            WindingConnection.WYE_GROUNDED, WindingConnection.WYE_GROUNDED, 0.0
        )
        xfmr = grid.branches[0]
        assert not is_sequence_aware(xfmr, 3)
        assert not is_sequence_aware(xfmr, 1)


# ---------------------------------------------------------------------------
# winding-resistance frequency law (harmonic_xr_constant / resistance_frequency)
# ---------------------------------------------------------------------------
def _leakage_impedance(grid: Grid, order: float) -> complex:
    """The transformer's LV-coil-referred leakage impedance at ``order`` [Ohm]."""
    yb = assemble_network_ybus(grid, [order * F0], dtype=torch.complex128)
    y = yb.Y[0].numpy()
    idx = yb.index
    hv = [idx.row(1, p) for p in ABC]
    lv = [idx.row(2, p) for p in ABC]
    # Coupling entry of phase a is -y_se/tau (wye/wye: tau = U_HV/U_LV).
    tau = U_HV / U_LV
    return -tau / y[np.ix_(hv, lv)][0, 0]


class TestWindingResistanceFrequencyLaw:
    """``harmonic_xr_constant`` and ``resistance_frequency`` reach the stamp."""

    @staticmethod
    def _ynyn(**fields) -> Grid:
        grid = _grid(
            WindingConnection.WYE_GROUNDED, WindingConnection.WYE_GROUNDED, 0.0
        )
        for key, value in fields.items():
            setattr(grid.branches[0], key, value)
        return grid

    def test_default_keeps_r_constant(self) -> None:
        """Default (XRConst=No): R fixed, X proportional to the order."""
        grid = self._ynyn()
        z1 = _leakage_impedance(grid, 1.0)
        z13 = _leakage_impedance(grid, 13.0)
        assert z13.real == pytest.approx(z1.real, rel=1e-10)
        assert z13.imag == pytest.approx(13.0 * z1.imag, rel=1e-10)

    def test_xr_constant_scales_r_with_the_order(self) -> None:
        """XRConst=Yes: R proportional to the order, so X/R is frequency-independent."""
        grid = self._ynyn(harmonic_xr_constant=True)
        z1 = _leakage_impedance(grid, 1.0)
        z13 = _leakage_impedance(grid, 13.0)
        assert z13.real == pytest.approx(13.0 * z1.real, rel=1e-10)
        assert z13.imag == pytest.approx(13.0 * z1.imag, rel=1e-10)
        assert z13.imag / z13.real == pytest.approx(z1.imag / z1.real, rel=1e-10)

    def test_resistance_multiplier_curve_is_applied(self) -> None:
        """A sampled ``resistance_frequency`` curve multiplies R per frequency."""
        from pgml.schemas.grid_schema import CurveParam, ResistanceFrequencyModel

        grid = self._ynyn(
            resistance_frequency=ResistanceFrequencyModel(
                multiplier=CurveParam(frequencies_hz=[F0, 13.0 * F0], values=[1.0, 3.0])
            )
        )
        z1 = _leakage_impedance(grid, 1.0)
        z13 = _leakage_impedance(grid, 13.0)
        assert z13.real == pytest.approx(3.0 * z1.real, rel=1e-10)

    def test_curve_and_xr_constant_compose(self) -> None:
        """The two laws multiply: R(h) = R * m(h) * h."""
        from pgml.schemas.grid_schema import CurveParam, ResistanceFrequencyModel

        grid = self._ynyn(
            harmonic_xr_constant=True,
            resistance_frequency=ResistanceFrequencyModel(
                multiplier=CurveParam(frequencies_hz=[F0, 13.0 * F0], values=[1.0, 3.0])
            ),
        )
        z1 = _leakage_impedance(grid, 1.0)
        z13 = _leakage_impedance(grid, 13.0)
        assert z13.real == pytest.approx(3.0 * 13.0 * z1.real, rel=1e-10)

    def test_configured_law_overrides_the_per_element_flag(
        self, tmp_path, monkeypatch
    ) -> None:
        """`transformer.harmonic_resistance.law` forces one law on every transformer."""
        import yaml

        from pgml import defaults

        def _with_law(value: str) -> None:
            data = yaml.safe_load(yaml.safe_dump(defaults.defaults()))
            data["transformer"]["harmonic_resistance"]["law"]["value"] = value
            path = tmp_path / f"law_{value}.yaml"
            path.write_text(yaml.safe_dump(data))
            monkeypatch.setenv("PGML_DEFAULTS", str(path))
            defaults.reload(str(path))

        grid_plain = self._ynyn()  # harmonic_xr_constant = False
        grid_flag = self._ynyn(harmonic_xr_constant=True)
        try:
            _with_law("xr_constant")
            z1, z13 = (
                _leakage_impedance(grid_plain, 1.0),
                _leakage_impedance(grid_plain, 13.0),
            )
            assert z13.real == pytest.approx(13.0 * z1.real, rel=1e-10)
            _with_law("constant")
            z1, z13 = (
                _leakage_impedance(grid_flag, 1.0),
                _leakage_impedance(grid_flag, 13.0),
            )
            assert z13.real == pytest.approx(z1.real, rel=1e-10)
        finally:
            monkeypatch.delenv("PGML_DEFAULTS", raising=False)
            defaults.reload()

    def test_unknown_law_raises(self, tmp_path, monkeypatch) -> None:
        import yaml

        from pgml import defaults
        from pgml.assembly._transformer import harmonic_resistance_law

        data = yaml.safe_load(yaml.safe_dump(defaults.defaults()))
        data["transformer"]["harmonic_resistance"]["law"]["value"] = "eddy_current"
        path = tmp_path / "bad_law.yaml"
        path.write_text(yaml.safe_dump(data))
        monkeypatch.setenv("PGML_DEFAULTS", str(path))
        try:
            defaults.reload(str(path))
            with pytest.raises(ModelingError, match="harmonic_resistance.law"):
                harmonic_resistance_law()
        finally:
            monkeypatch.delenv("PGML_DEFAULTS", raising=False)
            defaults.reload()

    def test_law_applies_to_the_sequence_aware_path_too(self) -> None:
        """A unit with Z0 != Z1 scales BOTH sequence resistances with the law."""
        grid = self._ynyn(
            harmonic_xr_constant=True,
            zero_sequence=TransformerZeroSeq(r0_ohm=0.4 * R_LV, x0_ohm=0.4 * X_LV),
        )
        yb1 = assemble_network_ybus(grid, [F0], dtype=torch.complex128)
        yb13 = assemble_network_ybus(grid, [13.0 * F0], dtype=torch.complex128)
        idx = yb1.index
        lv = [idx.row(2, p) for p in ABC]
        z_seq_1 = 1.0 / _sequence_eigenvalues(yb1.Y[0].numpy()[np.ix_(lv, lv)])
        z_seq_13 = 1.0 / _sequence_eigenvalues(yb13.Y[0].numpy()[np.ix_(lv, lv)])
        for s in range(3):
            assert z_seq_13[s].real == pytest.approx(13.0 * z_seq_1[s].real, rel=1e-10)
            assert z_seq_13[s].imag == pytest.approx(13.0 * z_seq_1[s].imag, rel=1e-10)


class TestZigzagExperimentalNotice:
    """A zigzag pairing announces its experimental status, once per process."""

    def test_zigzag_logs_the_notice_once(self, caplog) -> None:
        import pgml.assembly._transformer as xfmr_mod

        monkeyed = xfmr_mod._ZIGZAG_NOTICE_LOGGED
        xfmr_mod._ZIGZAG_NOTICE_LOGGED = False
        try:
            grid = _grid(
                WindingConnection.WYE, WindingConnection.ZIGZAG_GROUNDED, 150.0
            )
            with caplog.at_level("WARNING"):
                assemble_network_ybus(grid, [F0], dtype=torch.complex128)
                assemble_network_ybus(grid, [F0], dtype=torch.complex128)
            notices = [r for r in caplog.records if "EXPERIMENTAL zigzag" in r.message]
            assert len(notices) == 1
        finally:
            xfmr_mod._ZIGZAG_NOTICE_LOGGED = monkeyed

    def test_non_zigzag_pairing_is_silent(self, caplog) -> None:
        import pgml.assembly._transformer as xfmr_mod

        monkeyed = xfmr_mod._ZIGZAG_NOTICE_LOGGED
        xfmr_mod._ZIGZAG_NOTICE_LOGGED = False
        try:
            grid = _grid(WindingConnection.DELTA, WindingConnection.WYE_GROUNDED, 30.0)
            with caplog.at_level("WARNING"):
                assemble_network_ybus(grid, [F0], dtype=torch.complex128)
            assert not [r for r in caplog.records if "EXPERIMENTAL zigzag" in r.message]
        finally:
            xfmr_mod._ZIGZAG_NOTICE_LOGGED = monkeyed
