"""Parity tests: CIGRE LV harmonic flow vs LIVE OpenDSS oracle.

Tests :func:`pgml.evaluation.references.opendss_harmonic_voltages` — the live
OpenDSS harmonic oracle that reads OpenDSS ``SystemY(h)`` at each harmonic order
and solves the system.

Two paths are exercised:

Single-phase geometry path
    :func:`pgml.geometry.synthesis.synthesize_grid_geometry` attaches single-conductor
    Carson ``conductor_geometry`` to all 37 lines.  OpenDSS builds a circuit from the
    same ``WireData`` / ``LineGeometry`` data and applies its own Carson/Deri earth-
    return correction.  pgml's ``geometry`` path replicates this correction
    bit-exactly, so the parity is near machine precision (~1e-11 V absolute at
    harmonics 5 and 11).

Three-phase sequence-aware path
    :func:`pgml.geometry.synthesis.apply_default_harmonic_model` tags all lines with
    ``harmonic_line_model=sequence_aware``.  A stub OpenDSS circuit with R1/X1/R0/X0
    lines is built (no native Transformer elements, which would create neutral nodes
    incompatible with pgml's flat node ordering); OpenDSS applies its own Carson
    correction to the R1/X1 lines.  Switches and transformers are stamped with
    pgml-exact formulas.  Because the ``sequence_aware`` earth-return model and
    OpenDSS's Carson/Deri model for R1/X1 lines use the same underlying correction,
    parity is again near machine precision (~1e-8 V absolute).

Stub-subtract technique
    For both paths the OpenDSS circuit contains a near-zero-impedance stub Vsource
    (r1=1e-6, x1=1e-6).  Its actual Carson-corrected Norton admittance is read from
    ``Vsource.Source.YPrim`` and subtracted from the SystemY diagonal; then the
    pgml-exact source Norton is added back.  This cleanly separates the line
    contribution (captured by OpenDSS) from the source contribution (stamped by pgml).

Tolerances
----------
- Single-phase geometry: ``atol = 1e-9 V`` (empirically ~2e-12 V; tight enough to
  catch any alignment or formula error).
- Three-phase sequence-aware: ``atol = 1e-6 V`` (empirically ~1.6e-8 V; tighter than
  5-15 % physics gap from a naive OpenDSS comparison).
- Fundamental (order 1): returned verbatim from ``v1`` — bit-for-bit equal.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# numpy 2.x compat for pandapower 2.14
np.Inf = np.inf  # type: ignore[attr-defined]
np.in1d = np.isin  # type: ignore[attr-defined]

from pgml.assembly import node_phase_index  # noqa: E402
from pgml.convert.pandapower import PhaseMode  # noqa: E402
from pgml.evaluation.references import (  # noqa: E402
    cigre_lv_full_grid,
    opendss_dyn_transformer_harmonic_voltages,
    opendss_harmonic_voltages,
)
from pgml.geometry.synthesis import (  # noqa: E402
    apply_default_harmonic_model,
    synthesize_grid_geometry,
)
from pgml.schemas.grid_schema import Load, Phase  # noqa: E402
from pgml.solver import solve_harmonic_flow  # noqa: E402

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

ORDERS = [1, 5, 11]

# 6-pulse-converter-like spectrum: fundamental + 5th + 11th.
SPECTRUM = {o: (mag, 0.0) for o, mag in [(1, 1.0), (5, 0.20), (11, 0.09)]}

# Injection nodes on DIFFERENT feeders to exercise cross-feeder spread.
INJECTION_NODES = [3, 25]

# Tolerance for live oracle vs pgml (geometry path — near machine precision).
ATOL_V_GEOMETRY = 1e-9  # empirically ~2e-12 V

# Tolerance for live oracle vs pgml (sequence-aware path — near machine precision).
ATOL_V_SEQ_AWARE = 1e-6  # empirically ~1.6e-8 V


def _build_injection(grid, nodes: list[int]) -> dict:
    """Build a ``harmonic_injection`` dict for loads on the given nodes."""
    loads = [a for a in grid.appliances if isinstance(a, Load)]
    inj_loads = [ld for ld in loads if ld.node in nodes]
    assert inj_loads, f"No loads found at nodes {nodes}"
    return {ld.id: SPECTRUM for ld in inj_loads}


# ---------------------------------------------------------------------------
# Single-phase geometry path tests
# ---------------------------------------------------------------------------


class TestCigreLvLiveOracleSinglePhaseGeometry:
    """Live OpenDSS parity: SINGLE_PHASE_EQUIV + synthesized Carson geometry.

    pgml's geometry path uses Carson/Deri line constants bit-exactly equal to
    OpenDSS, so the live oracle matches pgml to near machine precision.
    """

    PHASE_MODE = PhaseMode.SINGLE_PHASE_EQUIV

    def _solve_and_compare(self, orders=None):
        """Build geometry grid, solve pgml, run live oracle, return arrays."""
        if orders is None:
            orders = ORDERS
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        synthesize_grid_geometry(grid)

        harmonic_injection = _build_injection(grid, INJECTION_NODES)

        hres = solve_harmonic_flow(
            grid,
            orders,
            slack="norton",
            harmonic_injection=harmonic_injection,
            dtype=torch.complex128,
        )
        assert hres.pf.converged, (
            f"solve_harmonic_flow did not converge "
            f"(residual={float(hres.pf.residual):.3e})"
        )

        v1_np = hres.pf.v.detach().cpu().numpy()
        v_oracle = opendss_harmonic_voltages(
            grid,
            harmonic_injection,
            orders,
            slack="norton",
            v1=v1_np,
        )
        v_pgml = hres.v.detach().cpu().numpy()
        return v_pgml, v_oracle, orders

    def test_fundamental_exact_match(self) -> None:
        """Order 1 returned verbatim from v1 — bit-for-bit identical."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        h1_k = orders.index(1)
        np.testing.assert_array_equal(
            v_oracle[h1_k],
            v_pgml[h1_k],
            err_msg="Geometry path order 1: oracle and pgml must be bit-for-bit identical",
        )

    def test_harmonic_5_tight_parity(self) -> None:
        """5th harmonic (geometry): live oracle matches pgml to < 1e-7 V."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        h5_k = orders.index(5)
        np.testing.assert_allclose(
            v_oracle[h5_k],
            v_pgml[h5_k],
            atol=ATOL_V_GEOMETRY,
            rtol=ATOL_V_GEOMETRY,
            err_msg="Geometry path h=5: live oracle vs pgml voltage mismatch",
        )

    def test_harmonic_11_tight_parity(self) -> None:
        """11th harmonic (geometry): live oracle matches pgml to < 1e-7 V."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        h11_k = orders.index(11)
        np.testing.assert_allclose(
            v_oracle[h11_k],
            v_pgml[h11_k],
            atol=ATOL_V_GEOMETRY,
            rtol=ATOL_V_GEOMETRY,
            err_msg="Geometry path h=11: live oracle vs pgml voltage mismatch",
        )

    def test_all_orders_allclose(self) -> None:
        """All orders (geometry): max absolute deviation < 1e-7 V."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        max_err = float(np.abs(v_oracle - v_pgml).max())
        assert max_err < ATOL_V_GEOMETRY, (
            f"Geometry live oracle max |DeltaV| = {max_err:.3e} V exceeds "
            f"atol={ATOL_V_GEOMETRY:.0e} V (orders={orders})"
        )

    def test_output_shape(self) -> None:
        """Geometry live oracle returns [H, 44] complex array."""
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        synthesize_grid_geometry(grid)
        harmonic_injection = _build_injection(grid, INJECTION_NODES)
        hres = solve_harmonic_flow(
            grid,
            ORDERS,
            slack="norton",
            harmonic_injection=harmonic_injection,
            dtype=torch.complex128,
        )
        v1_np = hres.pf.v.detach().cpu().numpy()
        index = node_phase_index(grid)
        v_oracle = opendss_harmonic_voltages(
            grid, harmonic_injection, ORDERS, slack="norton", v1=v1_np
        )
        assert v_oracle.shape == (len(ORDERS), index.size), (
            f"Geometry oracle shape {v_oracle.shape} != expected "
            f"({len(ORDERS)}, {index.size})"
        )
        assert np.iscomplexobj(v_oracle), "Oracle must return a complex array"


# ---------------------------------------------------------------------------
# Three-phase sequence-aware path tests
# ---------------------------------------------------------------------------


class TestCigreLvLiveOracleThreePhaseSeqAware:
    """Live OpenDSS parity: THREE_PHASE + sequence-aware harmonic model.

    This validates only the LINE harmonic model against OpenDSS.  OpenDSS builds
    R1/X1/R0/X0 lines from the 3x3 phase matrices and applies its own Carson/Deri
    corrections at harmonics; the ``sequence_aware`` pgml model uses the same
    earth-return earth-resistance mechanism, giving near-machine-precision parity
    (~1e-8 V) on the lines.

    The transformer is NOT validated against OpenDSS here: switches and
    transformers are stamped with pgml's OWN formulas (no OpenDSS Transformer
    elements, which would create neutral nodes incompatible with pgml's flat row
    ordering).  For the genuine OpenDSS transformer/vector-group oracle (real
    OpenDSS ``Transformer`` element, delta/wye zero-sequence blocking) see
    :class:`TestCigreLvDynTransformerOracle` below.

    Empirically achieved tolerances: h=5 ~ 1.6e-8 V, h=11 ~ 1e-8 V.
    """

    PHASE_MODE = PhaseMode.THREE_PHASE

    def _solve_and_compare(self, orders=None):
        """Build 3-phase seq-aware grid, solve pgml, run live oracle."""
        if orders is None:
            orders = ORDERS
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        apply_default_harmonic_model(grid)

        harmonic_injection = _build_injection(grid, INJECTION_NODES)

        hres = solve_harmonic_flow(
            grid,
            orders,
            slack="norton",
            harmonic_injection=harmonic_injection,
            dtype=torch.complex128,
        )
        assert hres.pf.converged, (
            f"THREE_PHASE solve_harmonic_flow did not converge "
            f"(residual={float(hres.pf.residual):.3e})"
        )

        v1_np = hres.pf.v.detach().cpu().numpy()
        v_oracle = opendss_harmonic_voltages(
            grid,
            harmonic_injection,
            orders,
            slack="norton",
            v1=v1_np,
        )
        v_pgml = hres.v.detach().cpu().numpy()
        return v_pgml, v_oracle, orders

    def test_fundamental_exact_match(self) -> None:
        """Order 1 returned verbatim from v1 — bit-for-bit identical."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        h1_k = orders.index(1)
        np.testing.assert_array_equal(
            v_oracle[h1_k],
            v_pgml[h1_k],
            err_msg="Seq-aware order 1: oracle and pgml must be bit-for-bit identical",
        )

    def test_harmonic_5_parity(self) -> None:
        """5th harmonic (seq-aware): live oracle matches pgml to < 1e-5 V.

        The sequence-aware earth-return model and OpenDSS's Carson/Deri model
        for R1/X1 lines yield near-identical corrections; switches and transformers
        are stamped identically.  Empirically: max |DeltaV| ~ 2e-8 V.
        """
        v_pgml, v_oracle, orders = self._solve_and_compare()
        h5_k = orders.index(5)
        np.testing.assert_allclose(
            v_oracle[h5_k],
            v_pgml[h5_k],
            atol=ATOL_V_SEQ_AWARE,
            rtol=ATOL_V_SEQ_AWARE,
            err_msg="Seq-aware h=5: live oracle vs pgml voltage mismatch",
        )

    def test_harmonic_11_parity(self) -> None:
        """11th harmonic (seq-aware): live oracle matches pgml to < 1e-5 V."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        h11_k = orders.index(11)
        np.testing.assert_allclose(
            v_oracle[h11_k],
            v_pgml[h11_k],
            atol=ATOL_V_SEQ_AWARE,
            rtol=ATOL_V_SEQ_AWARE,
            err_msg="Seq-aware h=11: live oracle vs pgml voltage mismatch",
        )

    def test_all_orders_allclose(self) -> None:
        """All orders (seq-aware): max absolute deviation < 1e-5 V."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        max_err = float(np.abs(v_oracle - v_pgml).max())
        assert max_err < ATOL_V_SEQ_AWARE, (
            f"Seq-aware live oracle max |DeltaV| = {max_err:.3e} V exceeds "
            f"atol={ATOL_V_SEQ_AWARE:.0e} V (orders={orders})"
        )

    def test_output_shape_three_phase(self) -> None:
        """THREE_PHASE seq-aware oracle returns [H, 132] complex array."""
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        apply_default_harmonic_model(grid)
        harmonic_injection = _build_injection(grid, INJECTION_NODES)
        hres = solve_harmonic_flow(
            grid,
            ORDERS,
            slack="norton",
            harmonic_injection=harmonic_injection,
            dtype=torch.complex128,
        )
        v1_np = hres.pf.v.detach().cpu().numpy()
        index = node_phase_index(grid)
        v_oracle = opendss_harmonic_voltages(
            grid, harmonic_injection, ORDERS, slack="norton", v1=v1_np
        )
        assert v_oracle.shape == (len(ORDERS), index.size), (
            f"Seq-aware oracle shape {v_oracle.shape} != expected "
            f"({len(ORDERS)}, {index.size})"
        )
        assert np.iscomplexobj(v_oracle), "Oracle must return a complex array"

    def test_invalid_slack_raises(self) -> None:
        """Non-norton slack raises ValueError."""
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        apply_default_harmonic_model(grid)
        with pytest.raises(ValueError, match="norton"):
            opendss_harmonic_voltages(grid, None, ORDERS, slack="thevenin")

    def test_plain_grid_raises(self) -> None:
        """Plain R/X grid (no geometry, no seq-aware tags) raises ValueError."""
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        # No synthesize_grid_geometry or apply_default_harmonic_model called
        with pytest.raises(ValueError, match="conductor_geometry|sequence_aware"):
            opendss_harmonic_voltages(grid, None, ORDERS)


# ---------------------------------------------------------------------------
# True live-OpenDSS Dyn transformer oracle (vector-group validation)
# ---------------------------------------------------------------------------

# Tolerance for this oracle vs pgml at non-triplen orders: the dominant residual
# is the Carson line-model gap (~16 % relative at h=3 on short LV lines, and
# ~1e-8 V on LV lines at h=5/11).  For non-triplen we use 1 V as a generous
# bound that catches transformer formula bugs while remaining robust to the
# line gap.
ATOL_V_DYN_NON_TRIPLEN = 1.0  # V — captures the seq-aware line Carson gap

# For triplen orders (h=3, h=9) the test primarily validates that the delta
# winding BLOCKS zero-sequence current at the MV bus.  The LV-side triplen
# voltage is non-zero (from the injection), but the MV bus must remain at
# effectively zero regardless of which model is used.
TRIPLEN_MV_ATOL_V = 1e-6  # V — both pgml and OpenDSS should agree MV ≈ 0

# Inclusion spectrum (with triplen orders for delta blocking validation)
TRIPLEN_ORDERS = [1, 3, 5, 9, 11]
TRIPLEN_SPECTRUM = {
    o: (mag, 0.0) for o, mag in [(1, 1.0), (3, 0.30), (5, 0.20), (9, 0.10), (11, 0.09)]
}


class TestCigreLvDynTransformerOracle:
    """Vector-group validation: true OpenDSS Dyn1 transformer vs pgml assembly.

    Uses :func:`opendss_dyn_transformer_harmonic_voltages` which includes the REAL
    OpenDSS ``Transformer`` element (``conn=delta``, ``conn=wye``, ``LeadLag=Lag``
    for the CIGRE LV Dyn1 transformers), unlike :func:`opendss_harmonic_voltages`
    which stamps transformers using pgml's formula.

    The primary assertion is triplen zero-sequence blocking:

    - At h=3 and h=9, injection at LV nodes 3 and 25 creates significant LV
      harmonic voltages (~2 V).
    - The delta HV winding traps these currents inside the delta loop; the MV bus
      (slack node 1 and the three HV transformer nodes 2, 21, 24) should carry
      essentially zero h=3 / h=9 voltage (< 1e-6 V), for BOTH pgml and OpenDSS.
    - Both models must agree on this property: the triplen voltages at the MV bus
      must be < ``TRIPLEN_MV_ATOL_V = 1e-6 V`` in BOTH pgml and the oracle.

    Non-triplen parity:

    - At h=5 and h=11 the residual is the Carson line-model gap (~16 % relative on
      short LV feeders, ~1e-8 V on medium-length LV cables).  The tolerance
      ``ATOL_V_DYN_NON_TRIPLEN = 1 V`` covers this physical gap while remaining
      tight enough to catch transformer formula regressions.
    """

    PHASE_MODE = PhaseMode.THREE_PHASE

    def _solve(self, orders=None):
        """Build seq-aware 3-phase grid, solve pgml + true-Dyn oracle."""
        if orders is None:
            orders = TRIPLEN_ORDERS
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        apply_default_harmonic_model(grid)

        loads = [a for a in grid.appliances if isinstance(a, Load)]
        inj_loads = [ld for ld in loads if ld.node in INJECTION_NODES]
        harmonic_injection = {ld.id: TRIPLEN_SPECTRUM for ld in inj_loads}

        hres = solve_harmonic_flow(
            grid,
            orders,
            slack="norton",
            harmonic_injection=harmonic_injection,
            dtype=torch.complex128,
        )
        assert hres.pf.converged, (
            f"THREE_PHASE solve_harmonic_flow did not converge "
            f"(residual={float(hres.pf.residual):.3e})"
        )

        v1_np = hres.pf.v.detach().cpu().numpy()
        v_oracle = opendss_dyn_transformer_harmonic_voltages(
            grid,
            harmonic_injection,
            orders,
            slack="norton",
            v1=v1_np,
        )
        v_pgml = hres.v.detach().cpu().numpy()
        return v_pgml, v_oracle, orders, grid

    def test_fundamental_exact_match(self) -> None:
        """Order 1 returned verbatim from v1 — bit-for-bit identical."""
        v_pgml, v_oracle, orders, _ = self._solve()
        k1 = orders.index(1)
        np.testing.assert_array_equal(
            v_oracle[k1],
            v_pgml[k1],
            err_msg="Dyn oracle order 1: must be bit-for-bit identical",
        )

    def test_triplen_zero_sequence_blocked_at_mv_pgml(self) -> None:
        """pgml: triplen orders h=3 and h=9 are blocked at the MV bus.

        The delta HV winding traps zero-sequence currents; both injecting-feeder
        HV nodes and the slack MV node must carry < 1e-6 V at triplen orders.
        """
        v_pgml, _, orders, grid = self._solve()
        index = node_phase_index(grid)

        from pgml.schemas.grid_schema import Transformer

        # HV nodes of all three transformers + slack
        mv_nodes: set[int] = {
            int(src.node) for src in grid.appliances if hasattr(src, "u_ref_v")
        }
        for b in grid.branches:
            if isinstance(b, Transformer):
                mv_nodes.add(int(b.from_node))

        for h_ord in [3, 9]:
            if h_ord not in orders:
                continue
            h_k = orders.index(h_ord)
            for nid in mv_nodes:
                for phase in [Phase.A, Phase.B, Phase.C]:
                    try:
                        r = index.row(nid, phase)
                    except (KeyError, ValueError):
                        continue
                    v_mv = abs(v_pgml[h_k, r])
                    assert v_mv < TRIPLEN_MV_ATOL_V, (
                        f"pgml h={h_ord} node {nid} {phase}: |V| = {v_mv:.3e} V "
                        f"exceeds {TRIPLEN_MV_ATOL_V:.0e} V — delta should block triplen"
                    )

    def test_triplen_zero_sequence_blocked_at_mv_oracle(self) -> None:
        """OpenDSS oracle: triplen h=3 and h=9 are blocked at the MV bus.

        Identical assertion using the true OpenDSS Dyn1 transformer model,
        confirming that OpenDSS's delta winding also blocks zero-sequence to
        the MV side.
        """
        _, v_oracle, orders, grid = self._solve()
        index = node_phase_index(grid)

        from pgml.schemas.grid_schema import Transformer

        mv_nodes: set[int] = {
            int(src.node) for src in grid.appliances if hasattr(src, "u_ref_v")
        }
        for b in grid.branches:
            if isinstance(b, Transformer):
                mv_nodes.add(int(b.from_node))

        for h_ord in [3, 9]:
            if h_ord not in orders:
                continue
            h_k = orders.index(h_ord)
            for nid in mv_nodes:
                for phase in [Phase.A, Phase.B, Phase.C]:
                    try:
                        r = index.row(nid, phase)
                    except (KeyError, ValueError):
                        continue
                    v_mv = abs(v_oracle[h_k, r])
                    assert v_mv < TRIPLEN_MV_ATOL_V, (
                        f"OpenDSS oracle h={h_ord} node {nid} {phase}: |V| = {v_mv:.3e} V "
                        f"exceeds {TRIPLEN_MV_ATOL_V:.0e} V — DSS delta should block triplen"
                    )

    def test_triplen_lv_injection_nonzero(self) -> None:
        """h=3 voltages at directly injected LV transformer busbars are non-trivial.

        Confirms the triplen current is flowing on the LV side (the injection
        is active), and that blocking is specifically at the transformer delta.
        The LV busbars (nodes 3 and 25) should carry ~2 V at h=3.
        """
        v_pgml, _, orders, grid = self._solve()
        index = node_phase_index(grid)
        h3_k = orders.index(3)
        for node_id in INJECTION_NODES:
            r = index.row(node_id, Phase.A)
            v_lv = abs(v_pgml[h3_k, r])
            assert v_lv > 0.1, (
                f"pgml h=3 node {node_id} (injected LV busbar): |V| = {v_lv:.4e} V "
                f"too small — triplen injection may not be active"
            )

    def test_non_triplen_parity(self) -> None:
        """Non-triplen orders h=5, h=11: oracle vs pgml within Carson-gap tolerance.

        The residual (~16 % relative at h=3 on short LV lines) is the seq-aware
        line-model Carson gap documented in the seq-aware parity tests.  The
        tolerance ``ATOL_V_DYN_NON_TRIPLEN = 1 V`` is generous but ensures
        that transformer formula regressions (which would cause ~kV errors as in
        the pre-fix state) are caught.
        """
        v_pgml, v_oracle, orders, _ = self._solve()
        for h_ord in [5, 11]:
            if h_ord not in orders:
                continue
            h_k = orders.index(h_ord)
            err = float(np.abs(v_oracle[h_k] - v_pgml[h_k]).max())
            assert err < ATOL_V_DYN_NON_TRIPLEN, (
                f"True Dyn oracle h={h_ord}: max |DeltaV| = {err:.3e} V "
                f"exceeds {ATOL_V_DYN_NON_TRIPLEN:.0e} V (Carson-gap tolerance)"
            )

    def test_output_shape(self) -> None:
        """True Dyn oracle returns [H, 132] complex array (44 nodes × 3 phases)."""
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        apply_default_harmonic_model(grid)
        loads = [a for a in grid.appliances if isinstance(a, Load)]
        inj_loads = [ld for ld in loads if ld.node in INJECTION_NODES]
        harmonic_injection = {ld.id: TRIPLEN_SPECTRUM for ld in inj_loads}
        hres = solve_harmonic_flow(
            grid,
            TRIPLEN_ORDERS,
            slack="norton",
            harmonic_injection=harmonic_injection,
            dtype=torch.complex128,
        )
        v1_np = hres.pf.v.detach().cpu().numpy()
        index = node_phase_index(grid)
        v_oracle = opendss_dyn_transformer_harmonic_voltages(
            grid, harmonic_injection, TRIPLEN_ORDERS, slack="norton", v1=v1_np
        )
        assert v_oracle.shape == (len(TRIPLEN_ORDERS), index.size), (
            f"Dyn oracle shape {v_oracle.shape} != ({len(TRIPLEN_ORDERS)}, {index.size})"
        )
        assert np.iscomplexobj(v_oracle)


# ---------------------------------------------------------------------------
# Three-phase Carson geometry oracle (apples-to-apples: same geometry in both engines)
# ---------------------------------------------------------------------------

# Tolerance for the geometry-oracle path (both non-triplen and triplen orders).
# The identical synthesized 3-conductor geometry is fed to both pgml (via
# conductor_geometry) and OpenDSS (via WireData / LineGeometry commands), so both
# engines run Carson/Deri on the same positions — the only residual comes from
# floating-point rounding in the respective implementations.
ATOL_V_3PH_GEOMETRY = 1e-9  # V — empirically ~1e-12; exact Carson parity

# Triplen/zero-sequence orders included to verify the h3/h9 gap collapses to
# numerical noise once both engines share the same 3-conductor geometry.
TRIPLEN_ORDERS_3PH = [1, 3, 5, 9, 11]
TRIPLEN_SPECTRUM_3PH = {
    o: (mag, 0.0) for o, mag in [(1, 1.0), (3, 0.30), (5, 0.20), (9, 0.10), (11, 0.09)]
}


class TestCigreLvThreePhaseCarsonGeometryOracle:
    """Bit-exact parity: THREE_PHASE + synthesized 3-conductor Carson geometry.

    Both pgml and the live OpenDSS oracle use the SAME synthesized equilateral
    3-conductor ``conductor_geometry`` (via ``synthesize_grid_geometry``).  OpenDSS
    receives the conductor positions as ``WireData`` / ``LineGeometry`` elements and
    runs its own Carson/Deri calculation; pgml uses the same positions in
    ``_stamp_geometry_lines``.  Because the Carson implementations are bit-exact
    (validated in ``tests/reference/test_carson_opendss.py``), the resulting
    harmonic voltages must agree to floating-point noise across ALL orders including
    the triplen / zero-sequence orders (h=3, h=9) which previously showed a
    ~60–180 % gap with the ``sequence_aware`` R1/X1 model.

    Switches and transformers are stamped with pgml-exact formulas (the geometry
    stub contains lines only; the stub-subtract technique removes the OpenDSS stub
    source Norton and replaces it with the pgml-exact Norton).

    Expected parity: < ``ATOL_V_3PH_GEOMETRY = 1e-5 V`` at all orders; the
    empirical residual is dominated by floating-point accumulation in the
    Carson computation and should be O(1e-8)–O(1e-6) V.
    """

    PHASE_MODE = PhaseMode.THREE_PHASE

    def _solve_and_compare(self, orders=None):
        """Build 3-phase geometry grid, solve pgml, run live oracle, return arrays."""
        import warnings

        if orders is None:
            orders = TRIPLEN_ORDERS_3PH
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            synthesize_grid_geometry(grid)

        loads = [a for a in grid.appliances if isinstance(a, Load)]
        inj_loads = [ld for ld in loads if ld.node in INJECTION_NODES]
        harmonic_injection = {ld.id: TRIPLEN_SPECTRUM_3PH for ld in inj_loads}
        assert inj_loads, f"No loads found at injection nodes {INJECTION_NODES}"

        hres = solve_harmonic_flow(
            grid,
            orders,
            slack="norton",
            harmonic_injection=harmonic_injection,
            dtype=torch.complex128,
        )
        assert hres.pf.converged, (
            f"THREE_PHASE geometry solve_harmonic_flow did not converge "
            f"(residual={float(hres.pf.residual):.3e})"
        )

        v1_np = hres.pf.v.detach().cpu().numpy()
        v_oracle = opendss_harmonic_voltages(
            grid,
            harmonic_injection,
            orders,
            slack="norton",
            v1=v1_np,
        )
        v_pgml = hres.v.detach().cpu().numpy()
        return v_pgml, v_oracle, orders, grid

    def test_fundamental_exact_match(self) -> None:
        """Order 1 returned verbatim from v1 — bit-for-bit identical."""
        v_pgml, v_oracle, orders, _ = self._solve_and_compare()
        h1_k = orders.index(1)
        np.testing.assert_array_equal(
            v_oracle[h1_k],
            v_pgml[h1_k],
            err_msg="3ph geometry oracle order 1: must be bit-for-bit identical",
        )

    def test_non_triplen_tight_parity(self) -> None:
        """Non-triplen orders h=5, h=11: live oracle matches pgml to < 1e-5 V.

        Both engines share the identical Carson geometry, so the discrepancy is
        floating-point noise (empirically O(1e-8) V), not a model-difference gap.
        """
        v_pgml, v_oracle, orders, _ = self._solve_and_compare()
        for h_ord in [5, 11]:
            if h_ord not in orders:
                continue
            h_k = orders.index(h_ord)
            err = float(np.abs(v_oracle[h_k] - v_pgml[h_k]).max())
            assert err < ATOL_V_3PH_GEOMETRY, (
                f"3ph geometry oracle h={h_ord}: max |DeltaV| = {err:.3e} V "
                f"exceeds {ATOL_V_3PH_GEOMETRY:.0e} V (expected numerical-noise-level parity)"
            )

    def test_triplen_tight_parity(self) -> None:
        """Triplen orders h=3, h=9: live oracle matches pgml to < 1e-5 V.

        This is the key validation: the h3/h9 gap present with the sequence-aware
        model (~60–180 %) collapses to numerical noise because both engines now share
        the same 3-conductor geometry (zero-sequence current sees the same Carson
        earth return in both pgml and OpenDSS).
        """
        v_pgml, v_oracle, orders, _ = self._solve_and_compare()
        for h_ord in [3, 9]:
            if h_ord not in orders:
                continue
            h_k = orders.index(h_ord)
            err = float(np.abs(v_oracle[h_k] - v_pgml[h_k]).max())
            assert err < ATOL_V_3PH_GEOMETRY, (
                f"3ph geometry oracle h={h_ord}: max |DeltaV| = {err:.3e} V "
                f"exceeds {ATOL_V_3PH_GEOMETRY:.0e} V — triplen gap should collapse "
                f"to numerical noise with shared 3-conductor geometry"
            )

    def test_all_orders_allclose(self) -> None:
        """All harmonic orders: max absolute deviation < 1e-5 V."""
        v_pgml, v_oracle, orders, _ = self._solve_and_compare()
        per_order_max = {}
        for k, h in enumerate(orders):
            per_order_max[h] = float(np.abs(v_oracle[k] - v_pgml[k]).max())
        max_err = max(per_order_max.values())
        assert max_err < ATOL_V_3PH_GEOMETRY, (
            f"3ph geometry oracle max |DeltaV| = {max_err:.3e} V exceeds "
            f"atol={ATOL_V_3PH_GEOMETRY:.0e} V\nPer-order: {per_order_max}"
        )

    def test_triplen_lv_voltages_nonzero(self) -> None:
        """h=3 LV voltages at injected nodes are non-trivial (injection is active).

        Confirms the triplen current is present on the LV side — the test is not
        trivially passing because no injection occurred.
        """
        v_pgml, _, orders, grid = self._solve_and_compare()
        index = node_phase_index(grid)
        h3_k = orders.index(3)
        for node_id in INJECTION_NODES:
            r = index.row(node_id, Phase.A)
            v_lv = abs(v_pgml[h3_k, r])
            assert v_lv > 0.1, (
                f"pgml 3ph geometry h=3 node {node_id}: |V| = {v_lv:.4e} V "
                f"— triplen injection appears inactive"
            )

    def test_output_shape(self) -> None:
        """Oracle returns [H, 132] complex array (44 nodes x 3 phases)."""
        import warnings

        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            synthesize_grid_geometry(grid)
        loads = [a for a in grid.appliances if isinstance(a, Load)]
        inj_loads = [ld for ld in loads if ld.node in INJECTION_NODES]
        harmonic_injection = {ld.id: TRIPLEN_SPECTRUM_3PH for ld in inj_loads}
        hres = solve_harmonic_flow(
            grid,
            TRIPLEN_ORDERS_3PH,
            slack="norton",
            harmonic_injection=harmonic_injection,
            dtype=torch.complex128,
        )
        v1_np = hres.pf.v.detach().cpu().numpy()
        index = node_phase_index(grid)
        v_oracle = opendss_harmonic_voltages(
            grid,
            harmonic_injection,
            TRIPLEN_ORDERS_3PH,
            slack="norton",
            v1=v1_np,
        )
        assert v_oracle.shape == (len(TRIPLEN_ORDERS_3PH), index.size), (
            f"3ph geometry oracle shape {v_oracle.shape} != expected "
            f"({len(TRIPLEN_ORDERS_3PH)}, {index.size})"
        )
        assert np.iscomplexobj(v_oracle)
