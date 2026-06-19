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
- Single-phase geometry: ``atol = 1e-7 V`` (empirically ~1e-11 V; tight enough to
  catch any alignment or formula error).
- Three-phase sequence-aware: ``atol = 1e-5 V`` (empirically ~1e-8 V; tighter than
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
    opendss_harmonic_voltages,
)
from pgml.geometry.synthesis import apply_default_harmonic_model, synthesize_grid_geometry  # noqa: E402
from pgml.schemas.grid_schema import Load  # noqa: E402
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
ATOL_V_GEOMETRY = 1e-7  # empirically ~1e-11 V

# Tolerance for live oracle vs pgml (sequence-aware path — near machine precision).
ATOL_V_SEQ_AWARE = 1e-5  # empirically ~1e-8 V


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

    OpenDSS builds R1/X1/R0/X0 lines from the 3x3 phase matrices and applies
    its own Carson/Deri corrections at harmonics.  The ``sequence_aware`` pgml
    model uses the same earth-return earth-resistance mechanism, giving near-
    machine-precision parity (~1e-8 V) when switches and transformers are stamped
    with pgml-exact formulas (no OpenDSS Transformer elements, which would create
    neutral nodes incompatible with pgml's flat row ordering).

    Empirically achieved tolerances: h=5 ~ 2e-8 V, h=11 ~ 1e-8 V.
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
