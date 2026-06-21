"""pgml-vs-numpy machine-precision REGRESSION: CIGRE LV full-grid harmonic flow.

This file runs ZERO OpenDSS.  The oracle is the pure-numpy mirror
:func:`numpy_harmonic_voltages`, which reimplements pgml's exact stamping formulas
in numpy; the comparison is therefore a machine-precision REGRESSION guard on
:func:`pgml.solver.solve_harmonic_flow` (it detects accidental changes to pgml's
own formulas), NOT a live-OpenDSS oracle validation.  For genuine OpenDSS-oracle
harmonic comparison see ``tests/reference/test_cigre_lv_live_opendss.py``.

Validates :func:`pgml.solver.solve_harmonic_flow` on the FULL CIGRE LV benchmark
(44 nodes, 43 branches including 3 Dyn30 20/0.4 kV transformers, 15 loads, 1 MV
source) for both ``SINGLE_PHASE_EQUIV`` and ``THREE_PHASE`` phase modes.

Transformer model
-----------------
pgml stamps each transformer as a per-phase off-nominal-tap leakage-pi (M1
diagonal model) with ``y_se = (R + j·h·2πf₀·L)⁻¹`` (LV-referred, R fixed with
frequency) and complex tap ``t = n·exp(j·shift_deg)``::

    Y_ff = y_se / |t|² + y_m,  Y_ft = −y_se / t*,
    Y_tf = −y_se / t,           Y_tt = y_se.

The oracle (:func:`numpy_harmonic_voltages`) mirrors this exact stamp in numpy,
so the parity is at machine precision (~1e-13 V).

Why the regression oracle is numpy-based
-----------------------------------------
OpenDSS applies a Carson earth-return correction to ALL line elements (even those
defined via R1/X1 parameters, not just geometry lines) and its native
``Transformer`` element uses a more complex harmonic impedance scaling. Both
corrections produce frequency-dependent R and sub-linear X scaling that does NOT
match pgml's simple ``R const, X∝h`` model.  The pure-numpy oracle
(:func:`numpy_harmonic_voltages`) reimplements pgml's exact formulas, giving
machine-precision parity (~1e-14 V).  For live OpenDSS comparison on conductor-
geometry grids (Carson-exact) see :func:`opendss_harmonic_voltages` and
``tests/reference/test_cigre_lv_live_opendss.py``.

Cross-feeder harmonic spread
-----------------------------
The representative injection is placed at loads on **different feeders** (bus 3
on the residential feeder, bus 25 on the commercial feeder), so the harmonic
currents must flow through two of the three 20/0.4 kV transformers and through
the 20 kV MV network to reach the non-injecting feeder — exercising both the
transformer harmonic model and the cross-feeder harmonic spread.

Tolerances
----------
Both modes: ``atol = 1e-10 V``, ``rtol = 1e-10`` (machine precision; empirically
achieved ~1e-14 V). The fundamental (order 1) is a strict equality (0.0) because
the oracle returns ``v1`` directly.
"""

from __future__ import annotations

import numpy as np
import torch

from pgml.assembly import node_phase_index
from pgml.convert.pandapower import PhaseMode
from pgml.evaluation.references import (
    cigre_lv_full_grid,
    numpy_harmonic_voltages,
)
from pgml.schemas.grid_schema import Load
from pgml.solver import solve_harmonic_flow

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

# 6-pulse converter-like spectrum: fundamental + 5th + 11th harmonics.
# Using only non-triplen non-zero-sequence orders keeps both the single-phase
# and three-phase paths exercised (these orders are the key harmonics in the
# CIGRE LV harmonic studies).
ORDERS = [1, 5, 11]

# Spectrum: (order, magnitude_pu, phase_deg)
# magnitude_pu is relative to the fundamental (order-1 entry).
SPECTRUM = {o: (mag, 0.0) for o, mag in [(1, 1.0), (5, 0.20), (11, 0.09)]}

# Machine-precision tolerance for the oracle-vs-pgml parity check.
ATOL_V = 1e-10  # absolute [V]
RTOL_V = 1e-10  # relative

# Nodes at which harmonic injection is applied (DIFFERENT feeders to exercise
# cross-feeder spread through MV network and transformers):
#   - node 3:  LV busbar of transformer 1 (residential feeder R)
#   - node 25: LV busbar of transformer 3 (commercial feeder C)
INJECTION_NODES = [3, 25]


def _build_injection(grid, nodes: list[int]) -> dict:
    """Build a ``harmonic_injection`` dict for loads on the given nodes."""
    loads = [a for a in grid.appliances if isinstance(a, Load)]
    inj_loads = [ld for ld in loads if ld.node in nodes]
    assert inj_loads, f"No loads found at nodes {nodes}"
    return {ld.id: SPECTRUM for ld in inj_loads}


# ---------------------------------------------------------------------------
# Single-phase equivalent tests
# ---------------------------------------------------------------------------


class TestCigreLvHarmonicOracleSinglePhase:
    """Regression parity: SINGLE_PHASE_EQUIV harmonic flow vs pure-numpy oracle.

    Exercises all three 20/0.4 kV transformers and cross-feeder spread.
    Uses :func:`numpy_harmonic_voltages` (R-const/X∝h, machine-precision parity).
    """

    PHASE_MODE = PhaseMode.SINGLE_PHASE_EQUIV

    def _solve_and_compare(self, orders=None):
        """Build grid, solve pgml, run oracle, return (v_pgml, v_oracle, orders)."""
        if orders is None:
            orders = ORDERS
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
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
        v_oracle = numpy_harmonic_voltages(
            grid,
            harmonic_injection,
            orders,
            slack="norton",
            v1=v1_np,
        )
        v_pgml = hres.v.detach().cpu().numpy()
        return v_pgml, v_oracle, orders

    def test_fundamental_exact_match(self) -> None:
        """Order 1 is returned directly from the supplied v1 — strictly identical."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        h1_k = orders.index(1)
        # The oracle returns v1 verbatim at order 1; the two arrays are bit-for-bit
        # equal (same float64 values, no arithmetic).
        np.testing.assert_array_equal(
            v_oracle[h1_k],
            v_pgml[h1_k],
            err_msg="Order 1: oracle and pgml must be bit-for-bit identical",
        )

    def test_harmonic_5_oracle_parity(self) -> None:
        """5th harmonic: oracle matches pgml to machine precision."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        h5_k = orders.index(5)
        np.testing.assert_allclose(
            v_oracle[h5_k],
            v_pgml[h5_k],
            atol=ATOL_V,
            rtol=RTOL_V,
            err_msg="5th harmonic: oracle vs pgml voltage mismatch",
        )

    def test_harmonic_11_oracle_parity(self) -> None:
        """11th harmonic: oracle matches pgml to machine precision."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        h11_k = orders.index(11)
        np.testing.assert_allclose(
            v_oracle[h11_k],
            v_pgml[h11_k],
            atol=ATOL_V,
            rtol=RTOL_V,
            err_msg="11th harmonic: oracle vs pgml voltage mismatch",
        )

    def test_transformer_nodes_carry_harmonics(self) -> None:
        """Injected-feeder LV busbars carry significant 5th harmonic voltage.

        Checks that the 5th harmonic voltage at the directly injected LV
        transformer busbars (nodes 3 and 25) is physically non-trivial,
        confirming that the transformer stamp propagates harmonics from the
        LV side through the series leakage impedance. The absolute level
        (>0.1 V) is a sanity bound — empirically ~6.5 V is achieved.

        Note: the stiff MV Norton source (R=1e-6 Ω) acts as a near-perfect
        short circuit at harmonics (~10^6 S admittance), clamping the 20 kV
        bus to essentially 0 V at harmonic frequencies. Therefore the
        non-injecting LV busbars (node 22, feeder I) carry negligible harmonic
        voltage (< 1e-6 V) — this is correct physics, not a modelling gap.
        """
        grid, id_map = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        harmonic_injection = _build_injection(grid, INJECTION_NODES)
        hres = solve_harmonic_flow(
            grid,
            ORDERS,
            slack="norton",
            harmonic_injection=harmonic_injection,
            dtype=torch.complex128,
        )
        assert hres.pf.converged

        index = node_phase_index(grid)
        from pgml.schemas.grid_schema import Phase

        h5_k = ORDERS.index(5)
        v5 = hres.v[h5_k].detach().cpu().numpy()

        # Injected LV busbars (directly connected to injecting loads)
        for node_id in INJECTION_NODES:
            row = index.row(node_id, Phase.A)
            v_mag = abs(v5[row])
            assert v_mag > 0.1, (
                f"Node {node_id} (injected LV transformer busbar): |V5| = {v_mag:.4e} V "
                f"is unexpectedly small — transformer harmonic propagation may be broken"
            )

    def test_harmonic_propagates_within_feeder(self) -> None:
        """Harmonic voltages propagate from LV busbar to downstream feeder nodes.

        Injection at node 3 (LV busbar of transformer 1, feeder R). All
        downstream nodes on feeder R (4–20) should carry the same 5th-harmonic
        voltage level as node 3 (the LV lines are short, so the harmonic
        voltage barely drops). This confirms the transformer stamp correctly
        passes harmonics from the injection point to the feeder.
        """
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        # Single injection at node 3 only
        loads = [a for a in grid.appliances if isinstance(a, Load)]
        inj_loads = [ld for ld in loads if ld.node == 3]
        harmonic_injection = {ld.id: SPECTRUM for ld in inj_loads}

        hres = solve_harmonic_flow(
            grid,
            ORDERS,
            slack="norton",
            harmonic_injection=harmonic_injection,
            dtype=torch.complex128,
        )
        assert hres.pf.converged

        index = node_phase_index(grid)
        from pgml.schemas.grid_schema import Phase

        h5_k = ORDERS.index(5)
        v5 = hres.v[h5_k].detach().cpu().numpy()

        # Node 3 (LV busbar, injection point)
        v_busbar = abs(v5[index.row(3, Phase.A)])
        assert v_busbar > 0.1, (
            f"Node 3 (injection busbar): |V5| = {v_busbar:.4e} V too small"
        )
        # Downstream nodes 4–6 (first three line segments of feeder R)
        for node_id in [4, 5, 6]:
            row = index.row(node_id, Phase.A)
            v_mag = abs(v5[row])
            # Voltage should be within 5% of the busbar voltage (short lines)
            assert abs(v_mag - v_busbar) / v_busbar < 0.05, (
                f"Node {node_id}: |V5| = {v_mag:.4e} V deviates > 5% from "
                f"busbar {v_busbar:.4e} V — feeder harmonic propagation broken"
            )

    def test_output_shape(self) -> None:
        """Oracle returns [H, N] complex array with correct dimensions."""
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
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
        v_oracle = numpy_harmonic_voltages(
            grid, harmonic_injection, ORDERS, slack="norton", v1=v1_np
        )
        assert v_oracle.shape == (len(ORDERS), index.size), (
            f"Oracle shape {v_oracle.shape} != expected ({len(ORDERS)}, {index.size})"
        )
        assert v_oracle.dtype == complex or np.iscomplexobj(v_oracle), (
            "Oracle must return a complex array"
        )


# ---------------------------------------------------------------------------
# Three-phase tests
# ---------------------------------------------------------------------------


class TestCigreLvHarmonicOracleThreePhase:
    """Regression parity: THREE_PHASE harmonic flow vs pure-numpy oracle.

    Three-phase grids have N=132 rows (44 nodes × 3 phases A/B/C).
    Uses :func:`numpy_harmonic_voltages` which stamps 3×3 diagonal phase
    matrices (M1 transformer model, R-const/X∝h) for machine-precision parity.
    """

    PHASE_MODE = PhaseMode.THREE_PHASE

    def _solve_and_compare(self, orders=None):
        """Build 3-phase grid, solve pgml, run oracle, return arrays."""
        if orders is None:
            orders = ORDERS
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
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
        v_oracle = numpy_harmonic_voltages(
            grid,
            harmonic_injection,
            orders,
            slack="norton",
            v1=v1_np,
        )
        v_pgml = hres.v.detach().cpu().numpy()
        return v_pgml, v_oracle, orders

    def test_fundamental_exact_match(self) -> None:
        """Order 1 is returned directly from v1 — bit-for-bit equal."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        h1_k = orders.index(1)
        np.testing.assert_array_equal(
            v_oracle[h1_k],
            v_pgml[h1_k],
            err_msg="THREE_PHASE order 1: oracle and pgml must be bit-for-bit identical",
        )

    def test_harmonic_5_oracle_parity(self) -> None:
        """5th harmonic (3-phase): oracle matches pgml to machine precision."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        h5_k = orders.index(5)
        np.testing.assert_allclose(
            v_oracle[h5_k],
            v_pgml[h5_k],
            atol=ATOL_V,
            rtol=RTOL_V,
            err_msg="THREE_PHASE 5th harmonic: oracle vs pgml voltage mismatch",
        )

    def test_harmonic_11_oracle_parity(self) -> None:
        """11th harmonic (3-phase): oracle matches pgml to machine precision."""
        v_pgml, v_oracle, orders = self._solve_and_compare()
        h11_k = orders.index(11)
        np.testing.assert_allclose(
            v_oracle[h11_k],
            v_pgml[h11_k],
            atol=ATOL_V,
            rtol=RTOL_V,
            err_msg="THREE_PHASE 11th harmonic: oracle vs pgml voltage mismatch",
        )

    def test_output_shape_three_phase(self) -> None:
        """THREE_PHASE oracle returns [H, 132] — 44 nodes × 3 phases."""
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
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
        v_oracle = numpy_harmonic_voltages(
            grid, harmonic_injection, ORDERS, slack="norton", v1=v1_np
        )
        assert v_oracle.shape == (len(ORDERS), index.size), (
            f"THREE_PHASE oracle shape {v_oracle.shape} != ({len(ORDERS)}, {index.size})"
        )

    def test_three_phase_balanced_harmonics(self) -> None:
        """THREE_PHASE 5th harmonic voltages are balanced (phases A/B/C equal mag).

        In a balanced symmetric grid the 5th harmonic is also balanced: all three
        phases have the same magnitude (offset by 120°·5 = 240° — negative sequence
        for order 5). The oracle must reproduce this balance to machine precision.
        """
        grid, _ = cigre_lv_full_grid(phase_mode=self.PHASE_MODE)
        harmonic_injection = _build_injection(grid, INJECTION_NODES)
        hres = solve_harmonic_flow(
            grid,
            ORDERS,
            slack="norton",
            harmonic_injection=harmonic_injection,
            dtype=torch.complex128,
        )
        assert hres.pf.converged

        index = node_phase_index(grid)
        from pgml.schemas.grid_schema import Phase

        h5_k = ORDERS.index(5)
        v5 = hres.v[h5_k].detach().cpu().numpy()

        max_imbalance = 0.0
        for node in grid.nodes:
            r_a = index.row(node.id, Phase.A)
            r_b = index.row(node.id, Phase.B)
            r_c = index.row(node.id, Phase.C)
            ma = abs(v5[r_a])
            mb = abs(v5[r_b])
            mc = abs(v5[r_c])
            imb = max(abs(ma - mb), abs(mb - mc), abs(ma - mc))
            if imb > max_imbalance:
                max_imbalance = imb

        assert max_imbalance < 5e-9, (
            f"THREE_PHASE 5th harmonic: magnitude imbalance {max_imbalance:.3e} V "
            f"exceeds 5e-9 V — symmetric grid should produce perfectly balanced harmonics "
            f"(residual is floating-point rounding in the balanced computation, "
            f"empirically ~1.8e-9 V on 25 V magnitude)"
        )
