"""Studies speak system power: one network gives the same numbers in every representation.

The network of :mod:`tests.studies.representations` is solved as a per-phase equivalent,
a legacy one-phase grid, a three-phase build and a four-wire phase-domain build.
Hosting capacity, P-V and Q-V curves and the enhancement sensitivities take and report
three-phase system powers, so they agree across the representations. The tolerance
``RTOL = 1e-9`` is far above the measured spread (complex128, CPU, 2026-11-07):
hosting capacity agreed exactly (the bisection runs the same trial powers in every
representation), P-V nose ``lambda`` and power within 2e-14, Q-V curve within 2e-13,
the enhancement stress and every sensitivity within 4e-14 (relative). A verdict flip in
the bisection would show as about 1e-3, a missing factor of three as 2/3.

A legacy grid (no ``phase_mode`` tag) carries line-to-line voltages and three-phase
totals on one-phase nodes, so its branch currents are sqrt(3) too large: it agrees
wherever line currents do not enter (voltage and transformer limits, P-V, Q-V), and
through :func:`pgml.representation.to_per_phase_equivalent` everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandapower")

from pgml.representation import to_per_phase_equivalent  # noqa: E402
from pgml.studies import (  # noqa: E402
    HostingLimits,
    branch_ratings,
    enhancement_sensitivities,
    hosting_capacity,
    pv_curve,
    qv_curve,
)

from .representations import LV_NODES, representations  # noqa: E402

RTOL = 1.0e-9
#: Representations that describe the conductors (branch currents are physical).
PHYSICAL = ("per_phase", "migrated", "three_phase", "four_wire")


@pytest.fixture(scope="module")
def grids():
    out = representations()
    out["migrated"] = to_per_phase_equivalent(out["legacy"])
    return out


def test_branch_ratings_report_the_three_phase_rating(grids):
    for name, grid in grids.items():
        ratings = branch_ratings(grid)
        trafo = [r.s_rated_va for r in ratings.values() if r.s_rated_va is not None]
        assert trafo == [pytest.approx(250.0e3, rel=1e-12)], name
        lines = [r.i_max_a for r in ratings.values() if r.i_max_a is not None]
        assert lines == [pytest.approx(270.0, rel=1e-12)] * 2, name


def _assert_same_hosting(results, reference):
    for name, result in results.items():
        np.testing.assert_allclose(
            result.p_max_w, reference.p_max_w, rtol=RTOL, err_msg=name
        )
        np.testing.assert_allclose(result.tol_w, reference.tol_w, rtol=RTOL)
        assert result.p_cap_w == reference.p_cap_w
        assert [b["kind"] for b in result.binding] == [
            b["kind"] for b in reference.binding
        ], name


@pytest.mark.parametrize("direction", ["generation", "load"])
def test_hosting_capacity_is_the_same_system_power(grids, direction):
    limits = HostingLimits(v_max_pu=1.03)
    results = {
        name: hosting_capacity(
            grids[name], LV_NODES, direction=direction, limits=limits
        )
        for name in PHYSICAL
    }
    reference = results["three_phase"]
    assert (reference.p_max_w > 10.0e3).all()
    _assert_same_hosting(results, reference)
    # The legacy grid agrees once line currents (sqrt(3) too large there) are left out.
    no_lines = HostingLimits(v_max_pu=1.03, line_loading=None)
    legacy = hosting_capacity(
        grids["legacy"], LV_NODES, direction=direction, limits=no_lines
    )
    _assert_same_hosting(
        {"legacy": legacy},
        hosting_capacity(
            grids["per_phase"], LV_NODES, direction=direction, limits=no_lines
        ),
    )


def test_hosting_capacity_cap_and_tolerance_are_system_power(grids):
    """A cap no limit reaches is reported as given, in every representation.

    100 kW of system power at the LV busbar keeps every limit; three times it (the cap
    read as a per-phase power) overloads the transformer.
    """
    limits = HostingLimits(line_loading=None)
    for name in ("per_phase", "three_phase", "four_wire", "legacy"):
        result = hosting_capacity(
            grids[name], [LV_NODES[0]], limits=limits, p_max_w=100.0e3, tol_w=30.0
        )
        assert result.p_max_w.tolist() == [100.0e3], name
        assert result.binding[0] == {"kind": "none"}
        assert result.tol_w.tolist() == [30.0]


def test_enhancement_sensitivities_are_per_system_kw(grids):
    limits = HostingLimits(
        v_min_pu=0.95, v_max_pu=1.05, line_loading=0.5, transformer_loading=0.5
    )
    results = {
        name: enhancement_sensitivities(grids[name], limits=limits, stress_onset=0.5)
        for name in PHYSICAL
    }
    reference = results["three_phase"]
    for kind in ("voltage", "line", "transformer"):
        assert reference.stress_by_kind[kind] > 0.0, kind
    for name, result in results.items():
        assert result.stress == pytest.approx(reference.stress, rel=RTOL), name
        for kind, value in reference.stress_by_kind.items():
            assert result.stress_by_kind[kind] == pytest.approx(
                value, rel=RTOL, abs=1e-300
            ), (name, kind)
        for attr in (
            "d_stress_d_p_per_kw",
            "d_stress_d_q_per_kvar",
            "line_impedance_elasticity",
            "line_rating_elasticity",
            "transformer_impedance_elasticity",
            "transformer_rating_elasticity",
            "transformer_tap_gradient",
        ):
            np.testing.assert_allclose(
                getattr(result, attr),
                getattr(reference, attr),
                rtol=RTOL,
                err_msg=f"{name}: {attr}",
            )


def test_pv_curve_reports_system_power(grids):
    curves = {name: pv_curve(grid, stop_at_nose=True) for name, grid in grids.items()}
    reference = curves["three_phase"]
    assert reference.nose_index is not None
    for name, curve in curves.items():
        assert curve.lambda_max == pytest.approx(reference.lambda_max, rel=RTOL), name
        # base point: the nameplate loads, 60 kW + 50 kW
        assert curve.p_total_w[0] == pytest.approx(110.0e3, rel=RTOL), name
        assert curve.p_total_w[curve.nose_index] == pytest.approx(
            reference.p_total_w[reference.nose_index], rel=RTOL
        ), name


def test_qv_curve_reports_system_reactive_power(grids):
    curves = {name: qv_curve(grid, LV_NODES[-1]) for name, grid in grids.items()}
    reference = curves["three_phase"]
    assert reference.q_margin_var > 0.0
    for name, curve in curves.items():
        assert curve.converged.tolist() == reference.converged.tolist(), name
        ok = curve.converged
        np.testing.assert_allclose(
            curve.q_var[ok], reference.q_var[ok], rtol=RTOL, err_msg=name
        )
        np.testing.assert_allclose(
            curve.q_phase_var[ok].sum(axis=1), curve.q_var[ok], rtol=1e-12
        )
        assert curve.q_margin_var == pytest.approx(reference.q_margin_var, rel=RTOL)
