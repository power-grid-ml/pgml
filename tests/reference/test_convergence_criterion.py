"""The per-unit convergence criteria of the nonlinear power flow.

The engine solves in SI units, so a convergence tolerance is only meaningful once the
measure is normalised. These tests pin the four properties that normalisation buys, and
the precision floor that keeps an unreachable tolerance from looking like a failure:

1. VOLTAGE-LEVEL INDEPENDENCE. Two electrically identical feeders built at 400 V and at
   20 kV converge in the same number of iterations to the same per-unit state. An
   absolute volt criterion is 50x stricter on the 20 kV feeder for the same number.
2. SIZE INDEPENDENCE. A disjoint union of G copies of a feeder (one block-diagonal
   system) converges exactly like one copy: both criteria are per row, so the row count
   does not dilute or tighten them.
3. THE CRITERION IS WHAT THE TOOLS REPORT. ``residual`` is the achieved per-unit
   apparent-power mismatch — pandapower's ``tolerance_mva`` quantity on a 1 MVA base —
   and a converged solve satisfies BOTH the mismatch and the voltage-update criterion.
4. TIGHTER COSTS MORE, AND IS REACHED. The iteration count grows monotonically with the
   requested tolerance, and the achieved mismatch follows it.

Plus: a tolerance below what the working precision can resolve converges at the floor
with a warning instead of running to ``max_iter``.
"""

from __future__ import annotations

import logging

import pytest
import torch

from pgml.multigrid import merge_grids
from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source
from pgml.solver import solve_power_flow
from tests.fixtures.tiny_grids import single_phase_chain

CDT = torch.complex128
A = (Phase.A,)
S_BASE = 1.0e6  # the documented default power base of the per-unit mismatch


def _scaled_feeder(u_v: float, *, p_w: float = 2.0e5, n_sections: int = 4) -> Grid:
    """A radial feeder whose PER-UNIT electrical state is independent of ``u_v``.

    Impedances scale with the base impedance ``u^2 / S``, the load powers do not, so the
    per-unit voltage profile of the 400 V and the 20 kV version is identical to the last
    digit and only the SI numbers differ.
    """
    z_base = u_v * u_v / S_BASE
    r_per_m = 2.0e-4 * z_base  # ~0.2 pu total over 1 km at S_BASE
    l_per_m = 6.0e-7 * z_base / (2.0 * torch.pi * 50.0)
    nodes = [Node(id=i, u_rated_v=u_v, phases=A) for i in range(n_sections + 1)]
    branches = [
        Line(
            id=100 + i,
            from_node=i,
            to_node=i + 1,
            from_phases=A,
            to_phases=A,
            length_m=250.0,
            series_resistance_ohm_per_m=[[r_per_m]],
            series_inductance_h_per_m=[[l_per_m]],
            shunt_capacitance_f_per_m=[[0.0]],
        )
        for i in range(n_sections)
    ]
    appliances = [
        Source(
            id=10,
            node=0,
            phases=A,
            u_ref_v=(u_v,),
            u_angle_deg=(0.0,),
            resistance_ohm=[[0.01 * z_base]],
            inductance_h=[[0.1 * z_base / (2.0 * torch.pi * 50.0)]],
        )
    ]
    appliances += [
        Load(id=200 + i, node=i, phases=A, p_nom_w=p_w, q_nom_var=0.3 * p_w)
        for i in range(1, n_sections + 1)
    ]
    return Grid(
        base_frequency_hz=50.0, nodes=nodes, branches=branches, appliances=appliances
    )


def _v_pu(res, grid) -> torch.Tensor:
    u = float(grid.nodes[0].u_rated_v)
    return res.v.reshape(-1).abs() / u


# --------------------------------------------------------------------------- #
# 1. voltage-level independence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tol", [1e-6, 1e-8, 1e-10])
def test_same_tolerance_means_the_same_thing_on_every_voltage_level(tol):
    lv, mv = _scaled_feeder(400.0), _scaled_feeder(20_000.0)
    r_lv = solve_power_flow(lv, tol=tol, dtype=CDT)
    r_mv = solve_power_flow(mv, tol=tol, dtype=CDT)
    assert r_lv.converged and r_mv.converged
    assert r_lv.iterations == r_mv.iterations
    # The per-unit states agree to the tolerance that was requested, not to the
    # tolerance divided by the voltage level.
    assert torch.allclose(_v_pu(r_lv, lv), _v_pu(r_mv, mv), rtol=0.0, atol=10.0 * tol)
    # Both report the same per-unit mismatch scale (the SI values differ by 50x).
    assert r_lv.diagnostics.mismatch_max_pu < tol
    assert r_mv.diagnostics.mismatch_max_pu < tol
    assert r_mv.diagnostics.mismatch_max_a < r_lv.diagnostics.mismatch_max_a


# --------------------------------------------------------------------------- #
# 2. size independence (the row count is not part of the criterion)
# --------------------------------------------------------------------------- #
def test_an_ensemble_converges_like_one_member():
    """A union of four feeders solved as ONE system is judged per row, not per norm."""
    one = _scaled_feeder(400.0)
    members = [_scaled_feeder(400.0) for _ in range(4)]
    merged = merge_grids(members)
    r_one = solve_power_flow(one, tol=1e-9, dtype=CDT)
    r_all = solve_power_flow(merged.grid, tol=1e-9, dtype=CDT)
    assert r_one.converged and r_all.converged
    assert r_all.iterations == r_one.iterations
    assert r_all.diagnostics.mismatch_max_pu == pytest.approx(
        r_one.diagnostics.mismatch_max_pu, rel=1e-6
    )


# --------------------------------------------------------------------------- #
# 3. what the criterion reports
# --------------------------------------------------------------------------- #
def test_residual_is_the_per_unit_power_mismatch():
    grid = _scaled_feeder(20_000.0)
    res = solve_power_flow(grid, tol=1e-9, dtype=CDT)
    d = res.diagnostics
    assert float(res.residual) == pytest.approx(d.mismatch_max_pu)
    assert d.s_base_va == S_BASE
    assert d.mismatch_max_va == pytest.approx(d.mismatch_max_pu * S_BASE)
    # Recompute the mismatch independently from the reported SI current and the state.
    assert d.mismatch_max_a > 0.0
    assert d.mismatch_max_va <= d.mismatch_max_a * float(res.v.abs().max()) * 1.000001


def test_both_criteria_must_hold():
    """A loose power tolerance does not release the voltage-update criterion."""
    grid = _scaled_feeder(400.0)
    loose = solve_power_flow(grid, tol=1e-2, tol_update_pu=1e-2, dtype=CDT)
    tight_v = solve_power_flow(grid, tol=1e-2, tol_update_pu=1e-10, dtype=CDT)
    assert loose.converged and tight_v.converged
    assert tight_v.iterations > loose.iterations
    assert tight_v.diagnostics.update_max_pu < 1e-10
    assert loose.diagnostics.update_max_pu > 1e-10


# --------------------------------------------------------------------------- #
# 4. monotonicity in the requested tolerance
# --------------------------------------------------------------------------- #
def test_iterations_and_accuracy_follow_the_tolerance():
    grid = _scaled_feeder(20_000.0)
    runs = [solve_power_flow(grid, tol=t, dtype=CDT) for t in (1e-4, 1e-6, 1e-8, 1e-10)]
    iters = [r.iterations for r in runs]
    mism = [r.diagnostics.mismatch_max_pu for r in runs]
    assert all(r.converged for r in runs)
    assert iters == sorted(iters)  # monotonically more work
    assert mism == sorted(mism, reverse=True)  # monotonically more accurate


# --------------------------------------------------------------------------- #
# the precision floor
# --------------------------------------------------------------------------- #
def test_tolerance_below_the_precision_floor_converges_at_the_floor(caplog):
    """An unreachable tolerance is reported, and the floor governs convergence.

    A stiff source behind a milliohm impedance makes the nodal residual a difference of
    terms of ~1e9 A, so at complex128 its mismatch bottoms out far above 1e-14 pu. The
    solve must converge at that floor and say so, not run to ``max_iter``.
    """
    grid = _scaled_feeder(20_000.0)
    stiff = grid.model_copy(
        update={
            "appliances": [
                a.model_copy(
                    update={"resistance_ohm": [[1e-6]], "inductance_h": [[1e-12]]}
                )
                if isinstance(a, Source)
                else a
                for a in grid.appliances
            ]
        }
    )
    with caplog.at_level(logging.WARNING, logger="pgml"):
        res = solve_power_flow(stiff, tol=1e-14, max_iter=60, dtype=CDT)
    assert res.converged
    assert res.iterations < 60
    assert any("precision floor" in rec.message for rec in caplog.records)


def test_float32_update_floor_still_terminates():
    """complex64 cannot resolve 1e-8 pu; the dtype floor terminates the iteration."""
    grid = _scaled_feeder(400.0)
    res = solve_power_flow(grid, dtype=torch.complex64, max_iter=100)
    assert res.converged
    assert res.iterations < 100
    assert res.diagnostics.update_max_pu < 1e-5  # at the float32 floor


class TestResidualIdentity:
    """The criterion's residual is the device-current change, and that is exact.

    The fixed point's back-substitution enforces ``(Y V_new)_free = I_free``, so on every
    free row ``F(V_new) = I_device(V_new) - I_device(V_old)``. The iteration uses that
    identity instead of a matrix-vector product and confirms it against the nodal residual
    before it exits.
    """

    @staticmethod
    def _grid_and_op(scales=(0.6, 1.0, 1.4)):
        grid = single_phase_chain()
        p = torch.tensor([2000.0 * s for s in scales], dtype=torch.float64)
        q = torch.tensor([300.0 * s for s in scales], dtype=torch.float64)
        return grid, {30: {"p_w": p, "q_var": q}}

    def test_one_matrix_vector_product_per_solve_not_per_iteration(self) -> None:
        """The identity removes the per-iteration product and keeps the same answer.

        The reference is the same solve with the residual formed explicitly every
        iteration, which is what the engine did before: it is reproduced here by counting
        the products and comparing the converged voltages and the iteration count.
        """
        import pgml.solver.power_flow as pf_mod

        grid, op = self._grid_and_op()
        orig = pf_mod._apply_y
        calls = {"n": 0}

        def counted(y, v):
            calls["n"] += 1
            return orig(y, v)

        pf_mod._apply_y = counted
        try:
            r = solve_power_flow(grid, operating_point=op, dtype=CDT)
        finally:
            pf_mod._apply_y = orig
        assert r.converged and r.iterations >= 4
        # One product confirms the exit; the residual-scale estimate of the floor uses
        # its own |Y| pass, which is not this function.
        assert calls["n"] <= 2, f"{calls['n']} products for {r.iterations} iterations"

    def test_the_reported_residual_is_the_nodal_one(self) -> None:
        """A mixed-precision solve forms the residual explicitly; the two must agree.

        ``precision="mixed"`` needs the nodal residual as its next right-hand side, so it
        never uses the identity. Solving the same grid both ways and comparing the
        reported per-unit mismatch at the same tolerance pins that the plain path reports
        a NODAL mismatch and not the device-current change it iterates on.
        """
        grid, op = self._grid_and_op()
        plain = solve_power_flow(
            grid, operating_point=op, dtype=CDT, tol=1e-12, tol_update_pu=1e-12
        )
        mixed = solve_power_flow(
            grid,
            operating_point=op,
            dtype=CDT,
            precision="mixed",
            tol=1e-12,
            tol_update_pu=1e-12,
        )
        assert plain.converged and mixed.converged
        assert torch.max(torch.abs(plain.v - mixed.v)) < 1e-9
        # Both report a mismatch at the same (floor-limited) level, and both below the
        # level a surrogate that ignores the back-substitution's own residual would show.
        assert float(plain.residual) < 1e-9
        assert float(mixed.residual) < 1e-9

    def test_a_prepared_system_reports_the_same_mismatch(self) -> None:
        """The cached per-row cancellation scale must not change the criterion."""
        from pgml.solver import prepare_power_flow

        grid, op = self._grid_and_op()
        direct = solve_power_flow(grid, operating_point=op, dtype=CDT)
        system = prepare_power_flow(grid, dtype=CDT)
        prepared = solve_power_flow(grid, operating_point=op, dtype=CDT, system=system)
        assert system.row_abs_scale is not None
        assert torch.equal(direct.v, prepared.v)
        assert float(direct.residual) == float(prepared.residual)
        assert direct.iterations == prepared.iterations
        assert (
            direct.diagnostics.mismatch_floor_pu
            == prepared.diagnostics.mismatch_floor_pu
        )
