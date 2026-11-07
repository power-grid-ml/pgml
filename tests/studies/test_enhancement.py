"""Enhancement sensitivities: the stress metric, finite differences, and the contract.

Tolerances: the finite-difference checks re-solve the grid with the parameter moved by
``±h`` (``h = 1e-4`` in log scale, 1e-5 on the tap ratio, 10 W / 10 var for new power)
at a power-flow tolerance of 1e-13 pu, so a central difference carries a truncation
error of order ``h^2`` (about 1e-8 relative) and a solve error of about 1e-13 / h. Every
elasticity is checked to 1e-5 relative (plus 1e-9 absolute for values near zero).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from pgml import SimulationConfig
from pgml.errors import ConvergenceError, InputError
from pgml.schemas.grid_schema import Load, Phase, Switch, WindingConnection
from pgml.studies import (
    EnhancementSensitivities,
    HostingLimits,
    enhancement_sensitivities,
    hosting_capacity,
)

from .grids import ABC, four_wire_two_bus, stressed_feeder

TIGHT = SimulationConfig(calculation="power_flow", tol=1e-13, tol_update_pu=1e-13)
ONSET = 0.5


def _stress(grid, **kw) -> float:
    kw = {"stress_onset": ONSET, "config": TIGHT, **kw}
    return enhancement_sensitivities(grid, **kw).stress


def _central(make, h: float, **kw) -> float:
    """Central difference of the stress over a grid factory ``make(step)``."""
    return (_stress(make(+h), **kw) - _stress(make(-h), **kw)) / (2.0 * h)


def _close(value: float, reference: float, rel: float = 1e-5) -> bool:
    return abs(value - reference) <= rel * abs(reference) + 1e-9


def _edit_branch(grid, branch_id: int, edit):
    g = grid.model_copy(deep=True)
    edit(next(b for b in g.branches if b.id == branch_id))
    return g


@pytest.fixture(scope="module")
def tapped():
    """The feeder at an off-nominal tap with a core branch, and its sensitivities."""
    grid = stressed_feeder(tap=1.02, magnetizing=True)
    return grid, enhancement_sensitivities(grid, stress_onset=ONSET, config=TIGHT)


def test_stress_terms_follow_the_hosting_quantities(tapped):
    grid, res = tapped
    kinds = {e["kind"] for e in res.stressed}
    assert kinds == {"voltage", "line", "transformer"}
    assert res.stress == pytest.approx(sum(res.stress_by_kind.values()), rel=1e-14)
    assert res.stress == pytest.approx(
        sum(e["stress"] for e in res.stressed), rel=1e-12
    )
    for e in res.stressed:
        s = max(0.0, (e["utilisation"] - ONSET) / (1.0 - ONSET)) ** 2
        assert e["stress"] == pytest.approx(s, rel=1e-12)
    # hosting_capacity judges the same quantities: its margins at a negligible added
    # power (1 uW) are the base case's, to the 1e-13 pu solve tolerance
    host = hosting_capacity(
        grid,
        [4],
        direction="load",
        base_operating_point="nameplate",
        p_max_w=1.0e-6,
        config=TIGHT,
    )
    top = {
        kind: max(e["value"] for e in res.stressed if e["kind"] == kind)
        for kind in ("line", "transformer")
    }
    v_min = min(e["value"] for e in res.stressed if e["kind"] == "voltage")
    assert 1.0 - host.margins["line_loading"][0] == pytest.approx(top["line"], abs=1e-9)
    assert 1.0 - host.margins["transformer_loading"][0] == pytest.approx(
        top["transformer"], abs=1e-9
    )
    assert 0.9 + host.margins["v_min_pu"][0] == pytest.approx(v_min, abs=1e-9)


def test_line_elasticities_match_finite_differences(tapped):
    grid, res = tapped
    h = 1e-4
    for line_id in res.line_ids:
        j = res.line_ids.index(line_id)

        def impedance(step, line_id=line_id):
            def edit(b):
                f = math.exp(-step)
                b.series_resistance_ohm_per_m = (
                    np.asarray(b.series_resistance_ohm_per_m) * f
                ).tolist()
                b.series_inductance_h_per_m = (
                    np.asarray(b.series_inductance_h_per_m) * f
                ).tolist()

            return _edit_branch(grid, line_id, edit)

        def ampacity(step):
            g = grid.model_copy(deep=True)
            g.types.lines["nayy"].max_i_a = 150.0 * math.exp(step)
            return g

        assert _close(res.line_impedance_elasticity[j], _central(impedance, h))
        length_km = next(b.length_m for b in grid.branches if b.id == line_id) / 1e3
        assert res.line_impedance_elasticity_per_km[j] == pytest.approx(
            res.line_impedance_elasticity[j] / length_km, rel=1e-14
        )
    # both lines share the catalog type: the ampacity difference moves both terms
    both = _central(ampacity, h)
    assert _close(float(np.sum(res.line_rating_elasticity)), both)
    assert np.all(res.line_rating_elasticity <= 0.0)
    # the loaded head section relieves the most stress when reinforced
    assert res.ranking("line_impedance_elasticity", 1)[0][0] == 31


def test_transformer_sensitivities_match_finite_differences(tapped):
    grid, res = tapped
    h = 1e-4

    def impedance(step):
        def edit(b):
            b.series_resistance_ohm *= math.exp(-step)
            b.series_inductance_h *= math.exp(-step)

        return _edit_branch(grid, 30, edit)

    def rating(step):
        return _edit_branch(
            grid, 30, lambda b: setattr(b, "s_rated_va", b.s_rated_va * math.exp(step))
        )

    def tap(step):
        return _edit_branch(
            grid, 30, lambda b: setattr(b.tap, "ratio_magnitude", 1.02 + step)
        )

    assert _close(res.transformer_impedance_elasticity[0], _central(impedance, h))
    assert _close(res.transformer_rating_elasticity[0], _central(rating, h))
    assert res.transformer_rating_elasticity[0] < 0.0
    assert _close(res.transformer_tap_gradient[0], _central(tap, 1e-5))
    # a higher HV/LV ratio lowers the LV voltage, which is the stressed side here
    assert res.transformer_tap_gradient[0] > 0.0


def test_tap_gradient_at_the_nominal_tap_is_the_central_derivative():
    # without a core branch both terminals carry the same loading at the nominal tap,
    # but the from-side loading falls with the tap; the reported gradient is the
    # central one (the mean of the two one-sided slopes), not a side picked by rounding
    grid = stressed_feeder()
    res = enhancement_sensitivities(grid, stress_onset=ONSET, config=TIGHT)

    def tap(step):
        return _edit_branch(
            grid, 30, lambda b: setattr(b.tap, "ratio_magnitude", 1.0 + step)
        )

    central = _central(tap, 1e-5)
    up = (_stress(tap(1e-5)) - res.stress) / 1e-5
    down = (res.stress - _stress(tap(-1e-5))) / 1e-5
    assert abs(up - down) > 0.02 * abs(central)  # a genuine kink
    assert _close(res.transformer_tap_gradient[0], central, rel=1e-4)


def test_injection_sensitivities_match_finite_differences(tapped):
    grid, res = tapped
    assert res.node_ids == (2, 3, 4)

    def added(node_id, p_w, q_var):
        g = grid.model_copy(deep=True)
        g.appliances.append(
            Load(
                id=99,
                node=node_id,
                phases=ABC,
                connection=WindingConnection.WYE,
                p_nom_w=p_w,
                q_nom_var=q_var,
            )
        )
        return g

    for k, node_id in enumerate(res.node_ids):
        fd_p = _central(lambda s, n=node_id: added(n, s, 0.0), 10.0) * 1e3
        fd_q = _central(lambda s, n=node_id: added(n, 0.0, s), 10.0) * 1e3
        assert _close(res.d_stress_d_p_per_kw[k], fd_p)
        assert _close(res.d_stress_d_q_per_kvar[k], fd_q)
    # new demand hurts most at the far end; generation there relieves the most
    assert res.ranking("d_stress_d_p_per_kw", 1, ascending=False)[0][0] == 4
    assert np.all(res.d_stress_d_p_per_kw > 0.0)


def test_neutral_and_unbalance_terms():
    # a single-phase load on a four-wire line loads the neutral and unbalances bus 2
    grid = four_wire_two_bus(i_max_a=60.0)
    grid = grid.model_copy(
        update={
            "appliances": [
                *grid.appliances,
                Load(
                    id=7,
                    node=2,
                    phases=(Phase.A,),
                    connection=WindingConnection.WYE,
                    p_nom_w=9.0e3,
                    q_nom_var=1.0e3,
                ),
            ]
        }
    )
    limits = HostingLimits(neutral_loading=1.0, vuf_max=0.004)
    res = enhancement_sensitivities(
        grid, limits=limits, stress_onset=ONSET, config=TIGHT
    )
    kinds = {e["kind"] for e in res.stressed}
    assert {"line", "line_neutral", "vuf"} <= kinds

    def ampacity(step):
        g = grid.model_copy(deep=True)
        g.types.lines["cable4"].max_i_a = 60.0 * math.exp(step)
        return g

    fd = _central(ampacity, 1e-4, limits=limits)
    assert _close(res.line_rating_elasticity[0], fd)


def test_a_branch_without_current_keeps_every_gradient_finite(tapped):
    # a cable bridged by a closed ideal switch has both ends fused into one bus and
    # carries exactly zero current, where the RMS norm has no slope; the study must not
    # turn that into NaN for every other sensitivity
    grid, reference = tapped
    bridged = next(b for b in grid.branches if b.id == 32).model_copy(
        update={"id": 33, "from_node": 4, "to_node": 5}
    )
    switch = Switch(
        id=34,
        from_node=4,
        to_node=5,
        from_phases=ABC,
        to_phases=ABC,
        default_closed=True,
    )
    node = next(n for n in grid.nodes if n.id == 4).model_copy(update={"id": 5})
    grid = grid.model_copy(
        update={
            "nodes": [*grid.nodes, node],
            "branches": [*grid.branches, bridged, switch],
        }
    )
    res = enhancement_sensitivities(grid, stress_onset=ONSET, config=TIGHT)
    for name in (
        "line_impedance_elasticity",
        "line_rating_elasticity",
        "transformer_impedance_elasticity",
        "transformer_tap_gradient",
        "d_stress_d_p_per_kw",
    ):
        assert np.isfinite(getattr(res, name)).all(), name
    assert res.line_impedance_elasticity[res.line_ids.index(33)] == 0.0
    # the bridged cable carries nothing: the loading stress upstream is unchanged
    for kind in ("line", "transformer"):
        assert res.stress_by_kind[kind] == pytest.approx(
            reference.stress_by_kind[kind], rel=1e-9
        )


def test_no_stress_below_the_onset_gives_zero_gradients():
    res = enhancement_sensitivities(stressed_feeder(load_scale=0.2))
    assert res.stress == 0.0 and res.stressed == ()
    for name in (
        "line_impedance_elasticity",
        "line_rating_elasticity",
        "transformer_impedance_elasticity",
        "transformer_rating_elasticity",
        "transformer_tap_gradient",
        "d_stress_d_p_per_kw",
        "d_stress_d_q_per_kvar",
    ):
        np.testing.assert_array_equal(getattr(res, name), 0.0)


def test_disabled_limits_and_unrated_branches_report_nan(tapped):
    grid, _ = tapped
    res = enhancement_sensitivities(
        grid,
        limits=HostingLimits(line_loading=None, transformer_loading=None),
        stress_onset=ONSET,
    )
    assert np.isnan(res.line_rating_elasticity).all()
    assert np.isnan(res.transformer_rating_elasticity).all()
    assert set(e["kind"] for e in res.stressed) == {"voltage"}
    assert np.isfinite(res.line_impedance_elasticity).all()
    assert res.ranking("line_rating_elasticity") == []


def test_worst_case_base_follows_the_direction(tapped):
    grid, nameplate = tapped
    kw = dict(stress_onset=ONSET, config=TIGHT)
    gen = enhancement_sensitivities(grid, base_operating_point="worst_case", **kw)
    load = enhancement_sensitivities(
        grid, base_operating_point="worst_case", direction="load", **kw
    )
    assert gen.base_operating_point == "worst_case" and gen.stress == 0.0
    # no generator in this grid: the load worst case is the nameplate case
    assert load.stress == pytest.approx(nameplate.stress, rel=1e-12)
    custom = enhancement_sensitivities(
        grid, base_operating_point={2: {"p_w": 0.0, "q_var": 0.0}}, **kw
    )
    assert custom.base_operating_point == "custom"
    assert 0.0 < custom.stress < nameplate.stress


def test_harmonic_calculation_without_sources_keeps_the_fundamental_answer(tapped):
    grid, fundamental = tapped
    cfg = SimulationConfig(
        calculation="harmonic", harmonic_orders=[1, 5], tol=1e-13, tol_update_pu=1e-13
    )
    harmonic = enhancement_sensitivities(grid, stress_onset=ONSET, config=cfg)
    assert harmonic.calculation == "harmonic"
    assert harmonic.stress == pytest.approx(fundamental.stress, rel=1e-10)
    np.testing.assert_allclose(
        harmonic.line_impedance_elasticity,
        fundamental.line_impedance_elasticity,
        rtol=1e-8,
    )


def test_complex64_agrees_with_complex128(tapped):
    grid, r128 = tapped
    r64 = enhancement_sensitivities(grid, stress_onset=ONSET, dtype="complex64")
    assert r64.stress == pytest.approx(r128.stress, rel=1e-3)
    np.testing.assert_allclose(
        r64.line_impedance_elasticity, r128.line_impedance_elasticity, rtol=1e-2
    )


def test_round_trip_ranking_and_input_is_not_mutated(tapped):
    grid, _ = tapped
    before = grid.model_dump_json()
    res = enhancement_sensitivities(grid)
    assert grid.model_dump_json() == before
    data = json.loads(json.dumps(res.to_dict()))
    assert data["study"] == "enhancement_sensitivities"
    back = EnhancementSensitivities.from_dict(data)
    assert back.to_dict() == res.to_dict()
    ranked = res.ranking("transformer_impedance_elasticity")
    assert [i for i, _ in ranked] == [30]
    with pytest.raises(InputError):
        res.ranking("stress")
    with pytest.raises(InputError):
        EnhancementSensitivities.from_dict({**data, "study": "hosting_capacity"})
    assert res.solve_time_s > 0.0 and res.backward_time_s > 0.0


def test_a_base_that_does_not_converge_raises():
    with pytest.raises(ConvergenceError, match="does not converge"):
        enhancement_sensitivities(stressed_feeder(load_scale=40.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"direction": "both"},
        {"stress_onset": 1.0},
        {"stress_onset": -0.1},
        {
            "limits": HostingLimits(
                v_max_pu=None,
                v_min_pu=None,
                line_loading=None,
                transformer_loading=None,
            )
        },
        {"limits": HostingLimits(v_min_pu=0.9, v_max_pu=0.99)},
        {"limits": HostingLimits(thd_max=0.05)},
        {"config": SimulationConfig(calculation="power_flow", operating_point={})},
        {"base_operating_point": "peak"},
        {"dtype": torch.float64},
    ],
)
def test_invalid_options_raise(kwargs):
    with pytest.raises(InputError):
        enhancement_sensitivities(stressed_feeder(), **kwargs)
