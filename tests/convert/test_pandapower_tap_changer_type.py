"""``net.trafo.tap_changer_type`` decides whether a tap position acts.

pandapower 3 applies a transformer's tap only for a recognised
``tap_changer_type`` (``pandapower.build_branch._calc_nominal_ratio_from_dataframe``):
``"Ratio"`` and ``"Symmetrical"`` move the tapped side's nominal voltage, ``"Ideal"``
shifts the angle only, and a row whose type is UNSET (NaN) gets NO tap at all,
however far from neutral its ``tap_pos`` sits. Datasets that predate the column
(SimBench, for example) therefore carry tap positions their own solver ignores.

The converter follows pandapower's rule so the two tools solve the same network, and
says so out loud: an off-neutral tap dropped for want of a changer type logs a
WARNING naming the transformer. A net whose ``trafo`` table has no
``tap_changer_type`` column at all (pandapower < 3.0 data) keeps the legacy
behaviour and applies the tap.

The two-bus 110/20 kV net below is the smallest case that shows it: one unit at
``tap_pos = -1`` with a 1.5 % step, i.e. a 1.5e-2 pu error if the tap is applied
where pandapower drops it.
"""

from __future__ import annotations

import copy

import pytest
import torch

from pgml.convert.pandapower import to_grid
from pgml.convert.pandapower.converter import _tap_ratio_magnitude
from pgml.errors import ConversionError
from pgml.schemas.grid_schema import Transformer
from pgml.solver import solve_power_flow

pp = pytest.importorskip("pandapower", exc_type=ImportError)

CDT = torch.complex128


def _two_bus_net(tap_changer_type=None, tap_pos: float = -1.0):
    """One tapped 110/20 kV transformer feeding a 5 MW load."""
    net = pp.create_empty_network(f_hz=50.0, sn_mva=1.0)
    hv = pp.create_bus(net, vn_kv=110.0, name="hv")
    lv = pp.create_bus(net, vn_kv=20.0, name="lv")
    pp.create_ext_grid(net, bus=hv, vm_pu=1.0, va_degree=0.0)
    kwargs = dict(
        hv_bus=hv,
        lv_bus=lv,
        sn_mva=25.0,
        vn_hv_kv=110.0,
        vn_lv_kv=20.0,
        vkr_percent=0.3,
        vk_percent=11.2,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=0.0,
        tap_side="hv",
        tap_neutral=0,
        tap_min=-9,
        tap_max=9,
        tap_step_percent=1.5,
        tap_pos=tap_pos,
    )
    if tap_changer_type is not None:
        kwargs["tap_changer_type"] = tap_changer_type
    pp.create_transformer_from_parameters(net, **kwargs)
    pp.create_load(net, bus=lv, p_mw=5.0, q_mvar=1.5)
    return net


def _tap_ratio(grid) -> float:
    return max(
        float(b.tap.ratio_magnitude)
        for b in grid.branches
        if isinstance(b, Transformer)
    )


def _max_dv_pu(net) -> float:
    """Largest per-bus ``|V|`` deviation (pu) from ``pp.runpp`` on the same net."""
    solved = copy.deepcopy(net)
    pp.runpp(solved, numba=False, calculate_voltage_angles=True)
    grid, id_map = to_grid(net)
    res = solve_power_flow(grid, slack="ideal", tol=1e-10, max_iter=200, dtype=CDT)
    assert res.converged
    worst = 0.0
    for pp_bus, node_id in id_map["bus"].items():
        row = res.index.row(node_id, grid.nodes[0].phases[0])
        base = float(net.bus.at[pp_bus, "vn_kv"]) * 1_000.0
        ours = float(res.v.reshape(-1)[row].abs()) / base
        worst = max(worst, abs(ours - float(solved.res_bus.at[pp_bus, "vm_pu"])))
    return worst


class TestTapChangerType:
    def test_unset_type_drops_the_tap_and_matches_runpp(self, caplog):
        """The SimBench case: tap columns set, ``tap_changer_type`` NaN."""
        net = _two_bus_net(tap_changer_type=None)
        with caplog.at_level("WARNING", logger="pgml"):
            grid, _ = to_grid(net)
        assert _tap_ratio(grid) == pytest.approx(1.0)
        assert any(
            "tap_changer_type is not set" in r.message and "trafo 0" in r.message
            for r in caplog.records
        ), f"expected a dropped-tap warning, got: {[r.message for r in caplog.records]}"
        # Both tools now solve the same network (the residual is solver tolerance).
        assert _max_dv_pu(net) < 1e-9

    def test_ratio_type_applies_the_tap_and_matches_runpp(self):
        net = _two_bus_net(tap_changer_type="Ratio")
        grid, _ = to_grid(net)
        assert _tap_ratio(grid) == pytest.approx(1.0 - 0.015)
        assert _max_dv_pu(net) < 1e-9

    def test_symmetrical_type_is_the_same_ratio_without_a_step_angle(self):
        """``"Symmetrical"`` differs from ``"Ratio"`` only through
        ``tap_step_degree``, which this converter rejects, so the real ratio is the
        same (pandapower treats both as its complex tap)."""
        ratio = _tap_ratio(to_grid(_two_bus_net(tap_changer_type="Symmetrical"))[0])
        assert ratio == pytest.approx(1.0 - 0.015)

    def test_ideal_type_raises(self):
        with pytest.raises(ConversionError, match="tap_changer_type"):
            to_grid(_two_bus_net(tap_changer_type="Ideal"))

    def test_unknown_type_raises(self):
        with pytest.raises(ConversionError, match="tap_changer_type"):
            to_grid(_two_bus_net(tap_changer_type="Wishful"))

    def test_unset_type_at_neutral_is_silent(self, caplog):
        """Nothing is dropped when the tap sits at neutral, so nothing is reported."""
        with caplog.at_level("WARNING", logger="pgml"):
            grid, _ = to_grid(_two_bus_net(tap_changer_type=None, tap_pos=0.0))
        assert _tap_ratio(grid) == pytest.approx(1.0)
        assert not any("tap_changer_type" in r.message for r in caplog.records)

    def test_absent_column_keeps_the_legacy_behaviour(self):
        """A pandapower < 3.0 net has no ``tap_changer_type`` column at all; its taps
        were always applied, and a row dict without the key reads that way."""
        row = {
            "tap_pos": -1.0,
            "tap_neutral": 0.0,
            "tap_step_percent": 1.5,
            "tap_side": "hv",
        }
        assert _tap_ratio_magnitude(row) == pytest.approx(1.0 - 0.015)

    def test_present_but_empty_value_drops_the_tap(self):
        row = {
            "tap_pos": -1.0,
            "tap_neutral": 0.0,
            "tap_step_percent": 1.5,
            "tap_side": "hv",
            "tap_changer_type": None,
        }
        assert _tap_ratio_magnitude(row) == pytest.approx(1.0)
