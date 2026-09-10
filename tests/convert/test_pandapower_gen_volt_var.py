"""``net.gen`` conversion: the default drop and the opt-in Volt-VAr approximation.

``net.gen`` is pandapower's PV bus (fixed P, regulated ``vm_pu``, free Q between
``min_q_mvar`` and ``max_q_mvar``). The converter's default
:data:`~pgml.convert.pandapower.GenMode.DROP` leaves it unread; the opt-in
:data:`~pgml.convert.pandapower.GenMode.VOLT_VAR_APPROX` maps each row onto a
:class:`~pgml.schemas.grid_schema.Generator` carrying a steep Volt-VAr droop.

Covered here:

- the default is byte-identical to converting the same net without the ``gen``
  rows at all, and still reports the table as dropped;
- the droop's parameters (setpoint, breakpoints, reactive base, saturation) come
  out of the row exactly as documented, including the fallbacks for a missing
  reactive limit and the ``scaling`` convention;
- the slack-bus rule (a row on the ``ext_grid`` bus, or one flagged ``slack``, is
  skipped and logged);
- the physics: a converted row holds its bus near the setpoint, more tightly as
  the droop steepens, and saturates at its reactive limit.

The end-to-end agreement against a live ``pp.runpp`` on transmission benchmarks is
``tests/reference/test_pandapower_gen_volt_var.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

# ---------------------------------------------------------------------------
# Optional pandapower guard (matches existing reference test conventions)
# ---------------------------------------------------------------------------
try:
    import pandapower as pp

    _PP_AVAILABLE = True
except ImportError:
    _PP_AVAILABLE = False

if not _PP_AVAILABLE:
    pytest.skip("pandapower not installed", allow_module_level=True)

from pgml.convert.pandapower import (  # noqa: E402
    DEFAULT_GEN_VOLT_VAR_SLOPE_PU,
    GenMode,
    PhaseMode,
    to_grid,
)
from pgml.errors import ConversionError  # noqa: E402
from pgml.schemas.grid_schema import Generator, Phase, QReference, VoltVarControl  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

CDT = torch.complex128


def _net(
    *,
    with_gen: bool = True,
    gen_at_slack: bool = False,
    load_mw: float = 2.0,
    **gen_kwargs,
):
    """Two-bus 20 kV feeder (20 km line — weak enough that the droop's reactive
    range can actually move the bus voltage): slack, line, load, one ``gen``."""
    net = pp.create_empty_network(f_hz=50.0)
    b0 = pp.create_bus(net, vn_kv=20.0, name="slack")
    b1 = pp.create_bus(net, vn_kv=20.0, name="pv")
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=20.0,
        r_ohm_per_km=0.3,
        x_ohm_per_km=0.4,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.create_load(net, bus=b1, p_mw=load_mw, q_mvar=0.8)
    if with_gen:
        kwargs = dict(p_mw=1.0, vm_pu=1.02, min_q_mvar=-3.0, max_q_mvar=3.0)
        kwargs.update(gen_kwargs)
        pp.create_gen(net, bus=b0 if gen_at_slack else b1, **kwargs)
    return net


def _gens(grid) -> list[Generator]:
    return [a for a in grid.appliances if isinstance(a, Generator)]


def _vm_pu(net, grid, id_map, pp_bus: int, *, method: str = "newton") -> float:
    """Converged |V| at ``pp_bus`` in per unit of that bus's nominal voltage."""
    res = solve_power_flow(
        grid, slack="ideal", method=method, tol=1e-9, max_iter=200, dtype=CDT
    )
    assert res.converged, f"solve did not converge (residual={float(res.residual):.3e})"
    row = res.index.row(id_map["bus"][pp_bus], Phase.A)
    v = res.v.reshape(-1)[row].item()
    return abs(v) / (float(net.bus.at[pp_bus, "vn_kv"]) * 1_000.0)


# ---------------------------------------------------------------------------
# 1. The default is unchanged
# ---------------------------------------------------------------------------
class TestDefaultDrops:
    def test_default_drops_gen_and_warns(self, caplog):
        with caplog.at_level("WARNING", logger="pgml"):
            grid, id_map = to_grid(_net())
        assert _gens(grid) == []
        assert id_map["gen"] == {}
        assert any(
            "'gen'" in r.message and "NOT converted" in r.message
            for r in caplog.records
        ), f"expected a gen drop warning, got: {[r.message for r in caplog.records]}"

    def test_default_matches_a_net_without_gen_rows(self):
        """The default conversion of a net WITH ``gen`` rows is identical to the
        conversion of the same net without them — the table changes nothing."""
        with_gen, _ = to_grid(_net(with_gen=True))
        without_gen, _ = to_grid(_net(with_gen=False))
        assert with_gen.model_dump() == without_gen.model_dump()

    def test_volt_var_mode_no_longer_reports_gen_as_dropped(self, caplog):
        with caplog.at_level("WARNING", logger="pgml"):
            to_grid(_net(), gen_mode=GenMode.VOLT_VAR_APPROX)
        assert not any(
            "'gen'" in r.message and "NOT converted" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# 2. The mapping
# ---------------------------------------------------------------------------
class TestVoltVarMapping:
    def test_row_becomes_a_volt_var_generator(self):
        grid, id_map = to_grid(_net(), gen_mode=GenMode.VOLT_VAR_APPROX)
        gens = _gens(grid)
        assert len(gens) == 1
        gen = gens[0]
        assert list(id_map["gen"].values()) == [gen.id]
        assert gen.name == "gen_0"
        assert gen.p_nom_w == pytest.approx(1.0e6)
        # A controlled generator's reactive nameplate is never read.
        assert gen.q_nom_var == 0.0
        assert isinstance(gen.control, VoltVarControl)
        assert gen.control.q_reference is QReference.RATED
        assert gen.control.smoothing == 0.0

    def test_curve_is_the_documented_clamped_droop(self):
        """``Q(V) = clamp(-slope * q_base * (v_pu - vm_pu), q_min, q_max)``, stored
        as the two endpoints of the linear segment with constant extrapolation."""
        slope = DEFAULT_GEN_VOLT_VAR_SLOPE_PU
        grid, _ = to_grid(_net(), gen_mode=GenMode.VOLT_VAR_APPROX)
        control = _gens(grid)[0].control
        q_base = math.hypot(1.0e6, 3.0e6)  # hypot(P, max(|q_min|, |q_max|))
        assert control.s_rated_va == pytest.approx(q_base)
        y_max, y_min = 3.0e6 / q_base, -3.0e6 / q_base
        assert control.characteristic.y_values == pytest.approx((y_max, y_min))
        assert control.characteristic.x_values == pytest.approx(
            (1.02 - y_max / slope, 1.02 - y_min / slope)
        )

    def test_slope_only_moves_the_breakpoints(self):
        """A steeper droop narrows the voltage band; the saturation levels and the
        reactive base are untouched."""
        soft, _ = to_grid(
            _net(), gen_mode=GenMode.VOLT_VAR_APPROX, gen_volt_var_slope_pu=50.0
        )
        stiff, _ = to_grid(
            _net(), gen_mode=GenMode.VOLT_VAR_APPROX, gen_volt_var_slope_pu=500.0
        )
        c_soft, c_stiff = _gens(soft)[0].control, _gens(stiff)[0].control
        assert c_soft.s_rated_va == pytest.approx(c_stiff.s_rated_va)
        assert c_soft.characteristic.y_values == pytest.approx(
            c_stiff.characteristic.y_values
        )
        width_soft = (
            c_soft.characteristic.x_values[1] - c_soft.characteristic.x_values[0]
        )
        width_stiff = (
            c_stiff.characteristic.x_values[1] - c_stiff.characteristic.x_values[0]
        )
        assert width_soft == pytest.approx(10.0 * width_stiff)

    def test_scaling_scales_p_but_not_the_reactive_limits(self):
        """pandapower's own build stage multiplies ``p_mw`` by ``scaling`` and reads
        ``min/max_q_mvar`` raw (``add_q_constraints``); the converter mirrors it."""
        grid, _ = to_grid(_net(scaling=0.5), gen_mode=GenMode.VOLT_VAR_APPROX)
        gen = _gens(grid)[0]
        assert gen.p_nom_w == pytest.approx(0.5e6)
        q_base = math.hypot(0.5e6, 3.0e6)
        assert gen.control.s_rated_va == pytest.approx(q_base)
        assert gen.control.characteristic.y_values == pytest.approx(
            (3.0e6 / q_base, -3.0e6 / q_base)
        )

    def test_out_of_service_row_is_skipped(self):
        grid, id_map = to_grid(_net(in_service=False), gen_mode=GenMode.VOLT_VAR_APPROX)
        assert _gens(grid) == [] and id_map["gen"] == {}

    def test_three_phase_mode_splits_the_rating_per_phase(self):
        """The control is evaluated per ELEMENT against that element's share of the
        active power, so the rating and limits divide by the phase count."""
        grid, _ = to_grid(
            _net(),
            phase_mode=PhaseMode.THREE_PHASE,
            gen_mode=GenMode.VOLT_VAR_APPROX,
        )
        gen = _gens(grid)[0]
        assert gen.phases == (Phase.A, Phase.B, Phase.C)
        assert gen.p_nom_w == pytest.approx(1.0e6)  # total nameplate, split by assembly
        assert gen.control.s_rated_va == pytest.approx(math.hypot(1.0e6, 3.0e6) / 3.0)


# ---------------------------------------------------------------------------
# 3. Missing / degenerate reactive limits
# ---------------------------------------------------------------------------
class TestReactiveLimitFallbacks:
    def test_missing_limits_fall_back_to_the_apparent_power_rating(self, caplog):
        with caplog.at_level("WARNING", logger="pgml"):
            grid, _ = to_grid(
                _net(min_q_mvar=float("nan"), max_q_mvar=float("nan"), sn_mva=2.5),
                gen_mode=GenMode.VOLT_VAR_APPROX,
            )
        envelope = math.sqrt((2.5e6) ** 2 - (1.0e6) ** 2)
        control = _gens(grid)[0].control
        assert control.s_rated_va == pytest.approx(math.hypot(1.0e6, envelope))
        assert any("no min_q_mvar/max_q_mvar" in r.message for r in caplog.records)

    def test_missing_limits_and_rating_fall_back_to_the_active_power(self):
        grid, _ = to_grid(
            _net(min_q_mvar=float("nan"), max_q_mvar=float("nan")),
            gen_mode=GenMode.VOLT_VAR_APPROX,
        )
        control = _gens(grid)[0].control
        assert control.s_rated_va == pytest.approx(math.hypot(1.0e6, 1.0e6))

    def test_one_sided_limit_uses_the_envelope_only_on_the_missing_side(self):
        grid, _ = to_grid(
            _net(min_q_mvar=-0.4, max_q_mvar=float("nan"), sn_mva=2.5),
            gen_mode=GenMode.VOLT_VAR_APPROX,
        )
        envelope = math.sqrt((2.5e6) ** 2 - (1.0e6) ** 2)
        control = _gens(grid)[0].control
        q_base = math.hypot(1.0e6, envelope)  # the wider (synthesised) side dominates
        assert control.s_rated_va == pytest.approx(q_base)
        assert control.characteristic.y_values == pytest.approx(
            (envelope / q_base, -0.4e6 / q_base)
        )

    def test_coincident_limits_convert_as_a_plain_pq_generator(self):
        """No reactive freedom means no PV bus: a fixed-Q injection, no control."""
        grid, _ = to_grid(
            _net(min_q_mvar=0.25, max_q_mvar=0.25), gen_mode=GenMode.VOLT_VAR_APPROX
        )
        gen = _gens(grid)[0]
        assert gen.control is None
        assert gen.q_nom_var == pytest.approx(0.25e6)

    def test_inconsistent_limits_raise(self):
        with pytest.raises(ConversionError, match="inconsistent"):
            to_grid(
                _net(min_q_mvar=1.0, max_q_mvar=-1.0),
                gen_mode=GenMode.VOLT_VAR_APPROX,
            )

    def test_non_positive_slope_raises(self):
        with pytest.raises(ConversionError, match="must be positive"):
            to_grid(_net(), gen_mode=GenMode.VOLT_VAR_APPROX, gen_volt_var_slope_pu=0.0)


# ---------------------------------------------------------------------------
# 4. The slack-bus rule
# ---------------------------------------------------------------------------
class TestSlackBusRule:
    def test_row_on_the_ext_grid_bus_is_skipped(self, caplog):
        with caplog.at_level("WARNING", logger="pgml"):
            grid, id_map = to_grid(
                _net(gen_at_slack=True), gen_mode=GenMode.VOLT_VAR_APPROX
            )
        assert _gens(grid) == [] and id_map["gen"] == {}
        assert any("the ext_grid bus" in r.message for r in caplog.records)

    def test_slack_flagged_row_is_skipped(self, caplog):
        with caplog.at_level("WARNING", logger="pgml"):
            grid, id_map = to_grid(_net(slack=True), gen_mode=GenMode.VOLT_VAR_APPROX)
        assert _gens(grid) == [] and id_map["gen"] == {}
        assert any("flagged slack=True" in r.message for r in caplog.records)

    def test_out_of_service_ext_grid_does_not_shadow_its_bus(self):
        net = _net(gen_at_slack=True)
        pp.create_bus(net, vn_kv=20.0, name="second slack")
        net.ext_grid.at[0, "in_service"] = False
        pp.create_ext_grid(net, bus=2, vm_pu=1.0, va_degree=0.0)
        pp.create_line_from_parameters(
            net,
            from_bus=2,
            to_bus=0,
            length_km=0.1,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.1,
            c_nf_per_km=0.0,
            max_i_ka=1.0,
        )
        grid, id_map = to_grid(net, gen_mode=GenMode.VOLT_VAR_APPROX)
        assert len(id_map["gen"]) == 1


# ---------------------------------------------------------------------------
# 5. The physics
# ---------------------------------------------------------------------------
class TestVoltVarPhysics:
    def test_droop_lifts_the_bus_towards_the_setpoint(self):
        """Without the ``gen`` the loaded bus sags well below 1.02 pu; with it the
        droop injects vars and pulls the bus up to the setpoint."""
        net = _net()
        sagged, id_map_drop = to_grid(net)
        held, id_map_gen = to_grid(net, gen_mode=GenMode.VOLT_VAR_APPROX)
        v_sag = _vm_pu(net, sagged, id_map_drop, 1)
        v_held = _vm_pu(net, held, id_map_gen, 1)
        assert v_sag < 1.0 < v_held
        assert abs(v_held - 1.02) < 2e-3

    @pytest.mark.parametrize(
        "slope,band", [(50.0, 2e-2), (500.0, 2e-3), (5000.0, 2e-4)]
    )
    def test_steeper_droop_holds_the_setpoint_tighter(self, slope, band):
        """The regulation error scales as 1 / slope — the documented accuracy vs
        conditioning trade, measured."""
        net = _net()
        grid, id_map = to_grid(
            net, gen_mode=GenMode.VOLT_VAR_APPROX, gen_volt_var_slope_pu=slope
        )
        assert abs(_vm_pu(net, grid, id_map, 1) - 1.02) < band

    def test_reactive_limit_saturates_like_a_pv_to_pq_switch(self):
        """A generator that cannot reach the setpoint within its reactive limit
        sticks at the limit: the bus lands BELOW the setpoint, and steepening the
        droop tenfold no longer moves it (the curve is saturated, not sloped)."""
        net = _net(max_q_mvar=0.05, min_q_mvar=-0.05)
        grid_a, map_a = to_grid(
            net, gen_mode=GenMode.VOLT_VAR_APPROX, gen_volt_var_slope_pu=500.0
        )
        grid_b, map_b = to_grid(
            net, gen_mode=GenMode.VOLT_VAR_APPROX, gen_volt_var_slope_pu=5000.0
        )
        v_a = _vm_pu(net, grid_a, map_a, 1)
        v_b = _vm_pu(net, grid_b, map_b, 1)
        assert v_a < 1.02 - 1e-3
        assert abs(v_a - v_b) < 1e-9

    def test_saturated_generator_matches_pandapower_q_limit_enforcement(self):
        """With the limit binding, the approximation reduces to pandapower's own
        PV->PQ result: the bus voltage agrees to within the solver tolerance."""
        net = _net(max_q_mvar=0.05, min_q_mvar=-0.05)
        pp.runpp(net, numba=False, enforce_q_lims=True)
        assert net.converged
        grid, id_map = to_grid(
            net, gen_mode=GenMode.VOLT_VAR_APPROX, gen_volt_var_slope_pu=5000.0
        )
        assert _vm_pu(net, grid, id_map, 1) == pytest.approx(
            float(net.res_bus.at[1, "vm_pu"]), abs=1e-6
        )
