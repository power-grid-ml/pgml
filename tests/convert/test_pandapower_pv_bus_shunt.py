"""``net.gen`` -> PV terminal and ``net.shunt`` -> fixed admittance (the mapping).

``GenMode.VOLTAGE_REGULATING`` (the converter default) turns each in-service
``net.gen`` row into a :class:`~pgml.schemas.grid_schema.Generator` carrying a
:class:`~pgml.schemas.grid_schema.VoltageRegulation` block, and ``net.shunt`` into a
WYE :class:`~pgml.schemas.grid_schema.ShuntAppliance`. Checked here, without a solve
unless stated:

- the gen mapping: ``vm_pu`` -> setpoint, ``min_q_mvar``/``max_q_mvar`` -> limits in
  var (a MISSING limit stays unbounded, as pandapower reads it), ``scaling`` on the
  active power only, the reactive nameplate left at zero, ``in_service``;
- several rows on ONE bus merging into a single regulating generator (one bus carries
  one setpoint), with summed active power and summed limits;
- the slack-bus rule (a row at the ``ext_grid`` bus is skipped and logged) and the
  ``slack=True`` note;
- the shunt mapping: ``G = p_mw/vn_kv**2`` and ``C = -q_mvar/(2*pi*f0*vn_kv**2)``
  referred to the SHUNT's own rated voltage, the ``step`` multiplier, the sign (a
  positive ``q_mvar`` consumes reactive power, hence a NEGATIVE capacitance, with a
  WARNING), and the physics (a capacitive shunt raises its bus voltage by the amount
  pandapower's own solve gives).

The cross-tool comparison on the MATPOWER benchmarks is
``tests/reference/test_pandapower_pv_bus.py``.
"""

from __future__ import annotations

import math

import pandapower as pp
import pytest
import torch

from pgml.assembly import assemble_network_ybus, node_phase_index
from pgml.convert.pandapower import GenMode, PhaseMode, to_grid
from pgml.schemas.grid_schema import (
    Generator,
    Phase,
    ShuntAppliance,
    VoltageRegulation,
    WindingConnection,
)
from pgml.solver import solve_power_flow

CDT = torch.complex128
_F0 = 50.0


def _net(*, gens=(), shunts=(), gen_at_slack=False, load_mw=2.0):
    """Two-bus 20 kV feeder (20 km line) with optional gen / shunt rows at bus 1."""
    net = pp.create_empty_network(f_hz=_F0)
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
    for kwargs in gens:
        pp.create_gen(net, bus=b0 if gen_at_slack else b1, **kwargs)
    for kwargs in shunts:
        pp.create_shunt(net, bus=b1, **kwargs)
    return net


def _gens(grid) -> list[Generator]:
    return [a for a in grid.appliances if isinstance(a, Generator)]


def _shunts(grid) -> list[ShuntAppliance]:
    return [a for a in grid.appliances if isinstance(a, ShuntAppliance)]


_GEN = dict(p_mw=1.0, vm_pu=1.02, min_q_mvar=-3.0, max_q_mvar=3.0)


# ---------------------------------------------------------------------------
# 1. net.gen -> PV terminal
# ---------------------------------------------------------------------------
class TestGenMapping:
    def test_row_becomes_a_regulating_generator(self):
        grid, id_map = to_grid(_net(gens=[_GEN]))
        gens = _gens(grid)
        assert len(gens) == 1
        gen = gens[0]
        assert list(id_map["gen"].values()) == [gen.id]
        assert gen.p_nom_w == pytest.approx(1.0e6)
        assert gen.q_nom_var == 0.0  # the reactive power is solved, not set
        assert gen.control is None
        reg = gen.voltage_regulation
        assert isinstance(reg, VoltageRegulation)
        assert reg.v_set_pu == pytest.approx(1.02)
        assert reg.q_min_var == pytest.approx(-3.0e6)
        assert reg.q_max_var == pytest.approx(3.0e6)

    def test_missing_limits_stay_unbounded(self):
        """pandapower reads an unset limit as effectively unbounded; so does this."""
        row = dict(_GEN, min_q_mvar=float("nan"), max_q_mvar=float("nan"), sn_mva=2.5)
        reg = _gens(to_grid(_net(gens=[row]))[0])[0].voltage_regulation
        assert reg.q_min_var is None and reg.q_max_var is None

    def test_one_sided_limit(self):
        row = dict(_GEN, max_q_mvar=float("nan"))
        reg = _gens(to_grid(_net(gens=[row]))[0])[0].voltage_regulation
        assert reg.q_min_var == pytest.approx(-3.0e6)
        assert reg.q_max_var is None

    def test_missing_setpoint_defaults_to_one_per_unit(self):
        row = {k: v for k, v in _GEN.items() if k != "vm_pu"}
        reg = _gens(to_grid(_net(gens=[dict(row, vm_pu=float("nan"))]))[0])[
            0
        ].voltage_regulation
        assert reg.v_set_pu == pytest.approx(1.0)

    def test_scaling_scales_p_only(self):
        """pandapower's own build scales ``p_mw`` and reads the limits raw."""
        gen = _gens(to_grid(_net(gens=[dict(_GEN, scaling=0.5)]))[0])[0]
        assert gen.p_nom_w == pytest.approx(0.5e6)
        assert gen.voltage_regulation.q_max_var == pytest.approx(3.0e6)

    def test_out_of_service_row_is_skipped(self):
        grid, id_map = to_grid(_net(gens=[dict(_GEN, in_service=False)]))
        assert _gens(grid) == [] and id_map["gen"] == {}

    def test_three_phase_mode_keeps_the_totals(self):
        """The setpoint is per unit either way; the limits stay machine TOTALS (the
        solver splits them over the phases)."""
        grid, _ = to_grid(_net(gens=[_GEN]), phase_mode=PhaseMode.THREE_PHASE)
        gen = _gens(grid)[0]
        assert gen.phases == (Phase.A, Phase.B, Phase.C)
        assert gen.p_nom_w == pytest.approx(1.0e6)
        assert gen.voltage_regulation.v_set_pu == pytest.approx(1.02)
        assert gen.voltage_regulation.q_max_var == pytest.approx(3.0e6)

    def test_rows_on_one_bus_merge_into_one_terminal(self):
        """One bus carries one voltage setpoint: the rows' active powers and limits
        add, and both source rows map to the merged generator's id."""
        grid, id_map = to_grid(
            _net(
                gens=[
                    dict(_GEN, p_mw=1.0, max_q_mvar=3.0, min_q_mvar=-3.0),
                    dict(_GEN, p_mw=0.4, max_q_mvar=1.0, min_q_mvar=-0.5),
                ]
            )
        )
        gens = _gens(grid)
        assert len(gens) == 1
        assert set(id_map["gen"].values()) == {gens[0].id}
        assert gens[0].p_nom_w == pytest.approx(1.4e6)
        assert gens[0].voltage_regulation.q_max_var == pytest.approx(4.0e6)
        assert gens[0].voltage_regulation.q_min_var == pytest.approx(-3.5e6)

    def test_merged_rows_with_one_unbounded_side_stay_unbounded(self):
        grid, _ = to_grid(_net(gens=[dict(_GEN), dict(_GEN, max_q_mvar=float("nan"))]))
        assert _gens(grid)[0].voltage_regulation.q_max_var is None

    def test_differing_setpoints_on_one_bus_warn(self, caplog):
        with caplog.at_level("WARNING", logger="pgml"):
            grid, _ = to_grid(
                _net(gens=[dict(_GEN, vm_pu=1.02), dict(_GEN, vm_pu=1.05)])
            )
        assert _gens(grid)[0].voltage_regulation.v_set_pu == pytest.approx(1.02)
        assert any("DIFFERENT vm_pu" in r.message for r in caplog.records)

    def test_row_at_the_slack_bus_is_skipped_and_logged(self, caplog):
        with caplog.at_level("WARNING", logger="pgml"):
            grid, id_map = to_grid(_net(gens=[_GEN], gen_at_slack=True))
        assert _gens(grid) == [] and id_map["gen"] == {}
        assert any("ext_grid bus" in r.message for r in caplog.records)

    def test_slack_flagged_row_is_converted_with_a_note(self, caplog):
        with caplog.at_level("WARNING", logger="pgml"):
            grid, _ = to_grid(_net(gens=[dict(_GEN, slack=True)]))
        assert _gens(grid)[0].voltage_regulation is not None
        assert any("distributed" in r.message for r in caplog.records)

    def test_solved_terminal_holds_the_setpoint(self):
        grid, id_map = to_grid(_net(gens=[_GEN]))
        res = solve_power_flow(grid, slack="ideal", tol=1e-9, max_iter=60, dtype=CDT)
        assert res.converged
        row = res.index.row(id_map["bus"][1], Phase.A)
        v_pu = float(res.v.reshape(-1)[row].abs()) / 20_000.0
        assert v_pu == pytest.approx(1.02, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. net.shunt -> fixed admittance
# ---------------------------------------------------------------------------
class TestShuntMapping:
    def test_capacitive_shunt_values(self):
        """``q_mvar < 0`` injects reactive power: a positive capacitance."""
        grid, id_map = to_grid(_net(shunts=[dict(q_mvar=-1.2, p_mw=0.05)]))
        sh = _shunts(grid)
        assert len(sh) == 1 and list(id_map["shunt"].values()) == [sh[0].id]
        v_sq = 20_000.0**2
        assert sh[0].conductance_s == pytest.approx((0.05e6 / v_sq,))
        assert sh[0].capacitance_f == pytest.approx(
            (1.2e6 / v_sq / (2.0 * math.pi * _F0),)
        )
        assert sh[0].connection is WindingConnection.WYE

    def test_inductive_shunt_carries_an_inductance(self):
        """``q_mvar > 0`` consumes reactive power: a reactor, stored as an inductance.

        ``L = U**2 / (2*pi*f0*q)``, which reproduces the fundamental susceptance
        ``B = -q/U**2`` exactly while falling as ``1/h`` above it, where a negative
        capacitance would rise as ``h``.
        """
        grid, _ = to_grid(_net(shunts=[dict(q_mvar=1.0, p_mw=0.0)]))
        sh = _shunts(grid)[0]
        v_sq = 20_000.0**2
        b_f0 = 1.0e6 / v_sq  # |B| at f0 [S]
        assert sh.capacitance_f == pytest.approx((0.0,))
        assert sh.inductance_h == pytest.approx((1.0 / (2.0 * math.pi * _F0 * b_f0),))

    def test_inductive_shunt_susceptance_falls_with_the_order(self):
        """The stamped susceptance is exact at f0 and falls as ``1/h`` above it."""
        grid, id_map = to_grid(_net(shunts=[dict(q_mvar=1.0, p_mw=0.0)]))
        plain, _ = to_grid(_net())
        index = node_phase_index(grid)
        row = index.row(id_map["bus"][1], Phase.A)
        orders = [1.0, 5.0, 13.0]
        freqs = torch.tensor([_F0 * h for h in orders], dtype=torch.float64)
        y_with = assemble_network_ybus(grid, freqs, dtype=CDT).Y
        y_without = assemble_network_ybus(plain, freqs, dtype=CDT).Y
        b_f0 = -1.0e6 / 20_000.0**2  # susceptance the shunt must add at f0 [S]
        for k, h in enumerate(orders):
            added = complex(y_with[k, row, row] - y_without[k, row, row])
            assert added.real == pytest.approx(0.0, abs=1e-15)
            assert added.imag == pytest.approx(b_f0 / h, rel=1e-12)

    def test_step_multiplies_the_admittance(self):
        grid, _ = to_grid(_net(shunts=[dict(q_mvar=-1.0, p_mw=0.0, step=3)]))
        v_sq = 20_000.0**2
        assert grid.appliances[-1].capacitance_f == pytest.approx(
            (3.0 * 1.0e6 / v_sq / (2.0 * math.pi * _F0),)
        )

    def test_referred_to_the_shunt_rated_voltage(self):
        """A shunt rated below its bus presents a LARGER admittance, exactly as
        pandapower's ``(vn_bus / vn_shunt)**2`` ratio does."""
        grid, _ = to_grid(_net(shunts=[dict(q_mvar=-1.0, p_mw=0.0, vn_kv=10.0)]))
        assert grid.appliances[-1].capacitance_f == pytest.approx(
            (1.0e6 / (10_000.0**2) / (2.0 * math.pi * _F0),)
        )

    def test_out_of_service_shunt_is_skipped(self):
        grid, id_map = to_grid(
            _net(shunts=[dict(q_mvar=-1.0, p_mw=0.0, in_service=False)])
        )
        assert _shunts(grid) == [] and id_map["shunt"] == {}

    def test_shunt_is_no_longer_reported_as_dropped(self, caplog):
        with caplog.at_level("WARNING", logger="pgml"):
            to_grid(_net(shunts=[dict(q_mvar=-1.0, p_mw=0.0)]))
        assert not any(
            "shunt" in r.message and "NOT converted" in r.message
            for r in caplog.records
        )

    def test_three_phase_mode_repeats_the_value_per_phase(self):
        grid, _ = to_grid(
            _net(shunts=[dict(q_mvar=-1.2, p_mw=0.0)]),
            phase_mode=PhaseMode.THREE_PHASE,
        )
        sh = _shunts(grid)[0]
        assert sh.phases == (Phase.A, Phase.B, Phase.C)
        v_sq = 20_000.0**2  # the per-phase value is the same number in both modes
        assert sh.capacitance_f == pytest.approx(
            (1.2e6 / v_sq / (2.0 * math.pi * _F0),) * 3
        )

    def test_capacitive_shunt_raises_the_bus_voltage_like_runpp(self):
        net = _net(shunts=[dict(q_mvar=-1.5, p_mw=0.0)])
        pp.runpp(net, numba=False, calculate_voltage_angles=True, tolerance_mva=1e-10)
        grid, id_map = to_grid(net)
        res = solve_power_flow(grid, slack="ideal", tol=1e-10, max_iter=80, dtype=CDT)
        assert res.converged
        row = res.index.row(id_map["bus"][1], Phase.A)
        v_pu = float(res.v.reshape(-1)[row].abs()) / 20_000.0
        assert v_pu == pytest.approx(float(net.res_bus.vm_pu.at[1]), abs=1e-10)

    def test_gen_and_shunt_together_match_runpp(self):
        """The two new elements on one bus, against a live solve."""
        net = _net(gens=[_GEN], shunts=[dict(q_mvar=-1.5, p_mw=0.02)])
        pp.runpp(
            net,
            numba=False,
            calculate_voltage_angles=True,
            tolerance_mva=1e-10,
            enforce_q_lims=True,
        )
        grid, id_map = to_grid(net)
        res = solve_power_flow(
            grid, slack="ideal", method="newton", tol=1e-9, max_iter=80, dtype=CDT
        )
        assert res.converged
        for pp_bus, node in id_map["bus"].items():
            row = res.index.row(node, Phase.A)
            v_pu = float(res.v.reshape(-1)[row].abs()) / (
                float(net.bus.at[pp_bus, "vn_kv"]) * 1_000.0
            )
            assert v_pu == pytest.approx(float(net.res_bus.vm_pu.at[pp_bus]), abs=1e-9)
        gid = id_map["gen"][0]
        q_pgml = float(res.regulation.q_var[gid]) / 1.0e6
        assert q_pgml == pytest.approx(float(net.res_gen.q_mvar.at[0]), abs=1e-7)


# ---------------------------------------------------------------------------
# 3. The other gen modes stay available
# ---------------------------------------------------------------------------
class TestModeSelection:
    def test_drop_mode_still_drops(self):
        grid, id_map = to_grid(_net(gens=[_GEN]), gen_mode=GenMode.DROP)
        assert _gens(grid) == [] and id_map["gen"] == {}

    def test_volt_var_mode_still_builds_the_droop(self):
        grid, _ = to_grid(_net(gens=[_GEN]), gen_mode=GenMode.VOLT_VAR_APPROX)
        gen = _gens(grid)[0]
        assert gen.voltage_regulation is None
        assert gen.control is not None

    def test_chosen_mode_is_logged(self, caplog):
        with caplog.at_level("INFO", logger="pgml"):
            to_grid(_net(gens=[_GEN]))
        assert any("EXACT PV terminal" in r.message for r in caplog.records)
