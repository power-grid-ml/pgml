"""Static-generator conversion + loud drop warnings (pandapower / pgm).

Two behaviours per converter:

- a static generator (pandapower ``sgen`` / pgm ``sym_gen``) becomes a pgml
  :class:`Generator` with the GENERATION-POSITIVE nameplate — verified through
  the physics: adding it must RAISE the load-bus voltage (it offsets the load);
- an element table the converter cannot read is never dropped silently — a
  WARNING names the kind and the count.
"""

from __future__ import annotations

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

from pgml.convert.pandapower import to_grid as pp_to_grid  # noqa: E402
from pgml.convert.pgm import to_grid as pgm_to_grid  # noqa: E402
from pgml.schemas.grid_schema import Generator  # noqa: E402
from pgml.solver import solve_power_flow  # noqa: E402

CDT = torch.complex128


def _pp_net(*, with_sgen: bool, with_motor: bool = False, with_shunt: bool = False):
    net = pp.create_empty_network(f_hz=50.0)
    b0 = pp.create_bus(net, vn_kv=20.0)
    b1 = pp.create_bus(net, vn_kv=20.0)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=0.2,
        x_ohm_per_km=0.3,
        c_nf_per_km=10.0,
        max_i_ka=1.0,
    )
    pp.create_load(net, bus=b1, p_mw=0.5, q_mvar=0.1)
    if with_sgen:
        pp.create_sgen(net, bus=b1, p_mw=0.3, q_mvar=0.05)
    if with_shunt:
        pp.create_shunt(net, bus=b1, q_mvar=-0.1, p_mw=0.0)
    if with_motor:
        pp.create_motor(net, bus=b1, pn_mech_mw=0.1, cos_phi=0.9)
    return net


def _load_bus_vmag(grid) -> float:
    res = solve_power_flow(grid, slack="ideal", dtype=CDT)
    assert res.converged
    return float(res.v.abs().min())  # the (only) load bus has the lowest |V|


class TestPandapowerSgen:
    def test_sgen_converts_to_generator(self):
        grid, id_map = pp_to_grid(_pp_net(with_sgen=True))
        gens = [a for a in grid.appliances if isinstance(a, Generator)]
        assert len(gens) == 1
        assert gens[0].p_nom_w == pytest.approx(0.3e6)
        assert gens[0].q_nom_var == pytest.approx(0.05e6)
        assert id_map["sgen"] and list(id_map["sgen"].values()) == [gens[0].id]

    def test_sgen_raises_load_bus_voltage(self):
        """Generation-positive sign, verified through the solve: injecting at
        the load bus must reduce the voltage drop."""
        v_without = _load_bus_vmag(pp_to_grid(_pp_net(with_sgen=False))[0])
        v_with = _load_bus_vmag(pp_to_grid(_pp_net(with_sgen=True))[0])
        assert v_with > v_without

    def test_unread_table_warns(self, caplog):
        """``motor`` is not read by the converter, so it is reported loudly."""
        with caplog.at_level("WARNING", logger="pgml"):
            pp_to_grid(_pp_net(with_sgen=False, with_motor=True))
        assert any(
            "motor" in r.message and "NOT converted" in r.message
            for r in caplog.records
        ), f"expected a motor drop warning, got: {[r.message for r in caplog.records]}"


class TestPgmSymGen:
    @staticmethod
    def _input(*, with_gen: bool, with_shunt: bool = False):
        pgm = pytest.importorskip("power_grid_model", exc_type=ImportError)
        LoadGenType, initialize_array = pgm.LoadGenType, pgm.initialize_array

        node = initialize_array("input", "node", 2)
        node["id"] = [1, 2]
        node["u_rated"] = [20000.0, 20000.0]
        line = initialize_array("input", "line", 1)
        line["id"] = [10]
        line["from_node"] = [1]
        line["to_node"] = [2]
        line["from_status"] = [1]
        line["to_status"] = [1]
        line["r1"] = [0.4]
        line["x1"] = [0.6]
        line["c1"] = [1e-8]
        line["tan1"] = [0.0]
        source = initialize_array("input", "source", 1)
        source["id"] = [20]
        source["node"] = [1]
        source["status"] = [1]
        source["u_ref"] = [1.0]
        source["u_ref_angle"] = [0.0]
        source["sk"] = [1e12]
        source["rx_ratio"] = [0.1]
        sym = initialize_array("input", "sym_load", 1)
        sym["id"] = [30]
        sym["node"] = [2]
        sym["status"] = [1]
        sym["type"] = [LoadGenType.const_power]
        sym["p_specified"] = [5e5]
        sym["q_specified"] = [1e5]
        data = {"node": node, "line": line, "source": source, "sym_load": sym}
        if with_gen:
            gen = initialize_array("input", "sym_gen", 1)
            gen["id"] = [50]
            gen["node"] = [2]
            gen["status"] = [1]
            gen["type"] = [LoadGenType.const_power]
            gen["p_specified"] = [3e5]
            gen["q_specified"] = [5e4]
            data["sym_gen"] = gen
        if with_shunt:
            shunt = initialize_array("input", "shunt", 1)
            shunt["id"] = [60]
            shunt["node"] = [2]
            shunt["status"] = [1]
            shunt["g1"] = [1e-6]
            shunt["b1"] = [-1e-5]
            data["shunt"] = shunt
        return data

    def test_sym_gen_converts_to_generator(self):
        grid, id_map = pgm_to_grid(self._input(with_gen=True), base_frequency_hz=50.0)
        gens = [a for a in grid.appliances if isinstance(a, Generator)]
        assert len(gens) == 1
        assert gens[0].p_nom_w == pytest.approx(3e5)
        assert id_map["sym_gen"] and list(id_map["sym_gen"].values()) == [gens[0].id]

    def test_sym_gen_raises_load_bus_voltage(self):
        v_without = _load_bus_vmag(
            pgm_to_grid(self._input(with_gen=False), base_frequency_hz=50.0)[0]
        )
        v_with = _load_bus_vmag(
            pgm_to_grid(self._input(with_gen=True), base_frequency_hz=50.0)[0]
        )
        assert v_with > v_without

    def test_unread_component_warns(self, caplog):
        with caplog.at_level("WARNING", logger="pgml"):
            pgm_to_grid(
                self._input(with_gen=False, with_shunt=True), base_frequency_hz=50.0
            )
        assert any(
            "shunt" in r.message and "NOT converted" in r.message
            for r in caplog.records
        ), f"expected a shunt drop warning, got: {[r.message for r in caplog.records]}"
