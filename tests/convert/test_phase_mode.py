"""Phase-mode integration tests for the pandapower + pgm converters.

Covers:
- ``SINGLE_PHASE_EQUIV`` (the default) reproduces the historical positive-sequence
  single-phase-equivalent output (phases, matrices, ids, id_map) — the bit-exact
  regression gate stays green via the reference oracle suite; here we assert the
  invariants on a small net.
- ``THREE_PHASE`` produces abc nodes, 3x3 line matrices with the correct
  self/mutual from sequence (vs a hand computation), and balanced source angles.
- A pandapower ``asymmetric_load`` and a pgm ``asym_load`` yield per-phase P/Q +
  connection on the converted Load.
- A ``THREE_PHASE`` grid assembles and solves the power flow without error.
- A per-phase ``spectrum_per_phase`` added to a converted abc load runs through
  ``solve_harmonic_flow``.
"""

from __future__ import annotations

import pytest

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

from pgml.convert._common import PhaseMode  # noqa: E402
from pgml.convert.pandapower import to_grid as pp_to_grid  # noqa: E402
from pgml.convert.pgm import to_grid as pgm_to_grid  # noqa: E402
from pgml.schemas.grid_schema import (  # noqa: E402
    HarmonicComponent,
    Line,
    Load,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
    WindingConnection,
)
from pgml.solver import solve_harmonic_flow, solve_power_flow  # noqa: E402

ABC = (Phase.A, Phase.B, Phase.C)


# ----------------------------------------------------------------------- #
# Small pandapower net builders
# ----------------------------------------------------------------------- #
def _pp_net(with_asym: bool = False):
    """A 2-bus pandapower net: slack -> line -> balanced load (+ optional asym)."""
    net = pp.create_empty_network(f_hz=50.0)
    b0 = pp.create_bus(net, vn_kv=20.0, name="slack")
    b1 = pp.create_bus(net, vn_kv=20.0, name="load")
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
    if with_asym:
        pp.create_asymmetric_load(
            net,
            bus=b1,
            p_a_mw=0.1,
            p_b_mw=0.2,
            p_c_mw=0.3,
            q_a_mvar=0.01,
            q_b_mvar=0.02,
            q_c_mvar=0.03,
            type="wye",
        )
    return net


# ----------------------------------------------------------------------- #
# 1. SINGLE_PHASE_EQUIV reproduces the historical single-phase output
# ----------------------------------------------------------------------- #
def test_pp_single_phase_equiv_default_phases_and_matrices():
    net = _pp_net()
    grid_default, idmap_default = pp_to_grid(net)
    grid_explicit, idmap_explicit = pp_to_grid(
        net, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV
    )

    for grid in (grid_default, grid_explicit):
        assert all(n.phases == (Phase.A,) for n in grid.nodes)
        line = next(b for b in grid.branches if isinstance(b, Line))
        assert line.from_phases == (Phase.A,)
        assert line.series_resistance_ohm_per_m == [[0.2 / 1000.0]]
        src = next(a for a in grid.appliances if isinstance(a, Source))
        assert src.phases == (Phase.A,)
        ld = next(a for a in grid.appliances if isinstance(a, Load))
        assert ld.phases == (Phase.A,)
        assert ld.p_nom_w == pytest.approx(0.5e6)

    # id_map buckets and node ids identical regardless of explicit/default arg.
    assert idmap_default["bus"] == idmap_explicit["bus"]
    assert idmap_default["line"] == idmap_explicit["line"]
    assert "slack_v_complex" in idmap_default


def test_pgm_single_phase_equiv_phases():
    input_data = _pgm_input()
    grid, id_map = pgm_to_grid(input_data, base_frequency_hz=50.0)
    assert all(n.phases == (Phase.A,) for n in grid.nodes)
    line = next(b for b in grid.branches if isinstance(b, Line))
    assert line.series_resistance_ohm_per_m == [[0.4]]  # r1 total, length=1
    src = next(a for a in grid.appliances if isinstance(a, Source))
    assert src.phases == (Phase.A,)


def test_pp_three_phase_transformer_vector_group(caplog):
    """A transformer under THREE_PHASE carries an explicit Dyn vector group and
    emits no "not modeled" warning (the phase-domain stamp models it)."""
    from pgml.schemas.grid_schema import Transformer, WindingConnection

    net = pp.create_empty_network(f_hz=50.0)
    b_hv = pp.create_bus(net, vn_kv=20.0, name="hv")
    b_lv = pp.create_bus(net, vn_kv=0.4, name="lv")
    pp.create_ext_grid(net, bus=b_hv, vm_pu=1.0, va_degree=0.0)
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b_hv,
        lv_bus=b_lv,
        sn_mva=0.4,
        vn_hv_kv=20.0,
        vn_lv_kv=0.4,
        vk_percent=6.0,
        vkr_percent=1.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=30.0,
        vector_group="Dyn1",
    )
    pp.create_load(net, bus=b_lv, p_mw=0.1, q_mvar=0.02)

    with caplog.at_level("WARNING", logger="pgml"):
        grid, _ = pp_to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
    assert not any("vector-group" in r.message for r in caplog.records)

    trafo = next(b for b in grid.branches if isinstance(b, Transformer))
    assert trafo.from_connection == WindingConnection.DELTA
    assert trafo.to_connection == WindingConnection.WYE_GROUNDED
    # Nominal ratio lives in u_rated; tap is off-nominal only.
    assert abs(float(trafo.tap.ratio_magnitude) - 1.0) < 1e-12
    assert abs(float(trafo.u_rated_from_v) / float(trafo.u_rated_to_v) - 50.0) < 1e-9


def test_pp_multi_ext_grid_first_slack_wins():
    """With several ext_grids the ideal-slack phasor takes the FIRST slack and is
    not overwritten by later ones."""
    net = pp.create_empty_network(f_hz=50.0)
    b0 = pp.create_bus(net, vn_kv=20.0, name="slack0")
    b1 = pp.create_bus(net, vn_kv=20.0, name="slack1")
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=0.2,
        x_ohm_per_km=0.3,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    # First ext_grid: 1.0 pu / 0 deg. Second: 0.95 pu / +10 deg (must NOT win).
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0)
    pp.create_ext_grid(net, bus=b1, vm_pu=0.95, va_degree=10.0)

    _, id_map = pp_to_grid(net)
    # 1.0 pu * 20 kV * 1000 = 20000 V at 0 deg -> purely real.
    assert id_map["slack_v_complex"] == pytest.approx(complex(20000.0, 0.0))


# ----------------------------------------------------------------------- #
# 2. THREE_PHASE produces abc nodes + 3x3 matrices + balanced source
# ----------------------------------------------------------------------- #
def test_pp_three_phase_abc_nodes_and_line_matrix():
    net = _pp_net()
    grid, _ = pp_to_grid(net, phase_mode=PhaseMode.THREE_PHASE)

    assert all(n.phases == ABC for n in grid.nodes)
    line = next(b for b in grid.branches if isinstance(b, Line))
    assert line.from_phases == ABC
    R = line.series_resistance_ohm_per_m
    assert len(R) == 3 and all(len(r) == 3 for r in R)

    # Self / mutual from the symmetric-component identity with config defaults.
    from pgml.convert._common import zero_sequence_ratios

    rr, _, _ = zero_sequence_ratios()
    r1 = 0.2 / 1000.0
    r0 = r1 * rr
    r_self = (r0 + 2 * r1) / 3.0
    r_mut = (r0 - r1) / 3.0
    assert R[0][0] == pytest.approx(r_self)
    assert R[0][1] == pytest.approx(r_mut)
    assert R[1][0] == pytest.approx(r_mut)


def test_pp_three_phase_balanced_source_angles():
    net = _pp_net()
    grid, _ = pp_to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
    src = next(a for a in grid.appliances if isinstance(a, Source))
    assert src.phases == ABC
    assert src.u_angle_deg == (0.0, -120.0, -240.0)
    assert len(src.u_ref_v) == 3


# ----------------------------------------------------------------------- #
# 3. Asymmetric-load capture
# ----------------------------------------------------------------------- #
def test_pp_asymmetric_load_three_phase_per_phase_and_connection():
    net = _pp_net(with_asym=True)
    grid, id_map = pp_to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
    assert id_map["asymmetric_load"]  # captured
    asym = next(
        a
        for a in grid.appliances
        if isinstance(a, Load) and a.p_nom_per_phase_w is not None
    )
    assert asym.connection == WindingConnection.WYE
    assert tuple(asym.p_nom_per_phase_w) == pytest.approx((0.1e6, 0.2e6, 0.3e6))
    assert tuple(asym.q_nom_per_phase_var) == pytest.approx((0.01e6, 0.02e6, 0.03e6))
    assert asym.p_nom_w == pytest.approx(0.6e6)


def test_pp_asymmetric_load_single_phase_collapsed(caplog):
    net = _pp_net(with_asym=True)
    with caplog.at_level("INFO", logger="pgml"):
        grid, id_map = pp_to_grid(net, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
    asym = next(
        a
        for a in grid.appliances
        if isinstance(a, Load) and a.phases == (Phase.A,) and a.p_nom_w > 0.55e6
    )
    assert asym.p_nom_w == pytest.approx(0.6e6)
    assert asym.p_nom_per_phase_w is None
    assert any("collapsed" in r.message for r in caplog.records)


def test_pgm_asym_load_three_phase_per_phase_wye():
    input_data = _pgm_input(with_asym=True)
    grid, id_map = pgm_to_grid(
        input_data, base_frequency_hz=50.0, phase_mode=PhaseMode.THREE_PHASE
    )
    assert id_map["asym_load"]
    asym = next(
        a
        for a in grid.appliances
        if isinstance(a, Load) and a.p_nom_per_phase_w is not None
    )
    assert asym.connection == WindingConnection.WYE  # pgm has no connection field
    assert tuple(asym.p_nom_per_phase_w) == pytest.approx((100.0, 200.0, 300.0))


# ----------------------------------------------------------------------- #
# 4. THREE_PHASE grid assembles + solves
# ----------------------------------------------------------------------- #
def test_pp_three_phase_solves_power_flow():
    net = _pp_net(with_asym=True)
    grid, _ = pp_to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
    res = solve_power_flow(grid, slack="ideal")
    assert res.v is not None
    import torch

    assert torch.isfinite(res.v.real).all()
    assert torch.isfinite(res.v.imag).all()


def test_pgm_three_phase_solves_power_flow():
    input_data = _pgm_input(with_asym=True)
    grid, _ = pgm_to_grid(
        input_data, base_frequency_hz=50.0, phase_mode=PhaseMode.THREE_PHASE
    )
    res = solve_power_flow(grid, slack="ideal")
    import torch

    assert torch.isfinite(res.v.real).all()


# ----------------------------------------------------------------------- #
# 5. Per-phase spectrum on a converted abc load runs through harmonics
# ----------------------------------------------------------------------- #
def test_pp_three_phase_load_spectrum_per_phase_harmonic_flow():
    net = _pp_net()
    grid, _ = pp_to_grid(net, phase_mode=PhaseMode.THREE_PHASE)

    spec_a = StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[HarmonicComponent(order=5, magnitude_pu=0.1, phase_deg=0.0)]
        )
    )
    # Attach a per-phase spectrum (phase A only) to a converted balanced load.
    loads = [a for a in grid.appliances if isinstance(a, Load)]
    base = loads[0]
    distorting = Load(
        id=base.id,
        node=base.node,
        phases=ABC,
        p_nom_w=base.p_nom_w,
        q_nom_var=base.q_nom_var,
        spectrum_per_phase={Phase.A: spec_a},
    )
    grid.appliances = [distorting if a is base else a for a in grid.appliances]

    res = solve_harmonic_flow(grid, [1, 5], slack="norton")
    import torch

    # voltages finite at every requested harmonic [H, N]
    assert torch.isfinite(res.v.real).all()
    assert torch.isfinite(res.v.imag).all()


# ----------------------------------------------------------------------- #
# pgm input_data builder
# ----------------------------------------------------------------------- #
def _pgm_input(with_asym: bool = False):
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
    sym["type"] = [LoadGenType.const_impedance]
    sym["p_specified"] = [5e5]
    sym["q_specified"] = [1e5]

    data = {"node": node, "line": line, "source": source, "sym_load": sym}

    if with_asym:
        asym = initialize_array("input", "asym_load", 1)
        asym["id"] = [40]
        asym["node"] = [2]
        asym["status"] = [1]
        asym["type"] = [LoadGenType.const_impedance]
        asym["p_specified"] = [[100.0, 200.0, 300.0]]
        asym["q_specified"] = [[10.0, 20.0, 30.0]]
        data["asym_load"] = asym

    return data
