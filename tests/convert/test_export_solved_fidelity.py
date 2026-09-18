"""An exported case, solved by the target tool, equals pgml's own solve.

Each grid is solved by pgml and, after ``from_grid``, by the reference tool itself.
The voltages agree to solver precision whenever the export report says the two
sides describe the same fundamental model.
"""

from __future__ import annotations

import math

import pytest

from pgml import defaults
from pgml.schemas.grid_schema import (
    ComplexTap,
    Generator,
    Grid,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    ShuntAppliance,
    Source,
    Switch,
    Transformer,
    VoltageRegulation,
    VoltVarControl,
    Characteristic,
    WindingConnection,
)

from ._solved import (
    max_voltage_error,
    pandapower_node_voltages_pu,
    pgm_node_voltages_pu,
    pgml_node_voltages_pu,
)

OMEGA = 2.0 * math.pi * 50.0
A = (Phase.A,)
ABC = (Phase.A, Phase.B, Phase.C)


def _line(id_, a, b, phases, *, r=0.25e-3, x=0.12e-3, c=0.3e-9, length=400.0):
    n = len(phases)

    def diag(value):
        return [[value if i == j else 0.0 for j in range(n)] for i in range(n)]

    return Line(
        id=id_,
        from_node=a,
        to_node=b,
        from_phases=phases,
        to_phases=phases,
        length_m=length,
        series_resistance_ohm_per_m=diag(r),
        series_inductance_h_per_m=diag(x / OMEGA),
        shunt_capacitance_f_per_m=diag(c),
    )


def _source(id_, node, phases, u_ll, *, r_ohm=0.0, x_ohm=0.0):
    n = len(phases)
    u = u_ll / math.sqrt(3.0) if n == 3 else u_ll

    def diag(value):
        return [[value if i == j else 0.0 for j in range(n)] for i in range(n)]

    return Source(
        id=id_,
        node=node,
        phases=phases,
        u_ref_v=tuple(u * 1.01 for _ in range(n)),
        u_angle_deg=tuple(-120.0 * k for k in range(n)),
        resistance_ohm=diag(r_ohm),
        inductance_h=diag(x_ohm / OMEGA),
    )


def switch_grid() -> Grid:
    """A feeder with a lossy switch that carries a per-end shunt pair."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[Node(id=k, u_rated_v=10e3, phases=A) for k in (1, 2, 3, 4)],
        branches=[
            _line(10, 1, 2, A, length=3000.0),
            Switch(
                id=11,
                from_node=2,
                to_node=3,
                from_phases=A,
                to_phases=A,
                closed=True,
                resistance_ohm=0.05,
                inductance_h=0.02 / OMEGA,
                shunt_capacitance_f=2.0e-6,
                shunt_conductance_s=3.0e-5,
            ),
            _line(12, 3, 4, A, length=2000.0),
        ],
        appliances=[
            _source(20, 1, A, 10e3),
            Load(id=21, node=4, phases=A, p_nom_w=1.5e6, q_nom_var=0.4e6),
            Load(
                id=22,
                node=3,
                phases=A,
                p_nom_w=0.5e6,
                q_nom_var=0.1e6,
                load_model=LoadModel.CONST_IMPEDANCE,
            ),
        ],
    )


def transformer_grid(
    *, regulated: bool = False, controlled: bool = False, weak_source: bool = False
) -> Grid:
    """A balanced three-phase MV/LV grid: tapped Dyn11 unit, shunt, DER."""
    der_extra = {}
    source = {"r_ohm": 0.4, "x_ohm": 3.0} if weak_source else {}
    if regulated:
        der_extra["voltage_regulation"] = VoltageRegulation(
            v_set_pu=1.0, q_min_var=-40e3, q_max_var=40e3
        )
    if controlled:
        der_extra["control"] = VoltVarControl(
            characteristic=Characteristic(
                x_values=[0.9, 0.97, 1.03, 1.1], y_values=[0.44, 0.0, 0.0, -0.44]
            ),
            s_rated_va=60e3,
        )
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=20e3, phases=ABC),
            Node(id=2, u_rated_v=400.0, phases=ABC),
            Node(id=3, u_rated_v=400.0, phases=ABC),
        ],
        branches=[
            Transformer(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=ABC,
                to_phases=ABC,
                s_rated_va=400e3,
                u_rated_from_v=20e3,
                u_rated_to_v=400.0,
                from_connection=WindingConnection.DELTA,
                to_connection=WindingConnection.WYE_GROUNDED,
                series_resistance_ohm=0.004,
                series_inductance_h=0.015 / OMEGA,
                magnetizing_conductance_s=2.0e-6,
                magnetizing_inductance_h=1.0 / (OMEGA * 6.0e-6),
                tap=ComplexTap(ratio_magnitude=1.025, shift_deg=330.0),
            ),
            _line(11, 2, 3, ABC, r=0.2e-3, x=0.08e-3, c=0.25e-9, length=300.0),
        ],
        appliances=[
            _source(20, 1, ABC, 20e3, **source),
            Load(id=21, node=3, phases=ABC, p_nom_w=120e3, q_nom_var=30e3),
            Generator(
                id=22, node=3, phases=ABC, p_nom_w=50e3, q_nom_var=0.0, **der_extra
            ),
            ShuntAppliance(
                id=23,
                node=2,
                phases=ABC,
                conductance_s=[0.0] * 3,
                capacitance_f=[2.0e-4] * 3,
                connection=WindingConnection.WYE,
            ),
        ],
    )


GRIDS = {
    "switch": switch_grid,
    "transformer": transformer_grid,
    "pv_terminal": lambda: transformer_grid(regulated=True, weak_source=True),
}


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_pgm_solves_the_exported_model_like_pgml(name):
    pytest.importorskip("power_grid_model", exc_type=ImportError)
    from pgml.convert.pgm import from_grid

    grid = GRIDS[name]()
    with defaults.use_preset("power-grid-model"):
        exported = from_grid(grid)
        open_keys = [e.key for e in exported.report.open_entries("fundamental")]
        # power-grid-model always solves behind the source impedance; the report
        # names the pgml argument that does the same
        weak = name == "pv_terminal"
        expected = (
            ["model.transformer.magnetizing_tap_reflection", "model.source.impedance"]
            if weak
            else ["model.transformer.magnetizing_tap_reflection"]
            if name == "transformer"
            else []
        )
        assert open_keys == expected
        slack = "norton" if weak else "ideal"
        if weak:
            (entry,) = exported.report.get("model.source.impedance")
            assert entry.match.arguments["solve_power_flow.slack"] == slack
        ours = pgml_node_voltages_pu(grid, slack=slack, tol=1e-12)
    theirs = pgm_node_voltages_pu(exported)
    # the finite 1e14 VA stand-in for an ideal source moves a 2 MVA feeder by 2e-8;
    # the tap reflection of the magnetizing half shows behind the weak source
    assert max_voltage_error(ours, theirs) < (1e-6 if weak else 1e-7)


@pytest.mark.parametrize("name", ["transformer", "pv_terminal"])
def test_pandapower_solves_the_exported_model_like_pgml(name):
    pytest.importorskip("pandapower", exc_type=ImportError)
    from pgml.convert.pandapower import from_grid

    grid = GRIDS[name]()
    weak = name == "pv_terminal"
    with defaults.use_preset("pandapower"):
        exported = from_grid(grid, allow_approximation=weak)
        open_keys = [e.key for e in exported.report.open_entries("fundamental")]
        ours = pgml_node_voltages_pu(grid, slack="ideal", tol=1e-12)
    # pandapower holds the ext_grid bus as an ideal slack and ignores reactive
    # limits unless asked; the report says both
    expected = (
        ["approx.source_impedance_omitted", "model.generator.q_limit_enforcement"]
        if weak
        else []
    )
    assert open_keys == expected
    theirs = pandapower_node_voltages_pu(exported, enforce_q_lims=True)
    assert max_voltage_error(ours, theirs) < 1e-8


def test_pgm_switch_shunt_keeps_capacitance_and_conductance():
    """The total shunt pair maps onto pgm's ``c1`` and loss tangent exactly."""
    pytest.importorskip("power_grid_model", exc_type=ImportError)
    from pgml.convert.pgm import from_grid

    exported = from_grid(switch_grid())
    rows = exported.input_data["line"]
    row = rows[rows["id"] == exported.pgm_of_branch[11]][0]
    assert float(row["c1"]) == pytest.approx(2.0e-6)
    assert float(row["tan1"]) == pytest.approx(3.0e-5 / (OMEGA * 2.0e-6))


def test_inverter_control_is_flagged_not_silently_exported():
    pytest.importorskip("power_grid_model", exc_type=ImportError)
    from pgml.convert.pgm import UnsupportedGridError, from_grid

    grid = transformer_grid(controlled=True)
    with pytest.raises(UnsupportedGridError, match="inverter control"):
        from_grid(grid)
    exported = from_grid(grid, allow_approximation=True)
    (entry,) = exported.report.get("approx.inverter_control")
    assert entry.ids == (22,)
    assert not exported.report.is_exact("fundamental")
