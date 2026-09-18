"""pandapower ``net.controller`` rows import as pgml inverter control laws.

A ``DERController`` Q-model on an sgen row becomes the matching control law;
pgml's closed-loop Newton solve then agrees with pandapower's own
``run_control`` fixed point. Controllers without a mapping are reported.
"""

from __future__ import annotations

import pytest

pp = pytest.importorskip("pandapower", exc_type=ImportError)
ppc = pytest.importorskip("pandapower.control", exc_type=ImportError)

from pandapower.control.controller.DERController import (  # noqa: E402
    DERController,
    QModelConstQ,
    QModelCosphiPCurve,
    QModelCosphiPQ,
    QModelQVCurve,
)
from pandapower.control.controller.DERController.DERBasics import (  # noqa: E402
    CosphiPCurve,
    QVCurve,
)

from pgml.convert._common import PhaseMode  # noqa: E402
from pgml.convert.pandapower import to_grid  # noqa: E402
from pgml.schemas.grid_schema import (  # noqa: E402
    ConstantPowerFactorControl,
    ConstantReactivePowerControl,
    PowerFactorWattControl,
    VoltVarControl,
)
from pgml.solver import solve_power_flow  # noqa: E402


def _net():
    net = pp.create_empty_network(f_hz=50, sn_mva=1.0)
    b0 = pp.create_bus(net, vn_kv=0.4)
    b1 = pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_line_from_parameters(
        net,
        b0,
        b1,
        length_km=1.0,
        r_ohm_per_km=0.3,
        x_ohm_per_km=0.15,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.create_load(net, b1, p_mw=0.003, q_mvar=0.001)
    sgen = pp.create_sgen(net, b1, p_mw=0.010, q_mvar=0.0, sn_mva=0.015)
    return net, sgen, b1


def _pgml_vm_pu(grid, id_map, bus):
    res = solve_power_flow(grid, method="newton", tol=1e-12)
    assert bool(res.converged)
    v = res.v.detach().cpu().numpy().reshape(-1)
    phases = grid.nodes[0].phases
    # The node rating is line-to-line; a three-phase node solves line-to-neutral.
    v_nom = 400.0 if len(phases) == 1 else 400.0 / 3.0**0.5
    return abs(v[res.index.row(id_map["bus"][bus], phases[0])]) / v_nom


@pytest.mark.parametrize(
    "q_model, law",
    [
        (
            QModelQVCurve(
                QVCurve(
                    vm_points_pu=[0.9, 1.0, 1.04, 1.1],
                    q_points_pu=[0.3, 0.0, -0.2, -0.3],
                )
            ),
            VoltVarControl,
        ),
        (QModelCosphiPQ(cosphi=-0.95), ConstantPowerFactorControl),
        (QModelCosphiPQ(cosphi=0.95), ConstantPowerFactorControl),
        (QModelConstQ(q_pu=-0.2), ConstantReactivePowerControl),
        (
            QModelCosphiPCurve(
                CosphiPCurve(
                    p_points_pu=[0.0, 0.5, 1.0], cosphi_points=[-0.99, -0.97, -0.9]
                )
            ),
            PowerFactorWattControl,
        ),
    ],
)
@pytest.mark.parametrize("phase_mode", list(PhaseMode), ids=lambda m: m.value)
def test_der_controller_matches_run_control(q_model, law, phase_mode):
    """``sn_mva`` is the rating of the whole sgen, so the three-phase conversion
    reproduces the balanced controlled solution exactly like the equivalent."""
    net, sgen, b1 = _net()
    DERController(net, element_index=sgen, q_model=q_model, max_q_error=1e-9)
    ppc.run_control(net, numba=False, max_iter=1000)
    assert net.converged
    theirs = float(net.res_bus.vm_pu[b1])
    grid, id_map, report = to_grid(net, phase_mode=phase_mode, return_report=True)
    gen = next(a for a in grid.appliances if a.id == id_map["sgen"][sgen])
    assert isinstance(gen.control, law)
    assert len(gen.phases) == (1 if phase_mode is PhaseMode.SINGLE_PHASE_EQUIV else 3)
    assert "dropped.controller" not in report
    assert abs(_pgml_vm_pu(grid, id_map, b1) - theirs) < 1e-6


def test_unmapped_controllers_are_reported(caplog):
    net, sgen, b1 = _net()
    char = ppc.Characteristic(net, x_values=[0.9, 1.1], y_values=[0.003, -0.003])
    ppc.CharacteristicControl(
        net,
        output_element="sgen",
        output_variable="q_mvar",
        output_element_index=sgen,
        input_element="res_bus",
        input_variable="vm_pu",
        input_element_index=b1,
        characteristic_index=char.index,
    )
    ppc.ConstControl(net, element="load", variable="p_mw", element_index=0)
    grid, id_map, report = to_grid(net, return_report=True)
    gen = next(a for a in grid.appliances if a.id == id_map["sgen"][sgen])
    assert gen.control is None
    (entry,) = report.get("dropped.controller")
    assert entry.count == 2
    assert any("net.controller" in r.getMessage() for r in caplog.records)


def test_measurement_table_is_reported_as_dropped():
    net, sgen, b1 = _net()
    pp.create_measurement(net, "v", "bus", 1.0, 0.01, element=b1)
    grid, id_map, report = to_grid(net, return_report=True)
    (entry,) = report.get("dropped.measurement")
    assert entry.count == 1
