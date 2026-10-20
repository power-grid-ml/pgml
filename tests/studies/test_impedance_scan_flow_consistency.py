"""Impedance scan vs the harmonic power flow's own system matrix.

At an integer order the scan's ``Y(f)`` is the matrix :func:`pgml.solver.solve_harmonic_flow`
factors for the same load-shunt options (:func:`pgml.solver.assemble_harmonic_system`, which
returns exactly that matrix), so the driving-point impedance from the scan equals the block of
``Y(h)^-1`` the harmonic flow implies. Checked on a four-wire feeder that carries every device
kind the harmonic shunt distinguishes: wye and delta loads, a leading single-phase load, a
device with its own motor ``harmonic_model``, a generator with an explicit
``harmonic_impedance``, a generator without one (no shunt under the shipped
``generation_model``) and a capacitor bank.

The two sides are assembled by different code paths (the flow evaluates the device shunts at
the solved fundamental voltage, the scan at the rated one, which agree for these
constant-power devices) and inverted differently (dense inverse here, a factored column solve
in the scan). Measured (complex128, x86-64): identical (0 difference) on every parametrised
option; tolerance 1e-12 relative.
"""

from __future__ import annotations

import pytest
import torch

from pgml.schemas.grid_schema import (
    Generator,
    HarmonicImpedance,
    HarmonicShuntModel,
    Load,
    Phase,
    ShuntAppliance,
    WindingConnection,
)
from pgml.simulation import SimulationConfig
from pgml.solver import assemble_harmonic_system, solve_power_flow
from pgml.studies import impedance_scan

from tests.studies.grids import four_wire_two_bus

ABC = (Phase.A, Phase.B, Phase.C)
ORDERS = [3, 5, 7, 11, 13]
RTOL = 1e-12


def _feeder():
    grid = four_wire_two_bus()
    grid.appliances.extend(
        [
            Load(id=20, node=2, phases=ABC, p_nom_w=9.0e3, q_nom_var=3.0e3),
            Load(
                id=21,
                node=2,
                phases=ABC,
                connection=WindingConnection.DELTA,
                p_nom_w=6.0e3,
                q_nom_var=1.0e3,
            ),
            Load(id=22, node=2, phases=(Phase.B,), p_nom_w=2.0e3, q_nom_var=-0.8e3),
            Load(
                id=23,
                node=2,
                phases=ABC,
                p_nom_w=4.0e3,
                q_nom_var=2.0e3,
                harmonic_model=HarmonicShuntModel(motor_x_harm_pu=0.2),
            ),
            Generator(
                id=30,
                node=2,
                phases=ABC,
                p_nom_w=5.0e3,
                q_nom_var=0.0,
                harmonic_impedance=HarmonicImpedance(
                    resistance_ohm=0.5, inductance_h=2.0e-3
                ),
            ),
            Generator(id=31, node=2, phases=ABC, p_nom_w=3.0e3, q_nom_var=0.0),
            ShuntAppliance(
                id=40,
                node=2,
                phases=ABC,
                conductance_s=(0.0,) * 3,
                capacitance_f=(60.0e-6,) * 3,
            ),
        ]
    )
    return grid


def _flow_driving_point(grid, node_id, *, operating_point, load_shunt, basis):
    """``Y(h)^-1`` block of ``node_id`` of the harmonic flow's assembled system."""
    pf = solve_power_flow(grid, operating_point=operating_point)
    y, _, index = assemble_harmonic_system(
        grid,
        ORDERS,
        pf.v,
        operating_point=operating_point,
        load_shunt=load_shunt,
        load_shunt_basis=basis,
    )
    rows = torch.as_tensor(index.rows(node_id))
    z = torch.stack([torch.linalg.inv(y[k]) for k in range(y.shape[0])])
    return z[:, rows][:, :, rows]


@pytest.mark.parametrize(
    ("load_shunt", "basis", "operating_point"),
    [
        (None, None, None),
        ("motor", None, None),
        ("none", None, None),
        (None, "operating_point", {20: {"p_w": 4.0e3, "q_var": 2.5e3}}),
        (None, "nameplate", {20: {"p_w": 4.0e3, "q_var": 2.5e3}}),
    ],
)
def test_scan_equals_the_harmonic_flow_system(load_shunt, basis, operating_point):
    grid = _feeder()
    f0 = float(grid.base_frequency_hz)
    reference = _flow_driving_point(
        grid,
        2,
        operating_point=operating_point,
        load_shunt=load_shunt,
        basis=basis,
    )
    scan = impedance_scan(
        grid,
        nodes=[2],
        frequencies_hz=[h * f0 for h in ORDERS],
        operating_point=operating_point,
        load_shunt=load_shunt,
        load_shunt_basis=basis,
    )
    torch.testing.assert_close(scan.driving_point(2), reference, rtol=RTOL, atol=0)


def test_config_load_shunt_is_the_flow_option():
    """``config.load_shunt`` selects the model for both the scan and the flow."""
    grid = _feeder()
    f0 = float(grid.base_frequency_hz)
    cfg = SimulationConfig(load_shunt="motor", load_shunt_basis="nameplate")
    reference = _flow_driving_point(
        grid, 2, operating_point=None, load_shunt="motor", basis="nameplate"
    )
    scan = impedance_scan(
        grid, nodes=[2], frequencies_hz=[h * f0 for h in ORDERS], config=cfg
    )
    torch.testing.assert_close(scan.driving_point(2), reference, rtol=RTOL, atol=0)
