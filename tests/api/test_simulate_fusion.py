"""The public ``simulate`` path on a grid with an ideal (zero-impedance) switch.

What a dashboard or a REST caller sees: node voltages on the grid's own ids, a current and
a power flow for EVERY branch including the fused one, and a serializable result set. The
fused branch's current comes from Kirchhoff's law at the fused node, which needs the
solve's nodal injection — :class:`~pgml.simulation.SolvedState` rebuilds it from the grid,
the solved voltages and its own configuration, so the lazy accessor stays self-contained at
the fundamental and at every harmonic order.
"""

from __future__ import annotations

import torch

from pgml import SimulationConfig, simulate
from pgml.solver import assemble_harmonic_ybus, harmonic_injections
from pgml.schemas.grid_schema import (
    Grid,
    HarmonicComponent,
    Line,
    Load,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
    Switch,
)

A = (Phase.A,)


def _spectrum(h5: float) -> StaticSpectrum:
    return StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
                HarmonicComponent(order=5, magnitude_pu=h5, phase_deg=0.0),
            ]
        )
    )


def _grid() -> Grid:
    """Source -- ideal switch -- load bus -- feeder -- load bus (20 kV, 1-phase)."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[Node(id=i, u_rated_v=20_000.0, phases=A) for i in (1, 2, 3)],
        branches=[
            Switch(
                id=10,
                from_node=1,
                to_node=2,
                from_phases=A,
                to_phases=A,
                closed=True,
            ),
            Line(
                id=11,
                from_node=2,
                to_node=3,
                from_phases=A,
                to_phases=A,
                length_m=1_000.0,
                series_resistance_ohm_per_m=[[2.0e-4]],
                series_inductance_h_per_m=[[8.0e-7]],
                shunt_capacitance_f_per_m=[[0.0]],
                harmonic_line_model="naive",
            ),
        ],
        appliances=[
            Source(
                id=20,
                node=1,
                phases=A,
                u_ref_v=(20_000.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.5]],
                inductance_h=[[1.0e-3]],
            ),
            Load(
                id=21,
                node=2,
                phases=A,
                p_nom_w=3.0e5,
                q_nom_var=1.0e5,
                spectrum=_spectrum(0.2),
            ),
            Load(
                id=22,
                node=3,
                phases=A,
                p_nom_w=9.0e5,
                q_nom_var=3.0e5,
                spectrum=_spectrum(0.1),
            ),
        ],
    )


def test_power_flow_reports_the_fused_nodes_and_every_branch_current():
    st = simulate(_grid(), SimulationConfig(calculation="power_flow"))
    assert st.converged
    assert st.fusion is not None and st.fusion.fused_branch_ids == (10,)
    # the grid's own rows, with the fused pair sharing one voltage exactly
    assert st.v.shape == (1, 3)
    assert st.voltage(1, Phase.A) == st.voltage(2, Phase.A)

    currents = {bc.branch_id: bc for bc in st.branch_currents()}
    assert set(currents) == {10, 11}
    sw, line = currents[10], currents[11]
    # an ideal conductor: no shunt path, so the terminal currents are opposite
    assert torch.allclose(sw.i_from, -sw.i_to, atol=0.0, rtol=0.0)
    # KCL at the fused node: the switch carries the feeder plus the load on node 2
    s_load2 = complex(3.0e5, 1.0e5)
    v2 = complex(st.voltage(2, Phase.A).reshape(-1)[0])
    i_load2 = (s_load2 / v2).conjugate()
    expected = complex(line.i_from.reshape(-1)[0]) + i_load2
    assert abs(complex(sw.i_from.reshape(-1)[0]) - expected) < 1e-9


def test_branch_flows_and_result_set_include_the_fused_branch():
    st = simulate(_grid(), SimulationConfig(calculation="power_flow"))
    flows = {bid: (s_from, s_to) for bid, s_from, s_to in st.branch_flows()}
    assert set(flows) == {10, 11}
    # the switch carries the whole downstream load, so its apparent power is finite
    assert abs(complex(flows[10][0].reshape(-1)[0])) > 1.0e6
    bundle = st.to_result_set()
    assert {b.branch_id for b in bundle.branches} == {10, 11}


def test_the_fused_switch_current_at_a_harmonic_order_is_the_injected_sum():
    """With the pure current-source device model the switch carries the injected sum.

    At h = 5 and ``load_shunt="none"`` the only sources are the two loads' spectra and
    no device draws a harmonic shunt current, so the switch carries the feeder current
    plus the injection of the load on the fused node (measured 7.8901 A).
    """
    st = simulate(
        _grid(),
        SimulationConfig(
            calculation="harmonic", harmonic_orders=[1, 5], load_shunt="none"
        ),
    )
    currents = {bc.branch_id: bc for bc in st.branch_currents()}
    sw, line = currents[10], currents[11]
    i_sw_h5 = complex(sw.i_from[1].reshape(-1)[0])
    i_line_h5 = complex(line.i_from[1].reshape(-1)[0])

    # each load injects (mag_h/mag_1) times its fundamental current
    i_load2_h5 = 0.2 * abs(
        (complex(3.0e5, 1.0e5) / complex(st.voltage(2, Phase.A)[0])).conjugate()
    )
    assert abs(abs(i_sw_h5) - (abs(i_line_h5) + i_load2_h5)) < 1e-6 * abs(i_sw_h5)
    assert torch.allclose(sw.i_from, -sw.i_to, atol=0.0, rtol=0.0)


def test_the_fused_switch_current_carries_the_shunt_of_the_fused_node():
    """With the default device shunt the fused node's own shunt current is in the KCL.

    The harmonic device shunt lives in ``Y(h)``, so at a fused node the current balance
    reads ``i_switch = i_feeder + Y_shunt(h)·V(h) - i_injected``. The shunt admittance
    here is the difference of the two assembled harmonic matrices (default model minus
    ``"none"``), i.e. it is read from the assembly rather than from the recovery under
    test. Measured at h = 5 on this feeder: the switch carries 7.8856 A against the
    7.8901 A of the pure current-source model, 5.8e-4 relative less, and the balance
    closes to machine precision.
    """
    grid = _grid()
    st = simulate(
        grid, SimulationConfig(calculation="harmonic", harmonic_orders=[1, 5])
    )
    assert st.fusion is not None
    currents = {bc.branch_id: bc for bc in st.branch_currents()}
    sw, line = currents[10], currents[11]

    v1, v5 = st.v[..., 0, :], st.v[..., 1, :]
    y_default, red = assemble_harmonic_ybus(grid, [5], v1=v1)
    y_none, _ = assemble_harmonic_ybus(grid, [5], v1=v1, load_shunt="none")
    i_shunt = torch.einsum(
        "hij,j->hi", y_default - y_none, st.fusion.sample(v5)
    )  # [1, M] on the reduced rows
    i_src = harmonic_injections(grid, v1, [5])  # [1, N], the grid's own rows

    row_full, row_red = st.index.row(2, Phase.A), red.row(2, Phase.A)
    expected = (
        complex(line.i_from[1].reshape(-1)[0])
        + complex(i_shunt[0, row_red])
        - complex(i_src[0, row_full])
    )
    i_sw_h5 = complex(sw.i_from[1].reshape(-1)[0])
    assert abs(i_sw_h5 - expected) < 1e-12 * abs(i_sw_h5)
    assert torch.allclose(sw.i_from, -sw.i_to, atol=0.0, rtol=0.0)
