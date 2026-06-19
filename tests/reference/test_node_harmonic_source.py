"""Behavioral / physics checks for the per-node harmonic "error" source.

Pins the model in ``references/error_injection.md``:

- A single VOLTAGE source (Thevenin) and a single CURRENT source (Norton) solve.
- The FUNDAMENTAL power flow is preserved EXACTLY (source applied only at h>1).
- A STIFF voltage source (large S_sc) drives the node harmonic voltage toward the
  spectrum EMF ``E_h``; a weak voltage source sees little.
- A CURRENT source injects ``I_N`` independent of the network (the node voltage is
  the bare divider ``I_N / Y_node``, unaffected by adding a parallel shunt elsewhere).
- The voltage-divider law ``V_node = E_h * Z_net / (Z_s + Z_net)`` matches.
"""

from __future__ import annotations

import math

import torch

from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
)
from pgml.solver import NodeHarmonicSource, solve_harmonic_flow

CDT = torch.complex128
F0 = 50.0
W0 = 2.0 * math.pi * F0


def _grid() -> Grid:
    return Grid(
        base_frequency_hz=F0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
        ],
        branches=[
            Line(
                id=1,
                from_node=1,
                to_node=2,
                from_phases=(Phase.A,),
                to_phases=(Phase.A,),
                length_m=1.0,
                series_resistance_ohm_per_m=[[0.5]],
                series_inductance_h_per_m=[[0.5 / W0]],
                shunt_capacitance_f_per_m=[[0.0]],
            )
        ],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[0.1 / W0]],
            ),
            Load(
                id=2,
                node=2,
                phases=(Phase.A,),
                p_nom_w=2000.0,
                q_nom_var=500.0,
                load_model=LoadModel.CONST_POWER,
            ),
        ],
    )


def test_none_is_byte_identical():
    """``node_sources=None`` is byte-identical to omitting the argument."""
    a = solve_harmonic_flow(_grid(), [1, 5, 7], slack="norton", dtype=CDT)
    b = solve_harmonic_flow(
        _grid(), [1, 5, 7], slack="norton", dtype=CDT, node_sources=None
    )
    assert torch.equal(a.v, b.v)


def test_fundamental_preserved():
    """A node source must NOT perturb the fundamental (applied only at h>1)."""
    base = solve_harmonic_flow(_grid(), [1, 5, 7], slack="norton", dtype=CDT)
    for kind in ("voltage", "current"):
        src = NodeHarmonicSource(
            node_id=2,
            spectrum={1: (1.0, 0.0), 5: (0.2, 0.0), 7: (0.1, 0.0)},
            source_power_va=1.0e6,
            kind=kind,
        )
        res = solve_harmonic_flow(
            _grid(), [1, 5, 7], slack="norton", dtype=CDT, node_sources=[src]
        )
        assert torch.equal(base.v[0], res.v[0])  # order 1 untouched


def test_no_source_no_injection():
    """Without any device spectrum and without a node source, h>1 voltages are 0."""
    res = solve_harmonic_flow(_grid(), [1, 5, 7], slack="norton", dtype=CDT)
    assert torch.allclose(res.v[1], torch.zeros_like(res.v[1]))
    assert torch.allclose(res.v[2], torch.zeros_like(res.v[2]))


def test_stiff_voltage_source_imposes_spectrum():
    """A STIFF voltage source drives ``V_node(h) -> E_h`` (imposes the spectrum)."""
    grid = _grid()
    res = solve_harmonic_flow(
        grid,
        [1, 5],
        slack="norton",
        dtype=CDT,
        node_sources=[
            NodeHarmonicSource(
                node_id=2,
                spectrum={1: (1.0, 0.0), 5: (0.2, 0.0)},
                source_power_va=1.0e12,  # very stiff
                kind="voltage",
            )
        ],
    )
    idx = res.index
    row2 = idx.row(2, Phase.A)
    v1_node2 = res.v[0, row2]
    e_h_mag = 0.2 * v1_node2.abs()  # |E_h| = (mag_h/mag_1)*|V1|
    v_h_mag = res.v[1, row2].abs()
    assert torch.isclose(v_h_mag, e_h_mag, rtol=1e-4)


def test_weak_voltage_source_sees_little():
    """A WEAK voltage source (small S_sc) yields a much smaller node harmonic V."""
    grid = _grid()

    def vmag(s_sc):
        res = solve_harmonic_flow(
            grid,
            [1, 5],
            slack="norton",
            dtype=CDT,
            node_sources=[
                NodeHarmonicSource(
                    node_id=2,
                    spectrum={1: (1.0, 0.0), 5: (0.2, 0.0)},
                    source_power_va=s_sc,
                    kind="voltage",
                )
            ],
        )
        row2 = res.index.row(2, Phase.A)
        return res.v[1, row2].abs()

    stiff = vmag(1.0e12)
    weak = vmag(1.0e2)
    assert weak < 0.05 * stiff


def test_current_source_independent_of_network():
    """A CURRENT source injects ``I_N`` independent of the network.

    Adding a parallel SHUNT at another node (changing the network) does NOT change
    the injected current; for the ideal Norton source the node voltage is the bare
    divider ``I_N / Y_node`` and depends only on the local network, so the analytic
    relation ``I_N = V_node * Y_node`` holds regardless. Here we verify the injected
    current equals the analytic ``E_h * Y_s`` by reconstructing it from the solve.
    """
    grid = _grid()
    s_sc = 3.0e5
    src = NodeHarmonicSource(
        node_id=2,
        spectrum={1: (1.0, 0.0), 5: (0.2, 30.0)},
        source_power_va=s_sc,
        kind="current",
    )
    res = solve_harmonic_flow(
        grid, [1, 5], slack="norton", dtype=CDT, node_sources=[src]
    )
    row2 = res.index.row(2, Phase.A)
    v1_node2 = res.v[0, row2]
    v_base = 230.0  # single-phase node L-N
    y_s = s_sc / (v_base * v_base)
    # E_h = (mag_h/mag_1)|V1| exp(j*(rad(ang_h) + h*(arg(V1)-rad(ang_1))))
    h = 5
    e_h = (0.2 * v1_node2.abs()) * torch.exp(
        1j
        * (
            torch.tensor(math.radians(30.0), dtype=torch.float64)
            + h * (torch.angle(v1_node2) - 0.0)
        )
    )
    i_n_expected = e_h * y_s

    # The node-2 row of Y(5) I(5) reconstruction: I = Y @ V (current source only,
    # no shunt at node 2), so injected current at node 2 equals (Y V)[row2].
    from pgml.assembly import assemble_network_ybus, node_phase_index
    from pgml.assembly._stamps import _cdtype, _rdtype
    from pgml.assembly.ybus import _stamp_sources

    index = node_phase_index(grid)
    fvec = torch.as_tensor([h * F0], dtype=_rdtype(CDT))
    y = assemble_network_ybus(grid, [h * F0], dtype=CDT).Y
    if y.ndim == 2:
        y = y.unsqueeze(0)
    y = _stamp_sources(grid, fvec, y, index, _cdtype(CDT), _rdtype(CDT), None, None)
    i_reconstructed = (y[0] @ res.v[1])[row2]
    assert torch.isclose(i_reconstructed, i_n_expected, rtol=1e-6, atol=1e-9)


def test_voltage_divider_law():
    """``V_node(h) = E_h * Z_net / (Z_s + Z_net)`` (finite-strength Thevenin)."""
    grid = _grid()
    s_sc = 4.0e5
    res = solve_harmonic_flow(
        grid,
        [1, 5],
        slack="norton",
        dtype=CDT,
        node_sources=[
            NodeHarmonicSource(
                node_id=2,
                spectrum={1: (1.0, 0.0), 5: (0.2, 0.0)},
                source_power_va=s_sc,
                kind="voltage",
            )
        ],
    )
    from pgml.assembly import assemble_network_ybus, node_phase_index
    from pgml.assembly._stamps import _cdtype, _rdtype
    from pgml.assembly.ybus import _stamp_sources

    h = 5
    index = node_phase_index(grid)
    row2 = index.row(2, Phase.A)
    fvec = torch.as_tensor([h * F0], dtype=_rdtype(CDT))
    y = assemble_network_ybus(grid, [h * F0], dtype=CDT).Y
    if y.ndim == 2:
        y = y.unsqueeze(0)
    y = _stamp_sources(grid, fvec, y, index, _cdtype(CDT), _rdtype(CDT), None, None)
    ynet = y[0]  # network Y(h) WITHOUT the node source

    v_base = 230.0
    y_s = s_sc / (v_base * v_base)
    # E_h from the (unchanged) fundamental node-2 voltage.
    v1_node2 = res.v[0, row2]
    e_h = 0.2 * v1_node2  # ang_h=0, ang_1=0 -> arg(E_h)=h*arg(V1) == arg(v1^? ) ...
    # Build E_h precisely with the documented convention:
    e_h = (0.2 * v1_node2.abs()) * torch.exp(1j * (h * torch.angle(v1_node2)))

    # Solve the augmented system V = (Ynet + Ys e_row e_row^T)^-1 (Ynet... ) manually:
    n = index.size
    ys_mat = torch.zeros((n, n), dtype=CDT)
    ys_mat[row2, row2] = y_s
    i_vec = torch.zeros((n,), dtype=CDT)
    i_vec[row2] = e_h * y_s
    v_expected = torch.linalg.solve(ynet + ys_mat, i_vec)
    assert torch.allclose(res.v[1], v_expected, rtol=1e-9, atol=1e-12)


def test_multiple_sources():
    """Multiple simultaneous node sources superpose (list input)."""
    grid = _grid()
    res = solve_harmonic_flow(
        grid,
        [1, 5],
        slack="norton",
        dtype=CDT,
        node_sources=[
            NodeHarmonicSource(
                node_id=1,
                spectrum={1: (1.0, 0.0), 5: (0.1, 0.0)},
                source_power_va=2.0e5,
                kind="current",
            ),
            NodeHarmonicSource(
                node_id=2,
                spectrum={1: (1.0, 0.0), 5: (0.2, 0.0)},
                source_power_va=5.0e5,
                kind="voltage",
            ),
        ],
    )
    # Both nodes carry a nonzero harmonic voltage at h=5.
    assert res.v[1, res.index.row(1, Phase.A)].abs() > 0
    assert res.v[1, res.index.row(2, Phase.A)].abs() > 0


def test_phases_subset_selection():
    """``phases=None`` injects on all node phases; an explicit subset injects only there."""
    abc = (Phase.A, Phase.B, Phase.C)
    grid = Grid(
        base_frequency_hz=F0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=abc),
            Node(id=2, u_rated_v=400.0, phases=abc),
        ],
        branches=[
            Line(
                id=1,
                from_node=1,
                to_node=2,
                from_phases=abc,
                to_phases=abc,
                length_m=1.0,
                series_resistance_ohm_per_m=[
                    [0.5 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                series_inductance_h_per_m=[
                    [0.5 / W0 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                shunt_capacitance_f_per_m=[[0.0] * 3 for _ in range(3)],
            )
        ],
        appliances=[
            Source(
                id=10,
                node=1,
                phases=abc,
                u_ref_v=(231.0, 231.0, 231.0),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=[
                    [0.1 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [0.1 / W0 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            ),
            Load(
                id=2,
                node=2,
                phases=abc,
                p_nom_w=4500.0,
                q_nom_var=900.0,
                load_model=LoadModel.CONST_POWER,
            ),
        ],
    )
    src = NodeHarmonicSource(
        node_id=2,
        phases=(Phase.A,),  # inject only phase A
        spectrum={1: (1.0, 0.0), 5: (0.2, 0.0)},
        source_power_va=1.0e12,
        kind="voltage",
    )
    res = solve_harmonic_flow(
        grid, [1, 5], slack="norton", dtype=CDT, node_sources=[src]
    )
    idx = res.index
    va = res.v[1, idx.row(2, Phase.A)].abs()
    vb = res.v[1, idx.row(2, Phase.B)].abs()
    vc = res.v[1, idx.row(2, Phase.C)].abs()
    # Phase A driven hard; B/C only see coupling (here decoupled) -> ~0.
    assert va > 1.0
    assert vb < 1e-6 and vc < 1e-6
