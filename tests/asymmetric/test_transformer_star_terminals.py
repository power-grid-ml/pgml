"""Transformer star points as terminals: neutral rows, earthing impedances, centre taps.

Closed-form and equivalence checks of :mod:`pgml.assembly._transformer_terminals`
(the live OpenDSS comparison is ``tests/reference/test_opendss_centre_tap_grounding.py``):

- a star point bonded to the neutral row plus a separate earth shunt equals the same
  star point earthed through that impedance inside the transformer;
- an internal star point earthed through ``Z_g`` equals the neutral-row star point
  with the row eliminated afterwards, and a floating internal star point reproduces
  the vector-group stamp's ungrounded-wye projection;
- the zero-sequence impedance behind an impedance-earthed YNyn star is ``Z0 + 3·Z_g``;
- a centre tap's open-circuit half voltages are ``±V_primary/(2τ)`` and its full
  short circuit is ``Z_a``; Kersting's default split equals the explicit values;
- the transformer's terminal currents sum to its earth currents (KCL);
- invalid combinations raise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from pgml.assembly import assemble_network_ybus, branch_currents, node_phase_index
from pgml.assembly._transformer_terminals import uses_terminal_model
from pgml.errors import ModelingError
from pgml.schemas.grid_schema import (
    CentreTap,
    ComplexTap,
    Grid,
    GroundingImpedance,
    Node,
    Phase,
    ShuntReactor,
    Transformer,
)

P = Phase
F0 = 50.0
ABC = (P.A, P.B, P.C)
ABCN = (P.A, P.B, P.C, P.N)
R_LEAK, X_LEAK = 0.002, 0.016  # LV-coil referred leakage [Ohm]


def _dyn(
    to_phases,
    to_connection,
    *,
    to_grounding=None,
    from_connection="delta",
    shift=330.0,
    from_grounding=None,
    lv_phases=ABCN,
    earth_shunt=None,
) -> Grid:
    branches = [
        Transformer(
            id=10,
            from_node=1,
            to_node=2,
            from_phases=ABC,
            to_phases=to_phases,
            s_rated_va=4e5,
            u_rated_from_v=20e3,
            u_rated_to_v=400.0,
            from_connection=from_connection,
            to_connection=to_connection,
            series_resistance_ohm=R_LEAK,
            series_inductance_h=X_LEAK / (2 * math.pi * F0),
            tap=ComplexTap(ratio_magnitude=1.0, shift_deg=shift),
            from_grounding=from_grounding,
            to_grounding=to_grounding,
        )
    ]
    if earth_shunt is not None:
        branches.append(
            ShuntReactor(
                id=11,
                from_node=2,
                to_node=2,
                from_phases=(P.N,),
                to_phases=(P.N,),
                conductance_s=[[earth_shunt]],
                capacitance_f=[[0.0]],
            )
        )
    nodes = [
        Node(id=1, u_rated_v=20e3, phases=ABC),
        Node(id=2, u_rated_v=400.0, phases=lv_phases),
    ]
    return Grid(base_frequency_hz=F0, nodes=nodes, branches=branches, appliances=[])


def _y(grid, orders=(1, 5)) -> np.ndarray:
    f = [F0 * h for h in orders]
    return assemble_network_ybus(grid, f, dtype=torch.complex128).Y.numpy()


def _kron_last(y: np.ndarray) -> np.ndarray:
    keep = list(range(y.shape[-1] - 1))
    i = y.shape[-1] - 1
    return y[np.ix_(keep, keep)] - np.outer(y[keep, i], y[i, keep]) / y[i, i]


def test_standard_units_keep_the_vector_group_stamp():
    assert not uses_terminal_model(_dyn(ABC, "wye_grounded").branches[0])
    solid = GroundingImpedance(r_ohm=0.0, x_ohm=0.0)
    assert not uses_terminal_model(
        _dyn(ABC, "wye_grounded", to_grounding=solid).branches[0]
    )
    assert uses_terminal_model(_dyn(ABCN, "wye").branches[0])


def test_bonded_star_plus_earth_shunt_equals_earthed_star_row():
    bonded = _y(_dyn(ABCN, "wye", earth_shunt=1 / 0.3))
    earthed = _y(_dyn(ABCN, "wye_grounded", to_grounding=GroundingImpedance(r_ohm=0.3)))
    # Same admittances summed in a different order: 1e-14 relative at most.
    assert np.abs(bonded - earthed).max() <= 1e-12 * np.abs(bonded).max()


def test_internal_earthed_star_equals_eliminated_neutral_row():
    zg = GroundingImpedance(r_ohm=5.0, x_ohm=1.0)
    internal = _y(_dyn(ABC, "wye_grounded", to_grounding=zg, lv_phases=ABC))
    row = _y(_dyn(ABCN, "wye_grounded", to_grounding=zg))
    for k in range(internal.shape[0]):
        reduced = _kron_last(row[k])
        assert np.abs(reduced - internal[k]).max() <= 1e-12 * np.abs(internal[k]).max()


def test_floating_star_on_neutral_row_reproduces_the_projection():
    projection = _y(_dyn(ABC, "wye", lv_phases=ABC), orders=(1,))[0]
    row = _y(_dyn(ABCN, "wye"), orders=(1,))[0]
    reduced = _kron_last(row)
    assert np.abs(reduced - projection).max() <= 1e-12 * np.abs(projection).max()


def test_zero_sequence_through_earthing_impedance():
    """YNyn with the LV star earthed through ``Z_g``: ``Z0_LV = Z_leak + 3·Z_g``."""
    zg = complex(5.0, 1.0)
    grid = _dyn(
        ABC,
        "wye_grounded",
        to_grounding=GroundingImpedance(r_ohm=zg.real, x_ohm=zg.imag),
        from_connection="wye_grounded",
        shift=0.0,
        lv_phases=ABC,
    )
    y = _y(grid, orders=(1,))[0]
    z_lv = np.linalg.inv(y[3:, 3:])  # HV terminals held at 0 V (shorted)
    z0 = z_lv.sum() / 3.0  # V per phase for I = 1 A on every phase
    assert abs(z0 - (complex(R_LEAK, X_LEAK) + 3.0 * zg)) < 1e-10


def _centre_tap_grid(centre_tap: CentreTap, *, to_phases=(P.A, P.B, P.N)) -> Grid:
    xfmr = Transformer(
        id=10,
        from_node=1,
        to_node=2,
        from_phases=(P.A,),
        to_phases=to_phases,
        s_rated_va=50e3,
        u_rated_from_v=7967.0,
        u_rated_to_v=440.0,
        from_connection="wye_grounded",
        to_connection="wye_grounded",
        series_resistance_ohm=0.0465,
        series_inductance_h=0.0774 / (2 * math.pi * F0),
        to_grounding=GroundingImpedance(r_ohm=0.3) if P.N in to_phases else None,
        centre_tap=centre_tap,
    )
    nodes = [
        Node(id=1, u_rated_v=13800.0, phases=ABC),
        Node(id=2, u_rated_v=440.0, phases=to_phases),
    ]
    return Grid(base_frequency_hz=F0, nodes=nodes, branches=[xfmr], appliances=[])


def test_kersting_default_equals_explicit_split():
    r_a, x_a = 0.0465, 0.0774
    explicit = CentreTap(
        half_resistance_ohm=r_a / 4.0,
        half_inductance_h=0.4 * x_a / 4.0 / (2 * math.pi * F0),
    )
    y_default = _y(_centre_tap_grid(CentreTap()))
    y_explicit = _y(_centre_tap_grid(explicit))
    assert np.abs(y_default - y_explicit).max() <= 1e-12 * np.abs(y_default).max()


def test_centre_tap_ratio_polarity_and_full_short_circuit():
    # Open circuit, centre tap earthed at the reference: V_X = -Y_XX^-1 Y_XH V_H.
    y = _y(_centre_tap_grid(CentreTap(), to_phases=(P.A, P.B)), orders=(1,))[0]
    v_h = np.array([1000.0, 0.0, 0.0])
    v_x = -np.linalg.solve(y[3:, 3:], y[3:, :3] @ v_h)
    half = 1000.0 / (7967.0 / 220.0)
    assert np.allclose(v_x, [half, -half], rtol=1e-12)  # X2 in antiphase

    # Full short circuit seen from X1-X2 with the primary shorted and the centre tap
    # open (bonded to an otherwise unconnected neutral row): Z_a.
    grid = _centre_tap_grid(CentreTap(), to_phases=(P.A, P.B, P.N))
    grid.branches[0].to_connection = "wye"
    grid.branches[0].to_grounding = None
    y = _kron_last(_y(grid, orders=(1,))[0])[3:, 3:]  # HV rows at 0 V, N open
    v = np.linalg.solve(y, np.array([1.0, -1.0]))
    assert abs((v[0] - v[1]) - complex(0.0465, 0.0774)) < 1e-12


def test_centre_tap_branch_currents_sum_to_the_earth_current():
    grid = _centre_tap_grid(CentreTap())
    idx = node_phase_index(grid)
    y = assemble_network_ybus(grid, [F0], dtype=torch.complex128).Y[0]
    v = torch.zeros(idx.size, dtype=torch.complex128)
    v[idx.row(1, P.A)] = 7967.0
    v[idx.row(2, P.A)] = 200.0
    v[idx.row(2, P.B)] = -230.0
    v[idx.row(2, P.N)] = 1.5
    bc = branch_currents(grid, v, [F0], idx)[0]
    # KCL against the assembled matrix (the only branch in the grid).
    nodal = y @ v
    assert torch.allclose(bc.i_to[0], nodal[[idx.row(2, p) for p in (P.A, P.B, P.N)]])
    # The half-coil and neutral currents cancel; what is left leaves through the
    # centre-tap earthing (V_N / 0.3 Ohm).
    assert abs(complex(bc.i_to.sum()) - 1.5 / 0.3) < 1e-9


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"to_phases": ABCN, "to_connection": "wye_grounded"}, "cannot be earthed"),
        (
            {
                "to_phases": ABC,
                "to_connection": "wye",
                "to_grounding": GroundingImpedance(r_ohm=1.0),
                "lv_phases": ABC,
            },
            "ungrounded",
        ),
        (
            {
                "to_phases": ABC,
                "to_connection": "zigzag_grounded",
                "to_grounding": GroundingImpedance(r_ohm=1.0),
                "lv_phases": ABC,
                "from_connection": "wye",
                "shift": 150.0,
            },
            "zigzag",
        ),
        ({"to_phases": ABCN, "to_connection": "delta", "shift": 0.0}, "star point"),
    ],
)
def test_invalid_star_point_combinations_raise(kwargs, match):
    with pytest.raises(ModelingError, match=match):
        _y(_dyn(**kwargs))


def test_schema_rejects_misplaced_neutral_and_bad_centre_taps():
    with pytest.raises(ValidationError, match="last entry"):
        _dyn((P.N, P.A, P.B, P.C), "wye")
    with pytest.raises(ValidationError, match="to_phases"):
        _centre_tap_grid(CentreTap(), to_phases=(P.A, P.N))
    with pytest.raises(ValidationError, match="more resistance"):
        _centre_tap_grid(CentreTap(half_resistance_ohm=1.0))
