"""Connection-aware load Y-block + nodal current injection.

Hand-built 3-phase grids whose const-Z Y-block and ZIP nodal injection are
recomputed independently in numpy:
(a) WYE unbalanced load,
(b) DELTA 3-phase load (circulant),
(c) WYE load on a node WITH Phase.N: phase currents return into the N row.

These pin the incidence model ``M^T diag(y) M`` (Y) and ``M^T i_elem`` (current).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pgml.assembly import (
    assemble_network_ybus,
    assemble_ybus,
    device_current_injections,
    node_phase_index,
)
from pgml.schemas.grid_schema import (
    Grid,
    Load,
    LoadModel,
    Node,
    Phase,
    ShuntAppliance,
    WindingConnection,
)

ABC = (Phase.A, Phase.B, Phase.C)
ABCN = (Phase.A, Phase.B, Phase.C, Phase.N)
CDT = torch.complex128


def _node_block(y, index, node_id):
    rows = index.rows(node_id)
    r = torch.as_tensor(rows, dtype=torch.int64)
    return y.index_select(-2, r).index_select(-1, r)


# --- (a) WYE unbalanced const-Z Y-block --------------------------------------
def test_wye_unbalanced_yblock_and_injection():
    p = (1500.0, 1000.0, 500.0)
    q = (300.0, 200.0, 100.0)
    grid = Grid(
        nodes=[Node(id=1, u_rated_v=400.0, phases=ABC)],
        appliances=[
            Load(
                id=1,
                node=1,
                phases=ABC,
                p_nom_w=3000.0,
                q_nom_var=600.0,
                p_nom_per_phase_w=p,
                q_nom_per_phase_var=q,
            ),
        ],
    )
    yb = assemble_ybus(grid, 50.0, dtype=CDT)
    index = yb.index
    block = _node_block(yb.Y, index, 1).numpy()

    u_ln = 400.0 / math.sqrt(3.0)
    y_np = (np.array(p) - 1j * np.array(q)) / (u_ln**2)
    expected = np.diag(y_np)
    assert np.allclose(block, expected, atol=1e-12)

    # ZIP injection (const-impedance) reproduces Y_block @ V exactly.
    v = torch.tensor(
        [u_ln + 0j, u_ln * np.exp(-2j * np.pi / 3), u_ln * np.exp(2j * np.pi / 3)],
        dtype=CDT,
    )
    grid_z = Grid(
        nodes=[Node(id=1, u_rated_v=400.0, phases=ABC)],
        appliances=[
            Load(
                id=1,
                node=1,
                phases=ABC,
                p_nom_w=3000.0,
                q_nom_var=600.0,
                p_nom_per_phase_w=p,
                q_nom_per_phase_var=q,
                load_model=LoadModel.CONST_IMPEDANCE,
            ),
        ],
    )
    i_dev = device_current_injections(grid_z, v, index, [50.0], dtype=CDT).squeeze(-2)
    yv = torch.as_tensor(expected, dtype=CDT) @ v
    assert torch.allclose(i_dev, yv, atol=1e-9)


# --- (b) DELTA 3-phase const-Z Y-block (circulant) ---------------------------
def test_delta3_yblock_and_injection():
    p = (1500.0, 1000.0, 500.0)
    q = (300.0, 200.0, 100.0)
    grid = Grid(
        nodes=[Node(id=1, u_rated_v=400.0, phases=ABC)],
        appliances=[
            Load(
                id=1,
                node=1,
                phases=ABC,
                p_nom_w=3000.0,
                q_nom_var=600.0,
                p_nom_per_phase_w=p,
                q_nom_per_phase_var=q,
                connection=WindingConnection.DELTA,
            ),
        ],
    )
    yb = assemble_ybus(grid, 50.0, dtype=CDT)
    index = yb.index
    block = _node_block(yb.Y, index, 1).numpy()

    # DELTA V0 = line-to-line = u_rated (no /sqrt(3)).
    u_ll = 400.0
    y_np = (np.array(p) - 1j * np.array(q)) / (u_ll**2)
    m = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]])
    expected = m.T @ np.diag(y_np) @ m
    assert np.allclose(block, expected, atol=1e-12)

    # ZIP const-Z injection == expected_block @ V.
    u_ln = u_ll / math.sqrt(3.0)
    v = torch.tensor(
        [u_ln + 0j, u_ln * np.exp(-2j * np.pi / 3), u_ln * np.exp(2j * np.pi / 3)],
        dtype=CDT,
    )
    grid_z = Grid(
        nodes=[Node(id=1, u_rated_v=400.0, phases=ABC)],
        appliances=[
            Load(
                id=1,
                node=1,
                phases=ABC,
                p_nom_w=3000.0,
                q_nom_var=600.0,
                p_nom_per_phase_w=p,
                q_nom_per_phase_var=q,
                connection=WindingConnection.DELTA,
                load_model=LoadModel.CONST_IMPEDANCE,
            ),
        ],
    )
    i_dev = device_current_injections(grid_z, v, index, [50.0], dtype=CDT).squeeze(-2)
    yv = torch.as_tensor(expected, dtype=CDT) @ v
    assert torch.allclose(i_dev, yv, atol=1e-9)


# --- (c) WYE load on a node WITH Phase.N: phase currents return into N --------
def test_wye_neutral_returns_into_n_row():
    p = (1500.0, 1000.0, 500.0)
    q = (300.0, 200.0, 100.0)
    grid = Grid(
        nodes=[Node(id=1, u_rated_v=400.0, phases=ABCN)],
        appliances=[
            Load(
                id=1,
                node=1,
                phases=ABC,
                p_nom_w=3000.0,
                q_nom_var=600.0,
                p_nom_per_phase_w=p,
                q_nom_per_phase_var=q,
                load_model=LoadModel.CONST_IMPEDANCE,
            ),
        ],
    )
    index = node_phase_index(grid)
    n = index.size  # 4 rows (A,B,C,N)

    u_ln = 400.0 / math.sqrt(3.0)
    # Non-zero neutral voltage to make the return non-trivial.
    vN = 5.0 + 2.0j
    v = torch.tensor(
        [
            u_ln + 0j,
            u_ln * np.exp(-2j * np.pi / 3),
            u_ln * np.exp(2j * np.pi / 3),
            vN,
        ],
        dtype=CDT,
    )
    i_dev = device_current_injections(grid, v, index, [50.0], dtype=CDT).squeeze(-2)

    # Independent numpy: element voltage = V_phase - V_N; y = conj(S)/u_ln^2.
    y_np = (np.array(p) - 1j * np.array(q)) / (u_ln**2)
    v_np = v.numpy()
    v_term = v_np[:3] - v_np[3]
    i_elem = y_np * v_term  # const-Z element current
    i_expected = np.zeros(n, dtype=complex)
    i_expected[:3] = i_elem
    i_expected[3] = -i_elem.sum()  # Kirchhoff: I_N = -sum I_phase
    assert np.allclose(i_dev.numpy(), i_expected, atol=1e-9)
    # Sanity: nodal currents sum to zero (no current to ground via N path).
    assert abs(i_dev.numpy().sum()) < 1e-9


# --- (d) return_path: grounded vs neutral-returning loads on ONE 4-wire bus ---
def _single_phase_load(load_id, return_path):
    return Load(
        id=load_id,
        node=1,
        phases=(Phase.A,),
        p_nom_w=2000.0 + 500.0 * load_id,
        q_nom_var=400.0,
        load_model=LoadModel.CONST_IMPEDANCE,
        return_path=return_path,
    )


def _injections(appliances, v, index):
    grid = Grid(
        nodes=[Node(id=1, u_rated_v=400.0, phases=ABCN)], appliances=list(appliances)
    )
    return device_current_injections(grid, v, index, [50.0], dtype=CDT).squeeze(-2)


def test_return_path_neutral_current_is_neutral_load_only():
    """On a 4-wire bus carrying a GROUNDED and a NEUTRAL-returning WYE load, the
    neutral-conductor current equals the neutral-returning load's current only; the
    grounded load's return is absent from the neutral (it goes to ground)."""
    # A grid with the node fixed so both single-phase loads share row layout.
    index = node_phase_index(
        Grid(nodes=[Node(id=1, u_rated_v=400.0, phases=ABCN)], appliances=[])
    )
    a_row = index.row(1, Phase.A)
    n_row = index.row(1, Phase.N)

    u_ln = 400.0 / math.sqrt(3.0)
    v = torch.zeros(index.size, dtype=CDT)
    v[a_row] = u_ln + 0j
    v[index.row(1, Phase.B)] = u_ln * np.exp(-2j * np.pi / 3)
    v[index.row(1, Phase.C)] = u_ln * np.exp(2j * np.pi / 3)
    v[n_row] = 5.0 + 2.0j  # a live neutral so the return current is non-trivial

    ground = _single_phase_load(1, "ground")
    neutral = _single_phase_load(2, "neutral")

    i_ground = _injections([ground], v, index)
    i_neutral = _injections([neutral], v, index)
    i_both = _injections([ground, neutral], v, index)

    # Grounded load contributes nothing to the neutral conductor.
    assert abs(i_ground[n_row].item()) < 1e-9
    # The neutral current is the neutral-returning load's current only.
    assert torch.allclose(i_both[n_row], i_neutral[n_row], atol=1e-9)
    # Kirchhoff for a single-phase neutral-returning load: I_N = -I_A.
    assert torch.allclose(i_neutral[n_row], -i_neutral[a_row], atol=1e-9)
    # Both loads still contribute to the phase-A row (superposition).
    assert torch.allclose(i_both[a_row], i_ground[a_row] + i_neutral[a_row], atol=1e-9)


# --- (e) DELTA shunt appliance: cyclic phase-to-phase bank -------------------
def test_delta_shunt_yblock_and_frequency_scaling():
    """A DELTA :class:`ShuntAppliance` stamps ``M^T diag(G + jB) M`` with the cyclic
    incidence, frequency-correct (``B = 2*pi*f*C``) at every harmonic order."""
    g = (1.0e-6, 2.0e-6, 0.5e-6)
    c = (3.0e-6, 1.0e-6, 2.0e-6)
    grid = Grid(
        nodes=[Node(id=1, u_rated_v=400.0, phases=ABC)],
        appliances=[
            ShuntAppliance(
                id=1,
                node=1,
                phases=ABC,
                conductance_s=g,
                capacitance_f=c,
                connection=WindingConnection.DELTA,
            )
        ],
    )
    m = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]])
    for f0 in (50.0, 150.0):  # h=1 and h=3
        yb = assemble_network_ybus(grid, f0, dtype=CDT)
        block = _node_block(yb.Y, yb.index, 1).numpy()
        y_np = np.array(g) + 1j * (2.0 * np.pi * f0) * np.array(c)
        expected = m.T @ np.diag(y_np) @ m
        assert np.allclose(block, expected, atol=1e-18)
    # A balanced WYE bank (default) still stamps the diagonal (no cross-coupling).
    wye = Grid(
        nodes=[Node(id=1, u_rated_v=400.0, phases=ABC)],
        appliances=[
            ShuntAppliance(id=1, node=1, phases=ABC, conductance_s=g, capacitance_f=c)
        ],
    )
    block = _node_block(
        assemble_network_ybus(wye, 50.0, dtype=CDT).Y, node_phase_index(wye), 1
    ).numpy()
    y_np = np.array(g) + 1j * (2.0 * np.pi * 50.0) * np.array(c)
    assert np.allclose(block, np.diag(y_np), atol=1e-18)
