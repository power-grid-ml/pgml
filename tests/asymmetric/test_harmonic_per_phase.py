"""Per-phase / connection-aware harmonic injection.

These pin the connection-aware harmonic CURRENT builder
(``pgml.solver.harmonic_flow._harmonic_injections``): the per-element fundamental
current uses the TERMINAL voltage ``V_term = M @ V_used`` (WYE phase row, WYE-N
``V_phase - V_N``, DELTA L-L), and the nodal injection scatters via ``M^T``. The
per-phase spectrum sources (``spectrum`` / ``spectrum_per_phase`` / runtime
``harmonic_injection`` override) are all exercised against an independent numpy
oracle. WYE-to-ground with a device-level ``spectrum`` is a bit-exact regression
of the historical device-level behavior.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import torch

from pgml.schemas.grid_schema import (
    Grid,
    HarmonicComponent,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
    WindingConnection,
)
from pgml.solver import solve_harmonic_flow
from pgml.solver.harmonic_flow import _harmonic_injections

CDT = torch.complex128
RDT = torch.float64
F0 = 50.0
W0 = 2.0 * math.pi * F0
ABC = (Phase.A, Phase.B, Phase.C)
ABCN = (Phase.A, Phase.B, Phase.C, Phase.N)
DEVICE = torch.device("cpu")


def _spec(comps):
    return StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=o, magnitude_pu=m, phase_deg=a)
                for o, m, a in comps
            ]
        )
    )


def _source(node_phases):
    n = len(node_phases)
    angles = [0.0, -120.0, 120.0, 0.0][:n]
    mags = [231.0, 231.0, 231.0, 0.0][:n]
    r = [[0.1 if i == j else 0.0 for j in range(n)] for i in range(n)]
    ll = [[0.1 / W0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return Source(
        id=10,
        node=1,
        phases=node_phases,
        u_ref_v=tuple(mags),
        u_angle_deg=tuple(angles),
        resistance_ohm=r,
        inductance_h=ll,
    )


def _line(node_phases):
    n = len(node_phases)
    r = [[0.5 if i == j else 0.0 for j in range(n)] for i in range(n)]
    ll = [[0.5 / W0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    c = [[0.0] * n for _ in range(n)]
    return Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=node_phases,
        to_phases=node_phases,
        length_m=1.0,
        series_resistance_ohm_per_m=r,
        series_inductance_h_per_m=ll,
        shunt_capacitance_f_per_m=c,
    )


def _grid(load, node_phases):
    return Grid(
        base_frequency_hz=F0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=node_phases),
            Node(id=2, u_rated_v=400.0, phases=node_phases),
        ],
        branches=[_line(node_phases)],
        appliances=[_source(node_phases), load],
    )


def _injection(grid, orders, harmonic_injection=None):
    """Run the PF then build the harmonic nodal injection vector `[Hh, N]`."""
    res = solve_harmonic_flow(
        grid,
        [1, *orders],
        slack="norton",
        dtype=CDT,
        harmonic_injection=harmonic_injection,
    )
    v1 = res.pf.v  # [N] complex (order 1)
    ih = _harmonic_injections(
        grid,
        v1,
        res.index,
        orders,
        None,  # operating_point
        harmonic_injection,
        CDT,
        RDT,
        DEVICE,
        True,  # asymmetric
    )
    return res, v1, ih


def _i1_elem(s0_elem: complex, v_term: complex) -> complex:
    return np.conj(s0_elem) / np.conj(v_term)


def _i_h_elem(i1: complex, h: int, mag1, ang1, mag_h, ang_h) -> complex:
    """OpenDSS per-element harmonic current from the fundamental phasor."""
    if mag_h == 0.0:
        return 0.0 + 0.0j
    return (
        (mag_h / mag1)
        * abs(i1)
        * cmath.exp(
            1j * (math.radians(ang_h) + h * (cmath.phase(i1) - math.radians(ang1)))
        )
    )


# ---------------------------------------------------------------------------
# 1. WYE 3-phase, spectrum_per_phase on phase A only.
# ---------------------------------------------------------------------------
def test_wye_spectrum_per_phase_only_a():
    spec_a = _spec([(1, 1.0, 0.0), (5, 0.3, 10.0)])
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
        q_nom_per_phase_var=(400.0, 300.0, 200.0),
        load_model=LoadModel.CONST_POWER,
        spectrum_per_phase={Phase.A: spec_a},
    )
    grid = _grid(load, ABC)
    res, v1, ih = _injection(grid, [5])

    ra = res.index.row(2, Phase.A)
    rb = res.index.row(2, Phase.B)
    rc = res.index.row(2, Phase.C)
    inj5 = ih[0]  # [N]

    # Only phase A carries the 5th harmonic; B/C ~ 0.
    assert abs(complex(inj5[ra])) > 1e-3
    assert abs(complex(inj5[rb])) < 1e-12
    assert abs(complex(inj5[rc])) < 1e-12

    # Oracle on phase A: WYE-ground -> terminal voltage = phase-row voltage.
    s0a = complex(2000.0, 400.0)
    va = complex(v1[ra])
    i1a = _i1_elem(s0a, va)
    ihA = _i_h_elem(i1a, 5, 1.0, 0.0, 0.3, 10.0)
    np.testing.assert_allclose(
        [complex(inj5[ra]).real, complex(inj5[ra]).imag],
        [(-ihA).real, (-ihA).imag],  # nodal injection = -I_drawn
        rtol=1e-9,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# 2. DELTA-3 load: I1 uses L-L terminal voltage; nodal scatter via M^T.
# ---------------------------------------------------------------------------
def test_delta_uses_line_to_line_terminal_voltage():
    spec = _spec([(1, 1.0, 0.0), (5, 0.2, 0.0), (7, 0.14, 30.0)])
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
        q_nom_per_phase_var=(400.0, 300.0, 200.0),
        connection=WindingConnection.DELTA,
        load_model=LoadModel.CONST_POWER,
        spectrum=spec,
    )
    grid = _grid(load, ABC)
    res, v1, ih = _injection(grid, [5, 7])

    rows = [res.index.row(2, ph) for ph in ABC]
    vph = np.array([complex(v1[r]) for r in rows])  # [3] phase-row voltages

    # DELTA incidence: element k = phase_k - phase_{(k+1)%3}; per-phase value k ->
    # delta branch k (documented convention).
    M = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]], dtype=complex)
    v_term = M @ vph  # [3] element (L-L) voltages
    s0_elem = np.array(
        [complex(2000.0, 400.0), complex(1500.0, 300.0), complex(1000.0, 200.0)]
    )
    i1_elem = np.conj(s0_elem) / np.conj(v_term)  # [3]

    spec_d = {1: (1.0, 0.0), 5: (0.2, 0.0), 7: (0.14, 30.0)}
    for k, h in enumerate([5, 7]):
        mag_h, ang_h = spec_d[h]
        i_drawn = np.array(
            [_i_h_elem(i1_elem[e], h, 1.0, 0.0, mag_h, ang_h) for e in range(3)]
        )
        i_node = -(M.T @ i_drawn)  # nodal injection = -(M^T @ i_elem)
        got = np.array([complex(ih[k][r]) for r in rows])
        np.testing.assert_allclose(
            got, i_node, rtol=1e-9, atol=1e-9, err_msg=f"DELTA order {h}"
        )


# ---------------------------------------------------------------------------
# 2b. DELTA-3 load with spectrum_per_phase: branch k <- phases[k] mapping.
# ---------------------------------------------------------------------------
def _delta_spectrum_per_phase_oracle(v1, res, spp_by_phase, orders):
    """Independent numpy oracle for a DELTA-3 spectrum_per_phase nodal injection.

    ``spp_by_phase`` maps ``Phase -> {order: (mag, phase_deg)}``; delta BRANCH ``k``
    keys off ``phases[k]`` (element k <-> phases[k]). A phase with no entry injects 0.
    Returns ``{order: np.ndarray[N]}`` of the expected nodal injection.
    """
    rows = [res.index.row(2, ph) for ph in ABC]
    vph = np.array([complex(v1[r]) for r in rows])  # [3] phase-row voltages
    # DELTA incidence: element k = phase_k - phase_{(k+1)%3}.
    M = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]], dtype=complex)
    v_term = M @ vph  # [3] element (L-L) voltages
    s0_elem = np.array(
        [complex(2000.0, 400.0), complex(1500.0, 300.0), complex(1000.0, 200.0)]
    )
    i1_elem = np.conj(s0_elem) / np.conj(v_term)  # [3]
    # Per-element spectrum: element k uses phases[k] (= ABC[k]).
    elem_spec = [spp_by_phase.get(ABC[k], {}) for k in range(3)]
    out = {}
    n = res.index.size
    for h in orders:
        i_drawn = np.zeros(3, dtype=complex)
        for e in range(3):
            sd = elem_spec[e]
            mag1, ang1 = sd.get(1, (1.0, 0.0))
            mag_h, ang_h = sd.get(h, (0.0, 0.0))
            i_drawn[e] = _i_h_elem(i1_elem[e], h, mag1, ang1, mag_h, ang_h)
        i_node = -(M.T @ i_drawn)  # nodal injection = -(M^T @ i_elem)
        vec = np.zeros(n, dtype=complex)
        for k, r in enumerate(rows):
            vec[r] = i_node[k]
        out[h] = vec
    return out


def test_delta_spectrum_per_phase_branch_mapping():
    """DELTA-3 ``spectrum_per_phase``: branch k <- phases[k], vs a numpy oracle."""
    spec_a = _spec([(1, 1.0, 0.0), (5, 0.2, 0.0)])
    spec_c = _spec([(1, 1.0, 0.0), (7, 0.14, 30.0)])
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
        q_nom_per_phase_var=(400.0, 300.0, 200.0),
        connection=WindingConnection.DELTA,
        load_model=LoadModel.CONST_POWER,
        # Branch 0 (phase A) carries 5th; branch 2 (phase C) carries 7th; branch 1
        # (phase B) has NO entry -> injects nothing.
        spectrum_per_phase={Phase.A: spec_a, Phase.C: spec_c},
    )
    grid = _grid(load, ABC)
    orders = [5, 7]
    res, v1, ih = _injection(grid, orders)

    oracle = _delta_spectrum_per_phase_oracle(
        v1,
        res,
        {
            Phase.A: {1: (1.0, 0.0), 5: (0.2, 0.0)},
            Phase.C: {1: (1.0, 0.0), 7: (0.14, 30.0)},
        },
        orders,
    )
    for k, h in enumerate(orders):
        got = np.array([complex(x) for x in ih[k]])
        np.testing.assert_allclose(
            got, oracle[h], rtol=1e-9, atol=1e-9, err_msg=f"DELTA spp order {h}"
        )


def test_delta_spectrum_per_phase_missing_branch_injects_zero():
    """A DELTA branch whose starting phase has no spectrum entry injects no harmonic."""
    spec_a = _spec([(1, 1.0, 0.0), (5, 0.25, 0.0)])
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
        q_nom_per_phase_var=(400.0, 300.0, 200.0),
        connection=WindingConnection.DELTA,
        load_model=LoadModel.CONST_POWER,
        spectrum_per_phase={Phase.A: spec_a},  # only branch 0 injects.
    )
    grid = _grid(load, ABC)
    res, v1, ih = _injection(grid, [5])

    # Only delta branch 0 injects (i_drawn = [i0, 0, 0]). The nodal injection is
    # i_node = -(M^T @ i_drawn); with M row 0 = [1, -1, 0], node A gets -i0, node B
    # gets +i0, node C is exactly zero.
    oracle = _delta_spectrum_per_phase_oracle(
        v1, res, {Phase.A: {1: (1.0, 0.0), 5: (0.25, 0.0)}}, [5]
    )
    got = np.array([complex(x) for x in ih[0]])
    np.testing.assert_allclose(got, oracle[5], rtol=1e-9, atol=1e-9)

    rc = res.index.row(2, Phase.C)
    assert abs(complex(ih[0][rc])) < 1e-12  # only branch 0 alive -> node C row zero


# ---------------------------------------------------------------------------
# 3. WYE on an ABCN node: harmonic returns into the N row (Kirchhoff).
# ---------------------------------------------------------------------------
def test_wye_neutral_harmonic_returns_into_n_row():
    spec = _spec([(1, 1.0, 0.0), (5, 0.2, 0.0)])
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
        q_nom_per_phase_var=(400.0, 300.0, 200.0),
        load_model=LoadModel.CONST_POWER,
        spectrum=spec,
    )
    grid = _grid(load, ABCN)
    res, v1, ih = _injection(grid, [5])

    rows = [res.index.row(2, ph) for ph in ABC]
    rn = res.index.row(2, Phase.N)
    inj5 = ih[0]
    n_inj = complex(inj5[rn])
    phase_sum = sum(complex(inj5[r]) for r in rows)
    # N injection = -(sum of phase injections) (Kirchhoff at the neutral).
    np.testing.assert_allclose(
        [n_inj.real, n_inj.imag],
        [-phase_sum.real, -phase_sum.imag],
        rtol=1e-9,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# 4. Regression: device-level spectrum on WYE-to-ground == the historical stamp.
# ---------------------------------------------------------------------------
def test_wye_ground_device_spectrum_matches_numpy_oracle():
    """Single-phase WYE-to-ground load reproduces the legacy numpy oracle exactly.

    The INJECTION convention is what is under test here, so the device shunt is off
    (``load_shunt="none"``); the shunt's own connection awareness is covered by
    ``tests/reference/test_harmonic_load_shunt.py``.
    """
    r_line, x_line = 0.5, 0.5
    r_src, x_src = 0.1, 0.1
    p_load, q_load = 2000.0, 500.0
    comps = [(1, 1.0, 0.0), (5, 0.2, 0.0), (7, 0.14, 0.0)]
    grid = Grid(
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
                series_resistance_ohm_per_m=[[r_line]],
                series_inductance_h_per_m=[[x_line / W0]],
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
                resistance_ohm=[[r_src]],
                inductance_h=[[x_src / W0]],
            ),
            Load(
                id=2,
                node=2,
                phases=(Phase.A,),
                p_nom_w=p_load,
                q_nom_var=q_load,
                load_model=LoadModel.CONST_POWER,
                spectrum=_spec(comps),
            ),
        ],
    )
    orders = [1, 5, 7]
    res = solve_harmonic_flow(
        grid, orders, slack="norton", dtype=CDT, load_shunt="none"
    )
    ld = res.index.row(2, Phase.A)
    v_fund = complex(res.v[orders.index(1), ld])
    s0 = complex(p_load, q_load)
    i1 = np.conj(s0) / np.conj(v_fund)
    spec = {o: (m, a) for o, m, a in comps}
    mag1, ang1 = spec[1]
    for k, h in enumerate(orders):
        if h == 1:
            continue
        z_line = r_line + 1j * h * x_line
        z_src = r_src + 1j * h * x_src
        y_line, y_src = 1.0 / z_line, 1.0 / z_src
        Y = np.array([[y_src + y_line, -y_line], [-y_line, y_line]], dtype=complex)
        mag_h, ang_h = spec.get(h, (0.0, 0.0))
        i_drawn = _i_h_elem(i1, h, mag1, ang1, mag_h, ang_h)
        v = np.linalg.solve(Y, np.array([0.0, -i_drawn], dtype=complex))
        got = complex(res.v[k, ld])
        np.testing.assert_allclose(
            [got.real, got.imag], [v[1].real, v[1].imag], rtol=1e-7, atol=1e-9
        )


# ---------------------------------------------------------------------------
# 5. harmonic_injection override: scalar broadcast == old; per-element differs.
# ---------------------------------------------------------------------------
def test_override_scalar_broadcast_matches_device_spectrum():
    """A scalar override equals the same device-level spectrum (broadcast on all elements)."""
    comps = [(1, 1.0, 0.0), (5, 0.2, 0.0)]
    load_spec = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
        q_nom_per_phase_var=(400.0, 300.0, 200.0),
        load_model=LoadModel.CONST_POWER,
        spectrum=_spec(comps),
    )
    grid_spec = _grid(load_spec, ABC)
    res_a = solve_harmonic_flow(grid_spec, [1, 5], slack="norton", dtype=CDT)

    load_plain = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
        q_nom_per_phase_var=(400.0, 300.0, 200.0),
        load_model=LoadModel.CONST_POWER,
    )
    grid_plain = _grid(load_plain, ABC)
    inj = {30: {1: (1.0, 0.0), 5: (0.2, 0.0)}}  # scalar -> all elements
    res_b = solve_harmonic_flow(
        grid_plain, [1, 5], slack="norton", dtype=CDT, harmonic_injection=inj
    )
    torch.testing.assert_close(res_a.v, res_b.v, rtol=1e-9, atol=1e-12)


def test_override_per_element_differs_from_scalar():
    """A length-n_elem override (per element) gives a different, A-only 5th harmonic."""
    load = Load(
        id=30,
        node=2,
        phases=ABC,
        p_nom_w=4500.0,
        q_nom_var=900.0,
        p_nom_per_phase_w=(2000.0, 1500.0, 1000.0),
        q_nom_per_phase_var=(400.0, 300.0, 200.0),
        load_model=LoadModel.CONST_POWER,
    )
    grid = _grid(load, ABC)
    # 5th harmonic only on element 0 (phase A).
    inj = {30: {1: (1.0, 0.0), 5: ([0.3, 0.0, 0.0], [10.0, 0.0, 0.0])}}
    res, v1, ih = _injection(grid, [5], harmonic_injection=inj)

    ra = res.index.row(2, Phase.A)
    rb = res.index.row(2, Phase.B)
    rc = res.index.row(2, Phase.C)
    inj5 = ih[0]
    assert abs(complex(inj5[ra])) > 1e-3
    assert abs(complex(inj5[rb])) < 1e-12
    assert abs(complex(inj5[rc])) < 1e-12

    # Matches the spectrum_per_phase A-only result (same A spectrum, same grid).
    s0a = complex(2000.0, 400.0)
    va = complex(v1[ra])
    i1a = np.conj(s0a) / np.conj(va)
    ihA = _i_h_elem(i1a, 5, 1.0, 0.0, 0.3, 10.0)
    np.testing.assert_allclose(
        [complex(inj5[ra]).real, complex(inj5[ra]).imag],
        [(-ihA).real, (-ihA).imag],
        rtol=1e-9,
        atol=1e-9,
    )
