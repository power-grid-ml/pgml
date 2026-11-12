"""Two-phase DELTA (phase-to-phase) appliances and the node phase-to-neutral rating.

A DELTA appliance on two phases is one element between them (``M = [1, -1]``): it is
stamped and drawn on both phase rows with opposite sign, its nominal voltage is the
node's ``u_rated_v`` and its per-phase values are summed onto the element. A node's
``u_rated_ln_v`` is the per-phase base of its rows. A node with fewer than three phase
rows stores ``u_rated_v`` line-to-neutral unless it sets ``u_rated_ln_v``, so a DELTA
appliance there needs that field.
"""

from __future__ import annotations

import math

import pytest
import torch

from pgml.assembly import (
    assemble_ybus,
    base_voltage_per_row,
    device_current_injections,
    node_phase_index,
    node_voltage_magnitude,
    phase_voltage_magnitude,
)
from pgml.errors import InputError
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    HarmonicComponent,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    ShuntReactor,
    Source,
    SpectrumPoint,
    StaticSpectrum,
    Storage,
    WindingConnection,
)
from pgml.solver import harmonic_injections, solve_power_flow

P = Phase
CDT = torch.complex128
ABN = (P.A, P.B, P.N)
ABCN = (P.A, P.B, P.C, P.N)
SPEC = StaticSpectrum(
    spectrum=SpectrumPoint(
        components=[
            HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
            HarmonicComponent(order=3, magnitude_pu=0.2, phase_deg=30.0),
        ]
    )
)


def _grid(appliances, *, phases=ABCN, u=400.0, ln=None) -> Grid:
    """Source node 1 and load node 2 joined by a short four-wire line; N earthed."""
    n = len(phases)
    r = [[3e-4 if i == j else 5e-5 for j in range(n)] for i in range(n)]
    ell = [[1e-6 if i == j else 3e-7 for j in range(n)] for i in range(n)]
    nodes = [
        Node(id=1, u_rated_v=u, phases=phases, u_rated_ln_v=ln),
        Node(id=2, u_rated_v=u, phases=phases, u_rated_ln_v=ln),
    ]
    branches = [
        Line(
            id=10,
            from_node=1,
            to_node=2,
            from_phases=phases,
            to_phases=phases,
            length_m=100.0,
            series_resistance_ohm_per_m=r,
            series_inductance_h_per_m=ell,
            shunt_capacitance_f_per_m=[[0.0] * n for _ in range(n)],
        ),
        ShuntReactor(
            id=11,
            from_node=1,
            to_node=1,
            from_phases=(P.N,),
            to_phases=(P.N,),
            conductance_s=[[10.0]],
            capacitance_f=[[0.0]],
        ),
    ]
    live = [p for p in phases if p != P.N]
    u_ln = ln if ln is not None else u / math.sqrt(3.0)
    angles = {P.A: 0.0, P.B: -120.0, P.C: 120.0}
    if len(live) == 2 and ln is not None and abs(u / ln - 2.0) < 1e-9:
        angles = {P.A: 0.0, P.B: 180.0}  # split phase
    src = Source(
        id=1,
        node=1,
        phases=tuple(live),
        u_ref_v=[u_ln] * len(live),
        u_angle_deg=[angles[p] for p in live],
        resistance_ohm=[[0.01 if i == j else 0.0 for j in live] for i in live],
        inductance_h=[[1e-5 if i == j else 0.0 for j in live] for i in live],
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=nodes,
        branches=branches,
        appliances=[src, *appliances],
    )


def _delta(aid=20, *, phases=(P.A, P.B), model=LoadModel.CONST_POWER, **kw) -> Load:
    kw.setdefault("p_nom_w", 9e3)
    kw.setdefault("q_nom_var", 3e3)
    return Load(
        id=aid,
        node=2,
        phases=phases,
        connection=WindingConnection.DELTA,
        load_model=model,
        **kw,
    )


def _random_v(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    re = torch.rand(n, generator=g, dtype=torch.float64) * 50 + 200
    im = torch.rand(n, generator=g, dtype=torch.float64) * 40 - 20
    return torch.complex(re, im)


def test_const_z_fold_is_one_leg_between_the_two_phases():
    """``Y_load = y·[[1, -1], [-1, 1]]`` on the A, B rows, ``y = conj(S)/u_rated_v²``."""
    grid = _grid([_delta(model=LoadModel.CONST_IMPEDANCE)])
    bare = _grid([])
    y_load = (assemble_ybus(grid, 50.0).Y - assemble_ybus(bare, 50.0).Y).numpy()
    idx = node_phase_index(grid)
    a, b = idx.row(2, P.A), idx.row(2, P.B)
    y = (9e3 - 3e3j) / 400.0**2
    expected = {(a, a): y, (b, b): y, (a, b): -y, (b, a): -y}
    for (i, j), val in expected.items():
        assert abs(y_load[i, j] - val) < 1e-13 * abs(val)
    y_load[a, a] = y_load[b, b] = y_load[a, b] = y_load[b, a] = 0.0
    assert abs(y_load).max() == 0.0


@pytest.mark.parametrize("model", [LoadModel.CONST_IMPEDANCE])
def test_const_z_device_current_reproduces_the_linear_stamp(model):
    grid = _grid([_delta(model=model)])
    bare = _grid([])
    idx = node_phase_index(grid)
    v = _random_v(idx.size)
    y_load = assemble_ybus(grid, 50.0).Y - assemble_ybus(bare, 50.0).Y
    i_dev = device_current_injections(grid, v, idx, 50.0).reshape(-1)
    assert torch.allclose(i_dev, y_load @ v, rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("cls,sign", [(Load, 1.0), (Generator, -1.0), (Storage, -1.0)])
def test_constant_power_element_draws_its_power_across_the_phases(cls, sign):
    app = cls(
        id=20,
        node=2,
        phases=(P.B, P.C),
        connection=WindingConnection.DELTA,
        p_nom_w=9e3,
        q_nom_var=3e3,
    )
    grid = _grid([app])
    idx = node_phase_index(grid)
    v = _random_v(idx.size, seed=1)
    i = device_current_injections(grid, v, idx, 50.0).reshape(-1)
    b, c = idx.row(2, P.B), idx.row(2, P.C)
    assert torch.allclose(i[b], -i[c], rtol=0, atol=1e-12)
    others = [r for r in range(idx.size) if r not in (b, c)]
    assert i[others].abs().max() == 0.0
    s = (v[b] - v[c]) * torch.conj(i[b])
    assert abs(complex(s) - sign * (9e3 + 3e3j)) < 1e-9


def test_per_phase_values_sum_onto_the_element():
    """Nameplate split, per-phase operating point and symmetry give the same element."""
    split = _grid(
        [_delta(p_nom_per_phase_w=(5e3, 4e3), q_nom_per_phase_var=(2e3, 1e3))]
    )
    total = _grid([_delta()])
    idx = node_phase_index(total)
    v = _random_v(idx.size, seed=2)
    ref = device_current_injections(total, v, idx, 50.0)
    for sym in ("symmetric", "asymmetric"):
        got = device_current_injections(split, v, idx, 50.0, symmetry=sym)
        assert torch.allclose(got, ref, rtol=1e-14, atol=0)
    op = {20: {"p_per_phase_w": [4e3, 5e3], "q_per_phase_var": [1.5e3, 1.5e3]}}
    got = device_current_injections(total, v, idx, 50.0, operating_point=op)
    assert torch.allclose(got, ref, rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    "model",
    [LoadModel.CONST_POWER, LoadModel.CONST_IMPEDANCE, LoadModel.CONST_CURRENT],
)
def test_power_flow_converges_and_balances(model):
    grid = _grid([_delta(model=model)])
    res = solve_power_flow(grid, tol=1e-12)
    assert res.converged
    idx = node_phase_index(grid)
    v_ab = res.v[idx.row(2, P.A)] - res.v[idx.row(2, P.B)]
    assert abs(abs(complex(v_ab)) - 400.0) / 400.0 < 0.05


def test_harmonic_current_follows_the_element_fundamental_current():
    grid = _grid([_delta(spectrum=SPEC)])
    res = solve_power_flow(grid, tol=1e-12)
    idx = node_phase_index(grid)
    v1 = res.v
    a, b = idx.row(2, P.A), idx.row(2, P.B)
    i1 = torch.conj((9e3 + 3e3j) / (v1[a] - v1[b]))
    ih = harmonic_injections(grid, v1, [3])[0]
    phase = torch.as_tensor(math.radians(30.0), dtype=torch.float64) + 3 * i1.angle()
    expected = 0.2 * torch.polar(i1.abs(), phase)
    # The nodal injection is the drawn current with a negative sign.
    assert torch.allclose(ih[a], -expected, rtol=1e-12)
    assert torch.allclose(ih[b], expected, rtol=1e-12)


def test_per_phase_spectrum_keys_off_the_first_phase():
    first = _grid([_delta(spectrum_per_phase={P.A: SPEC})])
    second = _grid([_delta(spectrum_per_phase={P.B: SPEC})])
    shared = _grid([_delta(spectrum=SPEC)])
    v1 = solve_power_flow(shared, tol=1e-12).v
    ref = harmonic_injections(shared, v1, [3])
    assert torch.allclose(harmonic_injections(first, v1, [3]), ref, rtol=1e-14)
    assert harmonic_injections(second, v1, [3]).abs().max() == 0.0


# --- node phase-to-neutral rating ----------------------------------------------
def test_split_phase_rating_sets_the_row_base_and_element_voltages():
    grid = _grid([], phases=ABN, u=440.0, ln=220.0)
    assert base_voltage_per_row(grid).tolist() == [220.0] * 6
    node = grid.nodes[1]
    assert node_voltage_magnitude(node) == 220.0
    assert node_voltage_magnitude(node, line_to_line=True) == 440.0
    legacy = _grid([], phases=ABN, u=440.0)
    assert base_voltage_per_row(legacy).tolist() == pytest.approx(
        [440.0 / math.sqrt(3.0)] * 6, rel=1e-15
    )


def test_node_voltage_magnitude_is_public_and_matches_the_private_name():
    from pgml.assembly import _params

    assert node_voltage_magnitude is _params.node_voltage_magnitude
    import pgml.assembly as asm

    assert "node_voltage_magnitude" in asm.__all__


def test_phase_voltage_magnitude_needs_the_ln_rating_on_two_conductor_nodes():
    """Without ``u_rated_ln_v`` the phase-row rule gives the wrong per-phase base."""
    split = Node(id=1, u_rated_v=240.0, u_rated_ln_v=120.0, phases=ABN)
    two = Node(id=2, u_rated_v=400.0, u_rated_ln_v=230.0, phases=(P.A, P.B))
    for node, ln in ((split, 120.0), (two, 230.0)):
        n = len(node.phases)
        assert node_voltage_magnitude(node) == ln
        assert phase_voltage_magnitude(node.u_rated_v, n, u_rated_ln_v=ln) == ln
        assert node_voltage_magnitude(node, line_to_line=True) == node.u_rated_v
    # The phase-row rule alone: 240/sqrt(3) and 400 V instead of 120 and 230 V.
    assert phase_voltage_magnitude(240.0, 3) == pytest.approx(240.0 / math.sqrt(3.0))
    assert phase_voltage_magnitude(400.0, 2) == 400.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_node_voltage_magnitude_tensor_ratings(dtype):
    """Tensor ratings stay on the tape (device/dtype kept); ``plain`` reads floats."""
    u = torch.tensor(240.0, dtype=dtype, requires_grad=True)
    ln = torch.tensor(120.0, dtype=dtype, requires_grad=True)
    node = Node(id=1, u_rated_v=u, u_rated_ln_v=ln, phases=ABN)
    v = node_voltage_magnitude(node)
    assert isinstance(v, torch.Tensor) and v.dtype == dtype and v.device == ln.device
    v.backward()
    assert ln.grad == 1.0
    delta = node_voltage_magnitude(node, line_to_line=True)
    assert delta is u
    plain = node_voltage_magnitude(node, plain=True)
    assert type(plain) is float and plain == 120.0
    no_ln = Node(id=2, u_rated_v=u, phases=(P.A, P.B, P.C))
    w = node_voltage_magnitude(no_ln)
    torch.autograd.gradcheck(
        lambda x: node_voltage_magnitude(
            Node(id=3, u_rated_v=x, phases=(P.A, P.B, P.C))
        ),
        (torch.tensor(400.0, dtype=torch.float64, requires_grad=True),),
    )
    assert float(w.detach()) == pytest.approx(240.0 / math.sqrt(3.0), rel=1e-6)


def test_split_phase_constant_impedance_halves_use_the_rating():
    """A wye const-Z half load on X1-N is referred to ``u_rated_ln_v``."""
    half = Load(
        id=21,
        node=2,
        phases=(P.A,),
        connection="wye",
        p_nom_w=4e3,
        q_nom_var=1e3,
        load_model=LoadModel.CONST_IMPEDANCE,
    )
    grid = _grid([half], phases=ABN, u=440.0, ln=220.0)
    bare = _grid([], phases=ABN, u=440.0, ln=220.0)
    y_load = assemble_ybus(grid, 50.0).Y - assemble_ybus(bare, 50.0).Y
    idx = node_phase_index(grid)
    a, n = idx.row(2, P.A), idx.row(2, P.N)
    y = (4e3 - 1e3j) / 220.0**2
    assert abs(complex(y_load[a, a]) - y) < 1e-13 * abs(y)
    assert abs(complex(y_load[a, n]) + y) < 1e-13 * abs(y)


def test_split_phase_delta_and_halves_solve():
    """X1-X2 delta load plus both halves on a 440/220 V split-phase node."""
    loads = [
        _delta(model=LoadModel.CONST_IMPEDANCE, spectrum=SPEC),
        Load(id=21, node=2, phases=(P.A,), p_nom_w=3e3, q_nom_var=5e2),
        Load(id=22, node=2, phases=(P.B,), p_nom_w=2e3, q_nom_var=5e2),
    ]
    grid = _grid(loads, phases=ABN, u=440.0, ln=220.0)
    res = solve_power_flow(grid, tol=1e-12)
    assert res.converged
    idx = node_phase_index(grid)
    v = res.v
    v_x1x2 = abs(complex(v[idx.row(2, P.A)] - v[idx.row(2, P.B)]))
    assert abs(v_x1x2 - 440.0) / 440.0 < 0.05


# --- DELTA appliance on a node with fewer than three phase rows ------------------
AB = (P.A, P.B)
U_LN = 400.0 / math.sqrt(3.0)


def _two_wire_grid(appliances, *, ln=None) -> Grid:
    """Source node 1 and load node 2 on two phase conductors ``(A, B)``, no neutral.

    Without ``ln`` the nodes carry the line-to-neutral rating (the schema rule for a
    node with fewer than three phase rows); with it they are rated 400/``ln`` V.
    """
    u = U_LN if ln is None else 400.0
    r = [[3e-4, 5e-5], [5e-5, 3e-4]]
    ell = [[1e-6, 3e-7], [3e-7, 1e-6]]
    nodes = [
        Node(id=1, u_rated_v=u, phases=AB, u_rated_ln_v=ln),
        Node(id=2, u_rated_v=u, phases=AB, u_rated_ln_v=ln),
    ]
    line = Line(
        id=10,
        from_node=1,
        to_node=2,
        from_phases=AB,
        to_phases=AB,
        length_m=100.0,
        series_resistance_ohm_per_m=r,
        series_inductance_h_per_m=ell,
        shunt_capacitance_f_per_m=[[0.0, 0.0], [0.0, 0.0]],
    )
    src = Source(
        id=1,
        node=1,
        phases=AB,
        u_ref_v=[U_LN, U_LN],
        u_angle_deg=[0.0, -120.0],
        resistance_ohm=[[0.01, 0.0], [0.0, 0.01]],
        inductance_h=[[1e-5, 0.0], [0.0, 1e-5]],
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=nodes,
        branches=[line],
        appliances=[src, *appliances],
    )


@pytest.mark.parametrize("cls", [Load, Generator, Storage])
def test_delta_on_two_row_node_without_ln_rating_is_refused(cls):
    app = cls(
        id=20,
        node=2,
        phases=AB,
        connection=WindingConnection.DELTA,
        p_nom_w=9e3,
        q_nom_var=3e3,
    )
    grid = _two_wire_grid([app])
    with pytest.raises(InputError, match=r"appliance 20.*node 2.*`u_rated_ln_v`"):
        solve_power_flow(grid, tol=1e-12)


def test_delta_on_two_row_node_is_refused_by_the_linear_assembly():
    grid = _two_wire_grid([_delta(model=LoadModel.CONST_IMPEDANCE)])
    with pytest.raises(InputError, match="Set `u_rated_ln_v` on the node"):
        assemble_ybus(grid, 50.0)


def test_delta_by_config_default_on_two_row_node_is_refused(monkeypatch, tmp_path):
    """The refusal follows the RESOLVED connection, not only an explicit field."""
    import yaml

    from pgml import defaults

    load = Load(id=20, node=2, phases=AB, p_nom_w=9e3, q_nom_var=3e3)
    grid = _two_wire_grid([load])
    assert solve_power_flow(grid, tol=1e-12).converged
    data = yaml.safe_load(yaml.safe_dump(defaults.defaults()))
    data["appliance"]["load"]["default_connection"]["value"] = "delta"
    custom = tmp_path / "defaults.yaml"
    custom.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("PGML_DEFAULTS", str(custom))
    try:
        defaults.reload(str(custom))
        with pytest.raises(InputError, match="u_rated_ln_v"):
            solve_power_flow(grid, tol=1e-12)
    finally:
        monkeypatch.delenv("PGML_DEFAULTS", raising=False)
        defaults.reload()


@pytest.mark.parametrize(
    "model",
    [LoadModel.CONST_POWER, LoadModel.CONST_IMPEDANCE, LoadModel.CONST_CURRENT],
)
def test_delta_on_two_row_node_with_ln_rating_is_rated_phase_to_phase(model):
    grid = _two_wire_grid([_delta(model=model)], ln=U_LN)
    assert node_voltage_magnitude(grid.nodes[1], line_to_line=True) == 400.0
    res = solve_power_flow(grid, tol=1e-12)
    assert res.converged
    idx = node_phase_index(grid)
    v_ab = res.v[idx.row(2, P.A)] - res.v[idx.row(2, P.B)]
    assert abs(abs(complex(v_ab)) - 400.0) / 400.0 < 0.05
    if model is LoadModel.CONST_IMPEDANCE:
        bare = _two_wire_grid([], ln=U_LN)
        y_load = assemble_ybus(grid, 50.0).Y - assemble_ybus(bare, 50.0).Y
        a = idx.row(2, P.A)
        y = (9e3 - 3e3j) / 400.0**2
        assert abs(complex(y_load[a, a]) - y) < 1e-13 * abs(y)


def test_wye_on_two_row_node_without_ln_rating_is_unaffected():
    wye = Load(
        id=20,
        node=2,
        phases=AB,
        connection=WindingConnection.WYE,
        p_nom_w=9e3,
        q_nom_var=3e3,
        load_model=LoadModel.CONST_IMPEDANCE,
    )
    grid = _two_wire_grid([wye])
    assert solve_power_flow(grid, tol=1e-12).converged
    bare = _two_wire_grid([])
    y_load = assemble_ybus(grid, 50.0).Y - assemble_ybus(bare, 50.0).Y
    idx = node_phase_index(grid)
    a = idx.row(2, P.A)
    y = (4.5e3 - 1.5e3j) / U_LN**2
    assert abs(complex(y_load[a, a]) - y) < 1e-13 * abs(y)


@pytest.mark.parametrize(
    "phases,app_phases",
    [(ABCN, (P.A, P.B, P.C)), (ABCN, AB), (ABN, AB)],
)
def test_delta_on_three_row_node_without_ln_rating_is_unaffected(phases, app_phases):
    """Three or more phase rows: ``u_rated_v`` is line-to-line, the delta rating."""
    grid = _grid([_delta(phases=app_phases)], phases=phases)
    assert grid.nodes[1].u_rated_ln_v is None
    assert solve_power_flow(grid, tol=1e-12).converged
