"""Harmonic device shunt: the closed form, its limits, and the nodal stamp.

The analytic half of the load-shunt validation (the live-OpenDSS half is
``test_opendss_load_shunt.py``). Every expectation here is hand-derived from the model
pinned in ``pgml.assembly._load_shunt``, per element, with ``s = series_rl_fraction``::

    Y_eq     = conj(P + jQ)/V_rated**2
    Y_par(h) = (1 - s)*Re(Y_eq) + j*(1 - s)*Im(Y_eq)/h
    Y_ser(h) = 1/(Re(Z_ser) + j*h*Im(Z_ser)),   Z_ser = 1/(s*Y_eq)

Checked: the fundamental identity ``Y_par(1) + Y_ser(1) = Y_eq`` for every split, the
two limits ``s = 0`` (``G + jB/h``) and ``s = 1`` (``1/(R + j h X)``), the motor branch,
the zero-power degenerate case, the WYE / DELTA / four-wire nodal stamp, the per-device
override precedence, and the generation policy (a Generator / Storage carries no shunt
under the shipped ``appliance.harmonic_shunt.generation_model``, because the expression's
conductance is negative for an injecting device).
"""

from __future__ import annotations

import math

import pytest
import torch

from pgml.assembly import node_phase_index
from pgml.assembly._load_shunt import (
    generation_shunt_is_neglected,
    harmonic_shunt_element_admittance,
    resolve_harmonic_shunt,
    resolve_shunt_model_name,
)
from pgml.errors import InputError
from pgml.schemas.grid_schema import (
    Generator,
    Grid,
    HarmonicComponent,
    HarmonicShuntModel,
    Line,
    Load,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
    Storage,
    WindingConnection,
)
from pgml.solver.harmonic_flow import assemble_harmonic_ybus, solve_harmonic_flow

CDT = torch.complex128
RDT = torch.float64
F0 = 50.0


def _element_admittance(p_w, q_var, v_rated, orders, s, *, x_pu=0.0, xr=1.0, kva=0.0):
    """The module's element admittance for ONE element, as a list of complex numbers."""
    y = harmonic_shunt_element_admittance(
        torch.tensor([[complex(p_w, q_var)]], dtype=CDT),
        torch.tensor([[v_rated]], dtype=RDT),
        torch.tensor([float(h) for h in orders], dtype=RDT),
        torch.tensor([[s]], dtype=RDT),
        motor_x_pu=torch.tensor([[x_pu]], dtype=RDT),
        motor_xr=torch.tensor([[xr]], dtype=RDT),
        motor_s_base=torch.tensor([[kva]], dtype=RDT),
        cdtype=CDT,
    )
    return [complex(v) for v in y.reshape(-1)]


@pytest.mark.parametrize("s", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_split_is_exact_at_the_fundamental(s):
    """``Y_par(1) + Y_ser(1) = Y_eq`` for every split: the split only sets the roll-off."""
    p, q, v = 30_000.0, 10_000.0, 230.94
    (y1,) = _element_admittance(p, q, v, [1], s)
    y_eq = complex(p, -q) / v**2
    assert abs(y1 - y_eq) <= 1e-15 * abs(y_eq)


def test_all_parallel_limit_is_g_plus_jb_over_h():
    """``s = 0``: a constant conductance with a ``1/h`` susceptance (the damped limit)."""
    p, q, v = 30_000.0, 10_000.0, 230.94
    y_eq = complex(p, -q) / v**2
    for h, y in zip((3, 5, 25), _element_admittance(p, q, v, [3, 5, 25], 0.0)):
        expect = complex(y_eq.real, y_eq.imag / h)
        assert abs(y - expect) <= 1e-15 * abs(expect)


def test_all_series_limit_is_one_over_r_plus_jhx():
    """``s = 1``: the whole shunt is the series R-L branch ``1/(R + j h X)``."""
    p, q, v = 30_000.0, 10_000.0, 230.94
    z_ser = 1.0 / (complex(p, -q) / v**2)
    for h, y in zip((3, 5, 25), _element_admittance(p, q, v, [3, 5, 25], 1.0)):
        expect = 1.0 / complex(z_ser.real, h * z_ser.imag)
        assert abs(y - expect) <= 1e-14 * abs(expect)


def test_motor_branch_replaces_the_derived_series_impedance():
    """``motor``: ``Z_ser(h) = X/xr + j h X`` with ``X = V**2/(S*s)*x_pu``."""
    p, q, v, s, x_pu, xr = 30_000.0, 10_000.0, 230.94, 0.5, 0.2, 6.0
    kva = math.hypot(p, q)
    y_eq = complex(p, -q) / v**2
    x = v**2 / (kva * s) * x_pu
    for h, y in zip(
        (5, 13), _element_admittance(p, q, v, [5, 13], s, x_pu=x_pu, xr=xr, kva=kva)
    ):
        expect = complex(
            (1.0 - s) * y_eq.real, (1.0 - s) * y_eq.imag / h
        ) + 1.0 / complex(x / xr, h * x)
        assert abs(y - expect) <= 1e-14 * abs(expect)


def test_motor_without_a_series_fraction_keeps_only_the_parallel_branch():
    """``s = 0`` drops the series branch, as OpenDSS's ``%SeriesRL <> 0`` guard does."""
    p, q, v = 30_000.0, 10_000.0, 230.94
    y_eq = complex(p, -q) / v**2
    (y,) = _element_admittance(p, q, v, [5], 0.0, x_pu=0.2, xr=6.0, kva=1.0)
    assert abs(y - complex(y_eq.real, y_eq.imag / 5)) <= 1e-15


def test_zero_power_device_has_no_shunt():
    """A device at zero power has ``Y_eq = 0``, so no shunt and no division by zero."""
    for s in (0.0, 0.5, 1.0):
        for y in _element_admittance(0.0, 0.0, 400.0, [1, 5], s):
            assert y == 0.0


def test_generator_sign_gives_a_negative_conductance():
    """The load convention carries the sign: a generator's shunt conductance is < 0.

    Which is why a generation device carries NO shunt under the shipped
    ``appliance.harmonic_shunt.generation_model`` — see
    :func:`test_a_generation_device_carries_no_shunt_by_default`. This pins the
    expression itself, which the ``load_style`` policy applies.
    """
    (y,) = _element_admittance(-5_000.0, 0.0, 230.0, [5], 0.5)
    assert y.real < 0.0


def test_a_generation_device_carries_no_shunt_by_default():
    """Generator / Storage are pure current sources; an identical Load is not.

    The shipped ``appliance.harmonic_shunt.generation_model: none`` refuses the load
    expression for an injecting device because its conductance is negative. A stored
    ``harmonic_model`` block does not override it (every grid written before the field
    was consumed carries the former default block on every device); naming the motor
    model does.
    """
    kwargs = dict(node=1, phases=[Phase.A], p_nom_w=1_000.0)
    load = Load(id=1, **kwargs)
    gen = Generator(id=2, **kwargs)
    storage = Storage(id=3, **kwargs)
    legacy = Generator(
        id=4,
        harmonic_model=HarmonicShuntModel(series_rl_fraction=0.5, neglect_shunt=False),
        **kwargs,
    )
    motor = Generator(
        id=5, harmonic_model=HarmonicShuntModel(motor_x_harm_pu=0.3), **kwargs
    )
    assert resolve_harmonic_shunt(load, "opendss").kind == "opendss"
    assert resolve_harmonic_shunt(gen, "opendss").kind == "none"
    assert resolve_harmonic_shunt(storage, "opendss").kind == "none"
    assert resolve_harmonic_shunt(legacy, "opendss").kind == "none"
    assert resolve_harmonic_shunt(motor, "opendss").kind == "motor"
    assert generation_shunt_is_neglected(gen) is True
    assert generation_shunt_is_neglected(load) is False


def test_the_load_style_generation_policy_is_reachable(tmp_path, monkeypatch):
    """``generation_model: load_style`` applies the load expression, sign included."""
    import yaml

    from pgml import defaults

    gen = Generator(id=1, node=1, phases=[Phase.A], p_nom_w=1_000.0)
    data = yaml.safe_load(yaml.safe_dump(defaults.defaults()))
    data["appliance"]["harmonic_shunt"]["generation_model"]["value"] = "load_style"
    path = tmp_path / "load_style.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("PGML_DEFAULTS", str(path))
    try:
        defaults.reload(str(path))
        spec = resolve_harmonic_shunt(gen, "opendss")
    finally:
        monkeypatch.delenv("PGML_DEFAULTS", raising=False)
        defaults.reload()
    assert (spec.kind, spec.series_rl_fraction) == ("opendss", 0.5)
    assert resolve_harmonic_shunt(gen, "opendss").kind == "none"


def test_an_unknown_generation_policy_raises(tmp_path, monkeypatch):
    import yaml

    from pgml import defaults

    data = yaml.safe_load(yaml.safe_dump(defaults.defaults()))
    data["appliance"]["harmonic_shunt"]["generation_model"]["value"] = "filter"
    path = tmp_path / "bad_generation.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("PGML_DEFAULTS", str(path))
    try:
        defaults.reload(str(path))
        with pytest.raises(InputError, match="generation-shunt model"):
            resolve_harmonic_shunt(
                Generator(id=1, node=1, phases=[Phase.A], p_nom_w=1.0), "opendss"
            )
    finally:
        monkeypatch.delenv("PGML_DEFAULTS", raising=False)
        defaults.reload()


# --------------------------------------------------------------------------- #
# model resolution
# --------------------------------------------------------------------------- #
def test_unknown_model_name_raises():
    with pytest.raises(InputError, match="unknown harmonic load-shunt model"):
        resolve_shunt_model_name("series_rl")


def test_run_level_none_wins_over_a_device_override():
    """``load_shunt="none"`` is OpenDSS's ``NeglectLoadY``: no shunt anywhere."""
    load = Load(
        id=1,
        node=1,
        phases=[Phase.A],
        p_nom_w=1000.0,
        harmonic_model=HarmonicShuntModel(series_rl_fraction=1.0),
    )
    assert resolve_harmonic_shunt(load, "none").kind == "none"


def test_device_override_wins_over_the_run_level_model():
    """A device block decides its own model; the run level only fills the gaps."""
    off = Load(
        id=1,
        node=1,
        phases=[Phase.A],
        p_nom_w=1000.0,
        harmonic_model=HarmonicShuntModel(neglect_shunt=True),
    )
    motor = Load(
        id=2,
        node=1,
        phases=[Phase.A],
        p_nom_w=1000.0,
        harmonic_model=HarmonicShuntModel(series_rl_fraction=0.25, motor_x_harm_pu=0.3),
    )
    plain = Load(id=3, node=1, phases=[Phase.A], p_nom_w=1000.0)
    assert resolve_harmonic_shunt(off, "opendss").kind == "none"
    spec = resolve_harmonic_shunt(motor, "opendss")
    assert (spec.kind, spec.series_rl_fraction, spec.motor_x_harm_pu) == (
        "motor",
        0.25,
        0.3,
    )
    # No device block: the documented defaults, under the run-level model.
    assert resolve_harmonic_shunt(plain, "opendss") == resolve_harmonic_shunt(
        plain, "opendss"
    )
    assert resolve_harmonic_shunt(plain, "motor").kind == "motor"
    assert resolve_harmonic_shunt(plain, "opendss").series_rl_fraction == 0.5


def test_contradictory_device_block_is_rejected():
    with pytest.raises(ValueError, match="contradicts motor_x_harm_pu"):
        HarmonicShuntModel(neglect_shunt=True, motor_x_harm_pu=0.2)


# --------------------------------------------------------------------------- #
# nodal stamp: connection awareness
# --------------------------------------------------------------------------- #
def _two_node_grid(phases, connection, *, neutral=False, p=9_000.0, q=3_000.0, u=400.0):
    """Source - line - load on a two-bus grid, with the load on the second node."""
    node_phases = tuple(phases) + ((Phase.N,) if neutral else ())
    n = len(node_phases)

    def diag(value: float) -> list[list[float]]:
        return [[value if i == j else 0.0 for j in range(n)] for i in range(n)]

    nodes = [
        Node(id=1, u_rated_v=u, phases=node_phases),
        Node(id=2, u_rated_v=u, phases=node_phases),
    ]
    source = Source(
        id=10,
        node=1,
        phases=node_phases,
        u_ref_v=tuple([u / math.sqrt(3.0)] * n) if n >= 3 else tuple([u] * n),
        u_angle_deg=(0.0, -120.0, 120.0, 0.0)[:n],
        resistance_ohm=diag(0.05),
        inductance_h=diag(1.6e-4),
    )
    line = Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=node_phases,
        to_phases=node_phases,
        length_m=100.0,
        series_resistance_ohm_per_m=diag(1.0e-3),
        series_inductance_h_per_m=diag(1.0e-6),
        shunt_capacitance_f_per_m=diag(1.0e-9),
    )
    load = Load(
        id=30,
        node=2,
        phases=tuple(phases),
        p_nom_w=p,
        q_nom_var=q,
        connection=connection,
    )
    return Grid(
        base_frequency_hz=F0,
        nodes=nodes,
        branches=[line],
        appliances=[source, load],
    )


def _shunt_block(grid, order, rows):
    """The admittance the device shunt adds to ``Y(h)`` at ``rows`` (a difference)."""
    y_on, index = assemble_harmonic_ybus(grid, [order], load_shunt="opendss")
    y_off, _ = assemble_harmonic_ybus(grid, [order], load_shunt="none")
    d = (y_on - y_off)[0]
    idx = torch.as_tensor([index.row(2, ph) for ph in rows], dtype=torch.int64)
    return d.index_select(0, idx).index_select(1, idx).numpy(), index


def test_wye_grounded_stamp_is_diagonal_per_phase():
    """WYE to ground: the element admittance sits on each phase diagonal, nothing else."""
    grid = _two_node_grid((Phase.A, Phase.B, Phase.C), WindingConnection.WYE)
    block, _ = _shunt_block(grid, 5, (Phase.A, Phase.B, Phase.C))
    (y_elem,) = _element_admittance(3_000.0, 1_000.0, 400.0 / math.sqrt(3.0), [5], 0.5)
    for i in range(3):
        for j in range(3):
            expect = y_elem if i == j else 0.0
            assert abs(block[i, j] - expect) <= 1e-12 * max(abs(y_elem), 1.0)


def test_delta_stamp_puts_each_leg_on_both_phase_diagonals():
    """DELTA: the nodal block is the circulant ``M^T diag(y) M`` — diagonal ``2*y``.

    This is OpenDSS's own delta ``YPrim`` (each leg admittance appears on the two phase
    diagonals it connects, with ``-y`` off-diagonal), and the element voltage base is
    line-to-line.
    """
    grid = _two_node_grid((Phase.A, Phase.B, Phase.C), WindingConnection.DELTA)
    block, _ = _shunt_block(grid, 7, (Phase.A, Phase.B, Phase.C))
    (y_leg,) = _element_admittance(3_000.0, 1_000.0, 400.0, [7], 0.5)
    for i in range(3):
        for j in range(3):
            expect = 2.0 * y_leg if i == j else -y_leg
            assert abs(block[i, j] - expect) <= 1e-12 * abs(y_leg)


def test_four_wire_wye_stamp_returns_through_the_neutral_row():
    """WYE with a neutral: ``M = [I | -1]``, so the neutral row carries the full return."""
    grid = _two_node_grid(
        (Phase.A, Phase.B, Phase.C), WindingConnection.WYE, neutral=True
    )
    block, _ = _shunt_block(grid, 5, (Phase.A, Phase.B, Phase.C, Phase.N))
    (y_elem,) = _element_admittance(3_000.0, 1_000.0, 400.0 / math.sqrt(3.0), [5], 0.5)
    for i in range(3):
        assert abs(block[i, i] - y_elem) <= 1e-12 * abs(y_elem)
        assert abs(block[i, 3] + y_elem) <= 1e-12 * abs(y_elem)
        assert abs(block[3, i] + y_elem) <= 1e-12 * abs(y_elem)
    assert abs(block[3, 3] - 3.0 * y_elem) <= 1e-12 * abs(y_elem)


def test_the_shunt_damps_the_harmonic_voltage():
    """The shunt draws part of the injected harmonic current, so ``|V(h)|`` falls."""
    grid = _two_node_grid((Phase.A, Phase.B, Phase.C), WindingConnection.WYE)
    load = next(a for a in grid.appliances if isinstance(a, Load))
    load.spectrum = StaticSpectrum(
        spectrum=SpectrumPoint(
            components=[
                HarmonicComponent(order=1, magnitude_pu=1.0, phase_deg=0.0),
                HarmonicComponent(order=5, magnitude_pu=0.2, phase_deg=0.0),
            ]
        )
    )
    index = node_phase_index(grid)
    row = index.row(2, Phase.A)
    v_off = solve_harmonic_flow(grid, [1, 5], slack="norton", load_shunt="none").v
    v_on = solve_harmonic_flow(grid, [1, 5], slack="norton", load_shunt="opendss").v
    assert abs(v_on[0, row] - v_off[0, row]) <= 1e-9 * abs(
        v_off[0, row]
    )  # h = 1 intact
    assert abs(v_on[1, row]) < abs(v_off[1, row])
