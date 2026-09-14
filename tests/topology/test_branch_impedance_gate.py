"""Pre-solve gate for a branch with ZERO series impedance.

The nodal formulation turns every branch's series impedance into a primitive admittance
by inverting it, so a branch with no series impedance has no STAMP at all. Published
network data contains such branches routinely — a bus coupler or jumper modelled as a
zero-impedance line, a zero-length line, a closed switch with no impedance data — and
without a gate they surface as a linear-algebra failure naming an internal batch index,
which names neither the branch nor a fix.

:func:`pgml.solver.check_branch_impedances` answers exactly the structural question
"is every in-service branch stampable?" and raises
:class:`~pgml.errors.ModelingError` naming every branch that is not.

A solve asks a WIDER question, because an ideal conductor is representable without a
stamp: it collapses the branch's terminal rows (exact bus fusion, the documented default
``branch.zero_impedance: fuse``) and passes the resulting map to the same gate, which
then reports only what neither a stamp nor a fused row can express. The policy
``branch.zero_impedance: error`` restores the refusal.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from pgml import defaults

from pgml.errors import ModelingError, PgmlError
from pgml.schemas.grid_schema import (
    GenericBranch,
    Grid,
    Line,
    Load,
    Node,
    Phase,
    Source,
    Switch,
)
from pgml.solver import (
    check_branch_impedances,
    prepare_power_flow,
    solve_harmonic_flow,
    solve_power_flow,
)

A = (Phase.A,)
CDT = torch.complex128


def _line(bid, u, v, *, r=2.0e-4, ell=8.0e-7, length=1000.0):
    return Line(
        id=bid,
        from_node=u,
        to_node=v,
        from_phases=A,
        to_phases=A,
        length_m=length,
        series_resistance_ohm_per_m=[[r]],
        series_inductance_h_per_m=[[ell]],
        shunt_capacitance_f_per_m=[[0.0]],
    )


def _grid(branches):
    """Three 20 kV nodes in a row, source at node 1, load at node 3."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[Node(id=i, u_rated_v=20_000.0, phases=A) for i in (1, 2, 3)],
        branches=branches,
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
            Load(id=21, node=3, phases=A, p_nom_w=1.0e6, q_nom_var=3.0e5),
        ],
    )


def _coupler_grid():
    """A bus coupler between nodes 1 and 2: a line with R = L = 0."""
    return _grid([_line(10, 1, 2, r=0.0, ell=0.0), _line(11, 2, 3)])


def test_finite_impedance_grid_passes_the_gate():
    grid = _grid([_line(10, 1, 2), _line(11, 2, 3)])
    check_branch_impedances(grid)  # no raise
    assert solve_power_flow(grid, dtype=CDT).converged


def test_zero_impedance_line_raises_naming_the_branch():
    with pytest.raises(ModelingError) as err:
        check_branch_impedances(_coupler_grid())
    msg = str(err.value)
    assert "line 10" in msg
    assert "zero series impedance" in msg
    assert "1e-04 Ohm" in msg or "0.0001 Ohm" in msg  # the documented near-ideal value


@pytest.mark.parametrize(
    "entry",
    [
        lambda g: solve_power_flow(g, dtype=CDT),
        lambda g: prepare_power_flow(g, dtype=CDT),
        lambda g: solve_harmonic_flow(g, [1, 5], dtype=CDT),
    ],
)
def test_every_solve_entry_point_fuses_it(entry):
    """A bus coupler is solved by collapsing its terminals, at every entry point."""
    out = entry(_coupler_grid())
    fusion = out.fusion if hasattr(out, "fusion") else out.fusion
    assert fusion is not None
    assert fusion.fused_branch_ids == (10,)
    assert fusion.size == 2  # three rows, two of them one


@pytest.mark.parametrize(
    "entry",
    [
        lambda g: solve_power_flow(g, dtype=CDT),
        lambda g: prepare_power_flow(g, dtype=CDT),
        lambda g: solve_harmonic_flow(g, [1, 5], dtype=CDT),
    ],
)
def test_every_solve_entry_point_refuses_it_under_the_error_policy(
    entry, zero_impedance_error_policy
):
    """A pgml error, not a raw torch linear-algebra error, from every entry point."""
    with pytest.raises(PgmlError) as err:
        entry(_coupler_grid())
    assert isinstance(err.value, ModelingError)
    assert "line 10" in str(err.value)


@pytest.fixture
def zero_impedance_error_policy(tmp_path, monkeypatch):
    """Run the test under ``branch.zero_impedance: error`` (the refusal policy).

    Points ``pgml.defaults`` at a copy of the shipped file with that one key changed and
    restores the packaged defaults afterwards — both the environment pointer and the
    loader's cache, so the policy cannot leak into another test.
    """
    import yaml

    data = yaml.safe_load(
        (pathlib.Path(defaults.__file__).parent / "data" / "defaults.yaml").read_text()
    )
    data["branch"]["zero_impedance"]["value"] = "error"
    path = tmp_path / "defaults.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("PGML_DEFAULTS", str(path))
    defaults.reload(str(path))
    yield
    monkeypatch.delenv("PGML_DEFAULTS", raising=False)
    defaults.reload()
    assert defaults.get("branch.zero_impedance") == "fuse"


def test_zero_length_line_is_reported_as_such():
    """A zero LENGTH scales the whole series impedance to zero.

    The schema's positivity validator covers plain floats only, so a tensor-valued
    length (the float/tensor duality every physical field supports) reaches the
    assembly — where it is the same singular stamp.
    """
    grid = _grid(
        [_line(10, 1, 2, length=torch.zeros((), dtype=torch.float64)), _line(11, 2, 3)]
    )
    with pytest.raises(ModelingError, match="length_m is zero"):
        check_branch_impedances(grid)


def test_ideal_closed_switch_is_reported():
    """A closed switch with the schema's zero R/L default has no stamp either."""
    sw = Switch(
        id=10,
        from_node=1,
        to_node=2,
        from_phases=A,
        to_phases=A,
        closed=True,
    )
    with pytest.raises(ModelingError, match="switch 10"):
        check_branch_impedances(_grid([sw, _line(11, 2, 3)]))


def test_open_ideal_switch_is_accepted():
    """An OPEN switch is never stamped, so its impedance is irrelevant."""
    sw = Switch(
        id=10,
        from_node=1,
        to_node=2,
        from_phases=A,
        to_phases=A,
        closed=False,
    )
    check_branch_impedances(_grid([sw, _line(11, 1, 3)]))  # no raise


def test_out_of_service_zero_impedance_branch_is_accepted():
    grid = _grid(
        [
            _line(10, 1, 2, r=0.0, ell=0.0).model_copy(update={"in_service": False}),
            _line(11, 1, 2),
            _line(12, 2, 3),
        ]
    )
    check_branch_impedances(grid)  # no raise


def test_zero_impedance_generic_branch_is_reported():
    gb = GenericBranch(
        id=10,
        from_node=1,
        to_node=2,
        from_phases=A,
        to_phases=A,
        series_resistance_ohm=[[0.0]],
        series_inductance_h=[[0.0]],
    )
    with pytest.raises(ModelingError, match="generic_branch 10"):
        check_branch_impedances(_grid([gb, _line(11, 2, 3)]))


def test_near_ideal_resistance_makes_it_solvable():
    """The documented stand-in where the branch has to stay stamped."""
    r_ideal = float(defaults.get("branch.near_ideal_series_resistance_ohm"))
    grid = _grid([_line(10, 1, 2, r=r_ideal, ell=0.0, length=1.0), _line(11, 2, 3)])
    res = solve_power_flow(grid, dtype=CDT)
    assert res.converged
    # The coupler carries a negligible voltage drop (that is what near-ideal means):
    # the feeder's load current across 1e-4 Ohm, well below 1e-6 per unit.
    v = res.v.reshape(-1)
    rows = (res.index.row(1, Phase.A), res.index.row(2, Phase.A))
    drop_pu = float((v[rows[0]] - v[rows[1]]).abs()) / 20_000.0
    assert drop_pu < 1.0e-6
