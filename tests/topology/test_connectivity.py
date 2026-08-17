"""Pre-solve connectivity check: report, dedicated error, and the "zero" mode.

Covers :func:`pgml.topology.connectivity_report` / :func:`energized_subgrid`,
the :class:`pgml.errors.ConnectivityError` raised by the solvers, and the
``on_disconnected="zero"`` reduced-solve-and-scatter path (0 V on dead rows,
full-grid row layout, gradients preserved on the live rows).
"""

from __future__ import annotations

import pytest
import torch

from pgml.assembly import node_phase_index
from pgml.errors import ConnectivityError
from pgml.schemas.grid_schema import Grid, Line, Load, Node, Phase, Source, Switch
from pgml.solver import solve_harmonic_flow, solve_power_flow
from pgml.topology import connectivity_report, energized_subgrid

A = (Phase.A,)


def _line(bid, u, v, *, in_service=True):
    return Line(
        id=bid,
        from_node=u,
        to_node=v,
        from_phases=A,
        to_phases=A,
        in_service=in_service,
        length_m=100.0,
        series_resistance_ohm_per_m=[[1.0e-3]],
        series_inductance_h_per_m=[[1.0e-6]],
        shunt_capacitance_f_per_m=[[0.0]],
    )


def _switch(bid, u, v, *, closed, in_service=True):
    return Switch(
        id=bid,
        from_node=u,
        to_node=v,
        from_phases=A,
        to_phases=A,
        closed=closed,
        in_service=in_service,
        resistance_ohm=1.0e-3,
    )


def _grid(branches, *, n_nodes=5, loads=(3, 4)):
    """Source at node 0; nodes 0..n_nodes-1; loads at the given nodes."""
    nodes = [Node(id=i, u_rated_v=230.0, phases=A) for i in range(n_nodes)]
    appliances = [
        Source(
            id=100,
            node=0,
            phases=A,
            u_ref_v=(230.0,),
            u_angle_deg=(0.0,),
            resistance_ohm=[[0.05]],
            inductance_h=[[1.0e-4]],
        )
    ]
    appliances += [
        Load(id=200 + i, node=i, phases=A, p_nom_w=500.0, q_nom_var=100.0)
        for i in loads
    ]
    return Grid(
        base_frequency_hz=50.0, nodes=nodes, branches=branches, appliances=appliances
    )


def _connected_grid():
    return _grid([_line(1, 0, 1), _line(2, 1, 2), _line(3, 2, 3), _line(4, 3, 4)])


def _open_switch_grid():
    """Nodes 3, 4 hang behind an OPEN switch (a de-energized island)."""
    return _grid(
        [_line(1, 0, 1), _line(2, 1, 2), _switch(3, 2, 3, closed=False), _line(4, 3, 4)]
    )


# ---------------------------------------------------------------------------
# connectivity_report
# ---------------------------------------------------------------------------
def test_connected_grid_reports_connected():
    report = connectivity_report(_connected_grid())
    assert report.connected
    assert report.has_source
    assert report.unenergized == ()
    assert report.islands == ()


def test_open_switch_island_detected_with_reconnect_hint():
    report = connectivity_report(_open_switch_grid())
    assert not report.connected
    assert report.has_source
    assert [n for n, _ in report.unenergized] == [3, 4]
    assert report.islands == ((3, 4),)
    assert len(report.reconnectable) == 1
    hint = report.reconnectable[0]
    assert hint.branch_id == 3
    assert hint.kind == "switch"
    assert hint.reason == "open"
    assert "close switch 3" in report.describe()


def test_out_of_service_line_island_detected():
    grid = _grid(
        [
            _line(1, 0, 1),
            _line(2, 1, 2),
            _line(3, 2, 3, in_service=False),
            _line(4, 3, 4),
        ]
    )
    report = connectivity_report(grid)
    assert not report.connected
    assert report.islands == ((3, 4),)
    hint = report.reconnectable[0]
    assert (hint.branch_id, hint.kind, hint.reason) == (3, "line", "not in service")
    assert "in_service=True" in report.describe()


def test_no_source_reports_everything_unenergized():
    grid = _connected_grid()
    grid = grid.model_copy(
        update={
            "appliances": [
                a.model_copy(update={"in_service": False})
                if isinstance(a, Source)
                else a
                for a in grid.appliances
            ]
        }
    )
    report = connectivity_report(grid)
    assert not report.connected
    assert not report.has_source
    assert len(report.unenergized) == 5
    assert "no in-service Source" in report.describe()


def test_partial_phase_disconnection_detected():
    """A 3-phase node fed only on phase A: rows B, C are floating."""
    nodes = [
        Node(id=0, u_rated_v=400.0, phases=(Phase.A, Phase.B, Phase.C)),
        Node(id=1, u_rated_v=400.0, phases=(Phase.A, Phase.B, Phase.C)),
    ]
    branches = [
        Line(
            id=1,
            from_node=0,
            to_node=1,
            from_phases=A,
            to_phases=A,
            length_m=10.0,
            series_resistance_ohm_per_m=[[1.0e-3]],
            series_inductance_h_per_m=[[1.0e-6]],
            shunt_capacitance_f_per_m=[[0.0]],
        )
    ]
    appliances = [
        Source(
            id=100,
            node=0,
            phases=(Phase.A, Phase.B, Phase.C),
            u_ref_v=(230.0, 230.0, 230.0),
            u_angle_deg=(0.0, -120.0, 120.0),
            resistance_ohm=[[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]],
            inductance_h=[[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]],
        )
    ]
    grid = Grid(
        base_frequency_hz=50.0, nodes=nodes, branches=branches, appliances=appliances
    )
    report = connectivity_report(grid)
    assert not report.connected
    assert report.unenergized == ((1, (Phase.B, Phase.C)),)
    # A partially energized node cannot be reduced away.
    with pytest.raises(ConnectivityError, match="partially energized"):
        energized_subgrid(grid)


# ---------------------------------------------------------------------------
# energized_subgrid
# ---------------------------------------------------------------------------
def test_energized_subgrid_drops_island_and_its_elements():
    sub, dropped = energized_subgrid(_open_switch_grid())
    assert dropped == (3, 4)
    assert [n.id for n in sub.nodes] == [0, 1, 2]
    assert [b.id for b in sub.branches] == [1, 2]  # switch 3 + line 4 dropped
    assert all(a.node not in (3, 4) for a in sub.appliances)
    connected_sub = connectivity_report(sub)
    assert connected_sub.connected


def test_energized_subgrid_is_identity_when_connected():
    grid = _connected_grid()
    sub, dropped = energized_subgrid(grid)
    assert dropped == ()
    assert sub is grid


# ---------------------------------------------------------------------------
# solver wiring
# ---------------------------------------------------------------------------
def test_solve_power_flow_raises_connectivity_error():
    with pytest.raises(ConnectivityError) as exc:
        solve_power_flow(_open_switch_grid())
    assert exc.value.unenergized_nodes == (3, 4)
    assert exc.value.islands == ((3, 4),)
    assert exc.value.reconnectable[0].branch_id == 3
    assert "close switch 3" in str(exc.value)


def test_solve_power_flow_zero_mode_matches_subgrid_solve():
    grid = _open_switch_grid()
    res = solve_power_flow(grid, on_disconnected="zero")
    assert res.converged
    index = node_phase_index(grid)
    assert res.index.size == index.size == 5
    v = res.v
    # Dead rows are exactly 0 V.
    assert torch.equal(v[index.row(3, Phase.A)], torch.zeros_like(v[0]))
    assert torch.equal(v[index.row(4, Phase.A)], torch.zeros_like(v[0]))
    # Live rows equal the plain solve of the energized sub-grid.
    sub, _ = energized_subgrid(grid)
    ref = solve_power_flow(sub)
    for nid in (0, 1, 2):
        assert torch.allclose(
            v[index.row(nid, Phase.A)], ref.v[ref.index.row(nid, Phase.A)]
        )


def test_solve_power_flow_zero_mode_gradients_flow():
    grid = _open_switch_grid()
    r = torch.tensor([[1.0e-3]], dtype=torch.float64, requires_grad=True)
    res = solve_power_flow(
        grid,
        on_disconnected="zero",
        param_overrides={("line", 1, "series_resistance_ohm_per_m"): r},
    )
    res.v.abs().sum().backward()
    assert r.grad is not None
    assert torch.isfinite(r.grad).all()
    assert float(r.grad.abs().sum()) > 0.0


def test_solve_harmonic_flow_raise_and_zero():
    grid = _open_switch_grid()
    with pytest.raises(ConnectivityError):
        solve_harmonic_flow(grid, [1, 5])
    res = solve_harmonic_flow(grid, [1, 5], on_disconnected="zero")
    index = node_phase_index(grid)
    assert res.v.shape == (2, index.size)
    assert torch.equal(res.v[:, index.row(3, Phase.A)], torch.zeros_like(res.v[:, 0]))
    sub, _ = energized_subgrid(grid)
    ref = solve_harmonic_flow(sub, [1, 5])
    for nid in (0, 1, 2):
        assert torch.allclose(
            res.v[:, index.row(nid, Phase.A)], ref.v[:, ref.index.row(nid, Phase.A)]
        )


def test_ignore_mode_skips_the_check():
    # "ignore" must not raise ConnectivityError; the solve itself reports
    # non-convergence / produces a best-effort result instead.
    grid = _open_switch_grid()
    try:
        res = solve_power_flow(grid, on_disconnected="ignore", max_iter=5)
    except ConnectivityError:  # pragma: no cover - the failure being tested
        pytest.fail("on_disconnected='ignore' must skip the connectivity check")
    except Exception:
        return  # a numerical failure is acceptable historical behavior
    assert res.v.shape[-1] == 5
