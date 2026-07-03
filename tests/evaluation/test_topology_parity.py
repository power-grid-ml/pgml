"""The core Dijkstra (`pgml.topology`) matches the networkx path on real feeders.

`pgml.topology.distance_from_slack` is the stdlib implementation the training
path uses (no networkx); `pgml.evaluation.topology.grid_graph` is the nx view
the plots use. The two must agree exactly — and both must survive PARALLEL
branches (a ring tie / doubled cable sharing both endpoints), where the shorter
branch defines the distance.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from pgml.evaluation.topology import grid_graph
from pgml.schemas.grid_schema import Grid, Line, Node, Phase, Source
from pgml.topology import branch_edges, distance_from_slack, slack_node_id

W = 2.0 * math.pi * 50.0


def _line(bid, a, b, km):
    return Line(
        id=bid,
        from_node=a,
        to_node=b,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=km * 1000.0,
        series_resistance_ohm_per_m=[[0.5e-3]],
        series_inductance_h_per_m=[[0.5e-3 / W]],
        shunt_capacitance_f_per_m=[[0.0]],
    )


def _ring_grid() -> Grid:
    """4 nodes; a ring 1-2-3-4-1 plus a LONGER parallel line duplicating 1-2."""
    nodes = [Node(id=i, u_rated_v=400.0, phases=(Phase.A,)) for i in (1, 2, 3, 4)]
    branches = [
        _line(1, 1, 2, 1.0),
        _line(2, 2, 3, 1.0),
        _line(3, 3, 4, 1.0),
        _line(4, 4, 1, 1.0),
        _line(5, 1, 2, 5.0),  # parallel to branch 1, longer — must NOT win
    ]
    src = Source(
        id=10,
        node=1,
        phases=(Phase.A,),
        u_ref_v=(230.0,),
        u_angle_deg=(0.0,),
        resistance_ohm=[[0.1]],
        inductance_h=[[1e-3]],
    )
    return Grid(
        base_frequency_hz=50.0, nodes=nodes, branches=branches, appliances=[src]
    )


def _nx_distances(grid: Grid) -> dict[int, float]:
    g = grid_graph(grid)
    lengths = nx.single_source_dijkstra_path_length(g, slack_node_id(grid), weight="km")
    return {int(n.id): lengths.get(int(n.id), float("inf")) for n in grid.nodes}


def test_core_dijkstra_matches_networkx_on_ring():
    grid = _ring_grid()
    core = distance_from_slack(grid)
    ref = _nx_distances(grid)
    assert core == pytest.approx(ref)


def test_parallel_branch_takes_shorter_length():
    grid = _ring_grid()
    dist = distance_from_slack(grid)
    assert dist[2] == pytest.approx(1.0)  # via the 1 km branch, not the 5 km one
    assert dist[3] == pytest.approx(2.0)
    assert dist[4] == pytest.approx(1.0)  # around the ring the short way


def test_core_dijkstra_matches_networkx_on_cigre_lv():
    pytest.importorskip("pandapower")
    from pgml.grids import cigre_lv_full_grid

    grid, _ = cigre_lv_full_grid()
    core = distance_from_slack(grid)
    ref = _nx_distances(grid)
    assert core == pytest.approx(ref)


def test_branch_edges_skip_only_open_switches():
    grid = _ring_grid()
    edges = branch_edges(grid)
    # all five branches are drawable lines (incl. the parallel one)
    assert len(edges) == 5
    assert all(e.kind == "line" for e in edges)
