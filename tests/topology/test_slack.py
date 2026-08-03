"""Slack anchoring: plural source discovery + nearest-slack distances."""

from __future__ import annotations

from pgml.schemas.grid_schema import Phase, Source
from pgml.topology import distance_from_slack, slack_node_id, slack_node_ids
from tests.fixtures.tiny_grids import single_phase_chain


def _second_source(node: int) -> Source:
    return Source(
        id=11,
        node=node,
        phases=(Phase.A,),
        u_ref_v=(230.0,),
        u_angle_deg=(0.0,),
        resistance_ohm=[[0.1]],
        inductance_h=[[1.0e-3]],
    )


def test_slack_node_ids_single_source():
    grid = single_phase_chain()
    assert slack_node_ids(grid) == [1]
    assert slack_node_id(grid) == 1
    # single-source grid: the default (all-slacks) distance equals the explicit one
    assert distance_from_slack(grid) == distance_from_slack(grid, 1)


def test_slack_node_ids_multiple_sources():
    grid = single_phase_chain()
    grid.appliances = [*grid.appliances, _second_source(3)]
    assert slack_node_ids(grid) == [1, 3]
    assert slack_node_id(grid) == 1  # the primary stays the first in-service source

    # distance is to the NEAREST slack: node 2 is 0.1 km from node 1 but only
    # 0.05 km from node 3 (line2), and both source nodes sit at 0.
    dist = distance_from_slack(grid)
    assert dist[1] == 0.0 and dist[3] == 0.0
    assert abs(dist[2] - 0.05) < 1e-12

    # an out-of-service source is not a slack
    grid.appliances[-1].in_service = False
    assert slack_node_ids(grid) == [1]


def test_distance_accepts_explicit_source_set():
    grid = single_phase_chain()
    dist = distance_from_slack(grid, [3])
    assert dist[3] == 0.0
    assert abs(dist[2] - 0.05) < 1e-12
    assert abs(dist[1] - 0.15) < 1e-12
