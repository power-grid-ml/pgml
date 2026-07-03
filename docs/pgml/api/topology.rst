pgml.topology
=============

Pure grid-topology bookkeeping — slack anchor, branch edges, electrical distance.

Core, dependency-free helpers over the :class:`~pgml.schemas.grid_schema.Grid` contract
(plain python + the standard library — no torch / networkx / plotting). This is what a
lean training process imports for graph structure and node features without pulling in
the plotting stack:

- :func:`~pgml.topology.slack_node_id` — the reference bus (the first in-service
  :class:`~pgml.schemas.grid_schema.Source`).
- :func:`~pgml.topology.branch_edges` — the drawable branch interconnections (closed
  switches kept, open switches dropped — they carry no current and define no path).
- :func:`~pgml.topology.distance_from_slack` — shortest-path line distance from the slack
  along the branch graph (Dijkstra over the standard library ``heapq``); the x-axis of the
  profile plots and a node feature of the graph learning layer (``pgl.data.build_graph``).

The networkx graph view (:func:`pgml.evaluation.topology.grid_graph`) stays in the
evaluation package with the plotting stack; :mod:`pgml.evaluation.topology` re-exports
everything in this module unchanged, so existing importers of
``pgml.evaluation.topology.{slack_node_id,branch_edges,distance_from_slack}`` keep working.

No differentiable quantities pass through this module — topology is off the autograd tape.

Quick start
-----------

::

    from pgml.topology import slack_node_id, branch_edges, distance_from_slack

    slack = slack_node_id(grid)
    dist = distance_from_slack(grid)          # {node_id: km from slack}
    edges = branch_edges(grid)                 # [ProfileEdge(a, b, kind), ...]

.. automodule:: pgml.topology
   :members:
   :show-inheritance:
