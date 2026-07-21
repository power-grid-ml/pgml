pgml.topology
=============

Pure grid-topology bookkeeping — slack anchor, branch edges, electrical distance.

Core, dependency-free helpers over the :class:`~pgml.schemas.grid_schema.Grid` contract
(plain python + the standard library — no torch / networkx / plotting). This is what a
lean training process imports for graph structure and node features without pulling in
the plotting stack:

- :func:`~pgml.topology.slack_node_ids` — the reference bus(es): every in-service
  :class:`~pgml.schemas.grid_schema.Source` node.  :func:`~pgml.topology.slack_node_id`
  is the single-slack convenience (the first entry).
- :func:`~pgml.topology.branch_edges` — the drawable branch interconnections (closed
  switches kept, open switches dropped — they carry no current and define no path).
- :func:`~pgml.topology.distance_from_slack` — shortest-path line distance to the
  NEAREST slack along the branch graph (multi-source Dijkstra over the standard
  library ``heapq`` — identical to the single-slack result on a one-source grid); the
  x-axis of the profile plots and a node feature of the graph learning layer
  (``pgl.data.build_graph``).
- :func:`~pgml.topology.connectivity_report` — which ``(node, phase)`` rows have a
  galvanic path to an in-service :class:`~pgml.schemas.grid_schema.Source`; the
  pre-solve structural check behind :func:`pgml.solver.check_connectivity`.
- :func:`~pgml.topology.energized_subgrid` — the energized part of a grid plus the
  ids of the dropped (dead) nodes; the reduction behind
  ``solve_power_flow(..., on_disconnected="zero")``.

The networkx graph view (:func:`pgml.evaluation.topology.grid_graph`) stays in the
evaluation package with the plotting stack; :mod:`pgml.evaluation.topology` re-exports
everything in this module unchanged, so existing importers of
``pgml.evaluation.topology.{slack_node_id,slack_node_ids,branch_edges,distance_from_slack}``
keep working.

No differentiable quantities pass through this module — topology is off the autograd tape.

Quick start
-----------

::

    from pgml.topology import slack_node_id, slack_node_ids, branch_edges, distance_from_slack

    slacks = slack_node_ids(grid)              # every in-service Source node
    slack = slack_node_id(grid)                # == slacks[0]
    dist = distance_from_slack(grid)           # {node_id: km to the NEAREST slack}
    edges = branch_edges(grid)                 # [ProfileEdge(a, b, kind), ...]

Connectivity checking
----------------------

Before a solve, every ``(node, phase)`` row must have a galvanic path to an
in-service :class:`~pgml.schemas.grid_schema.Source` — otherwise the nodal
system is singular there. :func:`~pgml.topology.connectivity_report` walks the
CONDUCTING branch graph (in-service lines/transformers, closed switches) and
returns a :class:`~pgml.topology.ConnectivityReport`:

- ``connected`` / ``has_source`` — the top-level verdict.
- ``unenergized`` — ``(node_id, phases)`` for every under-energized node.
- ``islands`` — the disconnected components, largest first.
- ``reconnectable`` — :class:`~pgml.topology.ReconnectHint` entries (branch id,
  kind, terminal nodes, and whether it is open or out of service) naming a
  concrete fix for each island.

``report.describe()`` renders this as the human-readable message that
:class:`~pgml.errors.ConnectivityError` carries.

:func:`~pgml.topology.energized_subgrid` uses the same report to strip every
fully unenergized node (and the branches/appliances touching it), returning a
grid that solves normally plus the dropped node ids — the reduction behind
``solve_power_flow(grid, on_disconnected="zero")`` and
``solve_harmonic_flow(grid, ..., on_disconnected="zero")``, which solve the
energized sub-grid and report 0 V on the disconnected rows while keeping the
full grid's row layout. A node that is only PARTIALLY energized (some but not
all of its phases) cannot be split this way and raises
:class:`~pgml.errors.ConnectivityError` instead::

    from pgml.topology import connectivity_report, energized_subgrid

    report = connectivity_report(grid)
    if not report.connected:
        print(report.describe())

    sub_grid, dropped_node_ids = energized_subgrid(grid)

:func:`pgml.solver.check_connectivity` wraps
:func:`~pgml.topology.connectivity_report` as the pre-solve gate that
:func:`~pgml.solver.solve_power_flow` and
:func:`~pgml.solver.solve_harmonic_flow` run by default; see the "Connectivity
checking" section of :doc:`solver` for the ``on_disconnected`` modes.

.. automodule:: pgml.topology
   :members:
   :show-inheritance:
