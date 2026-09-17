pgml.multigrid
===============

Multi-grid batching: solve an ensemble of independent grids in ONE call.

A disjoint union of grids is itself a valid :class:`~pgml.schemas.grid_schema.Grid`: no
branch connects the members, so the assembled admittance is exactly the block-diagonal
``Y = diag(Y_1, …, Y_G)`` — no special solver support is needed, and the sparse
factorization backend handles the union in ~O(Σ nnz) where a dense LU would pay
O((Σ N)³). Every member keeps its own :class:`~pgml.schemas.grid_schema.Source`, so the
pre-solve connectivity check passes per member, ideal-slack rows pin per member, and the
whole pgml pipeline (:func:`~pgml.solver.solve_power_flow`,
:func:`~pgml.solver.solve_harmonic_flow`, batched operating points,
:func:`~pgml.assembly.branch_currents`, :func:`~pgml.solver.prepare_power_flow`) applies
unchanged to the merged grid.

:func:`~pgml.multigrid.merge_grids` builds that union with per-member id remapping and
returns a :class:`~pgml.multigrid.MergedGrid` that translates between member-local and
merged identifiers.

Quick start
-----------

::

    from pgml.multigrid import merge_grids
    from pgml.solver import solve_power_flow

    merged = merge_grids([g1, g2, g3])
    op = merged.operating_point([op1, op2, op3])   # member-local appliance ids
    result = solve_power_flow(merged.grid, operating_point=op)
    v1, v2, v3 = merged.split(result.v)            # per-member row slices

The merged grid SHARES the members' parameter objects (and any tensor leaves they hold),
so gradients computed through a merged solve flow back to the original grids' own leaf
tensors — merging is transparent to the differentiable path.

Semantics to be aware of
-------------------------

- All members must share ``base_frequency_hz`` and be materialised consistently
  (identical ``types`` entries may repeat across members; conflicting definitions under
  one name raise).
- Convergence is evaluated on the union state vector: the fixed-point iteration advances
  all members together and both per-unit criteria (``tol``, ``tol_update_pu``) are maxima
  over the rows of the union, so the verdict does not depend on the member count and one
  hard member keeps iterating an already-settled easy member (cheap — the extra
  iterations are back-substitutions). Per-scenario ``converged_mask`` semantics are
  unchanged.
- The calculation symmetry resolves once for the union: one asymmetric member makes the
  whole batch solve asymmetric (correct for every member, marginally more work for the
  symmetric ones).
- :meth:`~pgml.multigrid.MergedGrid.split` slices any solved quantity whose LAST axis is
  the merged node-phase row axis — power-flow and harmonic-flow voltages, residuals, and
  similarly-shaped derived tensors all work; the returned views are ``narrow``-based, so
  gradients flow through unchanged.

.. automodule:: pgml.multigrid
   :members:
   :show-inheritance:
