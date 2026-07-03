pgml.evaluation
===============

Comparison and evaluation plots (reference libraries vs our solve).

The evaluation package provides paper-ready (≥300 DPI / SVG) and interactive
(Plotly HTML) figures for comparing the `pgml` solver to reference
implementations.

.. rubric:: Sub-modules

- **topology** — re-exports the dependency-free :mod:`pgml.topology` (``slack_node_id``,
  ``branch_edges``, ``distance_from_slack`` — the x-axis of voltage/harmonic profiles) and
  adds the one helper that genuinely needs networkx: ``grid_graph``, the topology-graph
  layout used by :func:`~pgml.evaluation.plot_grid_graph`.
- **data** — framework-agnostic plot containers (:class:`~pgml.evaluation.VoltageProfile`,
  :class:`~pgml.evaluation.HarmonicProfile`, :class:`~pgml.evaluation.LabeledMatrix`)
  and builder functions that consume solver results, including the shared row-building
  primitive :func:`~pgml.evaluation.harmonic_profile_from_array` (a plain complex
  ``[H, N]`` array in, a :class:`~pgml.evaluation.HarmonicProfile` out — the solver-result
  path here and the ML estimator/dataset path in ``pgl.evaluation`` both delegate to it).
- **oracles** — lazy adapters for pandapower, OpenDSS, and standalone oracle functions,
  plus the reference-grid builders re-exported from :mod:`pgml.grids`; all emit the same
  containers.
- **ybus_plots** — Y-bus heatmaps side-by-side and a difference heatmap.
- **profiles** — voltage-drop diagram and harmonic magnitude/angle plot.
- **harmonic3d** — interactive 3D harmonic surface plot (Plotly → HTML).
- **graph_plots** — topology graph coloured by a per-node value.
- **style** — shared styling: the high-quality :func:`~pgml.evaluation.save_figure` /
  :func:`~pgml.evaluation.save_html` save helpers and
  :func:`~pgml.evaluation.style.positive_log_norm`, the shared positive-data ``LogNorm``
  guard for log-scaled heatmaps.

.. note::

   Reference library adapters (``pandapower``, ``opendssdirect``) are imported
   lazily inside :mod:`pgml.evaluation.oracles` so the rest of the package
   is available even when those libraries are absent.

3D harmonic profile — dash_map
-------------------------------

:func:`~pgml.evaluation.plot_harmonic_profile_3d` accepts an optional ``dash_map``
argument that lets line DASH encode a third dimension (such as phase) while COLOR
continues to encode the harmonic order::

    from pgml.evaluation import plot_harmonic_profile_3d

    fig = plot_harmonic_profile_3d(
        profiles,
        grid=grid,
        dash_map={"L1": "solid", "L2": "dash", "L3": "dot"},
    )

``dash_map`` is a ``{label: plotly_dash}`` mapping where the label matches
``profile.label`` and the value is any Plotly dash string (``"solid"``,
``"dash"``, ``"dot"``, ``"dashdot"``, …).  When ``dash_map`` is given:

- The binary ``reference_labels`` dashing is ignored.
- All markers use ``"circle"`` regardless of whether the label is in
  ``reference_labels``.

When ``dash_map`` is ``None`` (default), the original binary behaviour applies:
labels in ``reference_labels`` are dashed, all others solid.

Reference builders and oracle functions
----------------------------------------

:mod:`pgml.evaluation.oracles` provides harmonic oracle functions for validation and
regression testing, plus the reference-grid builders re-exported from :mod:`pgml.grids`
(see :doc:`grids` for the canonical documentation of
:func:`~pgml.grids.ieee33_geometry_grid`, :func:`~pgml.grids.cigre_lv_geometry_grid`,
:func:`~pgml.grids.cigre_lv_full_grid`, :func:`~pgml.grids.add_pv_systems`, and
:func:`~pgml.grids.se_benchmark_scenario_config`) — ``from pgml.evaluation.oracles import
cigre_lv_full_grid`` and ``from pgml.grids import cigre_lv_full_grid`` import the identical
function.

Harmonic oracle functions
~~~~~~~~~~~~~~~~~~~~~~~~~

Two functions return complex ``[H, N]`` node voltages aligned to
:func:`pgml.assembly.node_phase_index`, for use as regression and ground-truth
oracles.

**Pure-numpy regression oracle** (:func:`~pgml.evaluation.oracles.numpy_harmonic_voltages`)

Reimplements pgml's EXACT Y-bus formulas (R const / X∝h) in pure numpy without
any live OpenDSS circuit.  Gives machine-precision parity (~1e-13 V absolute) vs
``solve_harmonic_flow`` on grids without conductor geometry.  Useful as a fast
regression oracle and to validate CIGRE LV (both ``SINGLE_PHASE_EQUIV`` and
``THREE_PHASE``) with plain R/X lines::

    from pgml.evaluation.oracles import numpy_harmonic_voltages

    v_ref = numpy_harmonic_voltages(
        grid, harmonic_injection, orders=[1, 5, 7, 11],
        v1=hres.pf.v.detach().cpu().numpy(),   # share converged fundamental
    )
    # v_ref  complex [H, N], H = len(orders)

**Live OpenDSS harmonic oracle** (:func:`~pgml.evaluation.oracles.opendss_harmonic_voltages`)

Builds and runs a live OpenDSS circuit using pgml's Carson/Deri line model,
returning complex ``[H, N]`` voltages.  Gives near-machine-precision parity
(~1e-11 V for single-phase geometry grids; ~1e-8 V for three-phase
sequence-aware grids).  Requires either:

- All lines carry ``conductor_geometry`` (set by ``synthesize_grid_geometry``);
  single-phase path using OpenDSS WireData/LineGeometry.
- All lines tagged ``harmonic_line_model=sequence_aware`` (set by
  ``apply_default_harmonic_model``); three-phase path using R1/X1/R0/X0 lines.

Plain R/X grids without either tag raise ``ValueError`` — use
``numpy_harmonic_voltages`` instead::

    from pgml.evaluation.oracles import opendss_harmonic_voltages

    v_dss = opendss_harmonic_voltages(
        grid, harmonic_injection, orders=[1, 5, 7],
        v1=hres.pf.v.detach().cpu().numpy(),
    )
    # v_dss  complex [H, N]

Both functions share the same signature:
``(grid, harmonic_injection, orders, *, slack="norton", v1=None, operating_point=None, node_sources=None) -> np.ndarray``.

The optional ``node_sources`` argument accepts the same list of
:class:`~pgml.solver.NodeHarmonicSource` objects passed to
:func:`~pgml.solver.solve_harmonic_flow` and stamps them into the oracle using
identical physics, so the oracle remains a machine-precision parity check even
when per-node harmonic disturbance sources are active.

.. automodule:: pgml.evaluation
   :members:
   :show-inheritance:
