pgml.evaluation
===============

Comparison and evaluation plots (reference libraries vs our solve).

The evaluation package provides paper-ready (≥300 DPI / SVG) and interactive
(Plotly HTML) figures for comparing the `pgml` solver to reference
implementations.

.. rubric:: Sub-modules

- **topology** — re-exports the dependency-free :mod:`pgml.topology` (``slack_node_id``,
  ``slack_node_ids``, ``branch_edges``, ``distance_from_slack`` — the x-axis of
  voltage/harmonic profiles) and adds the one helper that genuinely needs networkx:
  ``grid_graph``, the topology-graph layout used by :func:`~pgml.evaluation.plot_grid_graph`.
- **data** — framework-agnostic plot containers (:class:`~pgml.evaluation.VoltageProfile`,
  :class:`~pgml.evaluation.HarmonicProfile`, :class:`~pgml.evaluation.LabeledMatrix`)
  and builder functions that consume solver results, including the shared row-building
  primitive :func:`~pgml.evaluation.harmonic_profile_from_array` (a plain complex
  ``[H, N]`` array in, a :class:`~pgml.evaluation.HarmonicProfile` out — the solver-result
  path here and the ML estimator/dataset path in ``pgl.evaluation`` both delegate to it)
  and :func:`~pgml.evaluation.node_numbering` (the zero-based display numbering used by
  :func:`~pgml.evaluation.plot_grid_graph` and, optionally, by
  :func:`~pgml.evaluation.data.row_labels`).
- **oracles** — lazy adapters for pandapower, OpenDSS, and standalone oracle functions,
  plus the reference-grid builders re-exported from :mod:`pgml.grids`; all emit the same
  containers.
- **ybus_plots** — Y-bus heatmaps side-by-side and a difference heatmap.
- **profiles** — voltage-drop diagram and harmonic magnitude/angle plot.
- **harmonic3d** — interactive 3D harmonic surface plot (Plotly → HTML).
- **graph_plots** — topology graph coloured by a per-node value
  (:func:`~pgml.evaluation.plot_grid_graph`), the shared node-placement helper
  :func:`~pgml.evaluation.graph_layout`, and :func:`~pgml.evaluation.load_node_positions`
  for reading explicit (e.g. geographic) coordinates from a JSON file. See
  `Node positions and numbering`_ below.
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

Node positions and numbering
------------------------------

Grid ``Node.id`` values are unique integers allocated from one counter shared by every
element class in a :class:`~pgml.schemas.grid_schema.Grid` (nodes, branches,
appliances), so they are 1-based and carry no positional meaning. Every place a grid is
DISPLAYED — a graph drawing or a row label — instead numbers nodes zero-based by their
position in ``grid.nodes``, which for a converted grid matches the source tool's own
bus order (e.g. pandapower bus ``0..N-1``). :func:`~pgml.evaluation.node_numbering`
computes this mapping once; :func:`~pgml.evaluation.plot_grid_graph`'s ``with_labels``
and :func:`~pgml.evaluation.data.row_labels`'s ``numbering`` argument both use it so a
plotted node number always means the same thing as the corresponding heatmap row label.

:func:`~pgml.evaluation.graph_layout` is the single source of node placement for every
grid-graph figure — it returns ``{node_id: (x, y)}`` either from an automatic layout
(``"spring"`` or ``"kamada"``) or from explicit ``positions`` (e.g. geographic
coordinates). Pass the SAME returned mapping to :func:`~pgml.evaluation.plot_grid_graph`
and to anything drawn on top of it (sensor markers, annotations) so every layer aligns::

    from pgml.evaluation import graph_layout, plot_grid_graph

    pos = graph_layout(grid, positions=None)          # deterministic spring layout
    fig, ax = plot_grid_graph(grid, node_values=values, positions=pos)
    ax.scatter(*zip(*(pos[nid] for nid in sensor_node_ids)), marker="*", zorder=5)

``positions`` keys resolve, in order, as an exact node NAME, else a zero-based node
NUMBER, else a node ID; every grid node must be covered or ``graph_layout`` raises (a
partial layout would silently misplace the rest of the graph). An already
node-id-keyed, fully-covering dict passes through unchanged (idempotent).
:func:`~pgml.evaluation.load_node_positions` reads such a mapping from a JSON file (keys
as strings) — the CIGRE LV benchmark's node coordinates ship as
``run/configs/cigre_lv_geo.json``, keyed by its zero-based bus numbers::

    from pgml.evaluation import load_node_positions

    pos = load_node_positions("run/configs/cigre_lv_geo.json", grid)
    fig, ax = plot_grid_graph(grid, node_values=values, positions=pos)

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

OpenDSS scenario oracle (independent full-circuit export)
-----------------------------------------------------------

Unlike the live-parity adapters above (which overwrite selected OpenDSS
elements with pgml's own stamps to isolate one model component),
:mod:`pgml.evaluation.oracles.opendss_scenario_oracle` builds a **genuine,
independent** OpenDSS circuit — its own ``Vsource`` / ``Line`` / ``Transformer``
/ ``Load`` / ``Capacitor`` / ``Reactor`` elements — and runs OpenDSS's own
physics end to end, with no pgml formula anywhere in the OpenDSS solve. It
serves two purposes: numeric cross-validation of the whole solver against an
independent reference, and generating an OpenDSS-solved dataset a trained
state estimator never saw pgml produce.

:func:`~pgml.evaluation.oracles.export_grid_to_opendss` builds the circuit
once, returning an :class:`~pgml.evaluation.oracles.ExportedCircuit` handle
that :func:`~pgml.evaluation.oracles.run_opendss_scenarios` edits and re-solves
per scenario (and per step, for a node-coherent batch). Two assumption modes
control how closely the exported circuit matches pgml's own reduced harmonic
model:

- ``mode="matched"`` (default) sets ``NeglectLoadY=Yes`` (pgml's harmonic
  solver has no load Norton shunt at all), ``Rg=Xg=0`` on every line-like
  element (pgml's non-geometry line models carry no Carson earth-return
  correction), a tight snap-solve tolerance, and an effectively unbounded
  ``Vminpu``/``Vmaxpu`` band on every load (pgml's load laws apply at any
  voltage, unlike OpenDSS's default clipping band). This isolates genuine
  numeric agreement between the two harmonic engines — measured (2026-07
  comparison campaign) at roughly 1e-8 to 1e-9 relative voltage error on
  single-phase feeder cases and roughly 1e-6 on the three-phase CIGRE LV
  benchmark, with one documented, bounded exception (the triplen orders under
  ``mode="default"``, below).
- ``mode="default"`` leaves OpenDSS's own defaults (load Norton shunt
  included, imperial-calibrated earth return, the default voltage-clip band)
  — a deliberate, documented divergence characterizing how far a naive
  "just point OpenDSS at the grid" study would drift from pgml's reduced
  model, not a bug.

::

    from pgml.evaluation.oracles import (
        export_grid_to_opendss, run_opendss_scenarios, compare_to_pgml,
        write_opendss_dataset,
    )
    from pgml.scenarios import sample, ScenarioConfig

    sampled = sample(grid, ScenarioConfig(n_samples=64, method="sobol", parameters=[...]))

    # Numeric cross-validation: solve the SAME batch with both engines.
    report = compare_to_pgml(grid, sampled, harmonic_orders=[1, 5, 7, 11])
    # report["per_order"][5]  ->  {"rel_mean": ..., "rel_p95": ..., "rel_max": ..., ...}

    # An independent, provenance-stamped test set for pgl.
    write_opendss_dataset(grid, sampled, "data/opendss_testset", harmonic_orders=[1, 5, 7])
    # meta.json gains engine="opendss", oracle_mode, opendssdirect_version, ...

:func:`~pgml.evaluation.oracles.compare_to_pgml` runs
:func:`~pgml.evaluation.oracles.run_opendss_scenarios` (the ground truth) and
:func:`pgml.scenarios.run_scenarios` on the *identical*
:class:`~pgml.scenarios.SampledScenarios` and reports, per harmonic order, the
absolute and RMS-relative voltage error. Its ``slack`` argument defaults to
``"norton"`` rather than pgml's own library default (``"ideal"``): an OpenDSS
``Vsource`` always behaves as a finite-impedance Thévenin source, so comparing
against pgml's ideal-slack solve on a grid with non-negligible source
impedance would report a spurious mismatch that is really just two different
slack models. :func:`~pgml.evaluation.oracles.write_opendss_dataset` writes
the OpenDSS-solved result through :func:`pgml.scenarios.write_dataset`
unchanged and stamps ``meta.json`` with ``engine="opendss"`` plus the
OpenDSS/``opendssdirect`` version, so a dataset generated this way is never
mistaken for a pgml-generated one and reads back through
:func:`pgml.scenarios.read_dataset` / any ``pgl`` data source unmodified.

The exporter covers every branch and appliance type in the schema (a
``Generator``/``Storage`` exports as a negative-kW ``Load`` — a genuine
OpenDSS ``Generator`` element stamps its own admittance into the harmonics
solve regardless of ``NeglectLoadY``, so it cannot represent a pure current
injection) and raises :class:`~pgml.errors.ConversionError` with a specific
reason for grid features it does not (yet, or by design) represent —
conductor-geometry lines (use the geometry-parity oracle above instead),
zigzag transformer windings, a phase-coupled or unbalanced ``Source``, an
impedance-grounded transformer neutral, and a nonzero vector-group phase
shift on a single-phase transformer (OpenDSS has no delta/``LeadLag``
mechanism at ``phases=1``). See the module docstring for the complete
coverage and refusal list.

.. automodule:: pgml.evaluation
   :members:
   :show-inheritance:

Oracle subpackage reference
------------------------------

:mod:`pgml.evaluation.oracles` is documented separately below (a package-level
``automodule``, the same pattern used for ``pgl``'s multi-submodule re-exporting
packages — see e.g. :doc:`/pgl/api/data`): unlike :mod:`pgml.evaluation` itself, it
requires the ``oracles`` extra (``pandapower``/``opendssdirect``) and is never imported
by the plotting side.

.. automodule:: pgml.evaluation.oracles
   :members:
   :show-inheritance:
