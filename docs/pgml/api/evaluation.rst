pgml.evaluation
===============

Comparison and evaluation plots (reference libraries vs our solve).

The evaluation package provides paper-ready (≥300 DPI / SVG) and interactive
(Plotly HTML) figures for comparing the `pgml` solver to reference
implementations.

.. rubric:: Sub-modules

- **topology** — distance from slack along the branch graph (the x-axis of
  voltage/harmonic profiles).
- **data** — framework-agnostic plot containers (:class:`~pgml.evaluation.VoltageProfile`,
  :class:`~pgml.evaluation.HarmonicProfile`, :class:`~pgml.evaluation.LabeledMatrix`)
  and builder functions that consume solver results.
- **oracles** — lazy adapters for pandapower, OpenDSS, and standalone oracle
  functions; all emit the same containers.
- **ybus_plots** — Y-bus heatmaps side-by-side and a difference heatmap.
- **profiles** — voltage-drop diagram and harmonic magnitude/angle plot.
- **harmonic3d** — interactive 3D harmonic surface plot (Plotly → HTML).
- **graph_plots** — topology graph coloured by a per-node value.
- **style** — shared styling and the high-quality :func:`~pgml.evaluation.save_figure`
  helper.

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

:mod:`pgml.evaluation.oracles` provides reference-grid builders and harmonic
oracle functions for validation and regression testing.

Grid builders
~~~~~~~~~~~~~

- :func:`~pgml.evaluation.oracles.ieee33_geometry_grid` — IEEE 33-bus feeder
  with synthesized Carson conductor geometry.
- :func:`~pgml.evaluation.oracles.cigre_lv_geometry_grid` — residential CIGRE
  LV feeder (one feeder, Carson geometry).
- :func:`~pgml.evaluation.oracles.cigre_lv_full_grid` — the **full** CIGRE LV
  benchmark (all three feeders, three 20/0.4 kV transformers, MV ext-grid source).
  Unlike ``cigre_lv_geometry_grid``, this uses standard R/X lines and converts the
  complete pandapower network::

      from pgml.evaluation.oracles import cigre_lv_full_grid
      from pgml.convert.pandapower import PhaseMode

      grid, id_map = cigre_lv_full_grid()
      # or with explicit phase mode and source impedance:
      grid, id_map = cigre_lv_full_grid(
          phase_mode=PhaseMode.THREE_PHASE,
          source_impedance_ohm=5.0,   # None -> config default source.series_impedance_ohm
      )

  The ``source_impedance_ohm`` parameter applies a finite upstream-grid impedance to
  the converted MV source so harmonics are not fully absorbed at the slack bus.
  Passing ``0`` keeps the converted near-ideal source.  The split between R and X is
  controlled by the config key ``source.rx_ratio`` (default 10).

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
