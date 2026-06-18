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
- **references** — lazy adapters for pandapower, OpenDSS, and a standalone
  numpy harmonic oracle; all emit the same containers.
- **ybus_plots** — Y-bus heatmaps side-by-side and a difference heatmap.
- **profiles** — voltage-drop diagram and harmonic magnitude/angle plot.
- **harmonic3d** — interactive 3D harmonic surface plot (Plotly → HTML).
- **graph_plots** — topology graph coloured by a per-node value.
- **style** — shared styling and the high-quality :func:`~pgml.evaluation.save_figure`
  helper.

.. note::

   Reference library adapters (``pandapower``, ``opendssdirect``) are imported
   lazily inside :mod:`pgml.evaluation.references` so the rest of the package
   is available even when those libraries are absent.

.. automodule:: pgml.evaluation
   :members:
   :show-inheritance:
