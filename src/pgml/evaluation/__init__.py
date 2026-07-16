"""pgml.evaluation — comparison & evaluation plots (reference libraries vs our solve).

Paper-ready (>=300 DPI / svg) and interactive (plotly HTML) figures comparing our
differentiable solver to the reference implementations. See ``evaluation/CONTEXT.md``.

Layout
------
- ``topology``    distance-from-slack along the branch graph (x-axis of profiles).
- ``data``        framework-agnostic plot containers + builders from solver results.
- ``oracles``  lazy pandapower / OpenDSS adapters + an independent numpy harmonic
                  oracle, all emitting the same containers.
- ``ybus_plots``  Y-bus heatmaps side by side + a difference heatmap.
- ``profiles``    reusable voltage-drop diagram + the harmonic magnitude/angle plot.
- ``harmonic3d``  interactive 3D harmonic plot (plotly -> HTML).
- ``graph_plots`` topology graph colored by a per-node value.
- ``style``       shared styling + the ``save_figure`` / ``save_html`` save helpers.

Importing this package pulls in matplotlib/plotly/networkx only; the reference
libraries are imported lazily inside :mod:`pgml.evaluation.oracles`.
"""

from __future__ import annotations

from .data import (
    HarmonicProfile,
    LabeledMatrix,
    VoltageProfile,
    harmonic_profile,
    harmonic_profile_from_array,
    harmonic_profiles,
    labeled_matrix,
    node_numbering,
    voltage_profile,
)
from .graph_plots import graph_layout, load_node_positions, plot_grid_graph
from .harmonic3d import plot_harmonic_profile_3d
from .interactive import plot_harmonic_profile_interactive
from .profiles import (
    plot_harmonic_model_comparison,
    plot_harmonic_profile,
    plot_profile_error,
    plot_voltage_profile,
)
from .style import save_figure, save_html
from .topology import distance_from_slack, grid_graph, slack_node_id
from .ybus_plots import plot_ybus_difference, plot_ybus_heatmaps

__all__ = [
    # data
    "VoltageProfile",
    "HarmonicProfile",
    "LabeledMatrix",
    "voltage_profile",
    "harmonic_profile",
    "harmonic_profile_from_array",
    "harmonic_profiles",
    "labeled_matrix",
    "node_numbering",
    # topology
    "distance_from_slack",
    "grid_graph",
    "slack_node_id",
    "graph_layout",
    "load_node_positions",
    # plots
    "plot_ybus_heatmaps",
    "plot_ybus_difference",
    "plot_voltage_profile",
    "plot_harmonic_profile",
    "plot_harmonic_model_comparison",
    "plot_profile_error",
    "plot_harmonic_profile_3d",
    "plot_harmonic_profile_interactive",
    "plot_grid_graph",
    # io
    "save_figure",
    "save_html",
]
