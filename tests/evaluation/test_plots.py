"""Evaluation package: topology, data builders, plot smoke tests, oracle consistency.

No reference libraries needed here (tiny fixture grids); the IEEE-33 reference wiring
is exercised by ``test_ieee33_figures.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from pgml.assembly import assemble_ybus, node_phase_index
from pgml.evaluation import (
    distance_from_slack,
    harmonic_profile,
    harmonic_profiles,
    labeled_matrix,
    plot_grid_graph,
    plot_harmonic_profile,
    plot_harmonic_profile_3d,
    plot_profile_error,
    plot_voltage_profile,
    plot_ybus_difference,
    plot_ybus_heatmaps,
    save_figure,
    voltage_profile,
)
from pgml.evaluation import references as ref
from pgml.evaluation.topology import branch_edges
from pgml.schemas.grid_schema import (
    Grid,
    HarmonicComponent,
    Line,
    Load,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
    Switch,
)
from pgml.solver import solve_harmonic_flow, solve_power_flow

from tests.fixtures.tiny_grids import single_phase_chain, three_phase_two_bus

CDT = torch.complex128
SPEC = [(1, 1.0, 0.0), (5, 0.2, 0.0), (7, 0.14, 0.0)]


def _harmonic_grid():
    grid = single_phase_chain()
    comps = [
        HarmonicComponent(order=o, magnitude_pu=m, phase_deg=a) for o, m, a in SPEC
    ]
    for app in grid.appliances:
        if isinstance(app, Load):
            app.spectrum = StaticSpectrum(spectrum=SpectrumPoint(components=comps))
    return grid


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------
def test_distance_from_slack_chain():
    grid = single_phase_chain()  # source@1, lines 1-2 (100 m), 2-3 (50 m)
    dist = distance_from_slack(grid)
    assert dist[1] == 0.0
    assert dist[2] == 0.1  # 100 m
    assert abs(dist[3] - 0.15) < 1e-12  # 100 + 50 m


# ---------------------------------------------------------------------------
# data builders
# ---------------------------------------------------------------------------
def test_voltage_profile_builder_sorted_and_pu():
    grid = single_phase_chain()
    pf = solve_power_flow(grid, slack="ideal", dtype=CDT)
    prof = voltage_profile(pf, grid, label="pgml")
    assert prof.distances_km.tolist() == sorted(prof.distances_km.tolist())
    assert prof.v_pu.shape == (3,)
    # Slack bus ~ 1 pu; voltage drops downstream.
    assert abs(prof.v_pu[0] - 1.0) < 0.05
    assert prof.v_pu[-1] < prof.v_pu[0]


def test_labeled_matrix_from_ybus_shape():
    grid = three_phase_two_bus()
    index = node_phase_index(grid)
    yb = assemble_ybus(grid, [50.0], dtype=CDT)
    lm = labeled_matrix(yb.Y, index, label="pgml")
    assert lm.matrix.shape == (index.size, index.size)
    assert len(lm.row_labels) == index.size


# ---------------------------------------------------------------------------
# numpy harmonic oracle consistency (validates references + builders together)
# ---------------------------------------------------------------------------
def test_numpy_harmonic_oracle_matches_solver():
    grid = _harmonic_grid()
    orders = [1, 5, 7]
    hres = solve_harmonic_flow(grid, orders, slack="norton", dtype=CDT)
    index = hres.index
    ours = harmonic_profiles(hres, grid, [5, 7], label="pgml", unit="V")
    oracle = ref.numpy_harmonic_profiles(
        grid, hres.pf.v, index, [5, 7], label="oracle", unit="V"
    )
    for po, pr in zip(ours, oracle):
        # same node ordering, near-identical magnitudes/angles
        np.testing.assert_allclose(po.magnitude, pr.magnitude, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(po.angle_deg, pr.angle_deg, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------
# plot smoke tests (figures build + save high-res)
# ---------------------------------------------------------------------------
def test_voltage_profile_plot_and_save(tmp_path):
    grid = single_phase_chain()
    pf = solve_power_flow(grid, slack="ideal", dtype=CDT)
    p1 = voltage_profile(pf, grid, label="pgml")
    p2 = voltage_profile(
        pf, grid, label="copy"
    )  # second line to exercise overlay/alpha
    fig, ax = plot_voltage_profile([p1, p2], grid=grid, alpha=0.6)
    out = save_figure(fig, tmp_path / "vp.png")
    assert (tmp_path / "vp.png").exists()
    # svg path works too (vector, paper-ready)
    save_figure(fig, tmp_path / "vp.svg")
    assert (tmp_path / "vp.svg").exists()
    plt.close("all")
    assert out.endswith("vp.png")
    # error bar chart
    fig2, _ = plot_profile_error(p1, p2)
    save_figure(fig2, tmp_path / "err.png")
    assert (tmp_path / "err.png").exists()
    plt.close("all")


def test_ybus_heatmaps_and_difference(tmp_path):
    grid = three_phase_two_bus()
    index = node_phase_index(grid)
    yb = assemble_ybus(grid, [50.0], dtype=CDT)
    a = labeled_matrix(yb.Y, index, label="A")
    b = labeled_matrix(yb.Y, index, label="B")
    fig = plot_ybus_heatmaps([a, b], part="abs")
    save_figure(fig, tmp_path / "yb.png")
    assert (tmp_path / "yb.png").exists()
    fig2 = plot_ybus_heatmaps([a, b], part="real", share_scale=False)
    save_figure(fig2, tmp_path / "yb_real.png")
    fig3 = plot_ybus_difference(a, b)  # identical -> zero difference
    save_figure(fig3, tmp_path / "diff.png")
    assert (tmp_path / "diff.png").exists()
    plt.close("all")


def test_harmonic_profile_2d_and_3d(tmp_path):
    grid = _harmonic_grid()
    orders = [1, 5, 7]
    hres = solve_harmonic_flow(grid, orders, slack="norton", dtype=CDT)
    p5 = harmonic_profile(hres, grid, 5, label="pgml")
    oracle5 = ref.numpy_harmonic_profiles(
        grid, hres.pf.v, hres.index, [5], label="oracle"
    )[0]
    fig, ax = plot_harmonic_profile([p5, oracle5], grid=grid)
    save_figure(fig, tmp_path / "h5.png")
    assert (tmp_path / "h5.png").exists()
    plt.close("all")

    profs = harmonic_profiles(hres, grid, [5, 7], label="pgml")
    figp = plot_harmonic_profile_3d(
        profs, grid=grid, out_html=str(tmp_path / "h3d.html")
    )
    assert (tmp_path / "h3d.html").exists()
    assert figp is not None


def test_grid_graph_plot(tmp_path):
    grid = single_phase_chain()
    pf = solve_power_flow(grid, slack="ideal", dtype=CDT)
    prof = voltage_profile(pf, grid, label="pgml")
    vmap = dict(zip([int(i) for i in prof.node_ids], prof.v_pu))
    fig, ax = plot_grid_graph(grid, node_values=vmap, value_label="V [pu]")
    save_figure(fig, tmp_path / "graph.png")
    assert (tmp_path / "graph.png").exists()
    plt.close("all")


def test_differentiability_unaffected_by_plotting():
    """Building plot data detaches; it must NOT disturb a live autograd graph."""
    grid = single_phase_chain()
    r = torch.tensor([[1.0e-3]], dtype=torch.float64, requires_grad=True)
    grid.branches[0].series_resistance_ohm_per_m = r
    pf = solve_power_flow(grid, slack="ideal", dtype=CDT)
    _ = voltage_profile(pf, grid, label="pgml")  # detaches internally
    pf.v.abs().sum().backward()  # graph still intact
    assert r.grad is not None and torch.isfinite(r.grad).all()


# ---------------------------------------------------------------------------
# topology-driven connecting lines (branches, not data order) + switch state
# ---------------------------------------------------------------------------
def _switch_grid(switch_closed: bool) -> Grid:
    """Source@1 --line(100 m)--> 2 --switch--> 3, with a load at node 3."""
    ph = (Phase.A,)
    return Grid(
        base_frequency_hz=50.0,
        nodes=[Node(id=i, u_rated_v=230.0, phases=ph) for i in (1, 2, 3)],
        branches=[
            Line(
                id=20,
                from_node=1,
                to_node=2,
                from_phases=ph,
                to_phases=ph,
                length_m=100.0,
                series_resistance_ohm_per_m=[[1e-3]],
                series_inductance_h_per_m=[[1e-6]],
                shunt_capacitance_f_per_m=[[1e-9]],
            ),
            Switch(
                id=21,
                from_node=2,
                to_node=3,
                from_phases=ph,
                to_phases=ph,
                closed=switch_closed,
                resistance_ohm=1e-6,
                inductance_h=1e-9,
            ),
        ],
        appliances=[
            Source(
                id=10,
                node=1,
                phases=ph,
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[1e-3]],
            ),
            Load(id=30, node=3, phases=ph, p_nom_w=1000.0, q_nom_var=200.0),
        ],
    )


def test_branch_edges_closed_switch_is_dashed_kind():
    edges = branch_edges(_switch_grid(switch_closed=True))
    kinds = {(e.a, e.b): e.kind for e in edges}
    assert kinds == {(1, 2): "line", (2, 3): "switch"}


def test_branch_edges_omits_open_switch():
    edges = branch_edges(_switch_grid(switch_closed=False))
    assert {(e.a, e.b) for e in edges} == {(1, 2)}  # open switch (2,3) dropped
    # ...unless explicitly requested
    edges_all = branch_edges(
        _switch_grid(switch_closed=False), include_open_switches=True
    )
    assert {(e.a, e.b) for e in edges_all} == {(1, 2), (2, 3)}


def test_distance_does_not_traverse_open_switch():
    dist_closed = distance_from_slack(_switch_grid(switch_closed=True))
    assert dist_closed[3] == 0.1  # reachable through the closed switch (0 length)
    dist_open = distance_from_slack(_switch_grid(switch_closed=False))
    assert dist_open[3] == float("inf")  # unreachable across the open switch


def test_voltage_profile_with_switch_grid_plots(tmp_path):
    grid = _switch_grid(switch_closed=True)
    pf = solve_power_flow(grid, slack="ideal", dtype=CDT)
    prof = voltage_profile(pf, grid, label="pgml")
    fig, ax = plot_voltage_profile([prof], grid=grid)
    save_figure(fig, tmp_path / "sw.png")
    assert (tmp_path / "sw.png").exists()
    plt.close("all")
