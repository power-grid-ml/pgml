"""IEEE 33-bus integration: reference adapters produce correct, aligned plot data.

Validates that:
- ``references.pandapower_ybus`` (pu->SI, network) matches our ``assemble_network_ybus``
  (the two most-similar Y-bus versions) to floating-point — the difference heatmap is
  near zero.
- ``references.pandapower_voltage_profile`` aligns to our profile and our nonlinear PF
  agrees with pandapower in pu (this is what the voltage-profile figure shows).
- the full figure pipeline runs and writes paper-ready files on a real feeder.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import torch

from pgml.assembly import assemble_network_ybus, assemble_ybus, node_phase_index
from pgml.convert.pandapower import to_grid
from pgml.evaluation import (
    labeled_matrix,
    plot_voltage_profile,
    plot_ybus_difference,
    plot_ybus_heatmaps,
    save_figure,
    voltage_profile,
)
from pgml.evaluation import references as ref
from pgml.solver import solve_power_flow

CDT = torch.complex128


def _ieee33():
    net = pn.case33bw()
    pp.runpp(net, numba=False)
    grid, id_map = to_grid(net)
    return net, grid, id_map


def test_pandapower_network_ybus_matches_ours():
    net, grid, id_map = _ieee33()
    f0 = grid.base_frequency_hz
    index = node_phase_index(grid)
    ours = labeled_matrix(
        assemble_network_ybus(grid, [f0], dtype=CDT).Y, index, label="pgml net"
    )
    ppm = ref.pandapower_ybus(net, grid, id_map, index, label="pp net")
    diff = ours.matrix - ppm.matrix
    rel = np.linalg.norm(diff) / np.linalg.norm(ppm.matrix)
    assert rel < 1e-6, f"network Y mismatch (rel Frobenius {rel:.2e})"


def test_voltage_profiles_align_and_agree():
    net, grid, id_map = _ieee33()
    pf = solve_power_flow(grid, slack="ideal", dtype=CDT)
    ours = voltage_profile(pf, grid, label="pgml")
    ppv = ref.pandapower_voltage_profile(net, grid, id_map, label="pandapower")
    # Same nodes in the same (distance-sorted) order.
    assert ours.node_ids.tolist() == ppv.node_ids.tolist()
    # Nonlinear PF agrees with pandapower const-power solution in pu.
    np.testing.assert_allclose(ours.v_pu, ppv.v_pu, atol=2e-4)


def test_full_figure_pipeline_writes_files(tmp_path):
    net, grid, id_map = _ieee33()
    f0 = grid.base_frequency_hz
    index = node_phase_index(grid)
    ours_full = labeled_matrix(
        assemble_ybus(grid, [f0], dtype=CDT).Y, index, label="pgml full"
    )
    ours_net = labeled_matrix(
        assemble_network_ybus(grid, [f0], dtype=CDT).Y, index, label="pgml net"
    )
    ppm = ref.pandapower_ybus(net, grid, id_map, index, label="pandapower")

    fig = plot_ybus_heatmaps([ours_full, ppm], part="abs")
    save_figure(fig, tmp_path / "yb.svg")
    fig = plot_ybus_difference(ours_net, ppm)
    save_figure(fig, tmp_path / "diff.png")

    pf = solve_power_flow(grid, slack="ideal", dtype=CDT)
    ours = voltage_profile(pf, grid, label="pgml")
    ppv = ref.pandapower_voltage_profile(net, grid, id_map, label="pandapower")
    fig, _ = plot_voltage_profile([ours, ppv], alpha=0.7)
    save_figure(fig, tmp_path / "vp.svg")
    plt.close("all")

    for name in ("yb.svg", "diff.png", "vp.svg"):
        assert (tmp_path / name).exists() and (tmp_path / name).stat().st_size > 0
