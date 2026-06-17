# Interface ledger: evaluation (comparison & evaluation plots)

Reference-vs-ours and diagnostic plots. Paper-ready raster (>=300 DPI) or vector
(svg/pdf) via `save_figure`; the 3D harmonic plot is interactive plotly HTML.
Design: plot functions consume framework-agnostic DATA containers (plain numpy), so
"ours vs reference" is just a list of containers; builders adapt our solver outputs,
and `references` adapts the reference libraries to the SAME containers.

## Data containers (`pgml.evaluation.data`)
- `VoltageProfile(distances_km, v_pu, label, node_ids)` — sorted by distance.
- `HarmonicProfile(distances_km, magnitude, angle_deg, order, frequency_hz, label,
  node_ids, unit)`.
- `LabeledMatrix(matrix[N,N] complex, label, row_labels)`.
- Builders from OUR results: `voltage_profile(PowerFlowResult, grid, *, label, phase,
  slack, scenario)`, `harmonic_profile(HarmonicFlowResult, grid, order, *, ...)`,
  `harmonic_profiles(..., orders, ...)`, `labeled_matrix(Y_tensor, index, *, label,
  freq_index)`. Batched results reduce to `scenario` (default 0).

## Topology (`pgml.evaluation.topology`)
- `slack_node_id(grid)`, `grid_graph(grid, *, weight="km")` (networkx; line length in
  km, 0 for non-lines; edges tagged `kind`), `distance_from_slack(grid, slack=None) ->
  {node_id: km}`. OPEN switches are excluded everywhere (no path/merge across them).
- `branch_edges(grid, *, include_open_switches=False) -> [ProfileEdge(a, b, kind)]` —
  the drawable interconnections for profile lines; `kind` in {line, switch, other};
  open switches dropped. This is what makes profile lines follow real branches.

## Plots
- `plot_ybus_heatmaps(matrices, *, part="abs"|"real"|"imag", share_scale=True, ...)
  -> Figure` — versions side by side (log / symlog color scale).
- `plot_ybus_difference(a, b, *, ...) -> Figure` — `|Y_a - Y_b|` heatmap (the two
  MOST SIMILAR versions; annotates max & Frobenius error).
- `plot_voltage_profile(profiles, *, grid=None, ax=None, colors, alpha, markers, ...)
  -> (fig, ax)` — REUSABLE voltage-drop diagram; color per implementation; `alpha`
  keeps close lines visible; accepts an external `ax`. With `grid`, connecting lines
  follow the real branches (closed switches dashed, open omitted); without it, a
  distance-sorted polyline (legacy).
- `plot_harmonic_profile(profiles, *, grid=None, ax=None, linestyles, cmap, alpha, ...)
  -> (fig, ax)` — one order: x=distance, y=magnitude, COLOR=voltage angle (cyclic
  cmap), LINE STYLE per implementation; lines follow branches when `grid` given.
- `plot_profile_error(reference, ours, *, ax=None, ...) -> (fig, ax)` — per-node |Δpu|
  bar chart (aligned by node id).
- `plot_harmonic_profile_3d(profiles, *, grid=None, reference_labels=(), out_html=None,
  ...) -> plotly.Figure` — x=distance, y=magnitude, z=angle; COLOR=order (show 3,5,7,9
  at once), DASH=implementation; lines follow branches when `grid` given; writes
  self-contained interactive HTML.
- `plot_grid_graph(grid, *, node_values=None, layout="spring"|"kamada", ...) -> (fig,
  ax)` — topology colored by a per-node value; slack outlined.

## Reference adapters (`pgml.evaluation.references`, lazy heavy imports)
- `pandapower_ybus(net, grid, id_map, index, *, label)` -> LabeledMatrix (pu->SI,
  pure NETWORK admittance; compare to `assemble_network_ybus`).
- `pandapower_voltage_profile(net, grid, id_map, *, label, slack)` -> VoltageProfile.
- `dss_systemy() -> (Y, node_order)`, `opendss_ybus(Y, node_order, id_map, index, *,
  label)` -> LabeledMatrix (aligns single-phase `BUS<n>.1` SystemY to our rows).
- `numpy_harmonic_profiles(grid, v1, index, orders, *, ...)` -> [HarmonicProfile] —
  INDEPENDENT single-phase numpy harmonic solve (R-const/X∝h), shared fundamental v1.

## IO + style (`pgml.evaluation.style`)
- `save_figure(fig, path, *, dpi=300)` — format from extension; raster >=300 DPI.
- Constants: `DPI`, `COMPARE_ALPHA`, `LINESTYLES`, `ANGLE_CMAP`.

## Conventions / notes
- Plotting is OUTSIDE the differentiable path; `_util` detaches tensors to numpy
  (correct here — never on a loss tape). Honors float/tensor duality on grid fields.
- pu = `|V|` / line-to-neutral base (`assembly._params.phase_voltage_magnitude`).
- pandapower internal Ybus / OpenDSS SystemY exclude const-Z load shunts; the cleanest
  "most similar" Y-bus difference is OUR `assemble_network_ybus` vs `pandapower_ybus`.
- Demo: `examples/evaluate_ieee33.py` regenerates the IEEE-33 figure set.
- Matplotlib backend: tests/headless use `Agg` (set in `tests/evaluation/conftest.py`).
