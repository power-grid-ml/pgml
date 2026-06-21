# Interface ledger: evaluation (comparison & evaluation plots)

Reference-vs-ours and diagnostic plots. Paper-ready raster (>=300 DPI) or vector
(svg/pdf) via `save_figure`; the 3D harmonic plot is interactive plotly HTML.
Design: plot functions consume framework-agnostic DATA containers (plain numpy), so
"ours vs reference" is just a list of containers; builders adapt our solver outputs,
and `oracles` adapts the reference libraries to the SAME containers.

## Optional-import boundary
`import pgml.evaluation` pulls in ONLY matplotlib / plotly / networkx; it does NOT
import the oracle subpackage and does not require pandapower or opendssdirect.
Oracle modules are isolated under `pgml.evaluation.oracles`; pandapower and
opendssdirect are imported lazily INSIDE functions, so even
`import pgml.evaluation.oracles` succeeds without them installed.  Install
`pgml[oracles]` (pyproject.toml) to pull in all oracle dependencies.

The old `pgml.evaluation.references` module remains as a thin compatibility shim
that re-exports everything from `pgml.evaluation.oracles`; new code should import
from the canonical `pgml.evaluation.oracles` path.

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
- `plot_harmonic_model_comparison(a, b, *, grid=None, relative=False, ...) ->
  (fig_overlay, fig_diff)` — pairwise TWO-model diagnostic for one order, as TWO figures:
  `fig_overlay` = magnitude vs distance; `fig_diff` = per-node difference SCATTER (`|a|−|b|`)
  with node id on the x-axis (not distance — several nodes share a distance).
- `plot_harmonic_profile_3d(profiles, *, grid=None, reference_labels=(), out_html=None,
  ...) -> plotly.Figure` — x=distance, y=magnitude, z=angle; COLOR=order (show 3,5,7,9
  at once), DASH=implementation; lines follow branches when `grid` given; writes
  self-contained interactive HTML.
- `plot_harmonic_profile_interactive(profiles, *, grid=None, out_html=None,
  line_alpha=0.6, ...) -> plotly.Figure` (`evaluation.interactive`) — 2D companion for a
  fixed order: x=distance, y=magnitude, ONE colour per implementation, translucent lines;
  each model is its own legend group, click to toggle on/off (`groupclick=togglegroup`).
  Carries MULTIPLE models (and orders — `h{order} · {label}` groups) at once. For
  heavily-overlapping models. The matplotlib `plot_harmonic_profile` also takes `alpha`
  to keep overlaps legible in the static SVG.
- `plot_grid_graph(grid, *, node_values=None, layout="spring"|"kamada", ...) -> (fig,
  ax)` — topology colored by a per-node value; slack outlined.

## Oracle subpackage (`pgml.evaluation.oracles`)
Canonical home for all reference-library adapters and grid builders.
**Optional import** — requires `pgml[oracles]` extras.

### Grid builders (`oracles.grids`)
- `ieee33_geometry_grid(*, n_harmonic_loads, spectrum) -> (grid, id_map)` — IEEE-33
  converted from pandapower + synthesized Carson geometry + converter spectra.
- `cigre_lv_full_grid(*, phase_mode, source_impedance_ohm) -> (grid, id_map)` — full
  CIGRE LV benchmark (all 3 feeders + 3 Dyn1 transformers + MV source).
- `cigre_lv_geometry_grid(*, n_harmonic_loads, spectrum) -> (grid, id_map)` — single
  CIGRE LV residential feeder with synthesized geometry, MV side abstracted to a
  Thévenin source.
- `CONVERTER_SPECTRUM` — default 6-pulse converter spectrum constant.

### numpy oracle (`oracles.numpy_oracle`)
Pure-numpy harmonic oracle, no live OpenDSS required.
- `numpy_harmonic_profiles(grid, v1, index, orders, ...) -> [HarmonicProfile]` —
  single-phase, R-const/X∝h, shared fundamental v1.
- `numpy_harmonic_voltages(grid, harmonic_injection, orders, *, slack="norton",
  v1=None, operating_point=None, node_sources=None) -> np.ndarray` — full multi-element
  oracle returning complex `[H, N]`; machine-precision parity (~1e-13 V) vs
  `solve_harmonic_flow`; supports single-phase and three-phase, plain R/X lines.

### pandapower oracle (`oracles.pandapower_oracle`)
- `pandapower_ybus(net, grid, id_map, index, *, label)` -> LabeledMatrix (pu->SI,
  pure NETWORK admittance; compare to `assemble_network_ybus`).
- `pandapower_voltage_profile(net, grid, id_map, *, label, slack)` -> VoltageProfile.

### OpenDSS oracle (`oracles.opendss_oracle`)
Requires `opendssdirect` (imported lazily inside functions).
- `dss_systemy() -> (Y, node_order)` — extract SystemY from the active DSS circuit.
- `align_dss_systemy(y_dss, node_order, id_map, index)` — reorder single-phase SystemY
  to our rows (IEEE-feeder positive-sequence circuits).
- `opendss_ybus(y_dss, node_order, id_map, index, *, label)` -> LabeledMatrix.
- `build_opendss_geometry_circuit(grid, *, slack_node) -> busname` — passive Carson
  circuit (Vsource + WireData/LineGeometry lines); returns `{node_id: dss_bus_name}`.
- `opendss_geometry_systemy(grid, index, orders, ...) -> {order: aligned SystemY}`.
- `opendss_geometry_harmonic_profiles(grid, hres, orders, ...) -> [HarmonicProfile]` —
  OpenDSS line-model profiles for the SAME geometry as pgml.
- `opendss_harmonic_voltages(grid, harmonic_injection, orders, *, slack="norton",
  v1=None, operating_point=None, node_sources=None) -> np.ndarray` — LIVE OpenDSS
  harmonic oracle; returns complex `[H, N]`; supports single-phase geometry path
  (parity ~1e-11 V) and three-phase sequence-aware path (parity ~1e-8 V).
- `opendss_dyn_transformer_harmonic_voltages(grid, harmonic_injection, orders, ...) ->
  np.ndarray` — genuine vector-group validation using real OpenDSS Transformer elements.

## Compatibility shim (`pgml.evaluation.references`)
Re-exports all of `pgml.evaluation.oracles`; kept for backward compatibility with
existing importers.  New code should import from `pgml.evaluation.oracles` directly.

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
