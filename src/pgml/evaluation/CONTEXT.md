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

## Reference adapters (`pgml.evaluation.references`, lazy heavy imports)
- `pandapower_ybus(net, grid, id_map, index, *, label)` -> LabeledMatrix (pu->SI,
  pure NETWORK admittance; compare to `assemble_network_ybus`).
- `pandapower_voltage_profile(net, grid, id_map, *, label, slack)` -> VoltageProfile.
- `dss_systemy() -> (Y, node_order)`, `opendss_ybus(Y, node_order, id_map, index, *,
  label)` -> LabeledMatrix (aligns single-phase `BUS<n>.1` SystemY to our rows).
- `numpy_harmonic_profiles(grid, v1, index, orders, *, ...)` -> [HarmonicProfile] —
  INDEPENDENT single-phase numpy harmonic solve (R-const/X∝h), shared fundamental v1.
- Carson feeder builders (pandapower -> pgml grid + synthesized geometry + spectra):
  `ieee33_geometry_grid()`, `cigre_lv_geometry_grid()` -> `(grid, id_map)`.
- OpenDSS-from-geometry (Carson harmonic comparison): `build_opendss_geometry_circuit(grid)`,
  `opendss_geometry_systemy(grid, index, orders) -> {order: aligned SystemY(order·f0)}`,
  `opendss_geometry_harmonic_profiles(grid, hres, orders) -> [HarmonicProfile]` (OpenDSS
  line model solved with pgml's converged injection — true OpenDSS-vs-pgml harmonic check).
  Demo: `examples/evaluate_harmonics_carson.py` (IEEE-33 + CIGRE LV).
- **`numpy_harmonic_voltages(grid, harmonic_injection, orders, *, slack="norton",
  v1=None, operating_point=None, node_sources=None) -> np.ndarray`** — Pure-numpy regression oracle returning
  complex `[H, N]` aligned to `node_phase_index(grid)`. Supports the FULL CIGRE LV grid
  for both `SINGLE_PHASE_EQUIV` and `THREE_PHASE` phase modes with **plain R/X lines** (no
  conductor_geometry). Reimplements pgml's EXACT Y-bus formulas (R const, X∝h — no Carson
  correction) in numpy, giving machine-precision parity (~1e-13 V absolute error) vs
  `solve_harmonic_flow`. Pass `v1 = hres.pf.v.detach().cpu().numpy()` to share the nonlinear
  fundamental operating point. `node_sources`: optional list of `NodeHarmonicSource` (from
  `pgml.solver`) — per-node harmonic disturbance sources stamped at h>1 with same physics
  as pgml; parity ~1e-12 V. Validation: `tests/reference/test_cigre_lv_full_harmonic_opendss.py`,
  `tests/reference/test_node_harmonic_source_oracle.py`.
- **`opendss_harmonic_voltages(grid, harmonic_injection, orders, *, slack="norton",
  v1=None, operating_point=None, node_sources=None) -> np.ndarray`** — LIVE OpenDSS harmonic oracle for grids
  using the Carson/Deri earth-return line model. Returns complex `[H, N]` aligned to
  `node_phase_index(grid)`. Requires one of:
  (a) **Single-phase geometry path**: all lines carry `conductor_geometry` (set by
  `synthesize_grid_geometry`). OpenDSS builds a stub Vsource + WireData/LineGeometry circuit;
  the stub Norton (which carries OpenDSS's Carson correction) is subtracted from SystemY and
  replaced with pgml's exact source Norton; switches and transformers are stamped with
  pgml-exact formulas (R const, X∝h, complex tap). Parity: ~1e-11 V (near machine precision).
  (b) **Three-phase sequence-aware path**: lines tagged `harmonic_line_model=sequence_aware`
  (set by `apply_default_harmonic_model`). OpenDSS builds a stub circuit with R1/X1/R0/X0
  lines (no native Transformer — avoids neutral bus incompatibility); the same stub-subtract
  technique is used; transformers are stamped with pgml-exact formulas. Parity: ~1e-8 V
  (near machine precision). Plain R/X grids with neither tag raise `ValueError`.
  `node_sources`: optional list of `NodeHarmonicSource` — stamped after the OpenDSS stub-subtract
  and pgml-exact element stamps using identical physics (same formulas, same V1). Single-phase
  geometry parity vs pgml with node sources: ~1e-11 V (tight). Three-phase seq-aware parity
  with node sources: ~3.7 V at MV-bus phases B/C (diagonal transformer gap, TODO #4); LV-bus
  parity < 1e-5 V.
  Validation: `tests/reference/test_cigre_lv_live_opendss.py`,
  `tests/reference/test_node_harmonic_source_oracle.py`.

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
