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

The old `pgml.evaluation.oracles` module remains as a thin compatibility shim
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
- `plot_grid_graph(grid, *, node_values=None, layout="spring"|"kamada", positions=None,
  ...) -> (fig, ax)` — topology colored by a per-node value; slack outlined;
  `with_labels` writes zero-based display numbers (`node_numbering`, = position in
  `grid.nodes` = the source tool's bus index for a converted grid — node IDS are 1-based
  global identifiers, not a numbering).
- `graph_layout(grid, *, layout="spring", positions=None) -> {node_id: (x, y)}` — the
  single source of node placement; pass the SAME dict to `plot_grid_graph` and to any
  overlay on top so all layers align. Explicit `positions` keys resolve name → zero-based
  node number → id (idempotent for an id-keyed dict); every node must be covered.
- `load_node_positions(path, grid) -> {node_id: (x, y)}` — read a position JSON (e.g.
  `examples/configs/cigre_lv_geo.json`, keyed by the CIGRE LV zero-based bus numbers).
- `node_numbering(grid) -> {node_id: int}` (in `data.py`) — the zero-based display
  numbering above; `row_labels(index, *, numbering=None)` uses it for `<node>·<phase>`
  tick labels.

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

### OpenDSS SCENARIO oracle (`oracles.opendss_scenario_oracle`)
Requires `opendssdirect` (imported lazily). Unlike `oracles.opendss_oracle` above (which
overwrites transformer/source contributions with pgml's own stamps to isolate one model
component), this exports a GENUINE, independent OpenDSS circuit — own `Vsource`/`Line`
(Rmatrix/Xmatrix/Cmatrix)/`Transformer` (native windings/conn/tap/clock, reusing
`opendss_oracle`'s vetted clock-realisation helpers)/`Load`/`Generator`/`Capacitor`/
`Reactor` — then translates a `SampledScenarios` batch into per-scenario `Edit` commands +
native DSS `Spectrum` objects and solves `Solve` (snap) + `Solve mode=harmonics` per order.
The scenario oracle for numeric cross-validation AND an independent `pgl` test-set
generator (see `docs/pgml/modeling/references/opendss/harmonics.md` for the harmonic
injection convention this reproduces bit-for-bit).
- `ExportedCircuit(grid, mode, busname, node_order, rowmap, index, loads, generators,
  sources, spectra)` — a live-circuit handle; `spectra` starts empty, populated by
  `run_opendss_scenarios` once it knows the requested orders.
- `export_grid_to_opendss(grid, *, mode="matched"|"default", circuit_name=...) ->
  ExportedCircuit` — builds the circuit, solves an initial nominal snapshot (validates +
  captures the stable DSS row order), raises `pgml.errors.ConversionError` for any
  unsupported grid feature (conductor-geometry lines — use `opendss_oracle`'s geometry
  path instead; unresolved `type_ref`; a non-diagonal/unbalanced `Source`; zigzag or
  non-3-phase `Transformer` windings; an impedance-grounded transformer neutral or an
  explicit `zero_sequence` override — NOT yet consumed by `pgml.assembly`, so faithfully
  exporting them would silently diverge; a per-phase override on a DELTA appliance).
  `mode="matched"` sets `NeglectLoadY=Yes` (REQUIRED for physical equivalence — pgml's
  harmonic solver has no load-shunt model at all, `include_load_shunt` is hard-`False`)
  and `Rg=Xg=0` on every line (pgml's non-geometry harmonic line models carry no earth
  term). `mode="default"` leaves OpenDSS's own defaults.
- `run_opendss_scenarios(grid, sampled, *, harmonic_orders, mode="matched"|"default",
  dtype=complex128) -> pgml.scenarios.ScenarioResult` — exports once, attaches one native
  `Spectrum` per device carrying a harmonic injection, then per scenario (× per STEP for a
  node-coherent batch, detected via `sampled.samples["time_s"]`) edits the operating point
  (`kW`/`kvar`, per-phase where the appliance was split, `Vsource.pu` for a source
  `u_ref_scale`) and each device's `Spectrum` `%mag`(`=magnitude_pu*100`)/`angle`
  (`=phase_deg`, direct — the order-1 entry is the same self-relative reference pgml's own
  `arg(I_h)=ang_h+h*(arg(I1)-ang_1)` formula uses), then solves snap (order 1) + one
  `Solve mode=harmonics` per remaining order. `v` is `[B,H,N]` / `[B,T,H,N]`, aligned to
  `node_phase_index` rows exactly like `run_scenarios`'s own output.
- `write_opendss_dataset(grid, sampled, path, *, harmonic_orders, mode="matched",
  layout="wide", dtype=complex128) -> Path` — `run_opendss_scenarios` +
  `pgml.scenarios.write_dataset`, then stamps `meta.json` with `engine="opendss"`,
  `oracle_mode`, `opendssdirect_version`, `opendss_engine_version` (`Basic.Version()`'s
  full string). Byte-identical layout otherwise — `read_dataset`/every `pgl` data source
  consume it unchanged.
- `compare_to_pgml(grid, sampled, *, harmonic_orders, mode="matched"|"default",
  out_dir=None, slack="norton", symmetry=None, dtype=complex128) -> dict` — runs BOTH
  engines on the IDENTICAL `sampled` and reports, per order over the whole batch: abs
  `|V_opendss-V_pgml|` and that error RELATIVE TO the order's RMS voltage, each
  mean/p95/max, plus `opendss_converged`/`pgml_converged`. `slack` defaults to `"norton"`
  (NOT pgml's own library default `"ideal"`) — an OpenDSS `Vsource` always behaves as a
  finite-impedance Thevenin source, so comparing against pgml's ideal-slack solve on a
  grid with a non-negligible source impedance reports a spurious ~1e-3 relative "error"
  that is actually two different slack models (measured on `pgml.grids.synthetic_feeder`).
  Optionally writes `opendss_comparison.json`/`.csv` to `out_dir`. Measured on
  `synthetic_feeder` (4-node, 3 WYE loads, `mode="matched"`): relative error ~3e-10
  (fundamental) / ~1.5e-8 (injected harmonics h=3,5) — near machine precision.
  `mode="default"` diverges as designed, NOT a bug: the TRIPLEN order is dominated by
  OpenDSS's imperial-calibrated earth-return `Rg`/`Xg` (tens-to-hundreds of percent on a
  3-wire feeder, see `docs/pgml/modeling/conventions.md` §8); the non-triplen order by
  OpenDSS's default load Norton shunt absent from pgml's harmonic model (~0.4-0.5%).

Coverage note: `Storage` exports as a DSS `Generator` (its snapshot behaviour is identical
to `Generator`'s per the schema; no `pgml.scenarios` selector targets storage either, so
neither engine dispatches it per-scenario). `ShuntAppliance`/`ShuntReactor` export as a
`Capacitor` (C) + a diagonal-`Rmatrix` `Reactor` (G); implemented but not independently
oracle-tested by `tests/reference/test_scenario_oracle_opendss.py` (the primary test grid,
`synthetic_feeder`, carries neither). See that test file + this module's docstring for the
full refusal list.

## Compatibility shim (`pgml.evaluation.oracles`)
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
