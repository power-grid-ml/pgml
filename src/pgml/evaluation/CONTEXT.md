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
  `run/configs/cigre_lv_geo.json`, keyed by the CIGRE LV zero-based bus numbers).
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
- `numpy_harmonic_profiles(grid, v1, index, orders, *, load_shunt=None, ...) ->
  [HarmonicProfile]` — single-phase, R-const/X∝h, shared fundamental v1.
- `numpy_harmonic_voltages(grid, harmonic_injection, orders, *, slack="norton",
  v1=None, operating_point=None, node_sources=None, load_shunt=None) -> np.ndarray` —
  full multi-element oracle returning complex `[H, N]`; machine-precision parity
  (~1e-13 V) vs `solve_harmonic_flow`; supports single-phase and three-phase, plain R/X
  lines. `load_shunt` (`None` = the documented default) adds each device's harmonic
  Norton shunt, re-derived from the OpenDSS equations in plain python; it is stamped
  phase-to-ground, which is the connection these oracles support. NOTE the oracle
  implements the NAIVE line model (R const, X∝h): with every device shunted, harmonic
  current flows in every branch, so a grid converted with the documented
  `sequence_aware`/`positive_sequence` default (skin effect + a `resistance_frequency`
  law, which the oracle does not mirror) deviates by ~2e-2 V on CIGRE LV — convert with
  `harmonic_line_model="naive"` for a machine-precision comparison.
- `fusion_prolongation(grid, index) -> np.ndarray | None` and
  `solve_with_fusion(y, i, p_mat) -> np.ndarray` — the oracles' own DENSE form of exact bus
  fusion: an ideal (zero-impedance) branch has no admittance to stamp, so the oracle skips
  it and solves `(Pᵀ Y P) v = Pᵀ I` with the 0/1 prolongation `P` built by its own
  union-find, then expands `V = P v`. pgml's solver reaches the same system through a
  many-to-one row index instead of a matrix, so the oracle's machine-precision parity on
  CIGRE LV (whose three bus-bus switches are ideal) is an INDEPENDENT check of the fusion
  algebra. Used by every solve site of both the numpy and the live-OpenDSS oracle; an ideal
  switch is also left out of the DSS circuit the live oracle builds.

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
  v1=None, operating_point=None, node_sources=None, load_shunt=None) -> np.ndarray` —
  LIVE OpenDSS harmonic oracle; returns complex `[H, N]`; supports single-phase geometry
  path (parity ~1.1e-7 V with the device shunt, ~1e-11 V without) and three-phase
  sequence-aware path (parity ~1e-8 V without the device shunt; with it the Carson
  line-model difference reaches ~2e-2 V because every branch then carries harmonic
  current). `load_shunt` is stamped with the numpy oracle's own formula, so the
  comparison stays a LINE-model comparison.
- `opendss_dyn_transformer_harmonic_voltages(grid, harmonic_injection, orders, *,
  load_shunt=None, ...) -> np.ndarray` — genuine vector-group validation using real
  OpenDSS Transformer elements.

### OpenDSS SCENARIO oracle (`oracles.opendss_scenario_oracle`)
Requires `opendssdirect` (imported lazily). Unlike `oracles.opendss_oracle` above (which
overwrites transformer/source contributions with pgml's own stamps to isolate one model
component), this exports a GENUINE, independent OpenDSS circuit — own `Vsource`/`Line`
(Rmatrix/Xmatrix/Cmatrix)/`Transformer` (native windings/conn/tap/clock, reusing
`opendss_oracle`'s vetted clock-realisation helpers; a 1-phase/positive-sequence-equivalent
unit exports too, as a plain-ratio device)/`Load` (EVERY injection appliance — Load,
Generator, Storage — exports as a native `Load`, negated for the generation-type ones; see
"Coverage note" below)/`Capacitor`/`Reactor` — then translates a `SampledScenarios` batch
into per-scenario `Edit` commands + native DSS `Spectrum` objects and solves `Solve` (snap)
+ `Solve mode=harmonics` per order. The scenario oracle for numeric cross-validation AND an
independent test-set generator for downstream state-estimation work (see
`docs/pgml/modeling/references/opendss/harmonics.md` for the harmonic injection convention
this reproduces bit-for-bit).
- `ExportedCircuit(grid, mode, busname, node_order, rowmap, index, loads, generators,
  sources, spectra)` — a live-circuit handle; `spectra` starts empty, populated by
  `run_opendss_scenarios` once it knows the requested orders.
- `export_grid_to_opendss(grid, *, mode="matched"|"default", load_shunt=None,
  circuit_name=...) -> ExportedCircuit` — builds the circuit, solves an initial nominal snapshot (validates +
  captures the stable DSS row order), raises `pgml.errors.ConversionError` for any
  unsupported grid feature (conductor-geometry lines — use `opendss_oracle`'s geometry
  path instead; unresolved `type_ref`; a non-diagonal/unbalanced `Source`; zigzag windings;
  a NONZERO exact phase shift on a 1-phase transformer — OpenDSS has no delta/LeadLag
  mechanism at `phases=1` (verified live: `conn=delta` collapses to a degenerate
  near-zero-voltage result there), so only a zero-shift 1-phase unit is representable, even
  though pgml's own `p==1` stamp DOES apply the exact vector-group shift; a non-3-phase
  transformer winding count other than 1; an impedance-grounded transformer neutral or an
  explicit `zero_sequence` override — NOT yet consumed by `pgml.assembly`, so faithfully
  exporting them would silently diverge; a per-phase override on a DELTA appliance).
  `mode="matched"` sets: the harmonic DEVICE model `load_shunt` names (`None` = the
  documented default) — `%SeriesRL` (plus `puXharm`/`XRharm` for the motor model) on every
  exported `Load`, or `NeglectLoadY=Yes` for the pure current-source model; REQUIRED for
  physical equivalence, and a device that switches its own shunt off while the run keeps
  one is refused (`NeglectLoadY` is global, with no per-`Load` equivalent);
  `Rg=Xg=0` on every LINE-LIKE element including `Switch` (a sequence-form `r1/x1/r0/x0`
  Line under the hood — it picks up OpenDSS's earth-return default exactly like an
  `r1/x1`-defined Line; verified live, a switch-only repro alone desynced a comparison by
  ~0.4%); a tight `Set Tolerance=1e-10`/`Set MaxIterations=100` snap solve (OpenDSS's
  default `1e-4` tolerance is loose enough to show up, growing with system size/loading);
  and `Vminpu=0.0001 Vmaxpu=10000` on every `Load` (OpenDSS's default `0.95`/`1.05` band
  CLIPS the constant-power/current/ZIP law outside it — pgml's laws have none — measured
  live: a bus at 0.919 pu, an everyday LV drop, made a default-banded load deliver 6.8%
  less than nameplate). `mode="default"` leaves OpenDSS's own defaults for all of these.
- `run_opendss_scenarios(grid, sampled, *, harmonic_orders, mode="matched"|"default",
  load_shunt=None, dtype=complex128) -> pgml.scenarios.ScenarioResult` — exports once,
  attaches one native `Spectrum` per device carrying a harmonic injection, then per
  scenario × per STEP (the step count read from `sampled.n_steps`; `Vsource`/load `Edit`s
  happen once per step too, so a PER-STEP `[B, T]` operating point is sliced by step, not
  just by scenario) edits the operating point (`kW`/`kvar`, per-phase
  where the appliance was split, `Vsource.pu` for a source `u_ref_scale`) and each device's
  `Spectrum` `%mag`(`=magnitude_pu*100`)/`angle`(`=phase_deg`, direct — the order-1 entry is
  the same self-relative reference pgml's own `arg(I_h)=ang_h+h*(arg(I1)-ang_1)` formula
  uses), then solves snap (order 1) + one `Solve mode=harmonics` per remaining order. `v` is
  `[B,H,N]` / `[B,T,H,N]`, aligned to `node_phase_index` rows exactly like `run_scenarios`'s
  own output.
- `write_opendss_dataset(grid, sampled, path, *, harmonic_orders, mode="matched",
  load_shunt=None, layout="wide", dtype=complex128) -> Path` — `run_opendss_scenarios` +
  `pgml.scenarios.write_dataset`, then stamps `meta.json` with `engine="opendss"`,
  `oracle_mode`, `oracle_load_shunt`, `opendssdirect_version`, `opendss_engine_version` (`Basic.Version()`'s
  full string). Byte-identical layout otherwise — `read_dataset` and every downstream data
  loader consume it unchanged.
- `compare_to_pgml(grid, sampled, *, harmonic_orders, mode="matched"|"default",
  load_shunt=None, out_dir=None, slack="norton", symmetry=None, dtype=complex128) -> dict` — runs BOTH
  engines on the IDENTICAL `sampled` and reports, per order over the whole batch: abs
  `|V_opendss-V_pgml|` and that error RELATIVE TO the order's RMS voltage, each
  mean/p95/max, plus `opendss_converged`/`pgml_converged`. `slack` defaults to `"norton"`
  (NOT pgml's own library default `"ideal"`) — an OpenDSS `Vsource` always behaves as a
  finite-impedance Thevenin source, so comparing against pgml's ideal-slack solve on a
  grid with a non-negligible source impedance reports a spurious ~1e-3 relative "error"
  that is actually two different slack models (measured on `pgml.grids.synthetic_feeder`).
  Optionally writes `opendss_comparison.json`/`.csv` to `out_dir`. Measured `mode="matched"`
  (2026-07 comparison campaign, `feeder12`/`feeder8`/CIGRE-LV-3ph/devices-with-Generator-
  Storage-Shunt-Switch-ZIP-CONST_CURRENT cases, snapshot + coherent + coherent-with-profile,
  light/nominal/heavy load, high THD, asymmetric): ~1e-9 to ~1e-6 relative on every case
  EXCEPT a documented, bounded model gap (below). `mode="default"` diverges as designed, NOT
  a bug: the TRIPLEN order is dominated by OpenDSS's imperial-calibrated earth-return
  `Rg`/`Xg` (up to ~4x relative on the CIGRE LV benchmark's 3-wire, no-neutral feeders, see
  `docs/pgml/modeling/conventions.md` §8); the non-triplen order by OpenDSS's default load
  Norton shunt, absent from pgml's harmonic model.

Voltage-dependent load models agree at the tight matched-mode floor:
`pgml.solver.harmonic_flow` anchors each device's harmonic spectrum to its
MODEL-CONSISTENT fundamental current (control-resolved / ZIP-scaled `S_eff` at the
converged terminal voltage — the same power the nonlinear fundamental solve draws), so
`CONST_IMPEDANCE`/`CONST_CURRENT`/`ZIP` loads match OpenDSS's per-model
fundamental-current scaling; pinned tight in
`tests/reference/test_scenario_oracle_opendss.py`.

Coverage note (`_ApplianceExport.dss_class` is ALWAYS `"Load"` now): a genuine DSS
`Generator` element stamps its own linearized PQ admittance into the harmonics-mode linear
system REGARDLESS of `NeglectLoadY`/`Model`/`Xdpp` (verified empirically: no combination
zeroes it, unlike a `Load`), so `Generator`/`Storage` export as a negative-kW `Load` instead
(the OpenDSS negative-load generation idiom), which DOES become a true pure current source
under `NeglectLoadY=Yes` (post-solve `YPrim` ~1e-12 vs ~0.0375 S for a genuine `Generator`
on the same nameplate). `ShuntAppliance`/`ShuntReactor` export as a `Capacitor` (C) + a
diagonal-`Rmatrix` `Reactor` (G). See the module docstring + the test file for the full
refusal list and every measured figure above.

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
- Demo: `run/examples/evaluate_ieee33.py` regenerates the IEEE-33 figure set.
- Matplotlib backend: tests/headless use `Agg` (set in `tests/evaluation/conftest.py`).
