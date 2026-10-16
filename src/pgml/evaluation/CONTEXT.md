# evaluation — comparison plots and reference oracles

Reference-vs-pgml and diagnostic figures, plus the adapters that run the reference tools
(pandapower, OpenDSS) and an independent numpy harmonic oracle. Plot functions consume
framework-agnostic data containers (plain numpy), so "ours vs reference" is a list of
containers: builders adapt pgml's results, `oracles` adapts the reference tools to the
same containers.

Signatures: docstrings / `API_INDEX.md`.

## Codemap

- `data` — containers `VoltageProfile`, `HarmonicProfile`, `LabeledMatrix`; builders from
  pgml results (`voltage_profile`, `harmonic_profile(s)`, `labeled_matrix`);
  `node_numbering`, `row_labels`.
- `topology` — `slack_node_id`, `grid_graph`, `distance_from_slack`, `branch_edges` (the
  x-axis of every profile and the branches profile lines follow).
- `ybus_plots` — `plot_ybus_heatmaps`, `plot_ybus_difference`.
- `profiles` — `plot_voltage_profile`, `plot_harmonic_profile`, `plot_profile_error`,
  `plot_harmonic_model_comparison`.
- `harmonic3d`, `interactive` — plotly figures written as self-contained HTML
  (`plot_harmonic_profile_3d`, `plot_harmonic_profile_interactive`).
- `graph_plots` — `plot_grid_graph`, `graph_layout`, `load_node_positions`.
- `style` — `save_figure` / `save_html`, shared constants (`DPI`, `COMPARE_ALPHA`,
  `LINESTYLES`, `ANGLE_CMAP`).
- `_util` — tensor → numpy conversion for plotting.
- `oracles/` — reference adapters (optional extra `pgml[oracles]`):
  - `grids` — re-exports of the benchmark grid builders in `pgml.grids`
    (`ieee33_geometry_grid`, `cigre_lv_full_grid`, `cigre_lv_geometry_grid`,
    `CONVERTER_SPECTRUM`);
  - `numpy_oracle` — pure-numpy harmonic solve (`numpy_harmonic_voltages`,
    `numpy_harmonic_profiles`) and the oracles' own dense bus fusion
    (`fusion_prolongation`, `solve_with_fusion`);
  - `pandapower_oracle` — `pandapower_ybus`, `pandapower_voltage_profile`;
  - `opendss_oracle` — hybrid live-OpenDSS paths: SystemY extraction and alignment, the
    Carson geometry circuit, `opendss_harmonic_voltages`,
    `opendss_dyn_transformer_harmonic_voltages`;
  - `opendss_scenario_oracle` — an independent native OpenDSS circuit for a scenario batch
    (`export_grid_to_opendss`, `ExportedCircuit`, `run_opendss_scenarios`,
    `write_opendss_dataset`, `compare_to_pgml`, `OpenDSSConvergenceError`).

## Invariants

- Import boundary: `import pgml.evaluation` pulls in matplotlib / plotly / networkx only and
  never imports `oracles`. pandapower and opendssdirect are imported lazily inside the
  oracle functions, so even `import pgml.evaluation.oracles` succeeds without them.
- Plotting is outside the differentiable path; `_util` detaches tensors to numpy (correct
  here, never on a loss tape). Grid fields follow the float/tensor duality.
- Profile magnitudes in pu are `|V|` over the line-to-neutral base
  (`assembly._params.phase_voltage_magnitude`).
- Distances, edges and profile lines exclude open switches (no path across them); closed
  switches draw dashed.
- Display numbering is zero-based position in `grid.nodes` (the source tool's bus index for
  a converted grid); node ids are 1-based global identifiers, not a numbering.
  `graph_layout` is the single source of node placement: pass the same dict to every layer.
- Raster output is >= 300 DPI; vector by file extension.
- Oracle outputs are aligned to `assembly.node_phase_index` rows, `[H, N]` (or
  `[B, H, N]` / `[B, T, H, N]` for the scenario oracle), like `run_scenarios`.
- pandapower's internal Ybus and OpenDSS SystemY exclude const-Z load shunts; compare them
  with `assemble_network_ybus`, the pure network admittance.

## Oracle roles (what each comparison proves)

- `numpy_oracle` is a regression mirror of pgml's own formulas, not independent physics
  validation; its three-phase transformer stamp calls `assembly._transformer`. Its fusion
  algebra (0/1 prolongation from its own union-find) is independent of pgml's many-to-one
  row index, so CIGRE LV parity checks fusion.
- `opendss_oracle` replaces transformer/source contributions with pgml's stamps to isolate
  one model component (line model, vector group). The hybrid entry points reject explicit
  DER harmonic impedance/source blocks.
- `opendss_scenario_oracle` exports a genuine circuit with OpenDSS's own elements and
  refuses (`ConversionError`) any grid feature it cannot represent faithfully; the refusal
  list and the `matched` / `default` mode settings are in its module docstring.
  `mode="matched"` aligns OpenDSS with pgml's model; `mode="default"` diverges by design
  (earth-return `Rg`/`Xg`, see `docs/pgml/modeling/conventions.md`, "Harmonics and earth
  return").
- Native DER export: Generator / PVSystem / Storage are kept as native elements for
  explicit `opendss_voltage` + `opendss_admittance` blocks with scalar balanced impedance;
  pure current-source generation exports as a negative-kW `Load` (a native `Generator`
  stamps its own admittance in harmonics mode regardless of `NeglectLoadY`). Other
  impedance/source laws, native controls and per-phase DER overrides are rejected.
- `compare_to_pgml` defaults to `slack="norton"`: an OpenDSS `Vsource` is always a finite
  Thevenin source, so an ideal-slack pgml solve would report a slack-model difference.
- Conformance tests select `defaults.use_preset("opendss")` explicitly.

## Demo

`run/examples/pgml/evaluate_ieee33.py` regenerates the IEEE-33 figure set.
</content>
