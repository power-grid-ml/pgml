# convert — external formats to and from the pgml schema

Signatures: docstrings / API_INDEX.md. Cross-tool conventions (voltage bases, transformer
reference side, vector-group sign, power signs, earth return, converter coverage):
`docs/pgml/modeling/conventions.md`, the authoritative record; keep it in sync when a
converter's convention handling changes.

## Codemap

- `pandapower/`, `pgm/`, `opendss/` — one subpackage per reference tool. Each has
  `converter.py` (`to_grid`, inbound) and `exporter.py` (`from_grid`, outbound) plus its
  own `CONTEXT.md` for the tool-specific field mapping.
- `_common.py` — the shared inbound scaffold: `IdCounter`, `PhaseMode`, `phases_for`, the
  sequence-to-phase expansion, Thevenin helpers, zero-sequence fallbacks, and the emit
  helpers `build_node`/`build_load`/`build_generator`/`build_source`/
  `build_line_from_sequence`/`build_line_from_matrices`/`warn_dropped_elements`.
  Unit-tested in `tests/convert/test_common.py`.
- `_export.py` — shared outbound helpers: `UnsupportedGridError`, the balanced-layout
  validation, `record_reduction` and the `REDUCTIONS` sentence table.
- `_report.py` — `ConversionReport`, `ReportEntry`, `ModelMatch`, `ReportCategory`.
- `_model_differences.py` — `add_model_differences` / `finalize_report`: derives the
  `model.*` entries (where the two tools' default models differ) from grid content.

## Contract surface (used by pgg, pghub, pgl, pgd and the paper harness)

- `convert.<tool>.to_grid(source, ...)` is pure: `(Grid, id_map)`, or
  `(Grid, id_map, ConversionReport)` with `return_report=True`. The `id_map` is keyed by
  the source tool's element kind and source index/id; `id_map["slack_v_complex"]` is the
  line-to-line slack phasor in volts (a single-phase ideal-slack `v_fixed` convenience).
- `convert.<tool>.from_grid(grid, ...)` returns `PandapowerExport` / `PgmExport` /
  `OpenDSSExport` with element maps, `reductions` (strings) and `.report`.
- `pgml.convert` exports `ConversionReport`, `ReportEntry`, `ModelMatch`,
  `ReportCategory`, `PhaseMode`, `add_model_differences`.
- Report keys are stable and shared across tools and directions: `dropped.<kind>`,
  `approx.<topic>`, `model.<topic>`. `affects` is a subset of
  `("fundamental", "unbalanced", "harmonic")` and empty for a stand-in below solver
  tolerance. Serialised layout versioned by `REPORT_FORMAT_VERSION`.

## Invariants

- Converters are the only unit boundary: engineering units in, SI out. Never invent schema
  fields (`extra="forbid"`); sequence and nameplate inputs go through the schema's
  input-convention DTOs. Record the source convention in `Provenance`.
- Nothing is dropped silently. Every non-empty source table the converter does not read
  goes through `warn_dropped_elements` (one WARNING with kind and count) and a
  `dropped.<kind>` report entry. Dropped/approximated entries log at WARNING, model
  differences at INFO, plus one closing WARNING when an open entry touches a `comparable`
  result class. `announced=True` marks an entry the converter already logged.
- All converters stamp identical schema objects for the same physical input because the
  phase decision and per-phase mapping live only in the `_common` emit helpers. Add new
  plumbing there, not in a per-tool converter.
- `phase_mode=PhaseMode.SINGLE_PHASE_EQUIV` (default): one positive-sequence equivalent,
  `phases=(A,)`, 1x1 matrices. Its output must stay unchanged by edits made for
  `THREE_PHASE`; the reference oracle suite calls `to_grid(net)` with no `phase_mode`.
  Asymmetric loads collapse to a balanced total with an INFO log. A coupled multi-phase
  OpenDSS line reduces to its positive sequence `Z1 = Z_self - Z_mutual`.
- `THREE_PHASE`: genuine abc (OpenDSS keeps its native phases including `Phase.N`).
  Sequence data expands with `self=(Z0+2*Z1)/3`, `mutual=(Z0-Z1)/3`; an explicit native
  `r0/x0/c0` always wins over the `line.zero_sequence.*` ratios, and the use of those
  ratios is warned once per grid (`ZeroSequenceDefaults`). Sources are balanced Thevenins
  whose zero sequence comes from native data (OpenDSS `R0/X0`, pgm `z01_ratio`, pandapower
  `x0x_max`/`r0x0_max`) or the `source.zero_sequence.*` ratios with a WARNING;
  `Z0 == Z1` yields the plain diagonal.
- Voltage base: `Node.u_rated_v` is line-to-line for a node with three or more phases;
  every per-phase working voltage is line-to-neutral, derived only by
  `assembly._params.phase_voltage_magnitude`. Under `THREE_PHASE` `build_source` divides
  the L-L `u_ref_v` by sqrt(3); pinning the L-L magnitude per phase is the classic
  sqrt(3) error. OpenDSS `Bus.kVBase()` is L-N and is recovered as `kVBase*sqrt(3)*1000`.
- Transformer leakage is stored TO-coil-referred (x3 for a DELTA to-winding) in BOTH phase
  modes; assembly undoes the factor for the single-phase scalar stamp. A
  phase-mode-conditional factor is wrong (pinned by
  `tests/reference/test_pandapower_grid_matrix.py`, `test_pgm_transformer.py`).
- `Transformer.zero_sequence` is written only from native data: pandapower
  `vk0_percent`/`vkr0_percent`, pgm grounded zigzag (`0.1*Z1`, power-grid-model's
  hardcoded value). OpenDSS has no zero-sequence transformer input; the
  `transformer.zero_sequence.*` ratios (1.0 = OpenDSS's own model) apply.
- `harmonic_line_model` (every `to_grid`): `None` writes the modeling default
  (`line.harmonic_model.three_phase` / `.single_phase`) onto every converted R/X line via
  `resolve_converted_line_models`; `"none"` leaves lines unresolved. No source tool carries
  a harmonic line model, so skipping this step would silently give the naive model.
- `base_frequency_hz` comes from the source (`net.f_hz`, `Solution.Frequency()`); pgm has
  none, so the caller passes it and a wrong value silently scales every L and C.
- Ideal branches (pgm `link`, pandapower bus-bus `switch` under the default
  `branch.switch_model: ideal`) convert to closed `Switch`es with R = L = 0 whose rows the
  solve fuses; `near_ideal` writes `branch.near_ideal_series_resistance_ohm` instead.

## Outbound (`from_grid`)

- pandapower and pgm exports are fundamental-only and balanced: a consistent `(A,)` or
  `(A, B, C)` layout; partial-phase, mixed-layout and neutral-conductor grids raise
  `UnsupportedGridError`, as do unsupported elements. Harmonic fields (spectra, harmonic
  shunt/impedance, frequency-dependent lines) are named in `reductions`, never implied.
- A model-changing reduction raises unless `allow_approximation=True`, which records it.
  Unavoidable stand-ins (pgm finite source for an ideal boundary, 1 nOhm pgm line for an
  ideal switch) are always recorded. The input `Grid` is never mutated.
- The OpenDSS export builds a live multi-phase circuit through the scenario oracle's
  `export_grid_to_opendss` and adds the off-nominal tap and, in `mode="matched"`, each
  line's `Rg/Xg/rho`.

## Decisions an edit must not undo

- Reference-tool divergence is an option plus a report entry (`ModelMatch` names the
  preset/setting that closes it), not a changed default; flip a default only when pgml
  lacks the option.
- Converters never mode-switch stored transformer leakage or base voltages.
- OpenDSS `Vsource.basekv` is used as-is for both phase counts (L-L for `phases>=3`, the
  solved conductor EMF for `phases=1`); no phase-count branch
  (`tests/convert/test_opendss_vsource_basekv.py`).
