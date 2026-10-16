# convert.pgm — power-grid-model input_data to and from the pgml schema

Signatures: docstrings / API_INDEX.md. Shared converter rules: `convert/CONTEXT.md`;
cross-tool conventions: `docs/pgml/modeling/conventions.md`.

## Codemap

- `converter.py` — `to_grid(input_data, ...)`: power-grid-model structured `input_data`
  (one numpy structured array per component) to `(Grid, id_map[, report])`. The module
  docstring carries the full field mapping and the transformer derivation.
- `exporter.py` — `from_grid(grid, ...)` -> `PgmExport` (`input_data`, `pgm_of_node`,
  `pgm_of_branch`, `pgm_of_appliance`, `reductions`, `report`): the balanced fundamental
  inverse over the shared scope.
- Tests: `tests/reference/test_ieee33_pgm.py` (load flow), `test_pgm_transformer.py`
  (vector groups, taps, coil factor, zigzag, live-solve convention pins),
  `test_switch_fusion_pgm.py` (`link`), `tests/convert/test_pgm_voltage_regulator.py`.

## Invariants

- No runtime dependency on `power_grid_model`: `input_data` is read as a plain
  structured-array format, and pgm enum ints (`WindingType`, `BranchSide`, `LoadGenType`)
  are mapped by local tables. Keep it that way; the package is only needed by the tests.
- pgm stores TOTAL (lumped) line impedances with no length: every line is a virtual
  `length_m = 1.0` segment, so `r_per_m * length_m` is the pgm total.
- pgm has no base frequency field; `base_frequency_hz` must be passed by the caller.
- pgm has no load connection field: every converted load is WYE.
- `id_map` keys are pgm integer ids; only in-service elements are converted
  (`from_status`/`to_status` for branches, `status` otherwise).
  `id_map["load_types"]` keeps each load's original `LoadGenType` int.
- `load_model` is applied to every load (default `CONST_IMPEDANCE`, the linear reference
  solve); `"source"` keeps each load's own pgm type. Loads whose type differs from the
  applied model are named in the report.
- `link` (pgm's perfect connection, realised inside pgm by a 1e6 p.u. stand-in) converts
  to a closed `Switch` with R = L = 0; the solve fuses its rows, so pgml carries no
  stand-in. pgm reports symmetric results in three-phase quantities, so its link current
  is the pgml single-phase-equivalent current divided by sqrt(3).
- `voltage_regulator` on a converted `sym_gen` makes it a PV terminal
  (`VoltageRegulation`: `u_ref` -> `v_set_pu` on the node's rated voltage, `q_min`/`q_max`
  -> reactive limits); regulated generators sharing a node merge into one terminal
  (recorded as an approximation). A regulator on any other object is reported dropped.
- Transformer mapping is pinned against the installed power-grid-model C++ source
  (`transformer.hpp`, `branch.hpp`, `transformer_utils.hpp`) and live solves, not its docs:
  - `uk`/`pk`/`i0`/`p0` are defined on the EFFECTIVE (tap-adjusted) to-side voltage;
    `u_rated_from_v`/`u_rated_to_v` stay the untapped `u1`/`u2`; `tap.ratio_magnitude`
    carries only the off-nominal fraction. An absent/invalid `tap_side` behaves as
    to-side, like pgm.
  - Positive `clock` means TO lags FROM, the same sign as pgml (`shift_deg = clock*30`,
    no flip).
  - Leakage stored TO-coil-referred in both phase modes (x3 for a DELTA to-winding).
  - The magnetizing shunt is referred to the HV terminal by `(u2_eff/u1_eff)**2`; where
    it is stamped is `transformer.magnetizing_placement` (default `split`, which is pgm's
    own pi equivalent).
  - A grounded zigzag carries pgm's hardcoded zero-sequence `0.1*Z1` in
    `Transformer.zero_sequence`.
  - A nonzero grounding impedance passes through and is rejected at assembly
    (`ModelingError`), not at conversion.
- Source: Thevenin from `sk` + `rx_ratio` (`thevenin_from_sk`); zero sequence
  `Z0 = z01_ratio * Z1` (complex scaling) under `THREE_PHASE`, else the
  `source.zero_sequence.*` ratios. `id_map["slack_v_complex"]` comes from the first
  in-service source.
- Not read: `uk_min/uk_max/pk_min/pk_max` (uk/pk constant over the tap range),
  `i0_zero_sequence`/`p0_zero_sequence`. `three_winding_transformer`, `shunt`, `asym_gen`
  and `transformer_tap_regulator` are reported dropped.

## Known differences from power-grid-model (reported, not converter bugs)

- pgm reflects the from-side magnetizing half through the off-nominal tap; pgml's `split`
  uses the rated ratio (`model.transformer.magnetizing_tap_reflection`).
- The exporter writes an ideal switch as a 1 nOhm line and an ideal source as a large
  finite `sk`, because pgm accepts neither; both are recorded reductions.

## Outbound (`from_grid`)

- Scope: nodes, explicit R/L/C/G lines, switches, two-winding transformers, shunts,
  sources, loads, generators, storage; PV terminals export as pgm `voltage_regulator`.
- Geometry lines, unresolved `type_ref`, generic branches and source forms pgm cannot
  encode raise `UnsupportedGridError`. pgm has no ZIP element: ZIP, unbalanced P/Q,
  inverter controls and any dropped electrical term raise unless
  `allow_approximation=True`.
