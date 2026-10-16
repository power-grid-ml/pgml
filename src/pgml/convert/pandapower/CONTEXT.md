# convert.pandapower

pandapower network (`pandapowerNet`) ↔ pgml `Grid`. Fundamental-frequency, balanced
pandapower model on one side, the frozen grid schema on the other.

Signatures: docstrings / API_INDEX.md.

## Codemap

- `converter.py`: inbound `to_grid` plus `GenMode` and `DEFAULT_GEN_VOLT_VAR_SLOPE_PU`.
  It is one long function with numbered sections (nodes, lines, transformers, switches,
  loads, sgen, gen, storage, shunt, ext_grid, dropped tables). The per-element mapping
  rules are in the helper docstrings: `_resolve_transformer_connections` /
  `_parse_vector_group` (vector groups), `_tap_ratio_magnitude` (taps and
  `tap_changer_type`), `_parallel_count`, `_scaling_factor`, `_zip_coefficients`,
  `_open_switch_targets`, `_closed_switch_resistance_ohm`, `_shunt_admittance`,
  `_gen_raw_reactive_bounds` (exact PV), `_gen_volt_var_control` /
  `_gen_reactive_bounds` / `_gen_reactive_envelope` (droop approximation),
  `_sgen_controllers` / `_der_controller_inverter_control` (pandapower `DERController`),
  `_ext_grid_zero_sequence`, `_transformer_zero_sequence`.
- `exporter.py`: outbound `from_grid` → `PandapowerExport` (`net` plus Grid-id ↔
  pandapower-index maps and the named `reductions`). Balanced fundamental export only;
  what raises and what needs `allow_approximation=True` is in the `from_grid` docstring.
- Shared scaffold: `pgml.convert._common` (phase mode, builders, dropped-element
  warning). Tool-difference catalogue: `pgml.convert._model_differences` (`PANDAPOWER`),
  surfaced through the `ConversionReport`.

## Contract surface

- `to_grid(net, ...)` is pure (the `net` is not mutated). It returns `(Grid, id_map)`,
  or `(Grid, id_map, ConversionReport)` with `return_report=True`. The `Grid` is
  materialised: it carries no `type_ref`.
- `id_map` buckets are `bus`, `line`, `trafo`, `switch` (closed bus-bus switches only),
  `load`, `asymmetric_load` (THREE_PHASE only), `sgen`, `gen`, `shunt`, `storage`,
  `ext_grid`, `open_terminal` and `slack_v_complex`. `gen` is empty under
  `GenMode.DROP`, and rows merged on one bus share one id. `et='l'`/`'t'` switches have
  no bucket of their own. `slack_v_complex` is the first ext_grid's phasor in V, L-L.
  Downstream packages index results through these maps.
- `ideal` slack fixes every in-service `Source` row on its own. A network with several
  `ext_grid`s (`mv_oberrhein` has 2) therefore needs no special case, and
  `slack_v_complex` is only a convenience.

## Invariants and conventions

- Units: kV→V, km→m, Ω/km→Ω/m, MW/MVAr→W/var. Reactance becomes inductance at the
  network's `f_hz`.
- `SINGLE_PHASE_EQUIV` (default) nodes carry `u_rated_v = vn_kv·1000` (L-L), which
  matches pandapower's per-unit base. `vm_pu` setpoints and droop axes use the same
  base in both phase modes.
- Transformer orientation: `from` = HV, `to` = LV. `shift_degree` passes through
  unchanged: positive means the LV side lags the HV side, which is pgml's convention.
- **Delta-LV coil referral is phase-mode independent.** `series_resistance_ohm` and
  `series_inductance_h` are always the TO-coil value: 3× the vk%-derived terminal value
  when `to_connection == DELTA`. Assembly undoes this on both paths: the `p == 1`
  scalar pi through `k_ll` in `_transformer_block_groups`, and the `p == 3` incidence
  stamp. Never condition this factor on `phase_mode`. Guards:
  `test_mv_oberrhein_delta_lv_referral_factor_applied_in_both_phase_modes` and
  `test_mv_oberrhein_ybus_transformer_stamp_matches_pandapower`.
- `parallel` follows `pandapower.build_branch`. It divides the series impedance and
  multiplies the shunt admittance and the rating. Because `magnetizing_inductance_h` is
  an inductance, it *divides*. The coil factor and `parallel` commute.
- Vector group: a clock-bearing string must agree with `shift_degree`, otherwise
  `ConversionError`. A bare string (`'Dyn'`, the form `runpp_3ph` requires) takes its
  clock from `shift_degree`. With no string at all: a non-30° shift gives YNyn
  (MATPOWER phase shifter, exact only in SINGLE_PHASE_EQUIV), an even clock gives YNyn,
  and an odd clock gives Dyn.
- Taps act only where pandapower 3 applies them. An unset `tap_changer_type` means no
  tap, and a WARNING is logged when such a tap is off neutral. A table without the
  column (pandapower < 3 data) applies the tap. Angle taps (`tap_step_degree`,
  `tap_phase_shifter`, `"Ideal"`) raise.
- `scaling` multiplies the P/Q of `load`/`sgen`/`asymmetric_load`/`storage` and the
  `p_mw` of `gen`, as `runpp` does. It does not multiply `gen` reactive limits, which
  `add_q_constraints` reads raw.
- ZIP loads use pandapower 3's four independent columns
  (`const_{z,i}_{p,q}_percent`), not the two shared 2.x columns. All zero means no
  `zip_coefficients` is set.
- Sign conventions: `storage` is consumption-positive in pandapower and
  discharge-positive in pgml (`p_nom_w = −p_mw·1e6·scaling`). `sgen.q_mvar > 0` is
  overexcited, which matches pgml. `net.shunt` with `q_mvar < 0` becomes a capacitance
  and `q_mvar > 0` an inductance, so `|B(h)|` follows `h` or `1/h`. A negative
  capacitance would be wrong by `h²` at harmonic orders.
- Storage energy fields (`max_e_mwh` → `energy_capacity_wh`, `soc_percent` → `soc`,
  `min_e_mwh` → `soc_min`) record state and bounds only. They do not change the
  snapshot P/Q solve.
- ext_grid: Thevenin `Z1` is near-ideal (1e-6 Ω). `Z0` comes from
  `x0x_max`/`r0x0_max`/`s_sc_max_mva`/`rx_max` when present, stored physically
  (`c = 1`; pandapower's internal zero-sequence shunt is 1.1× this). Otherwise the
  `source.zero_sequence.*` defaults apply with a WARNING. `Z2 = Z1`.

## Decisions an edit must not undo

- `gen_mode` default `VOLTAGE_REGULATING` is the exact PV row (`|V|² − V_set²`).
  `VOLT_VAR_APPROX` is a droop for studies that want a droop law. It is not an import
  path for transmission benchmarks: it can converge silently onto a collapsed branch.
- A `gen` row on an in-service ext_grid bus is skipped and its `p_mw` dropped, with a
  WARNING. A `slack=True` row elsewhere converts with fixed P under
  `VOLTAGE_REGULATING` and is skipped under `VOLT_VAR_APPROX`. There is no distributed
  slack.
- Several `gen` rows on one bus merge into one regulating generator. The first
  `vm_pu` wins, with a WARNING if the rows differ. A missing reactive limit stays
  unbounded in exact mode. `min_q == max_q` gives a plain PQ generator.
- The controlled droop row keeps `q_nom_var = 0`. Seeding the warm start from the
  reactive range moves the Newton basin; it does not enlarge it.
- A closed bus-bus switch with `z_ohm = 0` defaults to an ideal switch (R = L = 0,
  fused by `assembly.fusion_map`), which is what pandapower does by merging buses.
  `branch.switch_model: near_ideal` keeps it a stamped branch for `branch_states`
  sweeps and logs a WARNING. A non-zero `z_ohm` is read as resistance only (see
  limitations).
- `open_switch_model` default `terminal`: a singly-open `et='l'`/`'t'` element stays
  connected at its other end through an auxiliary node, which reproduces pandapower's
  auxiliary bus. `drop_element` omits the whole element. An element open at both ends
  is always omitted.
- Unmapped tables are reported, never silently dropped (`warn_dropped_elements` plus
  the report): `trafo3w`, `impedance`, `ward`, `xward`, `dcline`, `motor`,
  `asymmetric_sgen`, `measurement`. Every controller except a `DERController` on `sgen`
  with a mapped Q-model is reported as `dropped.controller`.

## Tests

`tests/convert/test_pandapower_*.py` (mapping units) and
`tests/reference/test_pandapower_*.py` (live `runpp` oracles: grid matrix, PV bus, gen
droop, DER control, source zero sequence), `test_cigre_lv_*`, `test_ieee33_*pandapower*`.
Cross-tool conventions: `docs/pgml/modeling/conventions.md`. PV bus and droop:
`docs/pgml/modeling/der-pv-storage.md`. Magnetizing placement:
`docs/pgml/modeling/transformer.md`.
