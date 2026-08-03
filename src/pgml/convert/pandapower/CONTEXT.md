# Interface ledger: convert.pandapower

Converts a pandapower network (`pandapowerNet`) to our schema `Grid`.

## Public API

```python
from pgml.convert.pandapower import to_grid, PhaseMode

import pandapower as pp
import pandapower.networks as pn
net = pn.create_cigre_network_lv()
pp.runpp(net)

grid, id_map = to_grid(net)                                    # SINGLE_PHASE_EQUIV (default)
grid_3ph, id_map_3ph = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)
```

### Signature
```
to_grid(net: Any, *, phase_mode: PhaseMode = PhaseMode.SINGLE_PHASE_EQUIV) -> tuple[Grid, dict[str, Any]]
```

Pure function; `net` must already carry basic DataFrames (`bus`, `line`, `load`,
`ext_grid`); unmaterialised `std_type` line references are accepted as long as
explicit per-km parameters are present.

### `id_map` format
```python
{
    "bus":              {pp_bus_idx: Node.id, ...},
    "line":             {pp_line_idx: Line.id, ...},
    "trafo":            {pp_trafo_idx: Transformer.id, ...},
    "switch":           {pp_switch_idx: Switch.id, ...},   # et='b' bus-bus switches only (et='l'/'t' resolve into "line"/"trafo" in-service, not their own bucket)
    "load":             {pp_load_idx: Load.id, ...},
    "asymmetric_load":  {pp_asym_idx: Load.id, ...},        # THREE_PHASE only
    "sgen":             {pp_sgen_idx: Generator.id, ...},
    "ext_grid":         {pp_eg_idx: Source.id, ...},
    "slack_v_complex":  complex,  # FIRST ext_grid's phasor (V, L-L); ideal-slack convenience
}
```
`solve_power_flow(..., slack="ideal")` does NOT use `id_map["slack_v_complex"]` for
multi-ext_grid networks — it fixes every in-service `Source` appliance's own row
independently (`solver.power_flow._slack_rows_and_vref`), so multiple `ext_grid`s
in one network are each solved as an independent ideal-slack row with no
special-casing needed (verified on `mv_oberrhein`, which has 2).

## Transformer field mapping (two-winding, vector-group + tap-changer aware)

| pandapower field | Our schema field | Notes |
|---|---|---|
| `hv_bus`/`lv_bus` | `from_node`/`to_node` | `from`=HV, `to`=LV |
| `vn_hv_kv`/`vn_lv_kv` | `u_rated_from_v`/`u_rated_to_v` | ×1000 |
| `sn_mva`, `parallel` | `s_rated_va` | `sn_va·parallel` (single-unit rating × the identical-unit count) |
| `vk_percent`, `vkr_percent`, `sn_mva`, `vn_lv_kv`, `parallel` | `series_resistance_ohm`/`series_inductance_h` | `Z_base_LV=vn_lv_v²/sn_va` (single-unit `sn_va`, pandapower's own convention); `R_ll=vkr%·Z_base_LV`, `X_ll=√((vk%·Z_base_LV)²−R_ll²)`. Stored TO-COIL-referred: `3×` when `to_connection==DELTA`, unchanged otherwise (see "Delta-LV coil referral" below), then DIVIDED by `parallel` — see "`parallel`" below |
| `pfe_kw`, `i0_percent`, `vn_hv_kv`, `parallel` | `magnetizing_conductance_s`/`magnetizing_inductance_h` | HV-referred; `g_m=pfe_w/u_hv²·parallel`; `l_m` from `i0%`/`sn` divided by `parallel` (`None` when the no-load apparent power ≤ real power, e.g. a MATPOWER-synthesized negative `i0_percent` — see "case118" below) |
| `vector_group` (row or `std_types['trafo'][std_type]`) + `shift_degree` | `from_connection`/`to_connection` | see "Vector-group resolution" below |
| `tap_pos`/`tap_neutral`/`tap_step_percent`/`tap_side` | `tap.ratio_magnitude` | see "Tap changer" below |
| `shift_degree` | `tap.shift_deg` | pass-through (pandapower's own convention: positive ⇒ LV lags HV, identical to pgml's) |

### Vector-group resolution (`_resolve_transformer_connections`)

1. A vector-group string is read from `net.trafo['vector_group']` (when the column
   exists and is non-null for the row) else `net.std_types['trafo'][std_type]
   ['vector_group']` (`_vector_group_string`).
2. If found, `_parse_vector_group` splits it into `(from_connection, to_connection,
   clock)`: HV token (`YN`/`ZN`/`Y`/`D`/`Z`, longest-first) + LV token (`yn`/`zn`/`y`/
   `d`/`z`) + optional trailing clock digits, matched CASE-INSENSITIVELY by exact
   token (not literal HV-upper/LV-lower case) so unconventional casing still parses.
   The clock digits are OPTIONAL: pandapower's own `runpp_3ph` zero-sequence
   transformer model requires the BARE letter form (`'Dyn'`, `'Yzn'`, no digit — it
   explicitly rejects a digit-suffixed string, "specified in net.trafo.shift_degree",
   see `pandapower/pd2ppc_zero.py`), so a real network may carry `vector_group='Dyn'`
   with the clock only in `shift_degree`. A digit-bearing string is cross-checked
   against `shift_degree`'s implied clock (`round(shift_degree/30) % 12`); a mismatch
   raises `ConversionError` (pandapower's own balanced `runpp` uses only
   `shift_degree`, the string is otherwise-unread metadata, so silently preferring
   either source would disagree with the other for someone relying on it). A bare
   (clock-less) string skips the cross-check and combines directly with
   `shift_degree`.
3. With NO vector-group string anywhere (a plain MATPOWER import, e.g. `case118`, or
   any benchmark net that only stamps `shift_degree`): a non-multiple-of-30°
   `shift_degree` (a MATPOWER ideal phase shifter) falls back to
   `WYE_GROUNDED`/`WYE_GROUNDED` with the exact angle passed through unconstrained
   (exact under `SINGLE_PHASE_EQUIV`; a genuine 3-phase stamp rejects a
   non-multiple-of-30 shift); otherwise the connection is derived from the shift
   PARITY: an even clock (incl. `shift_degree==0`) -> `WYE_GROUNDED`/`WYE_GROUNDED`
   (a zero-sequence-transparent sequence-domain import — physically arbitrary but
   harmless for a positive-sequence-only study), an odd clock ->
   `DELTA`/`WYE_GROUNDED` (the physical Dyn reality of most MV/LV distribution
   transformers, e.g. CIGRE LV/MV).

### Tap changer (`_tap_ratio_magnitude`)

`delta = (tap_pos − tap_neutral) · tap_step_percent / 100`. `tap_side='hv'` ->
`ratio_magnitude = 1 + delta`; `tap_side='lv'` -> `ratio_magnitude = 1 / (1 + delta)`.
Verified against a live pandapower `runpp` on a 2-bus net
(`tests/convert/test_pandapower_vector_groups.py`): `tap_pos=+2` on the HV side
LOWERS the LV voltage (matches `ratio_magnitude>1`, since pgml's tap multiplies the
HV/LV ratio); `tap_pos=+2` on the LV side RAISES it. Any of
`tap_pos`/`tap_neutral`/`tap_step_percent`/`tap_side` missing (NaN or absent) means no
tap-changer configured -> `ratio_magnitude=1.0`. `tap_step_degree` nonzero, or
`tap_phase_shifter=True` (an ideal phase-shifter tap) is not modelled and raises
`ConversionError`.

### `parallel` — identical parallel systems (lines AND two-winding transformers)

pandapower's `parallel` column (default 1) is the count of electrically identical
systems wired in parallel; `pandapower.build_branch` divides the series impedance
by it and multiplies the shunt admittance and rated power by it
(`_calc_r_x_from_dataframe`/`_calc_y_from_dataframe`). The converter mirrors this
exactly (`_parallel_count`, NaN-safe, default 1.0):
- **Line**: `series_resistance_ohm_per_m`/`series_inductance_h_per_m` divide by
  `parallel`; `shunt_capacitance_f_per_m`/`shunt_conductance_s_per_m` (and the
  native zero-sequence `r0`/`x0`/`c0` when present) multiply/divide the same way.
- **Transformer**: the coil-referred `series_resistance_ohm`/`series_inductance_h`
  divide by `parallel` (applied AFTER the delta-LV coil factor, order-independent);
  `magnetizing_conductance_s` multiplies by `parallel`, `magnetizing_inductance_h`
  divides by it (so its susceptance `1/(2·pi·f0·L_m)` multiplies by `parallel`,
  matching the conductance); `s_rated_va` multiplies by `parallel`.
`parallel==1` (pandapower's own default) is an IEEE-754 exact no-op (`x/1.0==x`,
`x*1.0==x`), so every other converted network stays byte-identical.
`trafo3w`'s own `parallel` column is moot — three-winding units are not converted
at all (see "Coverage gaps" below). See
`tests/reference/test_pandapower_grid_matrix.py`'s
`test_parallel_line_and_transformer_matches_pandapower` (live-oracle power-flow
parity), `test_parallel_scales_series_impedance_and_shunt_admittance` (closed-form
value checks), and `test_parallel_default_is_byte_identical`.

### Bus-line / bus-transformer switches (`et='l'`/`'t'`) — accepted approximation

An OPEN `et='l'`/`'t'` switch (`net.switch`) converts its line/transformer as
out-of-service (`_open_switch_targets`, called once up front and consulted by
both the line and the trafo loop): the WHOLE element drops, not just the switched
terminal. This is an accepted approximation, not full fidelity: pandapower's own
solver instead keeps the still-connected terminal energized via an internal
auxiliary bus (dropping only the current path, not that terminal's shunt
admittance), so the converter's simplification also loses the line-charging
capacitance at that terminal. A closed switch, or no switch at all, changes
nothing; bus-bus (`et='b'`) switches are unaffected (see the field mapping table
below this section). Root-caused (not guessed) on `create_cigre_network_mv`/
`mv_oberrhein`: comparing a live `runpp` on the untouched net against the SAME net
with those lines forced `in_service=False` for pandapower itself shows a
~2.9e-4 pu / ~6.9e-3 deg (CIGRE MV) / ~8.9e-4 pu / ~1.4e-2 deg (`mv_oberrhein`)
difference — almost the entire residual the grid-matrix oracle tests see against
pandapower's OWN untouched `runpp` (see
`tests/reference/test_pandapower_grid_matrix.py`'s module docstring, third
tolerance family, and `test_cigre_mv_shift30_fallback`/
`test_mv_oberrhein_ynd5_delta_referral_and_taps`, which now exercise this
production path directly rather than a test-harness workaround).

### Delta-LV coil referral — PHASE-MODE INDEPENDENT (the important gotcha)

`series_resistance_ohm`/`series_inductance_h` are ALWAYS the TO-side COIL value:
`z_coil = 3·z_LL` when `to_connection==DELTA`, `z_coil = z_LL` otherwise, where
`z_LL = vk% · Z_base_LV` is the raw terminal (line-to-line-equivalent) quantity —
applied THE SAME WAY REGARDLESS OF `phase_mode`. This mirrors the (already correct)
`convert.pgm` and `convert.opendss` converters exactly (see their own CONTEXT.md
"coil referral" sections and `docs/pgml/modeling/conventions.md` §2) and is REQUIRED
because `assembly.ybus._transformer_block_groups` undoes the factor internally on
BOTH phase-mode paths, not just `THREE_PHASE`: the `p==1` (`SINGLE_PHASE_EQUIV`)
scalar-tap-pi branch has its own `k_ll = 3.0 if to_side.kind=="delta" else 1.0`
factor baked in (easy to miss reading that ~1400-line file — the FIRST version of
this converter applied the factor only under `THREE_PHASE`, which broke
`SINGLE_PHASE_EQUIV` by exactly 3x; caught via a live `mv_oberrhein` (YNd5) oracle
comparison, not by inspection). Do NOT reintroduce a `phase_mode` condition on this
factor — see `tests/reference/test_pandapower_grid_matrix.py`'s
`test_mv_oberrhein_delta_lv_referral_factor_applied_in_both_phase_modes` and
`test_mv_oberrhein_ybus_transformer_stamp_matches_pandapower` (the pinned regression
guards) and `.claude/agent-memory/reference-integrator/pandapower-transformer-
converter.md` (the full account of how the wrong, phase-mode-conditional version was
first written and then caught).

### Magnetizing branch — a real, expected T-vs-pi residual (not a bug)

Any transformer with `pfe_kw>0` or `i0_percent>0` (every Kerber std type,
`mv_oberrhein`, but NOT the CIGRE LV/MV trafos, which ship `pfe_kw=i0_percent=0`)
shows a genuine ~1e-5 pu / ~1e-2 deg residual against a live `runpp` (and ~1e-4
relative on the Y-bus off-diagonal entries directly). Root-caused (not guessed): a
live 2-engine comparison with `pfe_kw`/`i0_percent` zeroed on the SAME transformer
collapses the residual to machine precision. Cause: pandapower's default
`trafo_model="t"` places the magnetizing shunt INSIDE a T-equivalent (between two
leakage half-impedances, so it couples into the HV↔LV transfer admittance too);
`assembly._transformer` stamps it as a simple shunt at the external HV terminal
ONLY (a documented core simplification, outside the leakage transform). Identical
phenomenon to the already-documented OpenDSS magnetizing-branch gap (see
`src/pgml/convert/opendss/CONTEXT.md`) and the pgm magnetizing-split gap (see
`src/pgml/convert/pgm/CONTEXT.md`) — a genus of gap common to all three reference
tools, not specific to pandapower.

### `scaling` — per-element P/Q multiplier (read for `load`/`sgen`/`asymmetric_load`)

pandapower's own `runpp` multiplies each element's nameplate P/Q by its per-row
`scaling` column before solving (`res_load.p_mw = load.p_mw * load.scaling`).
`_scaling_factor` reads it NaN-safely (missing/None/NaN -> 1.0, pandapower's own
default). Most reference networks default to `scaling=1.0` everywhere (unaffected);
`mv_oberrhein` ships `load.scaling=0.6`, `sgen.scaling=0.0` by default — silently
ignoring this made every `mv_oberrhein` conversion wrong (61.86 MW converted instead
of pandapower's actual 37.116 MW, plus 22.07 MW of phantom DER injection) until
found via that grid's oracle comparison.

### Voltage-dependent (ZIP) loads (`_zip_coefficients`)

`net.load`'s `const_z_p_percent`/`const_i_p_percent`/`const_z_q_percent`/
`const_i_q_percent` columns (pandapower 3's per-load ZIP model — FOUR
independent percentages, not one shared P/Q pair; see
`pandapower.build_bus._calc_pq_elements_and_add_on_ppc`) map onto
`ZipCoefficients`: `z_p=const_z_p_percent/100`, `i_p=const_i_p_percent/100`,
`p_p=1-z_p-i_p` (and the same for Q). Passed to `build_load(zip_coefficients=...)`,
which sets `load_model=LoadModel.ZIP`. Honoured by pandapower's own `runpp`
whenever `voltage_depend_loads=True` (the default) and by pgml's NONLINEAR solver
(`device_current_injections`) — the linear const-Z assembler ignores it (uses base
P/Q only), same as any other `LoadModel`. All four percentages at zero
(pandapower's own default — a pure constant-power load) converts with NO
`zip_coefficients`/`load_model` set, so a plain load stays byte-identical to the
pre-ZIP-aware output. `net.asymmetric_load` has no ZIP percentage columns at all
(unaffected). See `tests/reference/test_pandapower_grid_matrix.py`'s
`test_mixed_zip_load_matches_pandapower` (live-oracle nonlinear-solve parity),
`test_mixed_zip_load_coefficients_mapped_correctly`, and
`test_zero_zip_percentages_stay_byte_identical`.

## Coverage gaps / known scope boundaries

- `gen` (PV / voltage-controlled buses) and `shunt` are NOT converted
  (`warn_dropped_elements`). `case118` (a 345/161/138 kV transmission benchmark)
  relies on 53 `gen` + 14 `shunt` for voltage support; a full nonlinear
  constant-power comparison is therefore NOT a valid oracle for that grid even
  after freezing pandapower's own converged `res_gen`/`res_shunt` P/Q as fixed
  `sgen`/`load` (the nonlinear solver converges, but to a badly wrong, inflated
  operating point at nearly every bus — a known "wrong solution branch"
  phenomenon from removing voltage-control feedback on a stressed network, not a
  transformer bug). `case118`'s transformer/tap conversion is instead validated
  via a LINEAR, exact Y-bus-stamp comparison against pandapower's own internal
  `Ybus` — see `tests/reference/test_pandapower_grid_matrix.py::
  TestCase118TransformerOnly`.
- Bus-LINE/bus-transformer (`et='l'`/`'t'`) switches (used by `mv_oberrhein` and
  `create_cigre_network_mv` to operate a meshed ring radially) ARE converted, but
  only via the accepted out-of-service approximation described in "Bus-line /
  bus-transformer switches" above — the still-connected terminal's shunt
  admittance is dropped along with the element, unlike pandapower's own
  auxiliary-bus model. Not a silent gap (`_open_switch_targets` is unconditional,
  applied to every network), but not full fidelity either.
- `trafo3w`, `impedance`, `ward`/`xward`, `dcline`, `storage`, `motor`,
  `asymmetric_sgen`: not converted (`warn_dropped_elements`).
- ext_grid zero/negative-sequence source impedance (`r0x0_max`/`x0x_max`) is not
  read; the `Source`'s zero-sequence impedance equals its positive-sequence value.
  Irrelevant to `slack="ideal"` solves (the slack fixes the exact 3-phase phasor
  set regardless of any Thevenin impedance) but relevant if a `norton`-mode slack
  or a short-circuit study is ever added.
- An ideal phase-shifter tap (`tap_step_degree` nonzero, or `tap_phase_shifter`
  True) is not modelled and raises `ConversionError`.

## Oracle test coverage

- `tests/convert/test_pandapower_vector_groups.py` — unit tests: vector-group
  string parsing (all forms, incl. the bare/clock-less form), `std_type` catalog
  lookup precedence, the mismatch error, tap ratio computation (incl. NaN handling,
  lv-side taps, `tap_step_degree` rejection).
- `tests/reference/test_pandapower_grid_matrix.py` — the grid-agreement matrix:
  `case33bw` (control), `case118` (Yy fallback + off-nominal taps, Y-bus-only),
  `create_cigre_network_lv`/`_mv` (Dyn1 fallback; `_mv`'s 3 open bus-line
  switches now exercise the production `_open_switch_targets` path directly, no
  test-harness workaround), 3x Kerber nets (Dyn5 from `std_type`), `mv_oberrhein`
  (YNd5 + taps + the delta-LV/scaling fixes + its 6 open bus-line switches, same
  production path), a hand-built Yzn5 net, a hand-built `parallel` net (line AND
  trafo, `parallel=2`/`parallel=3`), a hand-built mixed-ZIP-load net, plus a
  best-effort `runpp_3ph` asymmetric comparison (Dyn passes; Yzn is skipped with
  a documented definitional-gap reason).
- `tests/reference/test_cigre_lv_full_transformer.py`,
  `test_cigre_lv_pandapower.py`, `test_ieee33_pandapower.py`,
  `test_ieee33_power_flow_pandapower.py`, `tests/convert/test_phase_mode.py` — the
  pre-existing regression suite (kept green).

See `docs/pgml/modeling/conventions.md` §2/§3/§5/§9 for the cross-tool convention
record and `docs/pgml/modeling/transformer.md` for the core vector-group model this
converter targets.
