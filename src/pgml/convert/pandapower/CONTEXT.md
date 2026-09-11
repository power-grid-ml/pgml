# Interface ledger: convert.pandapower

Converts a pandapower network (`pandapowerNet`) to our schema `Grid`.

## Public API

```python
from pgml.convert.pandapower import to_grid, GenMode, PhaseMode

import pandapower as pp
import pandapower.networks as pn
net = pn.create_cigre_network_lv()
pp.runpp(net)

grid, id_map = to_grid(net)                                    # SINGLE_PHASE_EQUIV (default)
grid_3ph, id_map_3ph = to_grid(net, phase_mode=PhaseMode.THREE_PHASE)

# net.gen (PV buses) is converted EXACTLY by default; the other two modes stay available
grid_pv, id_map_pv = to_grid(pn.case118())                     # VOLTAGE_REGULATING
grid_approx, _ = to_grid(pn.case118(), gen_mode=GenMode.VOLT_VAR_APPROX)
grid_nogen, _ = to_grid(pn.case118(), gen_mode=GenMode.DROP)
```

### Signature
```
to_grid(net: Any, *,
        phase_mode: PhaseMode = PhaseMode.SINGLE_PHASE_EQUIV,
        gen_mode: GenMode = GenMode.VOLTAGE_REGULATING,
        gen_volt_var_slope_pu: float = DEFAULT_GEN_VOLT_VAR_SLOPE_PU,  # 500.0
        ) -> tuple[Grid, dict[str, Any]]
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
    "gen":              {pp_gen_idx: Generator.id, ...},  # EMPTY under gen_mode=DROP; rows on ONE bus share the merged id
    "shunt":            {pp_shunt_idx: ShuntAppliance.id, ...},
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

`tap_changer_type` decides WHETHER the tap acts, following pandapower 3's own rule
(`build_branch._calc_nominal_ratio_from_dataframe`): `"Ratio"` and `"Symmetrical"`
move the tapped side's nominal voltage by `delta` (they differ only through
`tap_step_degree`, which raises here, so both reduce to the same real ratio);
`"Ideal"` shifts the angle only and raises; a row whose type is UNSET (NaN / empty)
gets NO tap, however far from neutral its `tap_pos` sits — pandapower ignores it, so
this converter does too, and logs a WARNING naming the transformer whenever an
off-neutral position is dropped that way (set the column on the source net if the tap
is meant to act). A `trafo` table WITHOUT the column at all (pandapower < 3.0 data)
keeps the legacy behaviour and applies the tap. An unrecognised non-empty value raises.
Datasets written before the column exists (SimBench, for example) carry tap positions
their own solver ignores: reading them as active moved `1-MV-urban--0-sw` by 1.65e-2 pu
and a two-bus 110/20 kV unit at `tap_pos=-1` with a 1.5 % step by the full tap step.
Tests: `tests/convert/test_pandapower_tap_changer_type.py`.

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
guards against exactly this mistake).

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

### `gen` (PV buses) — `gen_mode`

`net.gen` is pandapower's PV bus: fixed `p_mw`, regulated voltage MAGNITUDE
`vm_pu`, reactive power free between `min_q_mvar` and `max_q_mvar`. The solver
implements exactly that (`solver/_pv_bus.py`: the terminal's reactive
power-balance row becomes `|V|² − V_set²`), so `gen_mode` selects between the exact
model and two legacy behaviours:

| `gen_mode` | behaviour |
|---|---|
| `GenMode.VOLTAGE_REGULATING` (**default**) | Each in-service row becomes a `Generator` carrying a `VoltageRegulation` block — the EXACT PV terminal. The solver holds `vm_pu` and solves the reactive power within the row's limits. |
| `GenMode.VOLT_VAR_APPROX` | Each in-service row becomes a `Generator` whose `VoltVarControl` is a steep `Q(|V|)` droop centred on `vm_pu` and saturating at the row's reactive limits (the earlier approximation; kept for a DER study that wants a real droop law). |
| `GenMode.DROP` | `net.gen` is not read; `warn_dropped_elements` reports it. The converted `Grid` is identical to converting the same net with the `gen` table emptied. |

**The exact mapping** (`GenMode.VOLTAGE_REGULATING`), per in-service row not caught
by the slack rule below:

| pandapower field | Our schema field | Notes |
|---|---|---|
| `p_mw`, `scaling` | `Generator.p_nom_w` | `p_mw·1e6·scaling` (pandapower's own build scales `gen.p_mw` too) |
| `vm_pu` | `VoltageRegulation.v_set_pu` | the SAME per-unit base: pgml's setpoint is per unit of the host node's rated voltage, and `phase_voltage_magnitude` reduces that to the L-N base the solver regulates, which equals pandapower's `vm_pu` in BOTH phase modes. A NaN `vm_pu` converts as 1.0 |
| `min_q_mvar` / `max_q_mvar` | `VoltageRegulation.q_min_var` / `q_max_var` | `·1e6`, read RAW (pandapower's `add_q_constraints` does not scale them); a MISSING (NaN) limit stays `None` = UNBOUNDED, which is how pandapower reads it (`q_lim_default` 1e9 MVAr, and its `runpp` default `enforce_q_lims=False` ignores the limits entirely) |
| — | `Generator.q_nom_var` | `0.0` — a regulating machine's reactive power is solved, not set |

Several in-service rows on ONE bus MERGE into a single regulating generator (one
bus carries one voltage setpoint, as in pandapower's own per-bus build): active
powers and limits add, an unbounded side stays unbounded, the first row's `vm_pu`
wins and a differing one logs a WARNING, and every merged row maps to the one
`Generator.id` in `id_map["gen"]`. Reactive limits are enforced by the solver
(`solve_power_flow(enforce_q_limits=...)`, default from `pgml.defaults`), NOT by the
converter. Measured agreement against `pp.runpp` on the MATPOWER benchmarks: see
"Accuracy" below and `tests/reference/test_pandapower_pv_bus.py`.

**The Volt-VAr approximation's mapping** (`_gen_volt_var_control`, under
`GenMode.VOLT_VAR_APPROX`), per in-service row not caught by the slack rule below:

| pandapower field | Our schema field | Notes |
|---|---|---|
| `p_mw`, `scaling` | `Generator.p_nom_w` | `p_mw·1e6·scaling` — the same generation-positive, `scaling`-aware convention as `sgen` (pandapower's own `build_gen` scales `gen.p_mw` too) |
| `vm_pu` | droop centre | the x-axis is `|V_terminal| / V0` with `V0` the node's L-N (or L-L for a delta element) nominal, which equals pandapower's `vm_pu` in BOTH phase modes |
| `min_q_mvar` / `max_q_mvar` | curve saturation levels | read RAW (NOT multiplied by `scaling` — pandapower's `add_q_constraints` does not scale them) |
| — | `VoltVarControl.s_rated_va` | `hypot(P, max(|q_min|,|q_max|))` per element, so the capability circle's reactive headroom `sqrt(S²−P²)` equals the widest limit exactly: the circle never binds before the curve, and the curve applies the ASYMMETRIC `[q_min, q_max]` saturation |
| — | `Generator.q_nom_var` | `0.0` for a controlled row (a controlled appliance's reactive nameplate is never read by the nonlinear solve) |

The curve is the two endpoints of the clamped droop line —
`Q(|V|) = clamp(−slope·q_base·(v_pu − vm_pu), q_min, q_max)` — stored as
`x_values = (vm_pu − y_max/slope, vm_pu − y_min/slope)`,
`y_values = (y_max, y_min)` with `y = Q/q_base`, `interpolation="linear"`,
`extrapolation="constant"` (the constant extrapolation IS the clamp).
`smoothing=0` (the exact hard clamp — a Q limit is a hard limit).
Per-phase: the control is evaluated per ELEMENT against that element's share of
the active power, so the limits and the rating divide by the phase count
(1 under `SINGLE_PHASE_EQUIV`, 3 under `THREE_PHASE`).

**`gen_volt_var_slope_pu`** (default `DEFAULT_GEN_VOLT_VAR_SLOPE_PU = 500.0`) is
the droop STEEPNESS in units of the reactive base per per-unit terminal voltage:
the droop sweeps one full `q_base` over `1/slope` pu of voltage. The regulated bus
settles off its setpoint by `Q_actual / (slope·q_base)` pu, so the |V| error falls
as `1/slope` — measured, see "Accuracy" below.

**Fallbacks for a missing reactive limit** (`_gen_reactive_bounds` /
`_gen_reactive_envelope`). pandapower leaves `min_q_mvar`/`max_q_mvar` NaN on a
generator created without limits and its own `runpp` then treats the machine as
effectively unbounded (`q_lim_default = 1e9` MVAr; the DEFAULT `enforce_q_lims=False`
ignores the limits entirely). An unbounded range is unusable here — the droop needs
a FINITE reactive base to size the circle and scale the slope — so a missing bound
falls back to ± a machine-sized envelope, in order:
1. `sqrt(sn_mva² − p²)` when `sn_mva` is set (the rating itself if `|P| ≥ sn`);
2. `|p_mw|` when there is no rating (a machine able to run to a 0.707 power factor);
3. `_GEN_ENVELOPE_FLOOR_VAR = 1.0 var` when the row carries no size information at
   all (no limits, no rating, no active power) — such a row regulates nothing.
Every conversion that used a fallback logs one aggregated WARNING with the count.
`min_q_mvar == max_q_mvar` is NOT a PV bus (no reactive freedom): the row converts
as a PLAIN PQ `Generator` with `q_nom_var` = that value and NO control.
`min_q_mvar > max_q_mvar` raises `ConversionError`.

**SLACK-BUS RULE.** A row on a bus that already carries an in-service `ext_grid` is
SKIPPED and logged at WARNING (its `p_mw` really is dropped). The ideal slack fixes
that bus's voltage phasor outright, so a regulator there would work against an
infinitely stiff reference; pandapower's own solve treats a generator at the
reference bus the same way, absorbing its `p_mw` into the slack dispatch rather than
injecting it. A row flagged `slack=True` away from the `ext_grid` bus (pandapower's
distributed slack) is SKIPPED under `VOLT_VAR_APPROX` and CONVERTED under
`VOLTAGE_REGULATING` with its `p_mw` fixed — pgml has one reference, so such a row
does not share the slack's active-power imbalance; both paths log it. The solver
additionally REFUSES a regulating generator on a Source's node
(`ModelingError`).

**What the approximation does and does NOT give.**
- It holds `|V|` APPROXIMATELY, never exactly. Steeper trades conditioning for
  fidelity, in direct proportion, and the trade has a hard ceiling (below).
- Reactive limits are enforced by the curve saturation (backed by the capability
  circle), so a limit-hitting generator behaves like pandapower's discrete PV→PQ
  switch only APPROXIMATELY — the switch happens along the last droop segment
  rather than as a discrete bus-type change.
- Active power is a fixed injection (pandapower's own model). No distributed
  slack, no active-power limit enforcement.
- **Conditioning / solution branch is the real limit.** Outside the `1/slope`-wide
  band the droop's `dQ/dV` is exactly zero, so a Newton iterate that starts far from
  the setpoint sees NO voltage-control feedback. On a heavily loaded transmission
  benchmark the linear const-Z warm start is far enough out that the solve either
  fails or converges onto the COLLAPSED low-voltage branch (every generator pinned
  at its maximum Q and the network still at ~0.6 pu — a genuine second solution of
  the approximated system, which a true PV row `|V| − V_set = 0` would exclude by
  construction). Non-convergence, or an implausibly low converged profile, is the
  signal to REDUCE `gen_volt_var_slope_pu`. Use `method="newton"`; the
  current-injection fixed point does not contract on a stiff droop.
  A controlled row's `q_nom_var` is 0.0 — the nonlinear solve never reads it, but
  the LINEAR const-Z assembler that builds Newton's warm start DOES, so a nonzero
  value is a free knob on the STARTING POINT. It was measured and REJECTED: seeding
  the midpoint of each row's reactive range rescues `case39` (4.9e-1 -> 1.7e-3 pu at
  slope 200) but breaks `case118` (slope 5 stops converging; slope 20 converges to a
  profile 8.1e-1 pu off at its worst bus). It MOVES the basin, it does not ENLARGE
  it, so it is not baked in — the only dependable fix is the residual row.

**Accuracy of the EXACT mode** (max per-bus |V| / angle deviation against a live
`pp.runpp(net, calculate_voltage_angles=True, tolerance_mva=1e-10)` on the benchmark
AS PUBLISHED — every element in service, nothing approximated; `method="newton"`,
`SINGLE_PHASE_EQUIV`, complex128, pgml `tol=1e-6` V; i7-12700, CPU, float64):

| case | buses | gen | shunt | max &#124;dV&#124; [pu] | max &#124;dθ&#124; [deg] | max &#124;dQ_gen&#124; [Mvar] | Newton its | pgml / runpp [ms] |
|---|---|---|---|---|---|---|---|---|
| case9 | 9 | 2 | 0 | 4.4e-16 | 7.1e-15 | 3.1e-13 | 4 | 30 / 50 |
| case14 | 14 | 4 | 1 | 6.2e-12 | 3.9e-10 | 7.8e-9 | 5 | 27 / 41 |
| case30 | 30 | 5 | 2 | 8.9e-12 | 6.4e-10 | 7.9e-9 | 5 | 100 / 42 |
| case39 | 39 | 9 | 0 | 1.8e-15 | 3.7e-14 | 6.6e-12 | 10 | 277 / 58 |
| case57 | 57 | 6 | 3 | 3.1e-15 | 2.2e-13 | 1.9e-12 | 5 | 291 / 154 |
| case118 | 118 | 53 | 14 | 6.7e-3 | 3.7e-2 | 8.0e+1 | 6 | 173 / 37 |
| case300 | 300 | 68 | 29 | 1.0e-2 | 3.8e-1 | 1.1e+2 | 8 | 2055 / 207 |

`case14`/`case30` sit at pandapower's own `tolerance_mva=1e-10` floor, not at a
modelling difference. `case118` and `case300` are the ONLY two benchmarks with a
deviation above 1e-11, and it is entirely the transformer MAGNETIZING-BRANCH
placement: 4 of case118's and 18 of case300's transformers carry a nonzero
`i0_percent` (MATPOWER's branch charging susceptance on a ratio branch, which
`from_ppc` stores as a negative magnetizing current), and pgml stamps the
magnetizing admittance on the external HV terminal while pandapower splits it across
the pi-model. Zeroing `i0_percent`/`pfe_kw` in BOTH tools removes the whole
deviation: case118 6.7e-16 pu / 2.6e-13 deg, case300 3.0e-14 pu / 5.8e-12 deg, with
the generator reactive powers agreeing to 1.2e-11 Mvar. With
`enforce_q_lims=True` in both tools the same two benchmarks agree to 1.2e-15 pu and
the SAME generators switch to PQ (case39: 1 of 9; case118: 6 of 53) at the same
reactive power (7.3e-12 Mvar). `case300` with limits enforced is not comparable:
pandapower's own solve does not converge there (50 iterations), and pgml does not
either.

**Accuracy of the Volt-VAr approximation** (max / median per-bus |V| deviation in pu
against a live `pp.runpp(net, enforce_q_lims=True)` — the apples-to-apples oracle for
that mode, since the
converter enforces the reactive limits while pandapower's DEFAULT
`enforce_q_lims=False` does not; shunt-carrying nets have `net.shunt` disabled in
BOTH tools so the comparison isolates the `gen` mapping from the unconverted
`shunt` table; `method="newton"`, `SINGLE_PHASE_EQUIV`, complex128):

| slope | case9 | case14 | case57 | case118 |
|---|---|---|---|---|
| 5 | 6.6e-3 / 9.6e-4 | 9.4e-2 / 7.6e-2 | 4.3e-2 / 3.3e-2 | 4.4e-2 / 2.1e-2 |
| 50 | 8.2e-4 / 2.1e-4 | 1.7e-2 / 1.5e-2 | 6.8e-3 / 4.7e-3 | (no convergence) |
| 500 | 8.4e-5 / 2.3e-5 | 1.8e-3 / 1.5e-3 | 7.6e-4 / 5.8e-4 | (no convergence) |
| 2000 | (no convergence) | 4.6e-4 / 3.9e-4 | 2.3e-4 / 1.6e-4 | (no convergence) |

The `1/slope` law holds wherever the solve lands on the correct branch; the
convergence failures are basin-of-attraction flukes, not a monotone ceiling
(case9 fails at 200 and 2000 but succeeds at 500). `case_ieee30` is an instructive
edge: above slope ~20 EVERY generator is pinned at a reactive limit, the problem
degenerates to the same pure-PQ system pandapower's `enforce_q_lims` solves, and
the agreement is exact (1.4e-10 pu) and slope-independent.

Two calibrating measurements:
- **The unconverted `shunt` table is NOT what dominates `case118`.** Re-running the
  same comparison with its 14 shunts LEFT IN pandapower (so the converter's gap is
  live) moves the numbers only from 8.1e-2 to 8.9e-2 pu (slope 2) and 4.4e-2 to
  5.1e-2 pu (slope 5). The residual is the PV-bus approximation, not the shunts.
- **The network conversion itself is exact.** Freeze every generator at
  pandapower's OWN converged `(P, Q)` as an `sgen` row and drop the `gen` table:
  the converted grid then reproduces `runpp` to 2.1e-13 pu on `case39` and 9.8e-11
  pu on `case9`. Lines, transformers, taps and the slack are not the error source.
- **`case39` is the cautionary case.** At every slope from 5 upward it CONVERGES —
  onto the collapsed branch, 4.9e-1 pu away, with all nine generators pinned at
  their maximum reactive limit. A converged-but-wrong answer, not a loud failure.
  Sanity-check the converged voltage profile against the source network's
  `res_bus` whenever this mode is used on a transmission grid.

**Verdict on the approximation.** Good for small and moderately loaded networks (a
sub-1e-3 pu operating point at slope 500 on case9/case14/case57) and as a
differentiable stand-in for a real droop-controlled DER. NOT a way to import
transmission benchmarks: on `case118` the branch problem caps the usable steepness at
~5 (a 2-5 % voltage error), and on `case39` every steepness converges silently onto
the collapsed branch. That is what `VOLTAGE_REGULATING` (the default) exists for —
the same `case39` agrees with `runpp` to 1.8e-15 pu through the exact row pair.

Tests: `tests/convert/test_pandapower_pv_bus_shunt.py` (the exact mapping, the merge,
the slack rule, `net.shunt`), `tests/reference/test_pandapower_pv_bus.py` (live
`runpp` oracle on all seven benchmarks, with and without reactive limits),
`tests/convert/test_pandapower_gen_volt_var.py` and
`tests/reference/test_pandapower_gen_volt_var.py` (the approximation's mapping,
fallbacks and steepness sweep).

### `shunt` (fixed bus admittance)

`net.shunt` is a fixed admittance at its bus. Each in-service row becomes a WYE
`ShuntAppliance` on the host node:

| pandapower field | Our schema field | Notes |
|---|---|---|
| `p_mw`, `step`, `vn_kv` | `ShuntAppliance.conductance_s` | `G = p_mw·1e6·step / (vn_kv·1e3)²`, referred to the SHUNT's own rated voltage (`vn_kv`, defaulting to the bus's) — algebraically identical to pandapower's `(G, B) = (p, −q)·step·(vn_bus/vn_shunt)²` per unit on the bus base (`build_bus._calc_shunts_and_add_on_ppc`) |
| `q_mvar < 0`, `step`, `vn_kv` | `ShuntAppliance.capacitance_f` | a CAPACITIVE bank: `C = −q_mvar·1e6·step / (vn_kv·1e3)² / (2πf0)`, `inductance_h = None` |
| `q_mvar > 0`, `step`, `vn_kv` | `ShuntAppliance.inductance_h` | a REACTOR: `L = (vn_kv·1e3)² / (2πf0·q_mvar·1e6·step)`, `capacitance_f = 0` |
| `in_service` | — | an out-of-service row is skipped |

Under `THREE_PHASE` the per-phase value is the same number repeated over A/B/C (a
balanced bank's per-phase admittance equals its positive-sequence value). The reactive
part becomes the reactive ELEMENT it is, so its susceptance carries the right frequency
trend: `|B(h)| = h·B` for a capacitor bank, `B/h` for a reactor. Both forms reproduce the
fundamental admittance exactly, so no load-flow result depends on the choice (verified on
the MATPOWER benchmarks, whose shunts include 6 inductive rows in `case118` and several in
`case300`); only the harmonic orders differ, and a negative capacitance would get them
wrong by `h²`. The conversion log names how many converted rows are inductive.

Tests: `tests/convert/test_pandapower_pv_bus_shunt.py`.

## Coverage gaps / known scope boundaries

- `case118`'s transformer/tap conversion is ALSO validated independently of the
  power flow, via a LINEAR exact Y-bus-stamp comparison against pandapower's own
  internal `Ybus` — see
  `tests/reference/test_pandapower_grid_matrix.py::TestCase118TransformerOnly`. A
  network whose generators are frozen at pandapower's OWN converged `(P, Q)` as
  `sgen` rows reproduces pandapower to 2e-13 pu on `case9` and `case39`, so the
  network conversion itself (lines, transformers, taps, the slack) is exact.
- `gen` rows at the `ext_grid` bus are dropped by design (see the slack-bus rule);
  pandapower's DISTRIBUTED slack (`slack=True` with `slack_weight`) is not modelled —
  such a row converts with a FIXED active power.
- Bus-LINE/bus-transformer (`et='l'`/`'t'`) switches (used by `mv_oberrhein` and
  `create_cigre_network_mv` to operate a meshed ring radially) ARE converted, but
  only via the accepted out-of-service approximation described in "Bus-line /
  bus-transformer switches" above — the still-connected terminal's shunt
  admittance is dropped along with the element, unlike pandapower's own
  auxiliary-bus model. Not a silent gap (`_open_switch_targets` is unconditional,
  applied to every network), but not full fidelity either.
- trafo ZERO-SEQUENCE leakage IS read: `vk0_percent`/`vkr0_percent` become
  `Transformer.zero_sequence` (per-unit values are base-invariant, so the same
  `Z = vk0% · Z_base_LV` formula the positive sequence uses applies, including the
  TO-side delta coil factor and the `parallel` divide). A zero or absent `vk0_percent` is
  pandapower's own "use the positive-sequence value", which is pgml's
  `transformer.zero_sequence.*` default; the grounded-wye/zigzag units that fall back are
  reported in ONE WARNING per conversion carrying the count and their indices
  (`_warn_defaulted_zero_sequence`). NOT modelled, each named in a WARNING when set: `mag0_percent`/`mag0_rx`
  (a finite zero-sequence MAGNETIZING impedance — the three-limb-core path through tank
  and air), `si0_hv_partial` (the HV/LV split of the zero-sequence leakage inside a T) and
  `xn_ohm`/`rn_ohm` (a neutral earthing impedance, `3·Z_N` in series).
- `trafo3w`, `impedance`, `ward`/`xward`, `dcline`, `storage`, `motor`,
  `asymmetric_sgen`: not converted (`warn_dropped_elements`).
- ext_grid NEGATIVE-sequence source impedance is not represented separately: the
  converted `Source` is one physical Thevenin with `Z2 = Z1` (a passive upstream
  network), whereas pandapower's own `runpp_3ph` pins the positive sequence as an
  ideal slack and puts its short-circuit impedance in the NEGATIVE-sequence network
  only. The converter follows pandapower's positive-sequence behaviour (near-ideal
  `Z1 = 1e-6` Ohm), so the negative-sequence boundary differs by that
  short-circuit impedance — negligible for a stiff feed, visible on a weak one
  (measured 5e-6 pu on the 2-bus case of
  `tests/reference/test_pandapower_source_zero_sequence.py` at `s_sc_max_mva = 1000`).
- ext_grid ZERO-sequence source impedance IS read: `x0x_max`/`r0x0_max` together
  with `s_sc_max_mva`/`rx_max` give `X0 = x0x_max * X1`, `R0 = r0x0_max * X0` with
  `X1 = (U_LL^2/S_sc)/sqrt(1+rx_max^2)`, carried into the `Source`'s per-phase
  matrix through `Z_self=(Z0+2*Z1)/3`, `Z_mutual=(Z0-Z1)/3`
  (`_ext_grid_zero_sequence`). pandapower multiplies its OWN zero-sequence shunt by
  the IEC voltage factor `c = 1.1` even in power-flow mode
  (`pd2ppc_zero._add_ext_grid_sc_impedance_zero`); pgml stores the physical
  impedance (`c = 1`), so pandapower's internal value is exactly 1.1x pgml's
  (asserted in the oracle test). Without the short-circuit columns (pandapower's
  own default: NaN) the documented `source.zero_sequence.*` ratios apply and a
  WARNING names the ext_grid. Irrelevant to `slack="ideal"` solves (the slack fixes
  the exact 3-phase phasor set regardless of any Thevenin impedance); it shapes
  `slack="norton"` solves and every harmonic order, where the source is a Norton
  shunt.
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
- `tests/convert/test_pandapower_gen_volt_var.py` — unit tests for the `net.gen`
  Volt-VAr approximation: the default drop (and its byte-identity with a net whose
  `gen` table is empty), the curve/rating mapping, the `scaling` convention, every
  reactive-limit fallback, the slack-bus rule, and the droop physics (setpoint held,
  error ∝ 1/slope, reactive saturation).
- `tests/reference/test_pandapower_gen_volt_var.py` — the live `runpp` oracle for
  that mode: `case9` at the default steepness, the 1/slope law on `case57`, and
  (marked `slow`) `case118`, where the default conversion does not even converge.
- `tests/reference/test_cigre_lv_full_transformer.py`,
  `test_cigre_lv_pandapower.py`, `test_ieee33_pandapower.py`,
  `test_ieee33_power_flow_pandapower.py`, `tests/convert/test_phase_mode.py` — the
  pre-existing regression suite (kept green).

See `docs/pgml/modeling/conventions.md` §2/§3/§5/§9 for the cross-tool convention
record and `docs/pgml/modeling/transformer.md` for the core vector-group model this
converter targets.
