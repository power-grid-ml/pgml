# Interface ledger: convert (external formats -> our schema)

One subpackage per source: `pandapower/`, `pgm/`, `opendss/`. Each exposes a pure
function producing a valid `grid_schema.Grid` (and, where relevant, the id map back
to the source so tests can align components).

Public API (all three IMPLEMENTED; per-source detail in each subpackage CONTEXT.md):
- [x] `convert.pandapower.to_grid(net, *, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV,
      gen_mode=GenMode.DROP, gen_volt_var_slope_pu=DEFAULT_GEN_VOLT_VAR_SLOPE_PU)
      -> (Grid, id_map)` — this file, below.
- [x] `convert.pgm.to_grid(input_data, *, base_frequency_hz=50.0,
      load_model=LoadModel.CONST_IMPEDANCE, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
      -> (Grid, id_map)` — see `pgm/CONTEXT.md`.
- [x] `convert.opendss.to_grid(dss_handle, *, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
      -> (Grid, id_map)` — see `opendss/CONTEXT.md`.
Conventions: convert engineering units -> SI; record source convention in
Provenance; map sequence/nameplate inputs via the schema's input-convention DTOs;
never invent fields (schema has extra="forbid").

## `convert._common` — the shared scaffold (library-agnostic)

The duplicated plumbing of every `to_grid` lives in `convert/_common.py`; each
per-library converter contains only the source-specific field reading and calls
these helpers so all sources stamp identical schema objects for the same physical
input. Designed to also serve the OpenDSS converter (native multi-phase + explicit
n x n line matrices). Every helper is unit-tested in `tests/convert/test_common.py`.

API:
- `IdCounter` — the single monotonic id allocator (one instance per conversion).
- `PhaseMode(str, Enum)` — `SINGLE_PHASE_EQUIV` (today's positive-sequence 1-phase
  equivalent, `phases=(Phase.A,)`, 1x1 line matrices) vs `THREE_PHASE` (genuine abc).
  Re-exported from `convert`, `convert.pandapower`, `convert.pgm`.
- `phases_for(mode, *, native=None) -> tuple[Phase, ...]` — the one place the phase
  tuple is decided: `SINGLE_PHASE_EQUIV -> (A,)`; `THREE_PHASE -> native` if given
  (OpenDSS passes real phases incl. `Phase.N`) else `(A, B, C)`.
- `zero_sequence_ratios() -> (r0/r1, x0/x1, c0/c1)` — read from `pgml.defaults`
  (`line.zero_sequence.*`).
- `sequence_to_phase_matrices(r1, x1, c1, *, r0=None, x0=None, c0=None, two_pi_f0,
  g0=0, g1=0) -> (R, L, C, G)` — sequence -> 3x3 phase matrices via
  `self=(Z0+2*Z1)/3`, `mutual=(Z0-Z1)/3` (applied to R, X, C, G); `L=X/two_pi_f0`.
  A balanced sequence input (Z0==Z1) yields a pure diagonal (decoupled) matrix; in
  general a symmetric circulant (off-diagonal == mutual). `r0/x0/c0` default to
  `r1*ratio` etc. from `zero_sequence_ratios()` when `None`.
- `single_phase_matrix(value) -> [[value]]` — the exact 1x1 wrapper used by the
  positive-sequence-equivalent line path (preserves `[[r1]]`/`[[l1]]`/`[[c1]]`).
- `thevenin_from_z(r_ohm, x_ohm, two_pi_f0) -> (R, L)` and
  `thevenin_from_sk(u_rated_v, sk_va, rx_ratio, two_pi_f0) -> (R, L)` — source
  Thevenin from an explicit impedance / from short-circuit power + R/X ratio (the
  latter moved verbatim from the pgm converter; same fallback floors).
- `make_metadata(name, description) -> GridMetadata`.
- Emit helpers (the single place the phase decision + per-phase mapping live):
  `build_node`, `build_load`, `build_generator` (generation-positive PQ, for
  pp `sgen` / pgm `sym_gen`), `warn_dropped_elements` (the loud-drop contract),
  `build_source`, `build_line_from_sequence`,
  `build_line_from_matrices`. `build_line_from_matrices` takes explicit n x n
  R/L/C(/G) matrices + a `phases` tuple — implemented and unit-tested for the
  OpenDSS converter (pp/pgm route through `build_line_from_sequence`).

**Zero-sequence assumption (THREE_PHASE line expansion).** When a positive-sequence
line is expanded to abc and the dataset has no native zero-sequence data, the
zero-sequence quantity defaults to `r1*(R0/R1)` etc. from config. An explicit native
`r0/x0/c0` always wins.

## `phase_mode` + asymmetric capture (pandapower / pgm / opendss)

`phase_mode=PhaseMode.SINGLE_PHASE_EQUIV` (the DEFAULT) reproduces the historical
positive-sequence single-phase-equivalent output BYTE-FOR-BYTE (same phases, 1x1
matrices, ids, id_map) — the bit-exact regression gate; the reference oracle suite
calls `to_grid(net)` with no `phase_mode` and stays green.

`phase_mode=PhaseMode.THREE_PHASE`:
- nodes/branches become `(A, B, C)`; lines use `sequence_to_phase_matrices`
  (pandapower: per-m `r1/x1/c1` from `r_ohm_per_km/x_ohm_per_km/c_nf_per_km`, with
  `r0/x0/c0` from `net.line` columns `r0_ohm_per_km/x0_ohm_per_km/c0_nf_per_km` IF
  present else config defaults; pgm: total `r1/x1/c1` and `r0/x0/c0` if present else
  defaults);
- sources become BALANCED 3-phase Thevenins (angles `u_angle / -120 / +120`,
  diagonal R/L); zero-seq source impedance = positive-seq (no short-circuit data read);
- standard balanced `net.load` / `sym_load`: `connection=None` (resolves to WYE from
  config), no per-phase split (the symmetric/auto calc splits the total equally);
- static generators: pandapower `sgen` and pgm `sym_gen` -> `Generator`
  (generation-positive nameplate; id_map buckets `"sgen"` / `"sym_gen"`); pandapower's
  voltage-controlled `gen` converts only under the opt-in
  `gen_mode=GenMode.VOLT_VAR_APPROX` (a `Generator` carrying a `VoltVarControl` droop —
  see `pandapower/CONTEXT.md`); every other
  non-empty pgm component (`transformer`, `three_winding_transformer`, `shunt`,
  `asym_gen`, `link`, `transformer_tap_regulator`) triggers a `warn_dropped_elements`
  WARNING — nothing is dropped silently;
- ASYMMETRIC loads captured: pandapower `net.asymmetric_load` -> `connection=WYE`
  (`type=="wye"`) or `DELTA`, `p_nom_per_phase_w=(p_a,p_b,p_c)*1e6`,
  `q_nom_per_phase_var=(q_a,q_b,q_c)*1e6`; pgm `asym_load` -> `p_specified`/
  `q_specified` shape (3,) per-phase, `connection=WYE` ALWAYS (pgm has no load
  connection field). Both add an `"asymmetric_load"` / `"asym_load"` id_map bucket.
- Under `SINGLE_PHASE_EQUIV` an asymmetric load cannot be represented; its phases are
  summed to a balanced 1-phase total and an INFO is logged on `logging.getLogger("pgml")`.

`phase_mode` for opendss (added with the scaffold migration):
- `SINGLE_PHASE_EQUIV` (default): every node/branch `phases=(A,)`, 1×1 line matrices
  (the `[0][0]` entry of the DSS matrix).  Load `connection=None`.
- `THREE_PHASE`: nodes carry real DSS phases (incl. `Phase.N`); lines carry the full
  n×n matrices from `RMatrix()/XMatrix()/CMatrix()`; loads get `IsDelta()` connection
  (`WYE` or `DELTA`); sources become balanced Thevenins with per-phase angles.
  Single-phase loads land on their real bus-suffix phase (`.1`→`(A,)`) with `WYE`.
  A delta load under `SINGLE_PHASE_EQUIV` logs INFO on `logging.getLogger("pgml")`.
OpenDSS provides native multi-phase matrices directly, so `build_line_from_matrices`
is used (not `build_line_from_sequence`); no sequence assumption is made.

## `convert.pandapower.to_grid` — final signature and id_map format

```python
from pgml.convert.pandapower import to_grid

grid, id_map = to_grid(net)
```

### Signature
```
to_grid(net: pandapowerNet, *, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV,
        gen_mode=GenMode.DROP, gen_volt_var_slope_pu=DEFAULT_GEN_VOLT_VAR_SLOPE_PU)
    -> tuple[Grid, dict[str, Any]]
```

Pure function. Converts a (materialised) pandapower network to a schema `Grid`
and an `id_map` dictionary.  Handles: `bus`, `line`, `load`, `asymmetric_load`,
`ext_grid`, `trafo`, bus-bus `switch`, and `sgen` (-> `Generator`,
generation-positive). `gen` (a PV bus) is DROPPED by default and converted only
under the explicit `gen_mode=GenMode.VOLT_VAR_APPROX`, which APPROXIMATES the PV
bus with a steep Volt-VAr droop centred on `vm_pu` (steepness
`gen_volt_var_slope_pu`, default 500) and saturating at `min/max_q_mvar` — it holds
|V| near, not at, the setpoint; a row on the `ext_grid` bus (or flagged `slack`) is
skipped. Every OTHER non-empty element table (`shunt`, `trafo3w`, `impedance`,
`ward`, `xward`, `dcline`, `storage`, `motor`, `asymmetric_sgen`, and `gen` under
the default mode) triggers a WARNING naming the kind and count — nothing is dropped
silently.

### id_map format
```python
{
    "bus":      {pp_bus_index: Node.id, ...},
    "line":     {pp_line_index: Line.id, ...},
    "load":     {pp_load_index: Load.id, ...},
    "sgen":     {pp_sgen_index: Generator.id, ...},
    "gen":      {pp_gen_index: Generator.id, ...},   # empty unless gen_mode=VOLT_VAR_APPROX
    "ext_grid": {pp_extgrid_index: Source.id, ...},
    "slack_v_complex": complex,   # phasor V (line-to-line, V) for ideal-slack solve
}
```
- Keys are pandas integer indices (int) into the respective net.* DataFrames.
- Only in-service elements whose buses are also in-service are included.
- `slack_v_complex`: complex slack phasor = `vm_pu * vn_kv*1000 * exp(j*va_deg)`
  ready to pass directly as `v_fixed` to `solve_harmonic(..., v_fixed=...)`.

### Unit conversions (engineering -> SI)
| pandapower field    | SI field                       | factor        |
|---------------------|--------------------------------|---------------|
| `vn_kv` [kV]        | `Node.u_rated_v` [V]           | × 1000        |
| `length_km`         | `Line.length_m`                | × 1000        |
| `r_ohm_per_km`      | `series_resistance_ohm_per_m`  | / 1000        |
| `x_ohm_per_km`      | `series_inductance_h_per_m`    | / (1000 · 2πf₀) |
| `c_nf_per_km`       | `shunt_capacitance_f_per_m`    | × 1e-9 / 1000 = 1e-12 |
| `g_us_per_km`       | `shunt_conductance_s_per_m`    | × 1e-6 / 1000 = 1e-9  |
| `p_mw` [MW]         | `Load.p_nom_w` [W]             | × 1e6         |
| `q_mvar` [MVAr]     | `Load.q_nom_var` [VAr]         | × 1e6         |

### Single-phase positive-sequence convention
Every node: `phases=(Phase.A,)`, `u_rated_v = vn_kv * 1000` (line-to-line).
The assembly's `phase_voltage_magnitude` returns `u_rated_v` unchanged for
1-phase nodes, so const-Z shunt = `(P - jQ) / V_LL^2` — identical to
pandapower's const-Z reference.

### numpy 2.x compatibility
### Validated on
IEEE 33-bus Baran & Wu (`pandapower.networks.case33bw()`), 60 Hz, 33 buses,
32 in-service lines + 5 tie-lines (out of service), 32 loads, 1 slack.
Oracle test: `tests/reference/test_ieee33_pandapower.py`.

## IEEE 33-bus oracle status (load-flow regression gate)
All three reference oracles pass on the single-phase positive-sequence IEEE33:
- pandapower (results): node V within <1e-4 pu. `test_ieee33_pandapower.py`.
- OpenDSS (Y matrix, absolute siemens): off-diagonal ~2e-15 S, diagonal ~1e-10 S
  after accounting for load shunts. `test_ieee33_opendss.py`.
- power-grid-model (results, 2nd oracle): node V within ~2e-10 pu. `test_ieee33_pgm.py`.
Comparison is apples-to-apples: loads set to CONSTANT IMPEDANCE on the reference
side (pp `const_z_p_percent=const_z_q_percent=100`, pgm `const_impedance`) so both solve the same
linear system; the source is an ideal slack (`solve_harmonic(fixed_rows, v_fixed)`).

## NONLINEAR power-flow oracle status (`solve_power_flow`, const-power)
- IEEE33, DEFAULT const-power `pp.runpp`: `solve_power_flow(slack="ideal")` matches
  `res_bus` to ~3.2e-9 pu / 1.3e-7 deg. `tests/reference/test_ieee33_power_flow_pandapower.py`.
- CIGRE LV (`create_cigre_network_lv`, 44 bus, 3 Dyn1 20/0.4 kV MV/LV trafos, bus-bus CBs):
  matches to ~1.1e-7 pu / 2.1e-6 deg. `tests/reference/test_cigre_lv_pandapower.py`.

## pandapower converter — element coverage (extended)
`to_grid` handles `bus`, `line`, `load`, `sgen`, `asymmetric_load`, `ext_grid`,
**`trafo`** (vector-group + tap-changer aware), and **bus-bus `switch`** (`et='b'`,
modelled as near-ideal `Switch`, R=1e-4 Ω). `id_map` adds `"trafo"` and `"switch"`.
Transformer connections come from `net.trafo['vector_group']` / the row's
`std_types['trafo'][std_type]['vector_group']` (parsed, clock digit optional — a
bare `'Dyn'`/`'Yzn'` is pandapower's own `runpp_3ph` form), cross-checked against
`shift_degree` (a mismatch raises `ConversionError`); with no vector-group string
anywhere the connection falls back on the shift parity (even clock ->
WYE_GROUNDED/WYE_GROUNDED, odd -> DELTA/WYE_GROUNDED). `tap.ratio_magnitude` reads
`tap_pos`/`tap_neutral`/`tap_step_percent`/`tap_side` (NaN-safe; 1.0 = no tap).
Leakage `series_resistance_ohm`/`series_inductance_h` is the TO-side COIL value
(`3×` the raw `vk_percent`-derived terminal quantity when `to_connection==DELTA`,
applied THE SAME WAY REGARDLESS OF `phase_mode` — assembly's own
`_transformer_block_groups` undoes the factor internally for the single-phase
scalar stamp too; a phase-mode-conditional version of this factor is WRONG, see
`src/pgml/convert/pandapower/CONTEXT.md` and the pinned oracle test
`tests/reference/test_pandapower_grid_matrix.py`). A line/trafo's `parallel`
count (identical parallel systems) divides the series impedance and multiplies
the shunt admittance (line C/G, trafo magnetizing) and the rated power
(`s_rated_va`); `parallel==1` is byte-identical to before. An OPEN bus-line/
bus-transformer switch (`et='l'`/`'t'`) takes the whole line/trafo out of
service (an accepted approximation — the still-connected terminal's shunt is
dropped too, unlike pandapower's own auxiliary-bus model); bus-bus (`et='b'`)
switches are unaffected. `load`/`sgen`/`asymmetric_load` P/Q are scaled by the
per-element `scaling` column (NaN-safe, default 1.0 — pandapower's own `runpp`
convention); `load` additionally maps `const_z_p_percent`/`const_i_p_percent`/
`const_z_q_percent`/`const_i_q_percent` onto `ZipCoefficients` (all-zero, the
pandapower default, stays byte-identical with no `zip_coefficients`/
`load_model` set). See `docs/pgml/modeling/transformer.md` and
`src/pgml/convert/pandapower/CONTEXT.md` for the full field-mapping table.

## OpenDSS converter — transformer element coverage
`convert.opendss.to_grid` converts two-winding DSS `Transformer` elements (winding 1 =
HV/from, winding 2 = LV/to). Leakage: DSS's per-winding `%R` and inter-winding `XHL` are
PERCENT (base-invariant) quantities, so `R_lv=(%R_wdg1+%R_wdg2)/100*Z_base_LV`,
`X_lv=XHL%/100*Z_base_LV` with `Z_base_LV=kV_lv²*1000/kVA` recovers the LV-referred R/L
directly (no HV/LV referral arithmetic needed) — requires both windings to share one kVA
rating (`ConversionError` otherwise). Connections: `IsDelta()` per winding; a wye winding
converts only when solidly grounded (OpenDSS's shorthand bus rule or an explicit `.0`
neutral); an explicit non-zero neutral node raises. Vector group: `LeadLag` (`Lag`->clock
1/shift 30, `Lead`->clock 11/shift 330) for a Dy/Yd pairing (verified against a live
solve), clock 0 for a matching Yy/Dd pairing (OpenDSS has no explicit clock parameter
beyond the binary `LeadLag` toggle, so `Yy6`/`Dd6` is not detected). Magnetizing:
`%noloadloss`/`%imag` -> `magnetizing_conductance_s`/`magnetizing_inductance_h` via the
same closed-form the pandapower converter uses (HV-referred). NOT read: 3-winding units,
`RegControl` regulators, `XfmrCode`/frequency-correction curves. See
`src/pgml/convert/opendss/CONTEXT.md` and `tests/reference/test_opendss_transformer.py`.

## OpenDSS converter — element scope extension (Line/Load/Capacitor/Reactor/Generator/PVSystem/Storage)
`convert.opendss.to_grid` also converts: **Line** — a phase-permuted terminal
(`bus1=a.1.2.3 bus2=b.3.2.1`) carries an independent `to_phases` (the
assembly's series-branch stamp already indexes the two terminals
independently, matching Transformer); the `SINGLE_PHASE_EQUIV` reduction of a
coupled multi-phase line now uses the POSITIVE-SEQUENCE `Z1 = Z_self -
Z_mutual` (and `C1 = C_self - C_mutual`) rather than the bare self entry (a
genuinely 1-phase line is unaffected — byte-identical). **Load/Generator/
Storage/PVSystem** bus parsing now reads `CktElement.NodeOrder()` (DSS's own
resolved conductor/return assignment) instead of re-parsing the bus string,
fixing a bug where every bus-string suffix (including an explicit neutral
tie) was read as a phase conductor; each WYE appliance's return conductor sets
its `InjectionAppliance.return_path` (`_resolve_wye_return_path`: explicit
`.4` neutral tie → `"neutral"`, solidly grounded on a `Phase.N`-carrying bus →
`"ground"`, else `"auto"`), so pgml reproduces OpenDSS's per-element return
routing even on a shared 4-wire bus (the previously-inexpressible
grounded-despite-neutral case is now exact, no warning). **Load** also
maps `Loads.Model()` (1/2/5/8) to `LoadModel`/`ZipCoefficients` (models
3/4/6/7 fall back to `CONST_POWER` with a warning; the ZIPV low-voltage
cutoff is not modeled). **Capacitor/Reactor** convert to `ShuntAppliance`
(WYE solidly-grounded, or DELTA phase-to-phase bank via `? conn`; a
non-grounded 2-bus terminal-2 reference and an explicitly coupled Reactor
Rmatrix/Xmatrix are out of scope and warned/skipped); a Reactor's series
R+X converts to the equivalent shunt admittance `Y=1/(R+jX)` (exact at
the fundamental only — the schema's `ShuntReactor`/`ShuntAppliance` have no
inductance field, so a genuinely inductive reactor's harmonic frequency
trend is not modeled). **Generator/PVSystem** convert via
`build_generator` (generation-positive; PVSystem's present `kW`/`kvar`
already reflect OpenDSS's own Pmpp/irradiance/pf derating).
**Storage** converts directly to `Storage` (signed, discharge-positive —
DSS's own `kW` sign under `state=CHARGING`/`DISCHARGING` already matches)
plus its inert energy-state fields. Every other unhandled DSS element class
(`Isource`, `Monitor`, `EnergyMeter`, `RegControl`, ...) is enumerated
generically from `Circuit.AllElementNames()` and triggers a
`warn_dropped_elements` WARNING; a non-negligible Vsource `R1`/`X1` warns
that the default `slack="ideal"` ignores it (`slack="norton"` reproduces it).
See `src/pgml/convert/opendss/CONTEXT.md` for the full field-mapping tables
and `tests/reference/test_opendss_line_phase_permutation.py`,
`tests/reference/test_opendss_load_model.py`,
`tests/convert/test_opendss_shunt_and_der_elements.py`, and the strengthened
`tests/convert/test_opendss_phase_mode.py` for the oracle/coverage tests.

## Cross-converter conventions (voltage base, slack, frequency)

> The authoritative cross-tool convention record (base voltage L-L/L-N, transformer
> reference side, vector-group/clock, power signs, SI/phase-domain, harmonic earth-return)
> for pgml vs pandapower / OpenDSS / power-grid-model is `docs/pgml/modeling/conventions.md`. Keep
> it in sync when a converter's convention handling changes.

- CANONICAL `u_rated_v` for any node is LINE-TO-LINE (`vn_kv*1000` / `u_rated*1`
  for nodes that are already L-L), validated against pandapower's const-Z reference
  (`y = conj(P+jQ)/V_LL^2`). All three converters follow this convention.
  OpenDSS `Bus.kVBase()` returns L-N (= `BasekV_LL/sqrt(3)`); the OpenDSS converter
  recovers L-L via `kVBase * sqrt(3) * 1000`. The old converter stored `kVBase * 1000`
  (L-N), which was a factor-of-sqrt(3) error in the const-Z shunt on the load-flow
  path. The Y-bus oracle test (passive, load-free) was unaffected; the fix was applied
  together with the phase_mode scaffold migration.
- VOLTAGE BASE vs WORKING VOLTAGE: `u_rated_v` is stored L-L, but every per-phase voltage
  the solver touches is LINE-TO-NEUTRAL. `assembly._params.phase_voltage_magnitude(u_rated_v,
  n)` is the single base helper — `u_rated_v/sqrt(3)` for a WYE element on a >=3-phase node,
  `u_rated_v` for DELTA / 1-phase — and feeds the const-Z/ZIP load, the harmonic source
  admittance, and the per-unit reporting in `pgml.evaluation`. Accordingly the slack/source
  EMF is L-N: under `THREE_PHASE` `build_source` divides the L-L `u_ref_v` by sqrt(3) for the
  balanced wye expansion (1-phase keeps it). `id_map["slack_v_complex"]` stays the L-L phasor
  (a single-phase ideal-slack `v_fixed` convenience). Pinning the L-L magnitude on each phase
  makes every 3-phase voltage sqrt(3) too high (~1.73 pu on the L-N base).
- `base_frequency_hz` is read from the source (`net.f_hz`,
  `dss.Solution.Frequency()`); pgm has no f0 field so the caller passes it. For
  IEEE33 (no line charging) the absolute f0 cancels in `X=2πf·L`; it matters once
  C≠0 / for harmonics. `dss.Solution.Frequency()` reflects OpenDSS's process-global
  `DefaultBaseFrequency`, which `Clear` does NOT reset — a prior test/circuit that
  changed it silently leaks into the next one unless it is reasserted explicitly
  (`set DefaultBaseFrequency=<f0>`), a shared gotcha for any test file that drives a live
  `opendssdirect` engine (see `tests/reference/test_opendss_transformer.py`'s `_dss_clear`).
- OpenDSS `Vsource.basekv` IS the L-L nominal for a `phases>=3` source, but is used
  DIRECTLY (no sqrt(3) anywhere) as the solved single-conductor EMF for a `phases=1`
  source — undocumented by OpenDSS's general docs, verified empirically
  (`tests/convert/test_opendss_vsource_basekv.py`). The converter's `u_ref_v =
  basekv*pu*1000` and `u_rated_v = kVBase()*sqrt(3)*1000` need no phase-count branch:
  both simply mirror whatever OpenDSS itself solves for at that phase count. See
  `docs/pgml/modeling/conventions.md` sec. 1/6.
