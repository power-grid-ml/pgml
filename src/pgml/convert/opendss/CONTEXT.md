# Interface ledger: convert.opendss

Converts a live OpenDSS circuit (via `opendssdirect`) to our schema `Grid`.

## Public API

```python
from pgml.convert.opendss import to_grid, PhaseMode

import opendssdirect as dss
dss.Text.Command("Redirect feeder.dss")
dss.Text.Command("Solve")
grid, id_map = to_grid(dss)                                      # SINGLE_PHASE_EQUIV (default)
grid_3ph, id_map_3ph = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
```

### Signature
```
to_grid(dss_handle: Any, *, phase_mode: PhaseMode = PhaseMode.SINGLE_PHASE_EQUIV) -> tuple[Grid, dict[str, Any]]
```

Pure function (reads from the active OpenDSS engine state). The circuit must
already be loaded and solved (or `Calcvoltagebases` called) before calling.

### `phase_mode` parameter

| Value                 | Node phases          | Line matrices      | Load `connection`           |
|-----------------------|----------------------|--------------------|-----------------------------|
| `SINGLE_PHASE_EQUIV`  | `(Phase.A,)` always  | 1×1 (diagonal [0][0]) | `None` (resolves from config) |
| `THREE_PHASE`         | Real DSS phases (incl. `Phase.N` for neutral buses) | Full n×n from `RMatrix()/XMatrix()/CMatrix()` | `WYE` or `DELTA` from `IsDelta()` |

`SINGLE_PHASE_EQUIV` is byte-identical to the historical converter output, except
for the `u_rated_v` fix below (which does not change the IEEE 33-bus numbers).

### id_map format
```python
{
    "bus":             {dss_bus_name_lower: Node.id, ...},
    "line":            {dss_line_name_lower: Line.id, ...},
    "trafo":           {dss_trafo_name_lower: Transformer.id, ...},
    "load":            {dss_load_name_lower: Load.id, ...},
    "vsource":         {dss_vsrc_name_lower: Source.id, ...},
    "capacitor":       {dss_cap_name_lower: ShuntAppliance.id, ...},
    "reactor":         {dss_reactor_name_lower: ShuntAppliance.id, ...},
    "generator":       {dss_gen_name_lower: Generator.id, ...},
    "pvsystem":        {dss_pv_name_lower: Generator.id, ...},
    "storage":         {dss_storage_name_lower: Storage.id, ...},
    "slack_v_complex": complex,  # slack phasor (V) from the first Vsource
}
```
- Keys are lowercase DSS element/bus names.
- Node ids are assigned in `YNodeOrder` bus sequence (bus.phase pairs, in
  DSS's internal order), guaranteeing our `node_phase_index` rows align to
  the DSS Y-matrix rows/columns for oracle tests.
- `"slack_v_complex"` is the complex voltage phasor of the first in-service
  Vsource (`BasekV * pu * 1000 * exp(j*angle_deg)`); pass it as `v_fixed` to
  `solve_harmonic(..., v_fixed=...)` for ideal-slack mode. For a `phases>=3`
  Vsource this is the L-L phasor; for a `phases=1` Vsource `BasekV` is used
  directly by OpenDSS (see "Vsource basekv semantics" below), so this is
  already the correct single-conductor EMF -- no extra scaling needed either
  way.

### Supported element types
| DSS type    | Schema type            | Notes                                      |
|-------------|-------------------------|--------------------------------------------|
| Line        | `Line`                  | 1- or multi-phase; n×n R/X/C from matrix API; `to_phases` carries a phase-permuted terminal independently of `from_phases` |
| Transformer | `Transformer`           | Two-winding only; see below                |
| Vsource     | `Source`                | R1/X1 via text commands; `thevenin_from_z`; warns if non-negligible (see below) |
| Load        | `Load`                  | kW/kvar total; `IsDelta()` -> `connection`; `Loads.Model()` -> `LoadModel`/`ZipCoefficients`; WYE `return_path` from the return conductor (see below) |
| Capacitor   | `ShuntAppliance`        | WYE (solidly grounded) or DELTA (phase-to-phase bank); see below |
| Reactor     | `ShuntAppliance`        | WYE (solidly grounded, uncoupled) or DELTA (phase-to-phase bank); see below |
| Generator   | `Generator`             | generation-positive; `build_generator`     |
| PVSystem    | `Generator`             | `consumer_type=ConsumerType.PV`; present kW/kvar output |
| Storage     | `Storage`               | signed, discharge-positive; energy-state fields (inert) |

Every OTHER DSS element class (`Isource`, `Fault`, `Monitor`, `EnergyMeter`,
`RegControl`, `CapControl`, `InvControl`, `StorageController`, `Relay`,
`Recloser`, `Fuse`, `Sensor`, ...) is enumerated generically from
`Circuit.AllElementNames()` (`"ClassName.elementname"`, grouped by class,
excluding the handled set above) and triggers one `warn_dropped_elements`
WARNING per class naming the kind and count -- nothing vanishes silently. A
3-winding `Transformer` is NOT counted here; it already raises
`ConversionError` (a hard scope boundary, not a silent drop).

**Disabled elements.** `First()`/`Next()` class iterators already skip
DISABLED elements (verified empirically against opendssdirect 0.9.4: a
disabled `Generator` is invisible to `Generators.First()`/`.Next()`) -- every
element loop in this converter therefore only ever sees in-service elements
without an explicit `CktElement.Enabled()` check.

### Line: phase-permuted terminals and the positive-sequence Z1 reduction

**`to_phases` (THREE_PHASE).** A DSS line whose two bus-connection strings
list a DIFFERENT phase-conductor order at each end (e.g. `bus1=a.1.2.3
bus2=b.3.2.1`) is a genuine phase-transposing connection: line conductor `k`
(row/column `k` of the R/X/C matrix) ties `from_phases[k]` at `from_node` to
`to_phases[k]` at `to_node`. `to_grid` parses `bus2`'s suffix list
independently (`_parse_bus_connection`, already used for `bus1`) and passes
it via `build_line_from_matrices(to_phases=...)`. No assembly change was
needed: `pgml.assembly.ybus._series_terminal_indices` already indexes a
series branch's two terminals independently from `b.from_phases`/
`b.to_phases` (the SAME mechanism a Transformer's differing terminal phases
already relies on) -- verified by a live-oracle numeric test rather than
asserted from reading the code alone
(`tests/reference/test_opendss_line_phase_permutation.py`).

**`SINGLE_PHASE_EQUIV` positive-sequence reduction.** A genuinely 1-phase DSS
line (`n_phases == 1`, e.g. the IEEE 33-bus oracle) keeps the exact `[0][0]`
matrix entry -- byte-identical to the historical converter. A COUPLED
multi-phase line reduces to `Z1 = Z_self - Z_mutual` (mean diagonal minus
mean off-diagonal, `_positive_sequence_scalar`), not the bare self entry
(the historical bug, which ignored the mutual coupling and overstated the
positive-sequence impedance). The identical reduction applies to the Maxwell
C matrix: `C1 = C_self - C_mutual`; C's off-diagonals are NEGATIVE (mutual
coupling reduces net charge), so subtracting them INCREASES C1 above
`C_self`, the physically correct direction. For a DSS line built from
`r1`/`x1` (with `r0`/`x0` defaulted or given), this reduction recovers `r1`/
`x1` EXACTLY (`R_self - R_mutual = ((r0+2r1)/3) - ((r0-r1)/3) = r1`, a
closed-form identity independent of `r0`), which is how the byte-identical
IEEE-33 case is preserved and how the reduction is unit-tested.

### Load: `Phases=`-aware bus parsing, load model, and the neutral-routing gap

**Bus connection.** `Loads.Phases()` (`CktElement.NumPhases()`) is the number
of PHASE conductors only -- OpenDSS silently appends exactly ONE extra return
conductor beyond that count for a (default) WYE connection (grounded to node
0 by default, or an explicit non-zero suffix, typically `4`, an explicit
neutral tie). The historical converter parsed EVERY bus-string suffix as a
phase, so a 3-phase WYE load written `bus1=b1.1.2.3.4` became a 4-element
ABCN load. `_parse_appliance_bus_connection(dss, n_phases)` fixes this by
reading `CktElement.NodeOrder()` -- OpenDSS's OWN resolved conductor/return
assignment (already applying its bus-string default-padding rules) -- rather
than re-parsing the bus string: the first `n_phases` entries are the phase
conductors, one further entry (if present) is the return conductor (`0` /
absent -> ground; anything else, e.g. `4`, -> that `Phase` explicitly, most
often `Phase.N`). Shared by Load, Generator, Storage and PVSystem (all
single-terminal WYE-or-DELTA PC elements with this same conductor-count
pattern); NOT needed for Line (a line's `Phases=` already IS its total
conductor count, no auto-appended return) or Vsource (a genuine 2-terminal
element, `NumConductors == n_phases` on its own first terminal).

**WYE return-conductor routing (`return_path`).** pgml's WYE/neutral routing
(`assembly._incidence.group_appliances`) is a property of the NODE, but the schema's
`InjectionAppliance.return_path` overrides it per appliance, so the converter
reproduces OpenDSS's own per-element return-conductor choice exactly.
`_resolve_wye_return_path(bus_name, bus_phases, explicit_return)` maps
`CktElement.NodeOrder()`'s return conductor to a `return_path`: an explicit `Phase.N`
tie (`.4`) -> `"neutral"`; a solidly grounded appliance (no explicit tie) on a bus that
ALSO carries `Phase.N` from another element -> `"ground"` (return stays at true ground
despite the shared neutral — previously an inexpressible, warned mismatch); otherwise
`"auto"` (reduces to ground on a 3-wire node). Threaded into `build_load`/
`build_generator`/`Storage` for WYE appliances only (DELTA has no neutral). See
`tests/convert/test_opendss_phase_mode.py` (`test_grounded_load_on_neutral_carrying_node_uses_return_path_ground`,
`test_four_wire_mixed_return_paths_convert_and_match_opendss` — live parity for a
4-wire bus carrying BOTH a grounded and a neutral-returning load).

**Load model (`Loads.Model()` -> `LoadModel`/`ZipCoefficients`,
`_resolve_load_model`).**
| DSS Model | Meaning | pgml mapping |
|---|---|---|
| 1 | constant P, Q | `LoadModel.CONST_POWER` (the schema default; `(None, None)` returned to keep the historical byte-identical output) |
| 2 | constant impedance | `LoadModel.CONST_IMPEDANCE` |
| 5 | constant current magnitude | `LoadModel.CONST_CURRENT` (both P and Q scale linearly with `\|V\|`, matching pgml's `CONST_CURRENT` ZIP triple exactly) |
| 8 | ZIPV (custom coefficients) | `ZipCoefficients` from the first 6 `Loads.ZipV()` values (`z_p,i_p,p_p,z_q,i_q,p_q`); the 7th (low-voltage cutoff) has no pgml equivalent (the ZIP law applies at every voltage) -- warns if non-zero |
| 3, 4, 6, 7 | asymmetric P-vs-Q voltage dependence (motor-like) | no faithful `ZipCoefficients` equivalent -- falls back to `CONST_POWER`, warns naming the model |

Only the NONLINEAR `solve_power_flow`/`solve_harmonic_flow` reads
`load_model`/`zip_coefficients` (`assembly.ybus.device_current_injections`);
the linear const-Z assembler always uses the base P/Q regardless. See
`tests/reference/test_opendss_load_model.py` (live-oracle parity for Model=2,
5, 8, and the 3/4/6/7 fallback).

### Capacitor / Reactor -> `ShuntAppliance`

Both are converted to a per-phase-to-GROUND
`~pgml.schemas.grid_schema.ShuntAppliance` (simpler than the `ShuntReactor`
`BranchBase` type for this uncoupled, single-node case -- no genuine
`to_node`/`to_phases` bookkeeping is needed). **Note:** despite its name,
`ShuntReactor` stores `conductance_s` + `capacitance_f` only (no inductance
field); it is the schema's general shunt-admittance branch type, not a
dedicated inductive-reactor primitive. `ShuntAppliance` (per-phase G/C `Vec`,
node-anchored) is the better fit for both a Capacitor bank and an uncoupled
shunt Reactor.

**Scope (`_grounded_shunt_phases` / `_shunt_phase_conductors`).** A WYE (grounded)
element is a 2-terminal branch whose 2nd terminal defaults to the SAME bus, every
conductor tied to node 0 -- OpenDSS's UNIVERSAL (never bus-scoped) ground reference, so
checking every terminal-2 conductor is 0 is sufficient regardless of which bus name
expresses it (`_grounded_shunt_phases`; this also covers the "grounding reactor" idiom,
`bus1=busname.k.0` phases=1, which OpenDSS resolves into `bus1=busname.k` / an
auto-generated `bus2=busname.0`). A DELTA element (`? Class.name.conn == "delta"`)
collapses to `NumTerminals()==1` (phase-to-phase legs) and converts to a DELTA
`ShuntAppliance` whose per-leg G/C is read from the single terminal's phase conductors
(`_shunt_phase_conductors`). A non-grounded 2-bus terminal-2 reference is still
warned/skipped, and an explicit coupled Reactor `Rmatrix`/`Xmatrix` (off-diagonal) is
out of scope (only the scalar `R()`/`X()` diagonal path converts). Under
`SINGLE_PHASE_EQUIV` a DELTA bank folds to the positive-sequence WYE equivalent
(`Y_wye = 3·Y_leg`).

**Capacitor.** `Cuf` (read via text query, µF PER STEP) summed over the
ACTIVE steps (`Capacitors.States()`) gives the capacitance; `conductance_s=0`.
For a WYE bank this is the per-phase C, for a DELTA bank the per-leg C —
verified live: OpenDSS's own resolved `Cuf` is the leg capacitance for a
delta bank (a delta bank's `Cuf` is exactly `1/3` of the wye bank's for the
same `kvar`/`kV`). Reading `Cuf` directly (rather than re-deriving from
`kv`/`kvar`) sidesteps any base-frequency-dependent unit subtlety.

**Reactor.** `R()`/`X()` (Ω, resolved by DSS regardless of whether the
element was specified via `R=`/`X=` or `kV=`/`kvar=`) give the per-phase
series impedance; a series impedance to ground is electrically IDENTICAL to
a shunt admittance to ground (there is no series-vs-shunt distinction when
one end is the fixed-zero reference), so `Y=1/(R+jX)`, `G=Re(Y)`,
`C=Im(Y)/(2*pi*f0)`. **Caveat (documented, not modeled):** this C-based
susceptance model is exact ONLY at the fundamental (h=1) -- a genuinely
inductive reactor's true susceptance `-1/(h*2*pi*f0*L)` DECREASES with
frequency, while the fixed-C model INCREASES linearly with h (the wrong
trend for a harmonic study); load-flow (fundamental-only) studies are exact,
harmonic studies are not. See
`tests/convert/test_opendss_shunt_and_der_elements.py`.

### Generator / PVSystem / Storage

**Generator** converts via `build_generator` (generation-positive P/Q from
`Generators.kW()`/`.kvar()`), `IsDelta()` -> `connection`, bus/phases via
`_parse_appliance_bus_connection`.

A `model=3` Generator (constant kW, constant |V| — OpenDSS's PV bus) additionally
carries a `VoltageRegulation` block (`_generator_voltage_regulation`), which the
solver holds exactly (`solver/_pv_bus.py`):

| DSS property | Our schema field | Notes |
|---|---|---|
| `Model` | — | `3` -> a regulating terminal; every other model stays a PQ injection |
| `Vpu` | `VoltageRegulation.v_set_pu` | per unit of the MACHINE's `kV` rating (L-L for >= 2 phases, L-N for 1), re-referred to the HOST NODE's rated voltage through the two L-N bases; a mismatch between the two ratings is logged at INFO |
| `Maxkvar` / `Minkvar` | `q_max_var` / `q_min_var` | `*1e3`. OpenDSS always resolves both (from `kVA` and `PF` when not given) and enforces them, so they are read as finite limits |
| `kvar` | — | not read for `model=3` (`q_nom_var = 0.0`): the reactive power is solved |

Properties outside the `Generators` accessor set (`Model`, `Vpu`, `Maxkvar`,
`Minkvar`) are read through `dss.Properties.Value` on the active element. A
DELTA-connected `model=3` machine raises `ConversionError` — the regulated row pair is
formed for a WYE terminal. Measured against a live solve on a two-bus 20 kV feeder
(`tests/reference/test_opendss_pv_bus.py`, opendssdirect 0.9.4, `Tolerance=1e-10`):
a regulating machine agrees to 6.6e-10 pu / 4.2e-4 kvar (pgml 4 Newton iterations,
OpenDSS 108 of its own), one pinned at a reactive limit to 1.7e-13 pu / 1.3e-8 kvar.
OpenDSS's own `model=3` loop starts from the PREVIOUS solution, so a cold solve can
hit `MaxIter` where a repeated one converges; the test brings the machine up as a PQ
injection first, and the oracle pins only the two limit cases for that reason.

**PVSystem** converts the same way with `consumer_type=ConsumerType.PV`;
`PVsystems.kW()`/`.kvar()` report the PRESENT solved output (after OpenDSS's
own Pmpp/irradiance/pf/kVA-limit derating), so no extra derating arithmetic
is needed -- reading the present output is the "snapshot" convention also
used for Storage. Connection is read via a text query (`? PVSystem.name.conn`,
since `PVsystems` has no `IsDelta()` accessor).

**Storage** has no `build_generator`-style `_common` helper; the
`~pgml.schemas.grid_schema.Storage` object is constructed directly, mirroring
`build_load`/`build_generator`'s mode/connection/native_phases decisions.
Present `kW`/`kvar` (text query) already carry the schema's
discharge-positive / charge-negative sign convention -- verified empirically:
`state=DISCHARGING` reads a positive `kW`, `state=CHARGING` reads a negative
one, no sign flip needed in the converter. The inert energy-state fields map
directly: `energy_capacity_wh = kWhrated*1000`, `soc = %stored/100`,
`soc_min = %reserve/100`, `efficiency_charge/discharge = %EffCharge/
%EffDischarge / 100`, `p_rated_w = kWrated*1000`, `consumer_type=
ConsumerType.BATTERY`.

### Vsource impedance under the default ideal slack

A non-negligible Thevenin impedance (`R1`/`X1`, read for `thevenin_from_z`)
is silently UNUSED under pgml's default `slack="ideal"` (`solve_power_flow`/
`solve_harmonic` pins the bus voltage exactly at `u_ref_v` regardless of
R1/X1; only `slack="norton"` folds the source's own Norton shunt into
`Y_eff` and actually loads the bus). `to_grid` logs a WARNING once per
Vsource whenever `R1 > 1e-4` or `X1 > 1e-4` Ohm (comfortably above the
near-zero, <=1e-6 Ohm, "ideal source" placeholders this test suite's own
fixtures use, and well below DSS's own un-set Vsource default, ~0.02+j0.08
Ohm) naming the values and recommending `slack="norton"`.

### Transformer conversion (two-winding, vector-group aware)

`to_grid` converts a two-winding DSS `Transformer` element per: winding 1 =
HV/`from`, winding 2 = LV/`to` (standard OpenDSS convention; matches the live
oracle's own `_build_circuit_with_real_transformer`). `NumWindings() != 2`
raises `ConversionError`.

**Leakage (referred to the LV/to coil, pgml's storage convention).** OpenDSS's
per-winding `%R` and inter-winding `XHL` are PERCENT (per-unit, base-invariant)
quantities -- `XHL` is documented "on the kVA base of winding 1" -- so, requiring
both windings to share one kVA rating (checked; `ConversionError` if they
differ):
```
Z_base_LV = kV_lv^2 * 1000 / kVA
R_ll_ohm  = (%R_wdg1 + %R_wdg2) / 100 * Z_base_LV
X_ll_ohm  = %XHL / 100 * Z_base_LV
```
`R_ll_ohm`/`X_ll_ohm` are on the STANDARD line-to-line base -- the same
quantity pandapower's `vkr%`/`vk%` recover. pgml stores the leakage referred to
the ACTUAL TO-side COIL, which coincides with the L-L base for a wye/zigzag LV
winding but is **3x LARGER for a DELTA LV winding** (a delta coil is rated at
the L-L voltage with 1/3 the per-phase kVA, so its natural impedance base is
`Z_base_coil = V_LL^2/(S/3) = 3*Z_base_LV`; pinned exactly by
`tests/reference/test_transformer_clock_matrix.py`'s `y_LL = (3 if
to_kind=="delta" else 1) / y_coil` assertion and re-derived in
`docs/pgml/modeling/references/opendss/index.md`):
```
factor    = 3.0 if LV winding is DELTA else 1.0
R_lv_ohm  = factor * R_ll_ohm
X_lv_ohm  = factor * X_ll_ohm
L_lv_h    = X_lv_ohm / (2*pi*f0)
```
This was a **latent bug** fixed alongside the rotation work below: the
converter previously stored `R_ll_ohm`/`X_ll_ohm` directly regardless of the
LV connection, which is correct for a wye/zigzag LV winding (the common case,
e.g. Dyn11/Yy0 -- untouched by the fix) but off by a factor of 3 for a DELTA LV
winding (e.g. `YNd*`). Live-oracle evidence
(`tests/reference/test_opendss_transformer.py::TestYNd5RotatedOracle`, a
WYE-grounded-HV / DELTA-LV 20/0.4 kV unit): the pre-fix (no factor-3) leakage
produces a ~7e-3 pu LV voltage-magnitude error vs a live OpenDSS solve; the
fix reduces it to ~2e-8 pu. The oracle direction
(`pgml.evaluation.oracles.opendss_oracle._build_circuit_with_real_transformer`)
had the mirror-image bug (dividing by the SAME factor before back-calculating
`%R`/`XHL` from pgml's stored R/L) and is fixed the same way.
(Gotcha: the sequential tilde-continuation syntax `~ wdg=1 ... kVA=x` / `~
wdg=2 ... kVA=y` silently re-syncs BOTH windings' kVA to the last value given
-- only the array form `kvas=[x, y]` actually creates differing per-winding
kVA, which is the case this guard is defending against.)

**Connections and grounding.** `IsDelta()` per winding selects
`WindingConnection.DELTA`. A wye winding converts to
`WindingConnection.WYE_GROUNDED` ONLY when solidly grounded, per OpenDSS's own
shorthand-bus rule: no explicit `(n_phases+1)`-th conductor in the bus string
(the missing neutral is auto-tied to ground node 0), or an explicit `.0`. An
explicit NON-ZERO `(n_phases+1)`-th conductor (e.g. `.4`, a genuinely floating
or `Rneut`/`Xneut`-impedance-grounded neutral) raises `ConversionError` --
out of scope (pgml's transformer assembly models solid grounding only).
**Zigzag** has no OpenDSS `Transformer` connection at all, so a DSS file can
never produce one -- nothing to detect in this direction (see "Oracle
direction" below for the reverse).

**Vector group / clock.** OpenDSS has no explicit clock parameter. Two
independent mechanisms combine to determine `tap.shift_deg`:

1. `LeadLag`, meaningful only for a Dy/Yd (delta-wye) pairing: `Lag`/`ANSI`
   (default) -> clock 1 baseline (`shift_deg=30`); `Lead`/`Euro` -> clock 11
   baseline (`shift_deg=330`) -- verified against a live solve
   (`Bus.puVmagAngle()` shows the LV bus leading/lagging the HV bus by ~30 deg
   accordingly). A matching Yy or Dd pairing baselines at clock 0
   (`shift_deg=0`) regardless of `LeadLag`.
2. **Cyclic winding-bus rotation.** Each transformer winding's bus string
   (`bus=lv.1.2.3.0` etc.) can list its phase conductors in a CYCLIC rotation
   of `(1, 2, 3)` -- e.g. `bus=lv.2.3.1.0` (rotation `r=1`) or `bus=lv.3.1.2.0`
   (`r=2`) -- OpenDSS's only OTHER mechanism for expressing a clock beyond the
   `LeadLag` baseline. `_cyclic_rotation_steps` detects `r` on each winding
   (`phase_nums[k] == ((k + r) % 3) + 1`); the converter then NORMALIZES the
   stored `from_phases`/`to_phases` to the canonical `(A, B, C)` tuple (never
   the raw rotated order -- the phase-domain transformer stamp,
   `pgml.assembly._transformer.block_incidence`, assumes winding position `k`
   IS bus phase `k`) and instead folds the rotation into the clock:
   ```
   shift_deg = (base_shift_deg + 120*from_rotation - 120*to_rotation) % 360
   ```
   **Sign, pinned against a live OpenDSS solve** (`src.1.2.3` HV, Yy0 base,
   30 kW/10 kvar LV load; `lag = (src_phaseA_angle - lv_phaseA_angle) % 360`):
   rotating the HV (`from`) winding's bus string by `r=1`
   (`bus=src.2.3.1`) gives `lag ≈ 120.09°` (clock 4, i.e. **+4** clock steps
   per FROM-side rotation step); rotating the LV (`to`) winding by `r=1`
   (`bus=lv.2.3.1.0`) gives `lag ≈ 240.09°` (clock 8 ≡ **-4** clock steps per
   TO-side rotation step); `r=2` on either side doubles the effect
   (`+8`/`-8` ≡ `-4`/`+4` mod 12). The same +-4-per-step rule was re-verified
   for a Dyn1 base (`LeadLag=Lag`, delta HV) -- LV rotated `r=2`
   (`bus=lv.3.1.2.0`) gives `lag ≈ 150.09°` (clock 5 = Dyn5, exactly
   `30 + 120*2 mod 360`) -- and for a Dd0 base (both delta) with the same LV
   rotations, confirming the rule is independent of which winding is delta.
   This reaches every clock of the pairing's correct parity EXCEPT the
   polarity-flip clocks `{2, 6, 10}` (the old `Yy6`/`Dd6` case), which need a
   genuinely reversed winding construction no bus-conductor permutation can
   express. A NON-cyclic permutation (e.g. `bus=lv.1.3.2.0`, swapping two
   conductors) reverses the phase-rotation sequence -- a different, unrelated
   winding -- and `_cyclic_rotation_steps` raises `ConversionError` rather
   than silently producing a wrong clock.

**Tap.** `ratio_magnitude = Tap(wdg=1) / Tap(wdg=2)` (both default 1.0).

**Magnetizing branch** (referred to the HV terminal, same formula the
pandapower converter uses for `pfe_kw`/`i0_percent`):
```
pfe_w = %noloadloss/100 * s_rated_va
G_m   = pfe_w / u_hv_v^2                          (0 if pfe_w == 0)
s_nl  = %imag/100 * s_rated_va
L_m   = 1 / (2*pi*f0 * sqrt(s_nl^2 - pfe_w^2) / u_hv_v^2)   (None if %imag == 0)
```
Both `%noloadloss`/`%imag` default to 0 (no magnetizing branch, `G_m=0`,
`L_m=None`) when not set on the DSS element. NOTE: this reproduces the same
formula (hence the same ORDER of voltage effect) as OpenDSS's own
`%noloadloss`/`%imag`, but OpenDSS's internal transformer model places the
magnetizing branch inside its leakage "T" (splitting current between the
windings' half-impedances) rather than as a pure external-terminal shunt (the
pgml/`assembly._transformer` simplification, see
`docs/pgml/modeling/transformer.md`) -- expect a small (~1e-3 pu on a typical
~0.5% magnetizing current) residual vs a live OpenDSS solve for transformers
that carry a magnetizing branch; the leakage-only (no-magnetizing) path
matches to ~1e-7 pu / 1e-5 deg. See
`tests/reference/test_opendss_transformer.py`'s module docstring.

**Not converted:** 3-winding transformers (`ConversionError`), `RegControl`
regulators, tap-changer control, `XfmrCode`/frequency-correction curves.

### Oracle direction (`pgml -> DSS`, live-oracle transformer builder)

`pgml.evaluation.oracles.opendss_oracle._build_circuit_with_real_transformer`
(used by `opendss_dyn_transformer_harmonic_voltages` and the live-oracle tests
in `tests/reference/test_opendss_transformer.py`) is the REVERSE direction: it
emits a real OpenDSS `Transformer` element reproducing a pgml
`Transformer`'s vector group, so pgml's own model can be validated against a
live OpenDSS solve. It shares the same scope as the forward direction, plus:

- **Clock realisation.** `_dss_leadlag_and_rotation(clock, shifting_pairing)`
  is the exact inverse of the forward-direction fold above: it always rotates
  the TO/LV winding only (`r_from=0` always) and picks whichever `LeadLag`
  baseline (`Lag`->1, `Lead`->11 for a Dy/Yd pairing; always `Lag`/baseline 0
  for a matching Yy/Dd pairing) leaves a residual that is a multiple of 4
  clock steps, then emits `bus=...` with that many rotation steps via
  `_dss_rotated_phase_suffix`. Every clock of the pairing's correct parity is
  reachable this way except `{2, 6, 10}` (needs a reversed winding polarity --
  no OpenDSS bus wiring can express it), which raises `NotImplementedError`
  with a message naming the clock.
- **Delta-LV coil referral.** Back-calculates OpenDSS's `%R`/`XHL` from pgml's
  stored (coil-referred) `series_resistance_ohm`/`series_inductance_h` by
  dividing by the SAME factor-of-3 the forward direction multiplies by when
  the TO/LV winding is `DELTA` (see "Leakage" above) -- the old code used
  pgml's stored value directly, which was 3x too large for a delta LV winding
  and produced the mirror image of the forward-direction bug.
- **Zigzag** (`ZIGZAG`/`ZIGZAG_GROUNDED` on either winding) raises
  `NotImplementedError`: OpenDSS's `Transformer` element has no zigzag
  connection at all, so there is no way to build an equivalent live circuit.
- Only 3-phase (`p == 3`) transformers are supported (the clock-realising
  rotation is specific to the 3-phase A/B/C cyclic group); other phase counts
  raise `NotImplementedError`.

### Vsource `basekv` semantics (phase-count dependent; verified empirically)

OpenDSS's general documentation describes `Vsource.basekv` as line-to-line.
That is only true for a `phases>=3` source. For a `phases=1` source, OpenDSS
uses `basekv` DIRECTLY, unscaled, as the solved single-conductor-pair EMF
magnitude -- no internal `sqrt(3)` anywhere (confirmed: `Bus.Voltages()`
magnitude == `BasekV*pu` exactly, for both a genuine line-to-neutral `basekv`
and the historical positive-sequence-equivalent style of feeding in the
parent 3-phase system's L-L nominal, e.g. the IEEE 33-bus fixtures). The
converter's `u_ref_v = BasekV*pu*1000` and `u_rated_v = Bus.kVBase()*
sqrt(3)*1000` formulas need NO phase-count branch: they simply reproduce
whatever magnitude OpenDSS itself solves for at that bus/source, which is
L-L-scaled for `phases>=3` and used as-is for `phases=1`. See
`tests/convert/test_opendss_vsource_basekv.py` for the live-circuit proof and
`docs/pgml/modeling/conventions.md` sec. 1/6.

### A shared `opendssdirect` engine gotcha (test-writing note)

`opendssdirect` wraps ONE process-global DSS engine; `Clear` resets the
active circuit but NOT every engine-wide setting -- notably
`DefaultBaseFrequency` persists across `Clear`. A test module that changes it
(`set DefaultBaseFrequency=50`) silently leaks into the NEXT DSS-driven test
module in the same pytest process, even though that module's own `New
Circuit ... frequency=60` looks like it should set 60 Hz (`Solution.
Frequency()` still reads 50). Any new `opendssdirect`-driven test module
should reassert `set DefaultBaseFrequency=<f0>` right after `Clear` (see
`tests/reference/test_opendss_transformer.py::_dss_clear`) rather than assume
a fresh 60 Hz default.

### Unit conversions (OpenDSS engineering -> SI)
| DSS field              | SI field                        | factor                       |
|------------------------|---------------------------------|------------------------------|
| `Lines.RMatrix()` [Ω/unit] * `Lines.Length()` [unit] | total R [Ω] / length_m | ÷ length_m |
| `Lines.XMatrix()` [Ω/unit] | series_inductance_h_per_m | ÷ (2πf₀ · length_m)    |
| `Lines.CMatrix()` [nF/unit] | shunt_capacitance_f_per_m | × 1e-9 ÷ length_m       |
| `Vsources.BasekV()*PU` [kV] | `Source.u_ref_v` [V]      | × 1000                   |
| `Vsource.r1` [Ω]       | `Source.resistance_ohm`         | via `thevenin_from_z`        |
| `Vsource.x1/(2πf₀)` [H] | `Source.inductance_h`         | via `thevenin_from_z`        |
| `Loads.kW()` [kW]      | `Load.p_nom_w` [W]              | × 1000                       |
| `Loads.kvar()` [kVAR]  | `Load.q_nom_var` [VAR]          | × 1000                       |

### `u_rated_v` convention (fix applied)

OpenDSS `Bus.kVBase()` always returns `BasekV_LL / sqrt(3)` (line-to-neutral kV),
regardless of the number of phases on the circuit element. The pgml schema (like
pandapower and pgm) stores the **line-to-line** rated voltage so the const-Z load
shunt formula `y = conj(S) / u_rated_v^2` is consistent across all converters.

The converter applies: `u_rated_v = kVBase * sqrt(3) * 1000` (recovering `BasekV_LL`).

The old converter stored `kVBase * 1000` (L-N), which was a factor-of-sqrt(3)
error in the const-Z shunt when the DSS-converted Grid is used on the load-flow
path. The Y-bus oracle test (passive network, load-free) was unaffected by this
bug. The fix changes the IEEE 33-bus `u_rated_v` from 7309 V to 12660 V (matching
pandapower). All oracle tests remain green.

### Load connection capture (`THREE_PHASE` only)

`dss.Loads.IsDelta()` determines the connection:
- `False` → `WindingConnection.WYE` (phase-to-neutral / L-N)
- `True`  → `WindingConnection.DELTA` (phase-to-phase / L-L)

Single-phase loads (regardless of `IsDelta()`) always receive `WYE` because
OpenDSS single-phase loads are inherently two-conductor L-N elements.

Under `SINGLE_PHASE_EQUIV`, delta loads are collapsed to `phases=(A,)` with
`connection=None` and an INFO message is logged on `logging.getLogger("pgml")`.

### Phase convention
- Each (bus, phase) in `YNodeOrder` becomes a `(Node.id, Phase.X)` slot.
- Phase numbers: 1=A, 2=B, 3=C, 0=N.
- Under `THREE_PHASE`, multi-phase buses get their real phase tuple (e.g.
  `(Phase.A, Phase.B, Phase.C)` or `(Phase.A,)` for single-phase buses).
- Under `SINGLE_PHASE_EQUIV`, every node has `phases=(Phase.A,)`.

### Length unit codes (OpenDSS → metres)
| Code | Unit | Factor       |
|------|------|--------------|
| 0    | none | 1.0 (total values, length=1 by convention) |
| 1    | mi   | 1609.344     |
| 2    | kft  | 304.8        |
| 3    | km   | 1000.0       |
| 4    | m    | 1.0          |
| 5    | ft   | 0.3048       |
| 6    | in   | 0.0254       |
| 7    | cm   | 0.01         |

### EE convention note (important for harmonics, Phase 3)
OpenDSS `SystemY` / `Export Y` **includes** the following shunt admittances on
the diagonal beyond the passive line pi-model:
1. **Vsource Norton shunt** `Y_s = (R1 + jX1)^-1` added to the source bus.
2. **Load shunt** (model-dependent):
   - model=1 (const-P): shunt evaluated at the current iteration voltage
     (approximately `conj(S)/|V_sol|^2`); changes each Newton iteration.
   - model=2 (const-Z): shunt `conj(S)/(kv*1000)^2` at rated voltage; static.
   - `NeglectLoadY=yes`: pure current source, no shunt in Y.
3. **No line G/C shunts** are included in `Export Y` if `c1=c0=0` (case33bw).

For the Y-bus oracle test: the cleanest comparison is to build the OpenDSS
circuit **without loads** (so only line series/shunt + Vsource Norton shunt
appear), then compare OpenDSS Y to our Y minus load shunts, accounting for the
Vsource shunt on the source bus diagonal. See `tests/reference/test_ieee33_opendss.py`.

### Validated on
IEEE 33-bus Baran & Wu circuit built from `pandapower.networks.case33bw()` data,
60 Hz, 33 buses, 32 in-service lines, 1 Vsource, 32 loads.
Oracle test: `tests/reference/test_ieee33_opendss.py`.
Phase-mode test: `tests/convert/test_opendss_phase_mode.py`.
Transformer oracle (MV source -> Dyn11 / Yy0 / Dyn5 / YNd5 20/0.4 kV
transformer -> LV load, live `Solve` vs pgml):
`tests/reference/test_opendss_transformer.py` -- Dyn11/Yy0/Dyn5 via the
nonlinear `solve_power_flow` (voltage magnitude ~1e-7 pu, Dyn5 ~1e-8 pu; angle
~1e-5 deg); YNd5 via the linear `assemble_ybus`+`solve_harmonic` path (see the
class docstring for why -- a delta-only LV secondary is an isolated island
with no absolute-voltage reference pgml's nonlinear solver can anchor, a
pre-existing solver gap orthogonal to this conversion, worked around with a
WYE grounding load + DSS `model=2`; ~2e-8 pu / ~4e-7 deg); a dedicated
sub-test reproduces the pre-fix (no-factor-3) leakage and confirms it misses
by > 1e-3 pu. Cyclic winding-bus rotation parsing (`_cyclic_rotation_steps`/
`_parse_transformer_winding_bus`, normalized-phases + folded-clock checks,
non-cyclic-permutation rejection) and the oracle-direction scope guards
(zigzag, clock-6 `NotImplementedError`) are covered in the same file.
Magnetizing-branch field conversion checked separately against its closed
form; scope guards for 3-winding, ungrounded-neutral and differing-kVA
transformers.
Vsource `basekv` semantics: `tests/convert/test_opendss_vsource_basekv.py`.
Line phase-permutation (differing `from_phases`/`to_phases`, both a 3-phase
and a degenerate 1-phase case) and the positive-sequence Z1 = self - mutual
reduction: `tests/reference/test_opendss_line_phase_permutation.py`.
Load model conversion (Model=2/5/8 live-oracle voltage parity, the ZIPV
cutoff warning, and the 3/4/6/7 fallback): `tests/reference/test_opendss_load_model.py`.
Four-wire (explicit neutral conductor) voltage parity, including the
grounding-`Reactor`-as-`ShuntAppliance` anchoring the neutral rail, and the WYE
`return_path` routing (grounded, neutral, and a BOTH-kinds 4-wire bus):
`tests/convert/test_opendss_phase_mode.py`
(`TestFourWire.test_voltage_parity_vs_live_opendss`,
`test_grounded_load_on_neutral_carrying_node_uses_return_path_ground`,
`test_four_wire_mixed_return_paths_convert_and_match_opendss`).
Capacitor/Reactor/Generator/PVSystem/Storage field conversion, the DELTA
capacitor/reactor conversion (+ live voltage parity) and the coupled-reactor
out-of-scope warning, the generic dropped-element warning
(`Isource`/`Monitor`/`EnergyMeter`), and the Vsource-impedance warning:
`tests/convert/test_opendss_shunt_and_der_elements.py`.
