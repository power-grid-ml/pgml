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

### Supported element types (IEEE 33-bus element set + transformer-bearing feeders)
| DSS type    | Schema type       | Notes                                      |
|-------------|-------------------|--------------------------------------------|
| Line        | `Line`            | 1- or multi-phase; n×n R/X/C from matrix API |
| Transformer | `Transformer`     | Two-winding only; see below                |
| Vsource     | `Source`          | R1/X1 via text commands; `thevenin_from_z` |
| Load        | `Load`            | kW/kvar total; `IsDelta()` -> `connection` |

Capacitor/Reactor extension is structurally straightforward.

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
