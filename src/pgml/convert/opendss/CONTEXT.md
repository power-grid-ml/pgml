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
R_lv_ohm  = (%R_wdg1 + %R_wdg2) / 100 * Z_base_LV
X_lv_ohm  = %XHL / 100 * Z_base_LV
L_lv_h    = X_lv_ohm / (2*pi*f0)
```
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

**Vector group / clock.** OpenDSS has no explicit clock parameter; its
`LeadLag` toggle is the only source of clock information, and only means
anything for a Dy/Yd (delta-wye) pairing: `Lag`/`ANSI` (default) -> clock 1
(`shift_deg=30`); `Lead`/`Euro` -> clock 11 (`shift_deg=330`) -- verified
against a live solve (`Bus.puVmagAngle()` shows the LV bus leading/lagging the
HV bus by ~30 deg accordingly). A matching Yy or Dd pairing gets clock 0
(`shift_deg=0`); `Yy6`/`Dd6` (180 deg reversed polarity) is NOT detectable
from a plain `Transformer` element and is not converted.

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
Transformer oracle (MV source -> Dyn11 / Yy0 20/0.4 kV transformer -> LV
load, live `Solve` vs `solve_power_flow`): `tests/reference/test_opendss_transformer.py`
(voltage magnitude ~1e-7 pu, angle ~1e-5 deg on the leakage/vector-group path;
magnetizing-branch field conversion checked separately against its closed form;
scope guards for 3-winding, ungrounded-neutral and differing-kVA transformers).
Vsource `basekv` semantics: `tests/convert/test_opendss_vsource_basekv.py`.
