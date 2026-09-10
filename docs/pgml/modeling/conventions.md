# Modelling conventions vs pandapower, OpenDSS and power-grid-model

pgml has one internal convention set. Every converter translates its source tool into that
form, so the solver never sees a foreign convention. This page pins those definitions and
records how the three reference tools differ. power-grid-model is abbreviated pgm below.

Deeper derivations have their own pages: the [two-winding transformer](transformer.md),
[Carson line constants](references/opendss/carson.md), [OpenDSS harmonics](references/opendss/harmonics.md),
the [harmonic line model](harmonic-line-model.md) and [asymmetric modelling](asymmetric.md).
The schema is the data source of truth, see the [Schemas API reference](../api/schemas.rst).

## pgml canonical form

- Phase domain. Quantities are stored per phase or as `n×n` phase matrices, never as
  sequence components. Sequence inputs are decomposed at conversion time,
  `self=(Z0+2·Z1)/3` and `mutual=(Z0−Z1)/3`.
- SI units throughout. Volts, ohms, henries, farads, siemens, watts, vars. No kV, MW,
  per-unit or per-km inside the library. Converters are the only unit boundary.
- Inductance and capacitance are stored, reactance and susceptance are not. Both follow
  from a frequency, `X(h)=2π·h·f₀·L` and `B(h)=2π·h·f₀·C`. Resistance
  frequency-dependence is an explicit law.
- Phasors are `(real, imag)` pairs. Results are indexed by `frequency_hz`.
- `Node.u_rated_v` is line-to-line for a node with three or more phases, and
  line-to-neutral for a one-phase node, where there is no √3 to apply.
- Every per-phase voltage the solver touches is line-to-neutral. One helper derives it from
  `Node.u_rated_v` and the element connection, returning `u_rated_v/√3` for a WYE element
  on a node with three or more phases and `u_rated_v` for a DELTA element or a one-phase
  node. The const-Z/ZIP load model, the slack EMF, the harmonic source admittance and the
  per-unit reporting all call it, so they cannot drift apart.
- The slack EMF is the line-to-neutral phasor. Converters pass the line-to-line magnitude
  and the source builder divides by √3 for the balanced three-phase wye expansion.

A source tool's choice of base (line-to-line or line-to-neutral, HV- or LV-referred
transformer impedance, per-km or total, imperial or metric earth return) is resolved inside
the converter and recorded in `Provenance`.

## Base and nominal voltage

| | nominal-voltage field | stored as | per-unit base / output voltage |
|---|---|---|---|
| pgml | `Node.u_rated_v` [V] | L-L (≥3φ), L-N (1φ) | working voltages are L-N; pu = `\|V_LN\|/(u_rated/√3)` |
| pandapower | `bus.vn_kv` [kV] | L-L | `res_bus.vm_pu = \|V_LL\|/vn_kv`; asymmetric `res_bus_3ph` uses the L-N base `vn_kv/√3` |
| OpenDSS | `Vsource.basekv` [kV] | L-L for `phases>=3`, used directly for `phases=1` | `Bus.kVBase()` always returns the matched voltage base over √3; `AllBusVolts` are L-N phasors |
| pgm | `node.u_rated` [V] | L-L | symmetric output `u` is L-L with `u_pu=u/u_rated`; asymmetric output `u` is L-N |

Storing the universal nameplate and deriving the line-to-neutral working voltage at one
point of use makes the const-Z load admittance `y=conj(S)/V²` agree with pandapower's
positive-sequence reference in the one-phase path (`V=V_LL`) and stay physically correct
per phase in the three-phase path (`V=V_LN`, total power preserved).

pandapower `vn_kv·1000` and pgm `u_rated` are already line-to-line. `Bus.kVBase()` is
line-to-neutral for every OpenDSS bus whatever its local phase count, so the converter
recovers the nameplate as `kVBase·√3·1000`.

OpenDSS documents `Vsource.basekv` as line-to-line, and that holds only for a `phases>=3`
source. For a `phases=1` source OpenDSS uses `basekv` directly and unscaled as the solved
single-conductor-pair EMF magnitude, with no √3 anywhere. The converter formulas
`u_ref_v = basekv·pu·1000` and `u_rated_v = kVBase()·√3·1000` therefore need no phase-count
branch. They mirror whatever magnitude OpenDSS itself solves for. This holds whether a
one-phase `basekv` carries a genuine line-to-neutral value or the positive-sequence
convention of feeding in the parent three-phase system's line-to-line nominal.

Two further traps. pgm asymmetric power flow reports `u` as line-to-neutral, so multiply by
√3, or use `u_pu` on the matching base, before comparing against pgml's `u_rated`. And the
three-phase slack EMF must be line-to-neutral; pinning the line-to-line magnitude on each
phase makes every three-phase voltage √3 too high, about 1.73 pu.

## Transformer impedance and the side it is referred to

This is the headline cross-tool difference.

| | short-circuit params | referred to | recovery of the pgml series R/L |
|---|---|---|---|
| pgml | `series_resistance_ohm`, `series_inductance_h` | TO/LV coil | stored directly |
| pandapower | `vk_percent`, `vkr_percent`, `sn_mva` | LV side | `Z_base_LV=vn_lv_v²/sn_va`, `R_ll=vkr%·Z_base_LV`, `\|Z_ll\|=vk%·Z_base_LV`, `X_ll=√(\|Z_ll\|²−R_ll²)` |
| pgm | `uk`, `pk`, `sn`, `u2` | to-side (LV) | `R=pk·u2_eff²/sn²`, `\|Z\|=uk·u2_eff²/sn`, `X=√(\|Z\|²−R²)`, `u2_eff` tap-adjusted |
| OpenDSS | per-winding `%R`, inter-winding `XHL` | percent, so base-invariant, on the standard L-L base | `R_ll=(%R_wdg1+%R_wdg2)/100·Z_base_LV`, `X_ll=XHL%/100·Z_base_LV`, `Z_base_LV=kV_lv²·1000/kVA` |

Every converter then stores the value referred to the TO coil, with `L=X/2πf₀`, in both
phase modes. A DELTA to-winding carries three times the line-to-line-base impedance,
because a delta coil is rated at the line-to-line voltage with a third of the per-phase
kVA, so its stored value is multiplied by 3. A wye or zigzag to-winding is stored
unchanged.

pgml refers the leakage to the TO/LV coil, the same side as pandapower and pgm, so
`vk/vkr` and `uk/pk` convert with one LV base and no extra referral. It is also the natural
side for the winding-incidence primitive `Y = Nᵀ·Y_winding·N`, where the leakage `y` sits
on the LV coil block and the HV self-block picks up `1/τ²` from the turns ratio. OpenDSS's
`%R` and `XHL` are percent quantities, base-invariant, so they need only the LV base
impedance to return to ohms. That recovery assumes both windings share one kVA rating, and
the converter raises `ConversionError` if they do not. Watch the OpenDSS input syntax
here: the sequential `~ wdg=1 ... kVA=x` form silently re-syncs both windings to the last
kVA given, so only the array form `kvas=[x, y]` creates a genuine per-winding mismatch.

Magnetizing branch. pgml refers the magnetizing shunt `y_m=G_m+jB_m` to the HV terminal
(`magnetizing_conductance_s`, `magnetizing_inductance_h`). The pandapower converter
computes `G_m=pfe_w/u_hv²` and `B_m` from `i0%` and `sn` on the HV base; the pgm converter
refers `i0` and `p0`, which pgm defines on the to-side `u2`, the same way by the square
nameplate ratio. pandapower internally keeps the branch on the LV base split into the
pi-shunt, physically equivalent after the turns ratio but numerically different, so do not
compare raw numbers without re-referring. pgm's own stamp is a different topology, half the
magnetizing admittance on the to-side and half, through the tap, on the from-side. The
residual against pgml's HV-only shunt is small, around 1.5e-4 pu for a realistic `i0` of
0.5%, and at machine precision when `i0=p0=0`.

## Transformer ratio, tap and vector group

| | nominal ratio | tap (off-nominal) | vector-group phase shift |
|---|---|---|---|
| pgml | from rated coil voltages and connections (delta coil = L-L, wye or zigzag coil = `u/√3`) | `ComplexTap.ratio_magnitude`, 1.0 on tap | `tap.shift_deg = clock·30`, positive means LV lags HV |
| pandapower | `vn_hv_kv/vn_lv_kv` | `tap_pos`, `tap_neutral`, `tap_step_percent`, `tap_side` | `shift_degree`, positive means LV lags, as in pgml |
| OpenDSS | ratio of winding coil kV | tap per winding | `LeadLag` (`Lag` gives 30°, `Lead` gives 330°, Dy/Yd only) plus a cyclic winding-bus rotation of ±120° per step |
| pgm | `u1/u2` | `tap_side`, `pos`, `nom`, `size`, `min`, `max` | `clock` 0 to 12, with `winding_from`/`winding_to` enums |

The nominal ratio and the ±30° clock shift come from the rated voltages and winding
connections, following OpenDSS. So `tap.ratio_magnitude` carries only the off-nominal
deviation, near 1.0, and `tap.shift_deg` carries the clock. The √3 of a delta winding
cancels against the delta incidence `M`, so the positive-sequence block reduces exactly to
the classical off-nominal-tap pi. The [transformer model](transformer.md) derives this and
explains how the incidence realising a given clock is selected.

Per-tool notes:

- pandapower. The vector group comes from `net.trafo['vector_group']`, else the `std_type`
  catalog entry, and a clock-less form such as `'Dyn'` is accepted. It is cross-checked
  against `shift_degree`, and a mismatch raises `ConversionError` rather than silently
  preferring one source. pandapower stores `shift_degree` as a positive `clock·30`, so
  Dyn11 is 330. With no vector-group string anywhere, as in a plain MATPOWER import, the
  connection falls back on the shift parity: even gives `WYE_GROUNDED`/`WYE_GROUNDED`, odd
  gives `DELTA`/`WYE_GROUNDED`. Tap position is read and is NaN-safe, with
  `tap_side='hv'` giving `ratio_magnitude=1+delta` and `'lv'` giving `1/(1+delta)`, so a
  positive HV-side tap lowers the LV voltage. An ideal phase-shifter tap
  (`tap_step_degree` nonzero, or `tap_phase_shifter=True`) is not modelled and raises.
- pgm. `clock*30` is identical to pgml's convention with no sign flip, and `tap_side`
  (0 for from, anything else for to) selects which nameplate voltage the tap volts are
  added to.
- OpenDSS. Winding 1 is HV/from, winding 2 is LV/to. `LeadLag` sets a clock 1 or 11
  baseline for a Dy or Yd pairing, and clock 0 for Yy or Dd, since OpenDSS has no explicit
  clock parameter. A winding whose bus string cyclically rotates the phase-conductor order,
  such as `bus=lv.2.3.1.0`, folds a further ±4 clock steps into `tap.shift_deg`; phases are
  normalised back to canonical A/B/C. Together these reach every clock of a pairing's
  parity.

The OpenDSS transformer scope is narrower than the core model. Only two-winding units
convert, and only solidly grounded wye windings (the shorthand bus, or an explicit `.0`) or
delta windings. Three-winding units, and an explicit non-zero neutral node that is floating
or impedance-grounded, raise `ConversionError`. So does a non-cyclic winding-bus
permutation, such as swapping two phase conductors, which reverses the phase rotation;
raising avoids silently producing a wrong clock. The polarity-flip clocks 2, 6 and 10 need
a reversed winding construction that no bus wiring can express, so this converter never
produces them, and zigzag has no OpenDSS `Transformer` connection at all. Regulators,
tap-changer control and frequency-correction curves (`XfmrCode`, `FreqMultCurve`) are not
read. The magnetizing branch (`%noloadloss`, `%imag`) converts with the same closed form as
the pandapower path, but pgml stamps it as a plain HV-terminal shunt while OpenDSS places
it inside the leakage T, so the two agree in direction and order of magnitude only.

## Power sign convention

| | load P/Q | generator P/Q |
|---|---|---|
| pgml | positive is consumption | `Generator` injects, sign −1 internally |
| pandapower | `load.p_mw>0` is consumption | `sgen`/`gen` `p_mw>0` is injection |
| OpenDSS | `Load` positive is consumption | `Generator` positive is injection |
| pgm | `sym_load`/`asym_load` positive is consumption | `sym_gen` positive is injection |

All four agree, so converters pass P and Q through unchanged. ZIP behaviour maps across
too. pgm `LoadGenType` (`const_power`, `const_impedance`, `const_current`) becomes pgml's
`LoadModel`, and pandapower's `const_z_p_percent` and `const_i_p_percent` columns, with
their `_q_` twins, become `ZipCoefficients`.

## Units and internal representation

| | input units | internal | base |
|---|---|---|---|
| pgml | converters only | SI, phase domain | none, absolute SI |
| pandapower | kV, MW, MVAr, Ω/km, nF/km, % | per-unit (MATPOWER) | `sn_mva` system base plus per-bus `vn_kv` |
| OpenDSS | actual engineering units, length in `units=` | actual units | per element |
| pgm | SI (V, W, VA, Ω, F, S) | SI | none |

pgm is the least lossy source, being pure SI; its converter only does `L=X/2πf₀` and
`G=tan·2πf₀·C`. pandapower needs kV to V, km to m, nF to F, MW to W and the per-km
divisions. OpenDSS needs the `units=` length conversion, the crux of the earth-return
calibration below, plus the `kVBase·√3` recovery. pandapower's exported `Ybus` is per-unit
on the ppc base, so scale it by `Z_base=vn_kv²/sn_mva` before comparing against pgml's SI Y.

## Source and slack

| | reference voltage | Thévenin impedance | zero-sequence source Z |
|---|---|---|---|
| pgml | `Source.u_ref_v`, L-N per phase for 3φ, with `u_angle_deg` | diagonal per-phase R/L | equal to the positive sequence |
| pandapower | `ext_grid.vm_pu`·`vn_kv` (L-L), `va_degree` | from `s_sc_max_mva`, `rx_max` | `r0x0_max`, `x0x_max`, not read |
| OpenDSS | `Vsource.basekv`·`pu`, `angle` | `R1/X1`, or `MVAsc3`/`MVAsc1` with `x1r1` | `R0`, `X0`, not read |
| pgm | `source.u_ref`·`u_rated` (L-L), `u_ref_angle` | from `sk`, `rx_ratio` | `z01_ratio`, not read |

Every converter passes the magnitude OpenDSS itself would use as the solved per-conductor
EMF. The source builder divides that by √3 under three-phase operation to get the
per-phase line-to-neutral EMF and leaves a one-phase source unchanged, matching OpenDSS's
own treatment. `id_map["slack_v_complex"]` keeps the same raw phasor as a convenience for
the single-phase ideal-slack `v_fixed`. At harmonics the source EMF is zero, a short, and
the source contributes only its Norton shunt `Y_s(h)=1/(R+j·2πh·f₀·L)`.

Known gap. The zero-sequence source impedance is taken equal to the positive-sequence
value, since none of `r0x0_max`, `R0`/`X0` or `z01_ratio` are consumed. This matters only
for three-phase asymmetric studies where the source zero-sequence path is significant.

## Line model

| | parameters | per length or total | form |
|---|---|---|---|
| pgml | `series_resistance_ohm_per_m`, `series_inductance_h_per_m`, `shunt_capacitance_f_per_m`, `shunt_conductance_s_per_m` | per metre | `n×n` phase matrices, or `conductor_geometry` |
| pandapower | `r_ohm_per_km`, `x_ohm_per_km`, `c_nf_per_km`, `g_us_per_km`, plus `r0`/`x0`/`c0` | per km | sequence |
| OpenDSS | `R1/X1/R0/X0`, `C1/C0`, or `Rmatrix`/`Xmatrix`/`Cmatrix`, or geometry | per `units=` | sequence, matrix or geometry; matrix and geometry win |
| pgm | `r1/x1/c1/tan1`, plus `r0/x0/c0/tan0` | total (Ω, F) | sequence |

Sequence inputs become 3×3 phase matrices through `self=(Z0+2·Z1)/3` and
`mutual=(Z0−Z1)/3`, applied to R, X, C and G, followed by `L=X/2πf₀`. When a dataset has no
native zero-sequence data, `r0`, `x0` and `c0` default to ratios from
`config.line.zero_sequence.*`, and an explicit native value always wins. pgm total values
use a `length_m=1` idiom so the per-metre times length product reproduces the total
exactly. OpenDSS native `n×n` matrices are read as matrices with no sequence assumption,
while pandapower and pgm route through the sequence path.

## Harmonics and earth return

Only OpenDSS and pgml model harmonics. pandapower and pgm are fundamental-only, with no
harmonic power flow, no frequency-dependent line constants and no earth-return model, so
they serve as load-flow oracles. OpenDSS is the harmonic reference.

OpenDSS recomputes line impedance at every harmonic with a Carson/Deri earth-return and
skin model, for both geometry- and R/X-defined lines. The naive "R constant, X proportional
to h" is therefore wrong. R rises from skin effect on the earth return, and X is
sub-linear, because the earth-return log term shrinks as penetration depth drops. The
default earth model is DERI.

pgml offers three line models, chosen per study:

- Geometry Carson/Deri (`conductor_geometry`) uses the full complex-penetration formula. It
  matches OpenDSS bit-exactly, relative Z error around 1e-13, on every order including
  triplen, because feeding the same geometry to both engines removes any earth-model
  ambiguity. Use it for OpenDSS parity.
- Positive-sequence (`apply_positive_sequence_harmonic_model`) applies `X1(h)=X1·h` plus
  skin effect on `R1`, with no earth term, which cancels in the positive sequence. This is
  physically representative for balanced R/X feeders.
- Sequence-aware (`apply_sequence_aware_harmonic_model`, the three-phase default for R/X
  lines) uses an earth-free `Z1` plus a zero-sequence `Z0` carrying the Carson earth
  resistance `3·(Re(f)−Re(f₀))`. It is analytic and never non-physical, but `X0` stays
  linear in h, so it diverges from OpenDSS's Carson `Z0` on the triplen orders.

The earth-return calibration trap is worth a factor of about 3.28. The classical Carson
earth resistance is `Re(f)=ω·μ₀/8 = π²·f·10⁻⁷ Ω/m`, geometry-independent and proportional
to frequency. pgml's configurable
`line.earth_return.resistance_coeff_ohm_per_m_per_hz` defaults to that physical metric
value, `π²·10⁻⁷`. OpenDSS exposes the same physics through per-LineCode `Rg` and `Xg`, but
its defaults are calibrated for imperial length units, so on a `units=m` line they are
about 3.28 times smaller, that being the number of metres per foot. Comparing pgml's
sequence-aware `Z0` against an OpenDSS R/X line therefore shows a zero-sequence gap from
two compounding causes, the units-calibrated earth resistance and the linear-versus-
sub-linear `X0`. Both vanish on the geometry path, where the geometry and the Carson model
are identical on both sides. The transformer vector group is independent of this, since a
Dyn delta traps the zero sequence identically in both engines.

Transformer frequency scaling follows OpenDSS `XRConst=No`, its default. R is fixed and the
leakage X scales with h, which pgml mirrors through `X(h)=2π·h·f₀·L` at constant R. A
frequency-correction curve is not modelled.

## Converter coverage

The core model supports more than the converters read. Where a source convention is not
read, a foreign network under-converts, and these are the current gaps.

| | converted | not read |
|---|---|---|
| OpenDSS | `Transformer` (scope above), `Line`, `Load`, `Capacitor`, `Reactor`, `Generator`, `PVSystem`, `Storage` | three-winding transformers, regulators and tap-changer control, frequency-correction curves, a coupled `Rmatrix`/`Xmatrix` reactor, a non-grounded or two-bus terminal-2 shunt reference |
| pandapower | `trafo`, `line`, `load`, `sgen`, `asymmetric_load`, `switch`, `ext_grid` | `shunt`, `trafo3w`, `impedance`, `ward`, `xward`, `dcline`, `storage`, `motor`, `asymmetric_sgen`, `r0x0_max`/`x0x_max` |
| pgm | `node`, `line`, `transformer`, `sym_load`, `asym_load`, `sym_gen`, `source` | `asym_gen`, `three_winding_transformer`, `transformer_tap_regulator`, `shunt`, `link`, `uk_min`/`uk_max`/`pk_min`/`pk_max`, `i0_zero_sequence`/`p0_zero_sequence`, `z01_ratio`, line `tan0` |

Details worth knowing before a conversion:

- OpenDSS `Line` reads the native `n×n` R/X/C matrices with a per-terminal phase
  permutation. `Load` maps models 1, 2, 5 and 8 to `LoadModel` and `ZipCoefficients`, while
  models 3, 4, 6 and 7 fall back to `CONST_POWER` with a warning. Each WYE element's own
  resolved return conductor carries over as `InjectionAppliance.return_path`. `Capacitor`
  and `Reactor` become a `ShuntAppliance`, solidly grounded WYE or delta, with per-leg G
  and C from OpenDSS's own resolved values.
- Anything else in a DSS file (`Isource`, `Monitor`, `EnergyMeter`, `CapControl`,
  `InvControl`, `Relay`, `Sensor` and the rest) is enumerated generically and warns with
  its kind and count, so nothing vanishes silently. A non-negligible Vsource `R1`/`X1`
  warns under the default `slack="ideal"`, where it is ignored, and is used under
  `slack="norton"`.
- pandapower `line` and `trafo` honour the `parallel` column, dividing the series impedance
  and multiplying the shunt admittance and rated power; `parallel==1` stays byte-identical.
  An open bus-line or bus-transformer switch takes the whole branch out of service. That is
  an approximation, since pandapower keeps the still-connected terminal energised through
  an internal auxiliary bus, so pgml drops that terminal's shunt too.
- pandapower `gen`, a PV bus with fixed P, regulated `vm_pu` and free Q within its reactive
  limits, is dropped by default. It converts only under
  `gen_mode=GenMode.VOLT_VAR_APPROX`, which approximates the PV bus with a steep Volt-VAr
  droop centred on `vm_pu` and saturating at the reactive limits, holding the voltage
  magnitude near rather than at the setpoint. A `gen` row on the `ext_grid` bus, or one
  flagged `slack`, is skipped. See the [DER decision record](der-pv-storage.md).
- pgm stores no base frequency, so the caller must pass the correct `base_frequency_hz`. A
  50 versus 60 Hz mismatch silently scales every L and C.
