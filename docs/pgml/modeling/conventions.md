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

Magnetizing branch. pgml stores the magnetizing shunt `y_m=G_m+jB_m` referred to the
HV/from terminal (`magnetizing_conductance_s`, `magnetizing_inductance_h`). Where it is
stamped is the documented choice `transformer.magnetizing_placement`, because the three
reference engines disagree. OpenDSS attaches the whole branch to its last winding's
terminal, verified on a live `Yprim` difference; power-grid-model splits it half onto `Y_tt`
and half through the tap onto `Y_ff`; pandapower keeps it on the LV base inside its pi
shunt. These are different topologies rather than different referrals, since the magnetizing
current either does or does not see a winding's leakage drop. Measured deviations from a
live OpenDSS solve on a 500 kVA 20/0.4 kV unit: `to_terminal` 3e-10 to 2e-9 pu, `split` half
of `from_terminal`, and `from_terminal` (the shipped default) 9.2e-5, 2.2e-4 and 8.2e-4 pu
at magnetizing currents of 0.1, 0.5 and 2 %.

The conversion of the percentages differs per tool as well. pandapower's `i0_percent` is the
total no-load current, so `B=√(I0²−G²)`; OpenDSS's `%imag` is the susceptance itself, read
as `B_m = %imag/100·S/u_hv²` with no Pythagorean subtraction, and `%imag` below
`%noloadloss` is accepted.

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
the pandapower path. Where pgml stamps it is the documented choice
`transformer.magnetizing_placement`, whose `to_terminal` setting reproduces OpenDSS's own
placement to 3e-10 pu; the shipped `from_terminal` default deviates by up to 8.2e-4 pu at a
2 % magnetizing current.

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
| pgml | `Source.u_ref_v`, L-N per phase for 3φ, with `u_angle_deg` | per-phase self and mutual R/L | `Z_self=(Z0+2·Z1)/3`, `Z_mutual=(Z0−Z1)/3` |
| pandapower | `ext_grid.vm_pu`·`vn_kv` (L-L), `va_degree` | from `s_sc_max_mva`, `rx_max` | `x0x_max`, `r0x0_max` |
| OpenDSS | `Vsource.basekv`·`pu`, `angle` | `R1/X1`, or `MVAsc3`/`MVAsc1` with `x1r1` | `R0`, `X0` |
| pgm | `source.u_ref`·`u_rated` (L-L), `u_ref_angle` | from `sk`, `rx_ratio` | `z01_ratio` |

Every converter passes the magnitude OpenDSS itself would use as the solved per-conductor
EMF. The source builder divides that by √3 under three-phase operation to get the
per-phase line-to-neutral EMF and leaves a one-phase source unchanged, matching OpenDSS's
own treatment. `id_map["slack_v_complex"]` keeps the same raw phasor as a convenience for
the single-phase ideal-slack `v_fixed`. At harmonics the source EMF is zero, a short, and
the source contributes only its Norton shunt `Y_s(h)=1/(R+j·2πh·f₀·L)`.

Source zero sequence. A `Source`'s per-phase Thévenin is sequence-aware: each converter
reads its tool's native zero-sequence data into `Z_self=(Z0+2·Z1)/3` and
`Z_mutual=(Z0−Z1)/3`, the same identity the line path uses. Without native data the
documented `source.zero_sequence.{r0_over_r1, x0_over_x1}` ratios apply, shipping at 1.0 so
that `Z0=Z1`, which is what power-grid-model and pandapower themselves default to, and a
warning names the source. The negative sequence is always `Z2=Z1`, correct for a passive
upstream network; a rotating-machine source with `Z2≠Z1` would need the third circulant
entry.

Two differences to pandapower's own unbalanced power flow are worth knowing. It multiplies
its zero-sequence ext-grid shunt by the IEC voltage factor `c=1.1` even in power-flow mode,
and it pins the positive sequence as an ideal slack while putting the short-circuit
impedance in the negative-sequence network. Measured on a four-wire LV feeder fed directly
by an OpenDSS Vsource with `Z0≠Z1`, the per-phase voltages agree with a live OpenDSS solve
to 7.8e-8 V at the fundamental and 5.6e-9 V at harmonics 3 to 9, where assuming `Z0=Z1`
misses the triplen voltage by more than 100 % of its magnitude.

Upstream harmonic distortion is an operating-point quantity rather than grid data, because
the upstream network's harmonic voltage changes minute by minute while the grid description
does not. A `Source` has no spectrum field. Supply the background per solve as
`solve_harmonic_flow(..., node_sources=[NodeHarmonicSource(node_id=..., kind="voltage",
spectrum=...)])` at the source's node, or reproducibly through `pgml.scenarios`'
`BackgroundHarmonicConfig`, which realizes one voltage-kind node source per in-service
`Source` from a config plus a seed. At orders `h > 1` the source itself contributes its
Norton shunt, plus that EMF when one is supplied.

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
`line.zero_sequence.*`, and an explicit native value always wins. The converter then logs a
warning naming the ratios it used and the number of lines affected, because every unbalanced
and triplen result on that grid rests on them. pgm total values use a `length_m=1` idiom so
the per-metre times length product reproduces the total exactly. OpenDSS native `n×n`
matrices are read as matrices with no sequence assumption, while pandapower and pgm route
through the sequence path.

Every line model is a lumped pi branch: `Z=z·length`, `Y=y·length` split half to each
terminal, with no hyperbolic long-line correction and no distributed-parameter model. pgml
is a frequency-domain steady-state engine, so standing-wave and travelling-wave phenomena
are outside its scope. Split a long line into segments when its electrical length stops
being small at the highest order of interest.

A shunt reactor is modelled with an inductance (`inductance_h`), so its susceptance
magnitude falls as `1/h`. An OpenDSS `Reactor` with `R=0` maps exactly at every order; one
with a series resistance is converted as the equivalent parallel pair at the fundamental,
which keeps its loss term flat where the series branch would decay as `1/h²`, and the
converter warns. A pandapower `net.shunt` converts to a fixed WYE `ShuntAppliance`, `G` from
`p_mw` and `C` from `−q_mvar/(2πf₀)`, both referred to the shunt's own rated voltage. An
inductive shunt (`q_mvar > 0`) therefore becomes a negative capacitance: exact at the
fundamental, but its susceptance magnitude rises with frequency where a real reactor's
falls, so harmonic results at such a bus are not faithful. The converter warns and names the
count.

## Ideal branches: switches, couplers and jumpers

A branch whose series impedance is exactly zero, such as a closed switch with no impedance
data, a bus coupler or jumper modelled as a zero-impedance line, or a zero-length line, is
an ideal conductor. The nodal formulation has no stamp for it, because it inverts every
branch's series impedance, so the solve imposes what the element actually states: the two
terminals have the same voltage. Their node-phase rows are collapsed into one row of the
solved system, the reduced system is solved, and the result is reported on the original node
ids, where every node of a fused group carries the group's voltage. The current through such
a branch follows from Kirchhoff's law at the fused node. Two ideal branches in parallel
leave a circulating current undetermined, and the reported split is then the minimum-norm
one, named in a warning.

This is exact, and it is what pandapower, which merges the buses of a closed bus-bus switch,
and power-grid-model, whose `link` is a perfect connection, describe. The alternative is a
small stand-in resistance, `branch.near_ideal_series_resistance_ohm`, which stays available
for a branch that has to remain stamped, above all one whose state a `branch_states` sweep
toggles. It is not free: it adds its own voltage drop, and it raises that row's admittance
scale and with it the smallest power mismatch the solve can reach. On a 132 kV network with
three coupler lines, 1e-4 Ω multiplies the condition number of the assembled system by 7600,
from 6.5e2 to 4.9e6. Under
`branch.zero_impedance: error` such a branch is refused by name instead of fused, and the
message points at both ways out.

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

- Geometry Carson/Deri (`conductor_geometry`) uses the full complex-penetration formula. On
  the same geometry it agrees with OpenDSS to 4.8e-8 relative on `Z` on every order
  including triplen, because feeding the same geometry to both engines removes any
  earth-model ambiguity. The residual is the `μ0` constant OpenDSS truncates. Use it for
  OpenDSS parity.
- Positive-sequence (`apply_positive_sequence_harmonic_model`) applies `X1(h)=X1·h` plus
  skin effect on `R1`, with no earth term, which cancels in the positive sequence. This is
  physically representative for balanced R/X feeders.
- Sequence-aware (`apply_sequence_aware_harmonic_model`, the three-phase default for R/X
  lines) uses an earth-free `Z1` plus a zero-sequence `Z0` carrying the Carson earth
  resistance `3·(Re(f)−Re(f₀))`. It is analytic and never non-physical. `X0` is linear in
  h by default, which suits cables and ratio-derived zero-sequence data; it differs from
  OpenDSS's `Xg`-corrected `Z0` on the triplen orders unless `carson_sublinear` is
  selected.

The same physics is implemented on both sides. OpenDSS corrects a sequence-defined line as
`R += Rg·(h−1)` and `X = h·(X − 0.5·KXg·ln h)` per matrix entry, which in sequence terms is
pgml's `R0(h) = R0 + 3·(Re(f) − Re(f₀))` and, with
`line.earth_return.x0_frequency = carson_sublinear`, `X0(h) = h·(X0 − 1.5·kx·f₀·ln h)`. What
differs is the value of the earth parameters. OpenDSS's defaults `Rg = 0.01805` and
`Xg = 0.155081` are the physical Carson values at 60 Hz in ohms per 1000 ft and are
reinterpreted in the line's `units=`, so on a metric line they are 3.28 times too small per
km, or 3280 times too large per metre. pgml's configurable
`line.earth_return.resistance_coeff_ohm_per_m_per_hz` instead defaults to the physical
metric value `π²·10⁻⁷`. Two further OpenDSS traps: those defaults are not rescaled for a
50 Hz base frequency, and every element's base frequency comes from the global
`DefaultBaseFrequency`, 60 Hz unless set, rather than from the circuit's `frequency=`.

Comparing pgml's sequence-aware `Z0` against an OpenDSS R/X line therefore shows a
zero-sequence gap whose causes are the units-calibrated earth resistance and, unless
`carson_sublinear` is selected, the linear-versus-sub-linear `X0`. Both vanish on the
geometry path, where the geometry and the Carson model are identical on both sides. The
transformer vector group is independent of this, since a Dyn delta traps the zero sequence
identically in both engines.

Transformer winding resistance follows `R(f) = R · m(f) · (f/f₀ if the unit holds X/R
constant else 1)`. Here `m(f)` is the `resistance_frequency` multiplier, a constant, the
Carson skin law or a sampled curve, and the second factor is OpenDSS's `XRConst`, carried
per transformer in `harmonic_xr_constant` and selectable globally through
`transformer.harmonic_resistance.law`. The shipped default is R constant with X
proportional to h, which matches OpenDSS's `XRConst=No` default and pandapower's and
power-grid-model's frequency-independent resistance. It understates transformer damping at
high orders, because no eddy-current or stray-loss rise is modelled. A frequency-correction
curve read from a source file is not modelled.

## Converter coverage

The core model supports more than the converters read. Where a source convention is not
read, a foreign network under-converts, and these are the current gaps.

| | converted | not read |
|---|---|---|
| OpenDSS | `Transformer` (scope above), `Line`, `Load`, `Capacitor`, `Reactor`, `Generator` (including `model=3`), `PVSystem`, `Storage` | three-winding transformers, regulators and tap-changer control, frequency-correction curves, a coupled `Rmatrix`/`Xmatrix` reactor, a non-grounded or two-bus terminal-2 shunt reference, a neutral earthing impedance (`Rneut`/`Xneut`) |
| pandapower | `trafo`, `line`, `load`, `sgen`, `gen`, `storage`, `shunt`, `asymmetric_load`, `switch`, `ext_grid` (including `x0x_max`/`r0x0_max`) | `trafo3w`, `impedance`, `ward`, `xward`, `dcline`, `motor`, `asymmetric_sgen`, `mag0_percent`/`mag0_rx`, `si0_hv_partial`, `xn_ohm`/`rn_ohm` |
| pgm | `node`, `line`, `transformer`, `link`, `sym_load`, `asym_load`, `sym_gen`, `source` (including `z01_ratio`) | `asym_gen`, `voltage_regulator`, `three_winding_transformer`, `transformer_tap_regulator`, `shunt`, `uk_min`/`uk_max`/`pk_min`/`pk_max`, `i0_zero_sequence`/`p0_zero_sequence` |

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
  By default, an open bus-line or bus-transformer switch rewires only its open terminal to
  an auxiliary node, retaining the connected-end shunt like pandapower. The explicit
  `open_switch_model="drop_element"` option selects the legacy whole-branch approximation;
  an element open at both ends is omitted in either mode.
- pandapower `gen`, a PV bus with fixed P, regulated `vm_pu` and free Q within its reactive
  limits, converts exactly by default (`gen_mode=GenMode.VOLTAGE_REGULATING`): the row
  becomes a `Generator` carrying a `VoltageRegulation` block that the solver holds at
  `vm_pu`. `GenMode.VOLT_VAR_APPROX` keeps the earlier steep Volt-VAr droop for a study
  that wants a real droop law, and `GenMode.DROP` skips the table. A `gen` row on the
  `ext_grid` bus, or one flagged `slack`, is skipped in every mode. See the
  [DER models](der-pv-storage.md) page.
- pgm stores no base frequency, so the caller must pass the correct `base_frequency_hz`. A
  50 versus 60 Hz mismatch silently scales every L and C.
