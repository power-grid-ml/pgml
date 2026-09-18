# Two-winding transformer, phase-domain vector groups

How a two-winding transformer is modelled in the phase domain, and what that buys at
harmonic orders.

## Why the vector group matters

The winding connections and the clock number decide whether zero-sequence current can pass
through a transformer. Triplen harmonics, orders 3, 9, 15 and so on, are co-phasal across the
three phases, so they are pure zero sequence. The vector group therefore dominates the
triplen result. A per-phase diagonal stamp is transparent to the zero sequence and lets
triplen current pass from the LV network to the MV side, which the standard Dyn distribution
transformer does not do. Its delta traps that current.

## The winding-incidence primitive

The nodal admittance is built in the winding-voltage domain and mapped to bus phase rows by a
constant real incidence `N`, giving `Y_node = Nᵀ · Y_winding · N`. The same construction
carries connection-aware loads.

With the leakage admittance `y` referred to the TO coil and a coil turns ratio `τ`:

```
Y_winding = [[ (y/τ²)·I3 , −(y/τ)·I3 ],
             [ −(y/τ)·I3 ,    y·I3   ]]      (6x6)
```

The per-winding incidence is `I3` for a grounded wye, `M` or `Mᵀ` for a delta, and
`P = I − (1/3)·11ᵀ` for an ungrounded wye, with

```
M = [[ 1,-1, 0],
     [ 0, 1,-1],
     [-1, 0, 1]]
```

Three properties give the model its behaviour. `M·[1,1,1]ᵀ = 0`, so a delta winding blocks
the zero sequence and the assembled coupling block has zero row sums. `Mᵀ M` is the familiar
`[[2,-1,-1],[-1,2,-1],[-1,-1,2]]` and `M·V⁺ = √3·∠+30°`, so the delta supplies both the √3 in
magnitude and the 30° clock step. `P` is idempotent, so `Pᵀ(y·I)P = y·P`, the textbook
ungrounded-wye self block, and the floating neutral blocks the zero sequence with no Kron
reduction.

A zigzag winding couples through the limb fluxes. Each phase leg is two half-coils in series
opposition on adjacent limbs, so its topology is the normalised circulant `Z = (I − C)/√3`
with `C` the cyclic phase permutation. Because the limb flux is what the far winding sees, `Z`
left-multiplies the other side's incidence block while the zigzag's own block keeps the plain
star topology. `Z·V⁺ = 1∠±30°` gives the clock shift, the `1/√3` keeps the leakage referral on
the line-to-neutral basis, and `Z·[1,1,1]ᵀ = 0` means no zero-sequence transfer. A grounded
zigzag keeps `y·I` in its own self block, which is the classic grounding-transformer property:
a leakage-limited zero-sequence path to ground on its own side.

This is the generalized transformer model of Chen and Dillon, written in winding-incidence
form (Arrillaga and Watson, *Computer Modelling of Electrical Power Systems*; Bazrafshan and
Gatsis, arXiv:1705.06782). The `Y_I = y·I3`, `Y_II = (y/3)·Mᵀ M` and `Y_III = (y/√3)·M`
submatrices those references tabulate per vector group follow from `Nᵀ Y_winding N`, and serve
as a validation oracle.

## Turns ratio and tap

The nominal ratio and the vector-group shift come from the rated voltages and the
connections, not from an explicit complex tap.

```
coil_rated = u_rated            (delta winding, rated line-to-line)
coil_rated = u_rated / sqrt(3)  (wye winding, rated line-to-neutral)
tau = (coil_rated_from / coil_rated_to) * tap.ratio_magnitude
```

For a Dyn 20 kV to 0.4 kV unit this gives `τ = √3·n_LL`. The √3 cancels against `M`, so the
positive-sequence block reduces to `y/n_LL²` on the HV self block and `y/n_LL` on the
coupling, identical to the classical off-nominal-tap pi.

`ComplexTap.ratio_magnitude` therefore carries only the off-nominal tap deviation, 1.0 meaning
on tap, and `ComplexTap.shift_deg` carries the clock angle `clock·30°`. OpenDSS derives the
ratio the same way and keeps the tap changer separate. Converters set the rated voltages and
the connections and leave the tap at `(1.0, clock·30)`.

## Clock numbers

`shift_deg = clock·30`, with a positive angle making the LV phasor lag the HV one, which is
the pandapower and MATPOWER convention.

| Vector group | Clock | `shift_deg` | LV against HV | Realisation |
|---|---|---|---|---|
| Dyn1 | 1 | 30 | lags 30° | HV delta `Mᵀ` |
| Dyn5 | 5 | 150 | lags 150° | HV delta, LV connection rotated one phase |
| Dyn11 | 11 | 330 | leads 30° | HV delta `M` |
| YNyn0 | 0 | 0 | in phase | `I3` on both sides |
| Yzn5 | 5 | 150 | lags 150° | LV zigzag `Z` with a phase rotation |
| Yy6, Dd6 | 6 | 180 | reversed | `−1` on the LV incidence |

Every IEC clock number consistent with the pairing parity is modelled. A delta or zigzag
winding contributes an intrinsic ±30°, a cyclic permutation of the TO-side bus connection
contributes a multiple of −120°, and a reversed TO-winding polarity contributes 180°. So Dy,
Yd, Yz and Zy pairings admit the odd clocks 1, 3, 5, 7, 9 and 11, while Yy, Dd, Dz and Zd
admit the even ones. A clock that contradicts the parity raises.

Which combination realises a requested clock is decided by matching the candidate incidence's
own positive-sequence rotation against `clock·30°`, so the sign convention is pinned by
construction. The positive-sequence coupling then equals the scalar off-nominal-tap pi,
`Y_ft = −y_se/conj(t)` with `t = n·e^{j·shift_deg}`.

Clock 6 is a genuine 180° group rather than a variant of clock 0. It is realised as a sign
flip on the LV incidence. Because `(−N)ᵀ·Y·(−N) = Nᵀ·Y·N` the flip cancels in both self
blocks and survives only in the coupling blocks, which matches the single-phase equivalent's
rotation by `e^{jπ}`.

A single-phase or positive-sequence-equivalent run collapses the group into that complex
scalar tap, and the two paths agree to machine precision. The scalar path honours an arbitrary
`tap.shift_deg`, so a positive-sequence phase shifter from a MATPOWER import is
representable there. The phase-domain stamp rejects a shift that is not a multiple of 30°,
because no constant three-phase winding topology realises one.

### Harmonic orders on the single-phase equivalent

The angle `shift_deg` is the positive-sequence shift. In a balanced three-phase system the
harmonic orders `3k+1` (4, 7, 10, 13, ...) rotate as a positive sequence, the orders `3k+2`
(2, 5, 8, 11, ...) as a negative sequence, and the orders `3k` as a zero sequence. A
negative-sequence quantity crosses the same windings with the opposite angle, so the scalar
tap is `t = n·e^{−j·shift_deg}` at the orders `3k+2`. The phase-domain stamp has this property
by construction, because its incidence is real, and the two paths agree to machine precision
at every positive- and negative-sequence order. The sign matters as soon as harmonic sources
sit on both sides of a Dy or Yd unit: the 5th harmonic of a converter behind a Dyn11
transformer arrives on the HV side 60° away from where a same-sign shift would put it, which
decides how it adds to an MV-connected source. Interharmonics have no sequence assignment and
keep the positive-sequence angle. With `shift_deg = 0`, as in a genuinely single-phase
network, the rule has no effect.

The triplen orders are the limit of the single-phase equivalent. As zero-sequence
quantities they would be blocked by a delta, zigzag or ungrounded-wye winding, and they would
travel on the lines' `Z0` rather than `Z1`. The equivalent carries neither a winding topology
nor zero-sequence line data, and a grid does not say whether it is an equivalent of a
three-phase system or a single-phase network, so triplen orders pass through the
positive-sequence pi unchanged. Assembly logs one warning when that happens on a pairing
other than YNyn. Solve triplen orders on a three-phase grid.

## Leakage referral

`series_resistance_ohm` and `series_inductance_h` are referred to the TO-side coil. With
`z_LL = (vk/100)·u_LL,to²/S`, the usual leakage on the LV line-to-line base, a wye or zigzag
TO winding stores `z_coil = z_LL` unchanged. A delta TO winding stores `z_coil = 3·z_LL`,
because a delta coil's own base is `3·u_LL²/S`.

## Scope

- Every winding pairing of wye, grounded wye, delta, zigzag and grounded zigzag except
  zigzag to zigzag, at every clock of the pairing's parity.
- Solid neutral grounding only. A grounding impedance is rejected rather than ignored.
- The zero-sequence path follows the winding topology, so a delta or zigzag winding blocks
  it. Its value is `Transformer.zero_sequence` when set, otherwise the documented
  `transformer.zero_sequence.{r0_over_r1, x0_over_x1}` ratios, which ship at 1.0 so that
  `Z0 = Z1`. That is also what OpenDSS and power-grid-model imply for a two-winding unit,
  since neither has a zero-sequence leakage input, while pandapower's `vk0_percent` and
  `vkr0_percent` are read into the field by the converter. Set it for a three-limb-core YNyn
  unit, where `X0/X1` is typically 0.3 to 1.0, and for a grounding zigzag, where `X0/X1` is
  well below 1. Validated against pandapower's own unbalanced power flow on YNyn and Dyn
  units at `vk0/vk` of 0.3, 0.5 and 2.0: the per-phase voltages agree to 2.0e-4 V, against
  0.93 to 1.88 V when the value is dropped.
- Not modelled, and named in a converter warning rather than dropped silently: a
  zero-sequence magnetizing branch (pandapower `mag0_percent`/`mag0_rx`, power-grid-model
  `i0_zero_sequence`/`p0_zero_sequence`, the path a three-limb core's zero-sequence flux
  takes through tank and air), the HV/LV split of the zero-sequence leakage inside a T
  (pandapower `si0_hv_partial`), and a neutral earthing impedance `3·Z_N` (pandapower
  `xn_ohm`/`rn_ohm`, OpenDSS `Rneut`/`Xneut`, which the OpenDSS converter refuses).
- Two-winding units only. No three-winding units and no regulators.
- The magnetizing branch is a shunt from each phase terminal to ground, outside the winding
  incidence. On a delta side that is a zero-sequence path to ground the real winding does
  not have; at a magnetizing current of 0.5 % its admittance is 200 times smaller than the
  rated admittance and has no practical effect. The referral to the to side uses the rated
  voltage ratio without the off-nominal tap.
- The magnetizing branch defaults to `split`: half on each terminal, referred to
  each terminal's voltage base. `from_terminal` places it entirely on the from/HV
  terminal; `to_terminal` places it on the to/LV terminal, as OpenDSS does for the last
  winding of a two-winding transformer. These are alternative equivalent-circuit
  choices. Use `defaults.use_preset("opendss")` for OpenDSS conformance and
  `defaults.use_preset("power-grid-model")` for power-grid-model conformance.
  On the measured 500 kVA 20/0.4 kV unit, the OpenDSS placement agrees within
  3e-10 to 2e-9 pu; the full from-terminal model differs by 9.2e-5 pu at 0.1 %
  magnetizing current, 2.2e-4 pu at 0.5 %, and 8.2e-4 pu at 2 %. The split model's
  difference is about half as large. Reference agreement tests select the reference's
  model rather than treating a default mismatch as solver error.
- The winding resistance follows `R(f) = R · m(f) · (f/f₀ if the unit holds X/R constant
  else 1)`. Here `m(f)` is the `resistance_frequency` multiplier, a constant, the Carson skin
  law or a sampled curve, and the second factor is OpenDSS's `XRConst`, carried per
  transformer in `harmonic_xr_constant` and selectable globally through
  `transformer.harmonic_resistance.law`. The shipped default is R constant with X
  proportional to h, which matches OpenDSS's `XRConst=No` default and pandapower's and
  power-grid-model's frequency-independent resistance. It understates transformer damping at
  high orders, because no eddy-current or stray-loss rise is modelled. Both settings were
  validated against OpenDSS's own `Yprim` at orders 1, 5 and 13 to 1.25e-6 S, which is
  OpenDSS's anti-float shunt, against 2e-2 S when the flag is ignored.
- The zigzag winding model is experimental. It reproduces the three properties a zigzag must
  have, a ±30° clock contribution, no zero-sequence transfer, and a low-impedance
  zero-sequence path to ground on its own side, and it agrees with power-grid-model on an
  unbalanced solve once the zero-sequence value is carried. Neither OpenDSS nor pandapower
  can express the same unit as a single two-winding element, so it has one independent
  reference only. Constructing one logs a warning.
- With no winding metadata in the source data the default vector group is Dyn11, the European
  LV default.

## OpenDSS mapping

A real OpenDSS Dyn transformer looks like this.

```
New Transformer.T1 windings=2 phases=3
~ wdg=1 bus=HV.1.2.3   conn=delta kV=20  kVA=<sn> %R=<vkr/2>
~ wdg=2 bus=LV.1.2.3.0 conn=wye   kV=0.4 kVA=<sn> %R=<vkr/2> Rneut=0 Xneut=0
~ XHL=<sqrt(vk^2-vkr^2)> LeadLag=Lag
```

`Rneut=0 Xneut=0` solidly grounds the LV neutral. The LV `.0` node is the ground reference and
never appears in `YNodeOrder`, so the exported system admittance rows are the phase rows only
and map cleanly onto pgml's node-phase rows. `LeadLag=Lag` gives clock 1 and `Lead` gives
clock 11.

Reading a DSS file inverts the same per-unit identity. `%R` and `XHL` are percent quantities,
so `R_lv = (%R_wdg1+%R_wdg2)/100 · Z_base_LV` and `X_lv = XHL%/100 · Z_base_LV` with
`Z_base_LV = kV_lv²·1000/kVA` recover the LV-referred leakage directly, provided both windings
share one kVA rating. Grounding follows OpenDSS's shorthand-bus rule, and only solidly
grounded wye or delta windings convert.

`%imag` and `%noloadloss` are the imaginary and real parts of the core admittance
separately, each in percent of the winding base admittance, rather than a total no-load
current with the loss component inside it, which is pandapower's `i0_percent` convention.
The converter therefore reads `B_m = %imag/100 · S/u_hv²` directly, with no Pythagorean
subtraction, and accepts a `%imag` below `%noloadloss`. Measured on a live `Yprim`
difference, the magnetizing contribution is exactly
`(%noloadloss + j·(−%imag))/100 · S/u_wdg2²` at the last winding's terminal.

The three transformers of the CIGRE LV benchmark are Dyn1 units, 20 kV to 0.4 kV, rated 0.5,
0.15 and 0.3 MVA with `shift_degree = 30`. The pandapower reader converts them to a delta
from-connection, a grounded-wye to-connection, a tap of `(1.0, 30°)` and the rated voltages.
