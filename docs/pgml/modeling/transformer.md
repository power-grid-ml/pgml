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

## Leakage referral

`series_resistance_ohm` and `series_inductance_h` are referred to the TO-side coil. With
`z_LL = (vk/100)·u_LL,to²/S`, the usual leakage on the LV line-to-line base, a wye or zigzag
TO winding stores `z_coil = z_LL` unchanged. A delta TO winding stores `z_coil = 3·z_LL`,
because a delta coil's own base is `3·u_LL²/S`.

## Scope

- Every winding pairing of wye, grounded wye, delta, zigzag and grounded zigzag except
  zigzag to zigzag, at every clock of the pairing's parity.
- Solid neutral grounding only. A grounding impedance is rejected rather than ignored.
- The zero-sequence path follows the winding topology, while its value equals the
  positive-sequence leakage. A separate zero-sequence value override is not consumed yet,
  which matters mainly for a grounded zigzag, whose true `Z0` is smaller, and for
  three-limb-core YNyn units.
- Two-winding units only. No three-winding units and no regulators.
- The magnetizing branch is a shunt on the HV terminal, referred to the HV line voltage.
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

The three transformers of the CIGRE LV benchmark are Dyn1 units, 20 kV to 0.4 kV, rated 0.5,
0.15 and 0.3 MVA with `shift_degree = 30`. The pandapower reader converts them to a delta
from-connection, a grounded-wye to-connection, a tap of `(1.0, 30°)` and the rated voltages.
