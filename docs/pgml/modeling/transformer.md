# Two-winding transformer modelling (phase-domain vector groups)

How pgml models a two-winding transformer in the phase domain, why, and how it maps
to OpenDSS. This is the authoritative modelling-decision record for
`pgml.assembly._transformer` + `_stamp_transformers`.

## The problem this solves
A transformer's winding connections (wye / grounded-wye / delta) and clock number
decide whether ZERO-SEQUENCE current can flow through it. Triplen harmonics
(h = 3, 9, 15, …) are co-phasal across the three phases, i.e. pure zero sequence, so
the vector group dominates the triplen-harmonic result. A model that ignores the
connections (a per-phase diagonal stamp) is zero-sequence-TRANSPARENT and lets
triplen currents pass straight from the LV network to the MV side — wrong for the
standard Dyn distribution transformer, which traps them in the delta.

## The model: winding-incidence primitive `Y = Nᵀ · Y_winding · N`
Build the nodal admittance in the winding-voltage domain, then map to the bus phase
rows by a constant real incidence `N` (the same construction used for connection-aware
loads in `pgml.assembly._incidence`).

- Winding primitive (leakage admittance `y` referred to the TO/LV coil, coil turns
  ratio `τ`):
  ```
  Y_winding = [[ (y/τ²)·I3 , −(y/τ)·I3 ],
               [ −(y/τ)·I3 ,    y·I3   ]]      (6×6)
  ```
- Incidence per winding (3-phase): `wye_grounded → I3`; `delta → M`; ungrounded
  `wye → P = I − (1/3)·11ᵀ`, where
  ```
  M = [[ 1,−1, 0],
       [ 0, 1,−1],
       [−1, 0, 1]]        (Kersting's [D]; columns sum to zero)
  ```
- `N = blockdiag(N_hv, N_lv)`; `Y_node = Nᵀ Y_winding N`.

Properties (all verified numerically):
- `M·[1,1,1]ᵀ = 0` ⇒ a delta winding BLOCKS the zero sequence (it circulates inside
  the delta). The assembled HV-LV coupling block then has zero row sums.
- `Mᵀ M = [[2,−1,−1],[−1,2,−1],[−1,−1,2]]` and `M·V⁺ = √3·∠+30°` ⇒ the delta supplies
  the intrinsic √3 magnitude and the ±30° clock shift.
- `P` is idempotent, so `Pᵀ(y·I)P = y·P` (the textbook `Y_II` self-block); an
  ungrounded-wye neutral floats and blocks the zero sequence with no Kron reduction.

This is the Chen/Dillon generalized transformer model (Arrillaga & Watson, *Computer
Modelling of Electrical Power Systems*; Bazrafshan & Gatsis, arXiv:1705.06782) written
in its winding-incidence form. The `Y_I = y·I3`, `Y_II = (y/3)·Mᵀ M`,
`Y_III = (y/√3)·M` submatrices that those references tabulate per vector group are a
consequence of `Nᵀ Y_winding N`, and are used as a validation oracle.

## Turns ratio and the tap convention (DECISION)
The NOMINAL ratio and the vector-group phase shift come from the rated voltages plus
the connections — NOT from an explicit complex tap:
```
coil_rated = u_rated            (delta winding, rated line-to-line)
coil_rated = u_rated / √3       (wye winding, rated line-to-neutral)
τ = (coil_rated_from / coil_rated_to) · tap.ratio_magnitude
```
For a Dyn 20 kV / 0.4 kV unit, `τ = 20000 / (400/√3) = √3·(20000/400) = √3·n_LL`. The
√3 then cancels against `M`, so the assembled positive-sequence block is exactly
`y/n_LL²` (HV self) and `y/n_LL` (coupling) — identical to the historical
off-nominal-tap pi.

Therefore `ComplexTap.ratio_magnitude` is the OFF-NOMINAL tap deviation (1.0 = on-tap)
and `ComplexTap.shift_deg` carries the vector-group clock angle (`clock·30°`). This is
OpenDSS-faithful: OpenDSS likewise derives the ratio from the winding kVs + connections
and keeps the tap changer separate. Converters set the rated voltages + connections and
leave `tap = (1.0, clock·30)`.

## Clock / phase-shift sign (pinned numerically)
pgml uses the pandapower / MATPOWER convention `shift_deg = clock·30`, a positive angle
making the LV phasor LAG the HV one:

| Vector group | clock | `shift_deg` | LV vs HV | delta incidence |
|---|---|---|---|---|
| Dyn1  | 1  | 30  | lags 30°  | `Mᵀ` |
| Dyn11 | 11 | 330 | leads 30° | `M`  |
| YNyn0 | 0  | 0   | in phase  | `I3` both sides |

`VectorGroup.clock_transpose = sin(clock·30°) > 0` (Dyn1 → `Mᵀ`, Dyn11 → `M`),
pinned so the positive-sequence coupling equals the scalar off-nominal-tap pi
`Y_ft = −y_se/conj(t)`, `t = n·e^{j·shift_deg}` (and against a live OpenDSS export).

Single-phase / positive-sequence-equivalent runs (P = 1) collapse the group into that
complex scalar tap directly, so the two modes agree to machine precision.

**Clock 6 (`Yy6` / `Dd6`) — reversed LV winding polarity.** Unlike clock 0, clock 6 is
NOT in phase: it is a genuine 180° group, realised as a `−1` on the LV incidence
(`N_lv → −N_lv`) rather than a phase-shifted delta incidence. Because
`(−N)ᵀ·Y·(−N) = Nᵀ·Y·N`, the sign flip cancels in both self blocks (HV-HV, LV-LV) and
survives only in the HV↔LV coupling blocks — matching the single-phase-equivalent
path's complex rotation `e^(jπ) = −1`.

## Scope (what is and isn't modelled)
- Supported: Dyn1 / Dyn11 (clock 1 / 11) for delta-wye pairings; and, for wye-wye /
  delta-delta pairings, clock 0 (in phase, `shift_deg = 0`) and clock 6 (180°
  reversed polarity, `shift_deg = 180` — see above). Other clocks raise
  `NotImplementedError` (they need a cyclic phase permutation of the winding pairing).
- Solid neutral grounding only (`*_grounding` = None / 0). A non-solid grounding
  impedance (`GroundingImpedance`) and zigzag windings are not modelled yet.
- Magnetizing branch is a simple shunt on the HV terminal (referred to the HV line
  voltage), unchanged.
- Default vector group when a source carries no winding metadata:
  `transformer.vector_group` in `pgml/data/defaults.yaml` (Dyn11, the IEC / European LV
  default).

## OpenDSS mapping (oracle)
A real OpenDSS Dyn transformer:
```
New Transformer.T1 windings=2 phases=3
~ wdg=1 bus=HV.1.2.3   conn=delta kV=20  kVA=<sn> %R=<vkr/2>
~ wdg=2 bus=LV.1.2.3.0 conn=wye   kV=0.4 kVA=<sn> %R=<vkr/2> Rneut=0 Xneut=0
~ XHL=<sqrt(vk²−vkr²)> LeadLag=Lag        ! Lag → Dyn1 (pgml shift_deg=30); Lead → Dyn11
```
`Rneut=0 Xneut=0` solidly grounds the LV neutral; the LV `.0` node is the ground
reference and never appears in `YNodeOrder`, so the exported `SystemY` rows are only
`.1/.2/.3` and map cleanly to pgml `(node, phase)` rows.

## CIGRE LV
The three `create_cigre_network_lv()` transformers are Dyn1 (delta HV / grounded-wye
LV, 20 kV / 0.4 kV, 0.5 / 0.15 / 0.3 MVA, `shift_degree = 30`). pgml's pandapower
converter sets `from_connection=DELTA`, `to_connection=WYE_GROUNDED`,
`tap=(1.0, 30°)`, and the rated voltages.
