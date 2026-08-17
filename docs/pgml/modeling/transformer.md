# Two-winding transformer modelling (phase-domain vector groups)

How pgml models a two-winding transformer in the phase domain, why, and how it maps
to OpenDSS. This is the authoritative modelling-decision record for
`pgml.assembly._transformer` + `_transformer_block_groups`.

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
- Incidence per winding (3-phase): `wye_grounded → I3`; `delta → M` (or `Mᵀ`);
  ungrounded `wye → P = I − (1/3)·11ᵀ`, where
  ```
  M = [[ 1,−1, 0],
       [ 0, 1,−1],
       [−1, 0, 1]]        (Kersting's [D]; columns sum to zero)
  ```
- A zigzag (interconnected-star) winding couples through the LIMB fluxes: each
  phase leg is two half-coils in series opposition on adjacent limbs, so its
  topology is the normalised circulant `Z = (I − C)/√3` (`C` = cyclic phase
  permutation) and — because the limb flux is what the far winding shares — `Z`
  left-multiplies the OTHER side's incidence block, while the zigzag's own block
  keeps the plain star topology (`I3` grounded, `P` ungrounded). `Z·V⁺ = 1∠±30°`
  (clock shift like a delta; the `1/√3` keeps the leakage referral at the
  physical line-to-neutral basis) and `Z·[1,1,1]ᵀ = 0` (no zero-sequence
  TRANSFER). A grounded zigzag's own self block stays `y·I` — the winding keeps a
  leakage-limited zero-sequence path to ground on its own side, the classic
  grounding-transformer property. The path VALUE equals the positive-sequence
  leakage; the true zigzag zero-sequence leakage (set by the half-coil geometry)
  is typically smaller and would need the `TransformerZeroSeq` value override,
  which is not consumed yet.
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

| Vector group | clock | `shift_deg` | LV vs HV | realisation |
|---|---|---|---|---|
| Dyn1  | 1  | 30  | lags 30°  | HV delta `Mᵀ` |
| Dyn5  | 5  | 150 | lags 150° | HV delta + LV connection rotated one phase |
| Dyn11 | 11 | 330 | leads 30° | HV delta `M`  |
| YNyn0 | 0  | 0   | in phase  | `I3` both sides |
| Yzn5  | 5  | 150 | lags 150° | LV zigzag `Z` + phase rotation |
| Yy6 / Dd6 | 6 | 180 | reversed | `−1` on the LV incidence |

Every IEC clock number consistent with the pairing PARITY is modelled: each delta or
zigzag winding contributes an intrinsic ±30° (the `M`/`Mᵀ`, `Z`/`Zᵀ` orientation), a
cyclic permutation `C^m` of the TO-side bus connection contributes `m·(−120°)` (±4
clock steps), and a reversed TO-winding polarity contributes 180° (6 steps). So Dy /
Yd / Yz / Zy pairings admit exactly the odd clocks {1,3,5,7,9,11} and Yy / Dd / Dz /
Zd the even clocks {0,2,4,6,8,10}; a parity-inconsistent clock raises. The concrete
combination is selected by matching the realised positive-sequence rotation of the
candidate incidence against `clock·30°`, so the sign convention is pinned by
construction: the positive-sequence coupling equals the scalar off-nominal-tap pi
`Y_ft = −y_se/conj(t)`, `t = n·e^{j·shift_deg}` (verified against a live OpenDSS
export, and to machine precision for every pairing × clock in
`tests/reference/test_transformer_clock_matrix.py`).

Single-phase / positive-sequence-equivalent runs (P = 1) collapse the group into that
complex scalar tap directly, so the two modes agree to machine precision. The P = 1
path honours the EXACT `tap.shift_deg`, so an arbitrary positive-sequence
phase-shifter angle (a MATPOWER import) is representable there; the phase-domain
stamp rejects a shift that is not a multiple of 30°, since no constant 3-phase
winding topology realises it.

**Clock 6 (`Yy6` / `Dd6`) — reversed LV winding polarity.** Unlike clock 0, clock 6 is
NOT in phase: it is a genuine 180° group, realised as a `−1` on the LV incidence
(`N_lv → −N_lv`) rather than a phase-shifted delta incidence. Because
`(−N)ᵀ·Y·(−N) = Nᵀ·Y·N`, the sign flip cancels in both self blocks (HV-HV, LV-LV) and
survives only in the HV↔LV coupling blocks — matching the single-phase-equivalent
path's complex rotation `e^(jπ) = −1`.

## Leakage referral (the converter contract)
The schema's `series_resistance_ohm` / `series_inductance_h` are referred to the
TO-side COIL. With `z_LL = (vk/100)·u_LL,to²/S` (the usual LV line-to-line-base
leakage): a wye or zigzag TO winding stores `z_coil = z_LL` unchanged; a DELTA TO
winding stores `z_coil = 3·z_LL` (the delta coil base is `3·u_LL²/S`). Pinned in
`tests/reference/test_transformer_clock_matrix.py`.

## Scope (what is and isn't modelled)
- Supported: every winding pairing of {wye, grounded wye, delta, zigzag, grounded
  zigzag} except zigzag-zigzag, at every clock number of the pairing's parity.
- Solid neutral grounding only (`*_grounding` = None / 0). A non-solid grounding
  impedance (`GroundingImpedance`) is rejected, not ignored.
- The zero-sequence PATH follows the winding topology; its VALUE equals the
  positive-sequence leakage. The explicit `TransformerZeroSeq` value override is
  not consumed yet — relevant mainly for grounded zigzag (true Z0 < Z1) and for
  three-limb-core YNyn units.
- Two-winding units only (no 3-winding transformers, no regulators).
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

This is the pgml -> DSS direction (`pgml.evaluation.oracles.opendss_oracle
._build_circuit_with_real_transformer`), used to validate the harmonic assembly against
a live OpenDSS `Transformer` element. The FORWARD direction, DSS -> pgml
(`pgml.convert.opendss.to_grid`), inverts the same per-unit leakage identity: OpenDSS's
`%R`/`XHL` are percent (base-invariant) quantities, so
`R_lv = (%R_wdg1+%R_wdg2)/100 * Z_base_LV`, `X_lv = XHL%/100 * Z_base_LV` with
`Z_base_LV = kV_lv²*1000/kVA` recovers the LV-referred leakage directly (both windings
must share one kVA rating). The clock comes from `LeadLag` (`Lag` -> clock 1, `Lead` ->
clock 11 for a Dy/Yd pairing; clock 0 for a matching Yy/Dd pairing, since OpenDSS has no
explicit clock parameter beyond that binary toggle). Grounding follows OpenDSS's own
shorthand-bus rule (no explicit `(n_phases+1)`-th conductor, or an explicit `.0`, solidly
grounds a wye winding); only solidly grounded wye or delta windings convert. Validated by
a live 2-bus MV-source -> transformer -> LV-load oracle (Dyn11 and Yy0 cases,
voltage magnitude + angle vs OpenDSS's own `Solve`) in
`tests/reference/test_opendss_transformer.py`. See `src/pgml/convert/opendss/CONTEXT.md`
for the full field mapping and the current scope (two-winding only; no regulators, no
3-winding units, no frequency-correction curves, no `Yy6`/`Dd6`).

## CIGRE LV
The three `create_cigre_network_lv()` transformers are Dyn1 (delta HV / grounded-wye
LV, 20 kV / 0.4 kV, 0.5 / 0.15 / 0.3 MVA, `shift_degree = 30`). pgml's pandapower
converter sets `from_connection=DELTA`, `to_connection=WYE_GROUNDED`,
`tap=(1.0, 30°)`, and the rated voltages.
