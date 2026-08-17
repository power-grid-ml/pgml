# Modelling conventions — pgml vs pandapower / OpenDSS / power-grid-model

The single place that pins **pgml's internal definitions** and records how they differ
from the three reference tools we convert from and validate against
(pandapower, OpenDSS, power-grid-model / "pgm"). The conversion boundary
(`pgml.convert.*`) is where a foreign convention enters the library, so it is also where
a silent factor-of-√3 or referred-to-the-wrong-side error is introduced. The rule is:
**every converter translates the source convention into pgml's canonical form below;
the core never sees a foreign convention.**

Companion decision records (deeper derivations): the [transformer model](transformer.md)
(vector-group two-winding model), [Carson line constants](references/opendss/carson.md)
(bit-exact earth-return), [OpenDSS harmonics](references/opendss/harmonics.md)
(injection + frequency scaling), the [harmonic line model](harmonic-line-model.md), and
[asymmetric modeling](asymmetric.md). The frozen schema is the data source of truth — see
the [Schemas API reference](../api/schemas.rst).

---

## 0. pgml canonical form (the invariants every consumer honours)

- **Phase-domain, fully per-phase.** Quantities are stored per phase / as `n×n` phase
  matrices, never as sequence components. Sequence inputs are decomposed at conversion
  time (`self=(Z0+2·Z1)/3`, `mutual=(Z0−Z1)/3`).
- **SI units throughout.** Volts, ohms, henries, farads, siemens, watts, vars — no kV /
  MW / per-unit / per-km inside the library. Converters are the only unit boundary.
- **Store L and C, never X and B.** Reactance/susceptance are derived at a frequency
  `X(h)=2π·h·f₀·L`, `B(h)=2π·h·f₀·C`; resistance frequency-dependence is an explicit law.
- **Phasors as `(real, imag)` pairs**; results indexed by `frequency_hz`.
- **Voltage convention (the consistency rule that matters most):**
  - `Node.u_rated_v` is stored **line-to-line** for any node with ≥3 phases, and
    **line-to-neutral** for a 1-phase node (where there is no √3 to apply).
  - **Every per-phase voltage the solver actually touches is line-to-neutral.** The one
    helper `assembly._params.phase_voltage_magnitude(u_rated_v, n_phases)` is the single
    point of truth: it returns `u_rated_v/√3` for a WYE element on a ≥3-phase node,
    `u_rated_v` for a DELTA element (sees the full L-L) or a 1-phase node. It is used by
    the const-Z/ZIP load model, the slack EMF, the harmonic source admittance, and the
    per-unit reporting in `pgml.evaluation` — so they cannot drift apart.
  - The slack/source EMF is the **phase-to-ground (line-to-neutral)** phasor. Converters
    pass the line-to-line magnitude; `convert._common.build_source` divides it by √3 for
    the balanced wye 3-phase expansion (1-phase keeps it unchanged).

These invariants are why a foreign tool's choice (L-L vs L-N base, HV- vs LV-referred
transformer impedance, per-km vs total, imperial vs metric earth return) must be resolved
**in the converter**, documented in `Provenance`, and never leak into the core.

---

## 1. Base / nominal voltage

| | nominal-voltage field | stored as | per-unit base / output voltage |
|---|---|---|---|
| **pgml** | `Node.u_rated_v` [V] | **L-L** (≥3φ) / L-N (1φ) | working voltages are **L-N** (`phase_voltage_magnitude`); pu = `\|V_LN\|/(u_rated/√3)` |
| **pandapower** | `bus.vn_kv` [kV] | **L-L** | `res_bus.vm_pu = \|V_LL\|/vn_kv`; asym `res_bus_3ph` uses the L-N base (`vn_kv/√3`) |
| **OpenDSS** | `Vsource.basekv` [kV] | **L-L for `phases>=3`; used directly (no √3) for `phases=1`** — see the gotcha below | but `Bus.kVBase()` **always returns `(matched voltagebases class)/√3`, regardless of local phase count** (`=basekv/√3` when `voltagebases` matches the source's own `basekv`); `AllBusVolts` are L-N phasors |
| **pgm** | `node.u_rated` [V] | **L-L** | symmetric output `u` = L-L, `u_pu=u/u_rated`; asymmetric output `u` = **L-N** |

**Decision & rationale.** Store the universal nameplate (line-to-line) and derive the
line-to-neutral working voltage at the single point of use. This makes the const-Z load
admittance `y=conj(S)/V²` agree with pandapower's positive-sequence reference in the
1-phase path (`V=V_LL`) and remain physically correct per phase in the 3-phase path
(`V=V_LN`, total power preserved).

**Converter mapping.**
- pandapower / pgm: `u_rated_v = vn_kv·1000` / `u_rated` directly (both already L-L).
- OpenDSS: `Bus.kVBase()` is L-N, so the converter recovers L-L as `kVBase·√3·1000`. (An
  earlier converter stored `kVBase·1000` — an L-N value — which was a √3 error in the
  const-Z shunt; the converter now recovers L-L correctly.)

**Gotchas.**
- OpenDSS `Bus.kVBase()` returns `(matched voltagebases class)/√3` for EVERY bus,
  regardless of local phase count — `×√3` is mandatory to recover the nameplate
  `Node.u_rated_v` the converter stores.
- **OpenDSS `Vsource.basekv` is documented as line-to-line, but that is only true for a
  `phases>=3` Vsource.** For a `phases=1` Vsource, OpenDSS uses `basekv` DIRECTLY,
  unscaled, as the solved single-conductor-pair EMF magnitude — no internal `×√3` or
  `/√3` anywhere (verified empirically: `Bus.Voltages()` magnitude equals `basekv·pu`
  exactly for a 1-phase source, whether `basekv` is a genuine line-to-neutral value or
  the historical positive-sequence-equivalent convention of feeding in the parent
  3-phase system's line-to-line nominal, e.g. the IEEE 33-bus fixtures). The converter's
  `u_ref_v = basekv·pu·1000` and `u_rated_v = kVBase()·√3·1000` formulas need NO
  phase-count branch — they simply mirror whatever magnitude OpenDSS itself solves for,
  which is L-L-scaled for `phases>=3` and used as-is for `phases=1`. See
  `tests/convert/test_opendss_vsource_basekv.py` for the live-circuit proof and
  `src/pgml/convert/opendss/converter.py`'s Vsource docstring section for the derivation.
- pgm asymmetric power flow reports `u` as L-N; a future asym oracle must multiply by √3
  (or use `u_pu` with the correct base) before comparing to pgml's L-L `u_rated`.
- The 3-phase slack EMF must be L-N (`u_rated/√3`); pinning the L-L magnitude on each
  phase makes every 3-phase voltage √3 too high (≈1.73 pu on the L-N base). This is the
  one place `build_source` must (and now does) divide by √3.

---

## 2. Transformer impedance — which side it is referred to

This is the headline cross-tool difference.

| | short-circuit params | **referred to** | → pgml series R/L (referred to **TO/LV coil**) |
|---|---|---|---|
| **pgml** | `series_resistance_ohm`, `series_inductance_h` | **TO / LV coil** | stored directly |
| **pandapower** | `vk_percent`, `vkr_percent`, `sn_mva` | **LV side** ✅ | `Z_base_LV=vn_lv_v²/sn_va`; `R_ll=vkr%·Z_base_LV`, `\|Z_ll\|=vk%·Z_base_LV`, `X_ll=√(\|Z_ll\|²−R_ll²)`; stored TO-coil-referred (`3x` for a DELTA-to winding, unchanged otherwise) in BOTH phase modes, `L=X/2πf₀` — same convention as pgm/OpenDSS below |
| **pgm** | `uk`, `pk`, `sn`, `u2` | **to-side (LV)** ✅ | `R=pk·u2_eff²/sn²`, `\|Z\|=uk·u2_eff²/sn`, `X=√(\|Z\|²−R²)`, `u2_eff`=tap-adjusted to-side voltage; stored TO-coil-referred (`3x` for a DELTA-to winding, unchanged otherwise) in BOTH phase modes — the single-phase scalar pi applies its own `y_LL=3·y_coil` referral in assembly, see `src/pgml/convert/pgm/CONTEXT.md` |
| **OpenDSS** | per-winding `%R`, inter-winding `XHL` | **percent, i.e. base-invariant, on the standard L-L base** — `XHL` documented "on the kVA base of winding 1"; `%R` per winding on its own kV/kVA base | `to_grid` recovers `R_ll=(%R_wdg1+%R_wdg2)/100·Z_base_LV`, `X_ll=XHL%/100·Z_base_LV`, `Z_base_LV=kV_lv²·1000/kVA` (requires both windings to share one kVA rating), then multiplies by **3 if the LV winding is DELTA** (its natural coil impedance base is `3·Z_base_LV` — a delta coil is rated at the L-L voltage with 1/3 the per-phase kVA, see the [transformer model](transformer.md) and `src/pgml/convert/opendss/CONTEXT.md`) to get the actual TO-coil-referred `R_lv`/`X_lv` pgml stores; the oracle direction back-calculates the same way (divides by the same factor first), see below |

**Decision & rationale.** pgml refers the leakage admittance to the **TO/LV coil**, the
same side as pandapower and pgm (the two load-flow oracles), so their `vk/vkr/uk/pk`
convert with a single LV base and no extra referral. It is also the natural side for the
phase-domain winding-incidence primitive `Y = Nᵀ·Y_winding·N`, where the leakage `y` sits
on the LV coil block and the HV self-block picks up the `1/τ²` from the turns ratio (see
the [transformer model](transformer.md)). OpenDSS's `%R`/`XHL` are PERCENT (per-unit)
quantities, so — unlike an absolute-ohm impedance — they are base-invariant and need no
HV/LV referral arithmetic, only the LV base impedance to convert back to ohms; this holds
as long as both windings share one kVA rating (`to_grid` raises `ConversionError`
otherwise, an OpenDSS quirk: the *sequential* `~ wdg=1 ... kVA=x` / `~ wdg=2 ... kVA=y`
tilde-continuation syntax silently re-syncs both windings to the LAST kVA given — only the
array form `kvas=[x, y]` actually creates a genuine per-winding kVA mismatch). What DOES
still need a referral is pgml's coil-vs-line-to-line distinction: `%R`/`XHL` are recovered
on the standard L-L base regardless of connection, but a DELTA LV winding's actual coil
carries `3x` that impedance (§"headline" row above) — `to_grid` applies that factor, a
latent bug fixed alongside the vector-group rotation work (see
`src/pgml/convert/opendss/CONTEXT.md` for the before/after live-oracle error). The live
OpenDSS Dyn oracle (`opendss_oracle._build_circuit_with_real_transformer`, the pgml -> DSS
direction used for the harmonic vector-group tests) back-calculates `%R`/`XHL` from pgml's
LV-referred R/L with the same formula inverted (dividing out the same delta-LV factor
first) — self-consistent because the total per-unit leakage is preserved either way. The
DSS -> pgml direction (`convert.opendss.to_grid`) is the forward conversion documented
here; see `src/pgml/convert/opendss/CONTEXT.md`.

**Magnetizing branch.** pgml refers the magnetizing shunt `y_m=G_m+jB_m` to the **HV**
terminal (`magnetizing_conductance_s`, `magnetizing_inductance_h`); the pandapower
converter computes `G_m=pfe_w/u_hv²`, `B_m` from `i0%`/`sn` on the HV base. pandapower
internally keeps it on the LV base split into the pi-shunt — physically equivalent after
the turns ratio, but the raw numbers differ, so do not compare them without re-referring.
The pgm converter refers `i0`/`p0` (defined by pgm on the to-side `u2`, per
`transformer.hpp`) to the HV/from terminal the same way, by the square nameplate ratio —
but pgm's OWN branch stamp (`calc_param_y_sym`) instead splits the magnetizing admittance
HALF onto its `Y_tt` and HALF (through the tap) onto `Y_ff`, a genuinely different topology
from pgml's HV-only shunt; the residual is small (~1.5e-4 pu for a realistic ~0.5% `i0`,
machine precision at `i0=p0=0`) and documented in `tests/reference/test_pgm_transformer.py`.

---

## 3. Transformer ratio, tap, vector group / clock

| | nominal ratio | tap (off-nominal) | vector-group phase shift |
|---|---|---|---|
| **pgml** | from rated **coil** voltages + connections (`nominal_turns_ratio`: delta coil=L-L, wye/zigzag coil=L-N=`u/√3`) | `ComplexTap.ratio_magnitude` (1.0 = on-tap) | `tap.shift_deg = clock·30`; **positive ⇒ LV lags HV**; the delta/zigzag orientation × cyclic permutation × polarity realising the clock is selected by matching the incidence's positive-sequence rotation (self-pinning, see the [transformer model](transformer.md)) |
| **pandapower** | `vn_hv_kv/vn_lv_kv` (MATPOWER off-nominal tap) | `tap_pos/tap_neutral/tap_step_percent`, `tap_side` | `shift_degree` (positive ⇒ LV lags, matches pgml) |
| **OpenDSS** | ratio of winding coil kV | tap per winding | `LeadLag` (`Lag`→30°/`Lead`→330° baseline, Dy/Yd only) **+** a cyclic winding-bus rotation (±120°/±4 clocks per step — the only way to reach any OTHER clock; see §2's row above and `src/pgml/convert/opendss/CONTEXT.md`) |
| **pgm** | `u1/u2` | `tap_side/pos/nom/size/min/max` | `clock` 0–12; `winding_from/to` enums |

**Decision & rationale.** The nominal ratio and the ±30° clock shift come from the rated
voltages + winding connections (OpenDSS-faithful), so `tap.ratio_magnitude` carries
**only the off-nominal deviation** (≈1.0) and `tap.shift_deg` carries the clock. The √3 of
a delta winding cancels against the delta incidence `M`, so the positive-sequence block
reduces exactly to the classical off-nominal-tap pi (see the [transformer model](transformer.md)).

**Gotchas / current converter limits.**
- pandapower converter parses `net.trafo['vector_group']` (row column, else the
  `std_type` catalog entry; a clock-less bare form like `'Dyn'` is accepted too —
  pandapower's own `runpp_3ph` zero-sequence model requires that form and rejects a
  digit-suffixed one), cross-checked against `shift_degree` (`ConversionError` on a
  mismatch — pandapower's own balanced `runpp` uses only `shift_degree`, so silently
  preferring one source is never done). With no vector-group string anywhere
  (plain MATPOWER imports, e.g. `case118`) the connection falls back on the shift
  parity: even clock -> `WYE_GROUNDED`/`WYE_GROUNDED`, odd -> `DELTA`/`WYE_GROUNDED`.
  Tap-changer position (`tap_pos`/`tap_neutral`/`tap_step_percent`/`tap_side`, NaN-safe)
  IS read; `tap_side='hv'` -> `ratio_magnitude=1+delta`, `'lv'` ->
  `ratio_magnitude=1/(1+delta)` (verified against a live pandapower `runpp`: a
  positive HV-side tap LOWERS the LV voltage). An ideal phase-shifter tap
  (`tap_step_degree` nonzero, or `tap_phase_shifter=True`) is not modelled and raises.
  See `src/pgml/convert/pandapower/CONTEXT.md` and
  `tests/reference/test_pandapower_grid_matrix.py`.
- pgm transformer conversion is pinned against the installed power-grid-model C++ source
  (`transformer.hpp`): `clock*30` is IDENTICAL to pgml's "positive ⇒ LV lags" (verified by
  a live `PowerGridModel.calculate_power_flow` solve, no sign flip), and `tap_side`
  (0=from, else=to, matching pgm's own default) selects which nameplate voltage the tap
  volts are added to (`tests/reference/test_pgm_transformer.py`).
- pandapower stores `shift_degree` as a positive clock·30 (Dyn11 = 330, not 30); the
  converter cross-checks it against the `vector_group` clock digit and raises on a
  contradiction rather than silently preferring either.
- The OpenDSS converter (`convert.opendss.to_grid`) converts two-winding `Transformer`
  elements (winding 1 = HV/from, winding 2 = LV/to; `LeadLag` -> clock 1/11 baseline for a
  Dy/Yd pairing verified against a live solve, clock 0 baseline for Yy/Dd since OpenDSS has
  no explicit clock parameter). On top of that baseline, a winding whose bus string
  cyclically rotates the phase-conductor order (e.g. `bus=lv.2.3.1.0`) folds a further ±4
  clock steps into `tap.shift_deg` (verified against a live solve; phases are normalized to
  canonical A/B/C, never left in the rotated row order) — this is how EVERY clock of a
  pairing's correct parity converts, not just 0/1/11. Scope, verified via live oracle tests
  (`tests/reference/test_opendss_transformer.py`):
  - only two-winding transformers (3-winding raises `ConversionError`);
  - only solidly grounded wye (OpenDSS's shorthand-bus or explicit `.0` neutral) or delta
    windings — an explicit non-zero neutral node (floating or impedance-grounded) raises;
    zigzag has no OpenDSS `Transformer` connection at all, so a DSS file can never produce
    one;
  - a NON-cyclic winding-bus permutation (e.g. swapping two phase conductors) reverses the
    phase-rotation sequence and raises `ConversionError` rather than silently producing a
    wrong clock; the polarity-flip clocks `{2, 6, 10}` (the old `Yy6`/`Dd6` case) need a
    genuinely reversed winding construction that NO bus wiring (cyclic or not) can express,
    so they are never produced by this converter;
  - regulators (`RegControl`), 3-winding units, and OpenDSS's own frequency-correction
    curves (`XfmrCode`/`FreqMultCurve`) are not read;
  - the magnetizing branch (`%noloadloss`/`%imag`) converts with the SAME closed-form
    used by the pandapower converter (`pfe_w`/`i0%` on the HV base), but pgml stamps it
    as a simple HV-terminal shunt while OpenDSS's own internal model places it inside the
    leakage "T" — the two agree in direction and order of magnitude but not to the
    tight tolerance the leakage-only (no-magnetizing) oracle achieves (documented in that
    test file's module docstring).
  - the reverse direction (`opendss_oracle._build_circuit_with_real_transformer`, pgml ->
    DSS, used by the live harmonic vector-group oracle) mirrors this scope: it emits the
    `LeadLag` + rotated-bus equivalent for any reachable clock and raises
    `NotImplementedError` for zigzag windings or the `{2, 6, 10}` clocks.

---

## 4. Power sign convention

| | load P/Q | generator P/Q |
|---|---|---|
| **pgml** | positive = **consumption** | `Generator` injects (sign −1 internally) |
| **pandapower** | `load.p_mw>0` = consumption | `sgen/gen.p_mw>0` = injection |
| **OpenDSS** | `Load` positive = consumption | `Generator` positive = injection |
| **pgm** | `sym_load/asym_load` positive = consumption | `sym_gen` positive = injection |

All four agree (consumer reference for loads, generator reference for generators); the
converters pass P/Q through unchanged. ZIP behaviour: pgm `LoadGenType`
(`const_power/const_impedance/const_current`) maps to pgml's `LoadModel`; pandapower
`const_z_p_percent`/`const_i_p_percent` (+ the `_q_` twins) map to pgml
`ZipCoefficients`. (Converter coverage: pgm
`sym_gen` converts (`Generator`, generation-positive); pandapower `sgen`/`gen` and pgm
`asym_gen` are **not yet** converted.)

---

## 5. Units & internal representation

| | input units | internal | base |
|---|---|---|---|
| **pgml** | — (converters only) | **SI, phase-domain** | none (absolute SI) |
| **pandapower** | kV, MW, MVAr, Ω/km, nF/km, % | per-unit (MATPOWER) | `sn_mva` system base + per-bus `vn_kv` |
| **OpenDSS** | actual eng. units; length in `units=` (ft/m/km/…) | actual units | per-element |
| **pgm** | **SI** (V, W, VA, Ω, F, S) | SI | none |

**pgm is the least-lossy source** (pure SI; the converter only does `L=X/2πf₀` and
`G=tan·2πf₀·C`). pandapower needs kV→V, km→m, nF→F, MW→W and the per-km divisions;
OpenDSS needs the `units=` length conversion (the crux of §8's earth-return calibration)
and the `kVBase·√3` recovery. pandapower's exported `Ybus` is per-unit on the ppc base —
scale by `Z_base=vn_kv²/sn_mva` before comparing to pgml's SI Y.

---

## 6. Source / slack

| | reference voltage | Thévenin impedance | zero-sequence source Z |
|---|---|---|---|
| **pgml** | `Source.u_ref_v` = **L-N per phase** (3φ), `u_angle_deg` | diagonal per-phase R/L | = positive-seq (no separate value read) |
| **pandapower** | `ext_grid.vm_pu`·`vn_kv` (L-L), `va_degree` | from `s_sc_max_mva`, `rx_max` | `r0x0_max`/`x0x_max` (not read) |
| **OpenDSS** | `Vsource.basekv`·`pu` (L-L for `phases>=3`; used directly, no √3, for `phases=1` — see §1's gotcha), `angle` | `R1/X1` or `MVAsc3/MVAsc1`+`x1r1` | `R0/X0` (not read) |
| **pgm** | `source.u_ref`·`u_rated` (L-L), `u_ref_angle` | from `sk`, `rx_ratio` | `z01_ratio` (not read) |

**Decision.** All converters pass the magnitude OpenDSS itself would use as the solved
per-conductor EMF to `convert._common.build_source`, which divides by √3 under
THREE_PHASE to produce the per-phase **line-to-neutral** EMF for a `phases>=3` source
(1-phase keeps it unchanged — `build_source`'s `n<3` branch, matching OpenDSS's own
no-√3 treatment of a `phases=1` Vsource). `id_map["slack_v_complex"]` keeps that same
raw phasor (L-L for a `phases>=3` source; the solved 1-phase EMF for a `phases=1` one)
as a convenience for the single-phase ideal-slack `v_fixed`. At harmonics the source EMF
is zero (a short); the source contributes only its Norton shunt
`Y_s(h)=1/(R+j·2πh·f₀·L)`.

**Gotcha.** The zero-sequence source impedance is currently taken equal to the
positive-sequence value (`r0x0_max`/`R0/X0`/`z01_ratio` are not consumed) — relevant only
for 3-phase asymmetric studies where the source zero-sequence path matters.

---

## 7. Line model

| | parameters | per-length / total | form |
|---|---|---|---|
| **pgml** | `series_resistance_ohm_per_m`, `series_inductance_h_per_m`, `shunt_capacitance_f_per_m`, `_conductance_s_per_m` | **per-metre** | `n×n` phase matrices (or `conductor_geometry`) |
| **pandapower** | `r/x_ohm_per_km`, `c_nf_per_km`, `g_us_per_km` (+`r0/x0/c0`) | per-km | sequence |
| **OpenDSS** | `R1/X1/R0/X0`, `C1/C0` **or** `Rmatrix/Xmatrix/Cmatrix` **or** geometry | per `units=` | sequence / matrix / geometry (matrix & geometry take precedence) |
| **pgm** | `r1/x1/c1/tan1` (+`r0/x0/c0/tan0`) | **total** (Ω, F) | sequence |

**Mapping.** Sequence inputs → 3×3 phase matrices via `self=(Z0+2·Z1)/3`,
`mutual=(Z0−Z1)/3` (applied to R, X, C, G), then `L=X/2πf₀`. When a dataset has no native
zero-sequence data, `r0/x0/c0` default to `r1·(R0/R1)` etc. from
`config.line.zero_sequence.*` (an explicit native value always wins). pgm "total" values
use the `length_m=1` idiom so the per-metre×length product reproduces the total exactly.
OpenDSS native `n×n` matrices route through `build_line_from_matrices` (no sequence
assumption); pandapower/pgm route through `build_line_from_sequence`.

---

## 8. Harmonics & earth return (the key model-fidelity axis)

Only OpenDSS and pgml model harmonics. pandapower and pgm are **fundamental-only** (no
harmonic power flow, no frequency-dependent line constants, no earth-return model) — they
are load-flow result oracles, not harmonic oracles. pgml's harmonic model is entirely its
own; OpenDSS is the harmonic ground truth.

**OpenDSS.** Recomputes line impedance at **every** harmonic with a Carson/Deri
earth-return + skin model — for both geometry- and R/X-defined lines. So the naïve
"R const, X∝h" is wrong: R rises (skin on earth return) and X is **sub-linear** (the
earth-return log term shrinks as penetration depth drops). Default earth model is DERI.

**pgml — three line models (choose per study):**
- **Geometry Carson/Deri** (`conductor_geometry`, `geometry.carson`): the full
  complex-penetration formula — **bit-exact vs OpenDSS** (relZ ~1e-13) on every order,
  including triplen, because feeding the *same* geometry to both engines removes any
  earth-model ambiguity. Use this for OpenDSS parity. (See [Carson line constants](references/opendss/carson.md).)
- **Positive-sequence** (`apply_positive_sequence_harmonic_model`): `X1(h)=X1·h` + skin on
  `R1`, **no earth term** (it cancels in the positive sequence). Physically representative
  for balanced R/X feeders. (See the [harmonic line model](harmonic-line-model.md).)
- **Sequence-aware** (`apply_sequence_aware_harmonic_model`, the 3-phase config default for
  R/X lines): earth-free `Z1` + a zero-sequence `Z0` carrying the Carson earth
  **resistance** `3·(Re(f)−Re(f₀))`. Analytic and never non-physical, but `X0∝h` is
  **linear** (the sub-linear earth reactance is left to the geometry path), so it diverges
  from OpenDSS's Carson `Z0` on the zero-sequence (triplen) orders.

**The earth-return calibration gotcha (≈3.28×).** The classical Carson earth resistance is
`Re(f)=ω·μ₀/8 = π²·f·10⁻⁷ Ω/m`, geometry-independent and ∝f. pgml's configurable
`line.earth_return.resistance_coeff_ohm_per_m_per_hz` defaults to that **physical metric**
value (`π²·10⁻⁷`). OpenDSS exposes the same physics through per-LineCode `Rg`/`Xg`, but its
defaults are calibrated for **imperial length units (feet)**, so on a `units=m` line they
are ≈3.28× (= metres-per-foot) **smaller**. Comparing pgml's `sequence_aware` `Z0`
directly against an OpenDSS R/X line therefore shows a zero-sequence (triplen) gap from two
compounding causes: the units-calibrated earth resistance and the linear-vs-sub-linear
`X0`. Both vanish on the **geometry** path (identical geometry, identical Carson). The
transformer vector group is independent of this: a Dyn delta traps the zero sequence
identically on both sides (validated bit-for-bit by
`opendss_dyn_transformer_harmonic_voltages`). (See [OpenDSS harmonics](references/opendss/harmonics.md).)

**Transformer frequency scaling.** OpenDSS `XRConst=No` (default): R fixed, leakage X∝h;
pgml mirrors this (`X(h)=2π·h·f₀·L`, constant R) — a frequency-correction curve is not yet
modelled yet (tracked as open work in `src/pgml/STATUS.md`, "frequency-dependent device models").

---

## 9. Converter coverage gaps (consolidated — conversion scope, not core-model, issues)

These are places a source convention is **not yet** read, so a foreign network silently
under-converts. The core model supports each; only the converter intake is missing.

- **OpenDSS converter:** converts `Transformer` (two-winding, solidly grounded wye or
  delta windings; the clock is reachable for every value of the pairing's correct
  parity — via `LeadLag` combined with the source's cyclic winding-bus rotation, not
  just clock 0/1/11 — EXCEPT the polarity-flip clocks `{2, 6, 10}`, which need a
  reversed winding no bus permutation can express), `Line` (n×n matrices from the
  native R/X/C matrix API, per-terminal phase-permuted `to_phases`), `Load` (models
  1/2/5/8 → `LoadModel`/`ZipCoefficients`; models 3/4/6/7 fall back to `CONST_POWER`
  with a warning; each WYE element's own resolved return conductor carries over as
  `InjectionAppliance.return_path`), `Capacitor`/`Reactor` (→ `ShuntAppliance`,
  solidly-grounded WYE OR delta-connected, per-leg G/C from OpenDSS's own resolved
  values — an explicitly coupled `Rmatrix`/`Xmatrix` reactor and a non-grounded/2-bus
  terminal-2 reference are still skipped with a warning), and `Generator`/`PVSystem`/
  `Storage` (generation-positive / signed discharge-positive, plus `Storage`'s inert
  energy-state fields); NOT
  read/converted: 3-winding transformer units, `RegControl` regulators, tap-changer
  control, `XfmrCode`/frequency-correction curves, and an explicit non-zero
  (floating or impedance-grounded) neutral node (raises). Every other DSS element
  class (`Isource`, `Monitor`, `EnergyMeter`, `CapControl`, `InvControl`,
  `StorageController`, `Relay`, `Recloser`, `Fuse`, `Sensor`, …) is enumerated
  generically and triggers a `warn_dropped_elements` warning naming the kind and
  count — nothing vanishes silently; a non-negligible Vsource `R1`/`X1` under the
  default `slack="ideal"` likewise warns (it is ignored unless `slack="norton"`).
- **pgm converter:** `transformer` converts (two-winding, full vector-group support
  including zigzag — see §2/§3 rows above and `src/pgml/convert/pgm/CONTEXT.md`); `sym_gen`
  converts (`Generator`, generation-positive); NOT read/converted: `asym_gen`,
  `three_winding_transformer`, `transformer_tap_regulator`, `shunt`, `link`; transformer
  `uk_min`/`uk_max`/`pk_min`/`pk_max` (tap-dependent short-circuit parameters) and
  `i0_zero_sequence`/`p0_zero_sequence` ignored; `source.z01_ratio` and line `tan0` ignored;
  pgm stores no `f0`, so the caller must pass the correct `base_frequency_hz` (a 50/60 Hz
  mismatch silently scales every L and C).
- **pandapower converter:** `transformer` converts (two-winding, vector-group +
  tap-changer aware — see §2/§3 rows above and
  `src/pgml/convert/pandapower/CONTEXT.md`); `sgen`/`asymmetric_load` convert
  (per-element `scaling` honored); `line`/`trafo` honor the `parallel` column
  (identical parallel systems — divides the series impedance, multiplies the shunt
  admittance and rated power; `parallel==1` stays byte-identical); bus-bus `switch`
  (`et='b'`) converts to a near-ideal `Switch`, and an OPEN bus-line/bus-transformer
  `switch` (`et='l'`/`'t'`) takes the whole line/transformer out of service (an
  accepted approximation — pandapower itself keeps the still-connected terminal
  energized via an internal auxiliary bus, so this drops that terminal's shunt too);
  `load` maps the four-column `const_z_p_percent`/`const_i_p_percent`/
  `const_z_q_percent`/`const_i_q_percent` onto `ZipCoefficients` (all-zero, the
  pandapower default, stays byte-identical). NOT read/converted: `gen` (PV/voltage-controlled
  buses), `shunt`, `trafo3w`, `impedance`, `ward`/`xward`, `dcline`, `storage`,
  `motor`, `asymmetric_sgen`; an ideal phase-shifter tap
  (`tap_step_degree`/`tap_phase_shifter`) is not modelled and raises; source
  zero-sequence (`r0x0_max`/`x0x_max`) not read.

When extending a converter, conform to §0 and record the source convention in `Provenance`;
update this file if a new cross-tool convention difference is discovered.
