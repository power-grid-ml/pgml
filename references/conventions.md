# Modelling conventions — pgml vs pandapower / OpenDSS / power-grid-model

The single place that pins **pgml's internal definitions** and records how they differ
from the three reference tools we convert from and validate against
(pandapower, OpenDSS, power-grid-model / "pgm"). The conversion boundary
(`pgml.convert.*`) is where a foreign convention enters the library, so it is also where
a silent factor-of-√3 or referred-to-the-wrong-side error is introduced. The rule is:
**every converter translates the source convention into pgml's canonical form below;
the core never sees a foreign convention.**

Companion decision records (deeper derivations): `opendss/transformer.md` (vector-group
two-winding model), `opendss/carson.md` (bit-exact earth-return), `opendss/harmonics.md`
(injection + frequency scaling), `positive_sequence_harmonic_line_model.md`,
`asymmetric_modeling.md`. Schema source of truth: `src/pgml/schemas/CONTEXT.md`.

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
| **OpenDSS** | `Vsource.basekv` [kV] | **L-L** | but `Bus.kVBase()` **always returns L-N** (`=basekv/√3`); `AllBusVolts` are L-N phasors |
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
  const-Z shunt; corrected with the phase-mode migration. See `convert/opendss/CONTEXT.md`.)

**Gotchas.**
- OpenDSS `Bus.kVBase()` is L-N regardless of phase count — `×√3` is mandatory.
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
| **pandapower** | `vk_percent`, `vkr_percent`, `sn_mva` | **LV side** ✅ | `Z_base_LV=vn_lv_v²/sn_va`; `R=vkr%·Z_base_LV`, `\|Z\|=vk%·Z_base_LV`, `X=√(\|Z\|²−R²)`, `L=X/2πf₀` |
| **pgm** | `uk`, `pk`, `sn`, `u2` | **to-side (LV)** ✅ | `R=pk·u2²/sn²`, `\|Z\|=uk·u2²/sn`, `X=√(\|Z\|²−R²)` *(transformer path not yet implemented in the converter)* |
| **OpenDSS** | per-winding `%R`, inter-winding `XHL` | **winding 1 (HV)** ❌ | oracle back-calculates from pgml's LV-referred R/L on the LV base |

**Decision & rationale.** pgml refers the leakage admittance to the **TO/LV coil**, the
same side as pandapower and pgm (the two load-flow oracles), so their `vk/vkr/uk/pk`
convert with a single LV base and no extra referral. It is also the natural side for the
phase-domain winding-incidence primitive `Y = Nᵀ·Y_winding·N`, where the leakage `y` sits
on the LV coil block and the HV self-block picks up the `1/τ²` from the turns ratio (see
`opendss/transformer.md`). OpenDSS instead references `XHL` to winding 1 (HV); the live
OpenDSS Dyn oracle (`opendss_oracle._build_circuit_with_real_transformer`) back-calculates
`%R`/`XHL` from pgml's LV-referred R/L — self-consistent because the total per-unit
leakage is preserved.

**Magnetizing branch.** pgml refers the magnetizing shunt `y_m=G_m+jB_m` to the **HV**
terminal (`magnetizing_conductance_s`, `magnetizing_inductance_h`); the pandapower
converter computes `G_m=pfe_w/u_hv²`, `B_m` from `i0%`/`sn` on the HV base. pandapower
internally keeps it on the LV base split into the pi-shunt — physically equivalent after
the turns ratio, but the raw numbers differ, so do not compare them without re-referring.

---

## 3. Transformer ratio, tap, vector group / clock

| | nominal ratio | tap (off-nominal) | vector-group phase shift |
|---|---|---|---|
| **pgml** | from rated **coil** voltages + connections (`nominal_turns_ratio`: delta coil=L-L, wye coil=L-N=`u/√3`) | `ComplexTap.ratio_magnitude` (1.0 = on-tap) | `tap.shift_deg = clock·30`; **positive ⇒ LV lags HV**; `clock_transpose=sin(shift)>0` picks `Mᵀ` (Dyn1) vs `M` (Dyn11) |
| **pandapower** | `vn_hv_kv/vn_lv_kv` (MATPOWER off-nominal tap) | `tap_pos/tap_neutral/tap_step_percent`, `tap_side` | `shift_degree` (positive ⇒ LV lags, matches pgml) |
| **OpenDSS** | ratio of winding coil kV | tap per winding | `LeadLag` = `Lag`→Dyn1 (`shift 30`) / `Lead`→Dyn11 (`shift 330`) |
| **pgm** | `u1/u2` | `tap_side/pos/nom/size/min/max` | `clock` 0–12; `winding_from/to` enums |

**Decision & rationale.** The nominal ratio and the ±30° clock shift come from the rated
voltages + winding connections (OpenDSS-faithful), so `tap.ratio_magnitude` carries
**only the off-nominal deviation** (≈1.0) and `tap.shift_deg` carries the clock. The √3 of
a delta winding cancels against the delta incidence `M`, so the positive-sequence block
reduces exactly to the classical off-nominal-tap pi (`opendss/transformer.md`).

**Gotchas / current converter limits.**
- pandapower converter **hard-codes** `from_connection=DELTA`, `to_connection=WYE_GROUNDED`
  (correct for the CIGRE LV Dyn1 units, wrong for Yyn/Yzn/etc. — the `vector_group` string
  is recorded in `Provenance`, not parsed).
- Tap-changer position (`tap_pos`/`tap_step`) is **not read**; off-nominal ratio stays 1.0.
- pgm transformers are **not converted at all** yet; when added, the `clock` sign must be
  pinned against a known case (pgm's `clock·30` is the to-side shift; confirm it maps to
  pgml's "positive ⇒ LV lags" before trusting it).
- pandapower stores `shift_degree` as a positive clock·30; verify it stores Dyn11 as 330
  (or −30) and not 30 before relying on `clock_transpose` for a non-Dyn1 group.
- The OpenDSS converter does **not** emit `Transformer` elements (DSS→pgml transformer
  parsing is a gap); CIGRE/IEEE feeders enter via pandapower.

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
`const_z_percent`/`const_i_percent` map to pgml `ZipCoefficients`. (Converter coverage:
pandapower/pgm `sym_gen`/`sgen`/`gen` are **not yet** converted.)

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
| **OpenDSS** | `Vsource.basekv`·`pu` (L-L), `angle` | `R1/X1` or `MVAsc3/MVAsc1`+`x1r1` | `R0/X0` (not read) |
| **pgm** | `source.u_ref`·`u_rated` (L-L), `u_ref_angle` | from `sk`, `rx_ratio` | `z01_ratio` (not read) |

**Decision.** All converters pass the **line-to-line** magnitude to
`convert._common.build_source`, which divides by √3 under THREE_PHASE to produce the
per-phase **line-to-neutral** EMF (1-phase keeps it). `id_map["slack_v_complex"]` keeps the
**L-L** phasor as a convenience for the single-phase ideal-slack `v_fixed`. At harmonics
the source EMF is zero (a short); the source contributes only its Norton shunt
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
  earth-model ambiguity. Use this for OpenDSS parity. (`opendss/carson.md`)
- **Positive-sequence** (`apply_positive_sequence_harmonic_model`): `X1(h)=X1·h` + skin on
  `R1`, **no earth term** (it cancels in the positive sequence). Physically representative
  for balanced R/X feeders. (`positive_sequence_harmonic_line_model.md`)
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
`opendss_dyn_transformer_harmonic_voltages`). (`opendss/harmonics.md`)

**Transformer frequency scaling.** OpenDSS `XRConst=No` (default): R fixed, leakage X∝h;
pgml mirrors this (`X(h)=2π·h·f₀·L`, constant R) — a frequency-correction curve is not yet
modelled (see `HANDOFF.md`, "frequency-dependent device models").

---

## 9. Converter coverage gaps (consolidated — conversion scope, not core-model, issues)

These are places a source convention is **not yet** read, so a foreign network silently
under-converts. The core model supports each; only the converter intake is missing.

- **OpenDSS converter:** does not parse `Transformer` elements.
- **pgm converter:** no `transformer`, no `sym_gen`/`asym_gen`; `source.z01_ratio` and line
  `tan0` ignored; pgm stores no `f0`, so the caller must pass the correct
  `base_frequency_hz` (a 50/60 Hz mismatch silently scales every L and C).
- **pandapower converter:** transformer connection hard-coded to Dyn (`DELTA`/
  `WYE_GROUNDED`); `tap_pos`/`tap_step` not read; `sgen`/`gen` not converted; source
  zero-sequence (`r0x0_max`) not read.

When extending a converter, conform to §0 and record the source convention in `Provenance`;
update this file if a new cross-tool convention difference is discovered.
