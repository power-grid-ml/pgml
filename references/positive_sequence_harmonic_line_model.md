# Positive-sequence-aware harmonic line model — decision record

**Status:** implemented. Module:
`src/pgml/geometry/sequence.py`; wired into assembly via
`Line.resistance_frequency` (`carson_skin_multiplier` law) and
`geometry.synthesis.apply_positive_sequence_harmonic_model`.

## The problem
The R/X→geometry synthesis (`geometry/synthesis.synthesize_line_geometry`) reproduces a
line's `R1 + jX1` at the fundamental by reverse-fitting a **single conductor with earth
return**. But a single overhead conductor with earth return has a self-reactance
**floor** (the Deri earth term, ~0.4 Ω/km at 50 Hz) that EXCEEDS the positive-sequence
`X1` of cables and low-X feeders. To hit a small `X1` the closed-form GMR therefore has
to grow past the conductor radius (GMR ≥ radius → non-physical; CIGRE-LV: GMR ≈ 270 m,
26/32 IEEE-33 lines too) and the reactance can even go negative at high harmonics. A
`warnings.warn` + `provenance.extra["synth_unphysical"]` flag surfaced this, but the
harmonic *magnitudes* on those feeders were not physically representative.

## The physics
A line's phase impedance splits into three parts:

| part | scales as | appears in |
|---|---|---|
| conductor **internal** (skin effect, Bessel `I0/I1`) | √f-ish growth of R | Z1 **and** Z0 |
| **geometric** (Maxwell, `∝ ln(D/GMR)`) | reactance `∝ f` | Z1 **and** Z0 |
| **earth return** (Carson/Deri, ground path) | sub-linear, large R0 | **Z0 only** |

For a **balanced positive-sequence** current the three phase currents sum to zero, so
there is **no net ground current** and the earth-return terms **cancel**. Hence:

* `Z1(h)` carries only internal + geometric → `X1(h) = X1·(f/f0)` (geometric ∝ frequency)
  with a skin-effect rise on `R1`, and **NO earth floor**.
* `Z0(h)` (zero sequence / ground-return loops) carries the earth return → the floor.

This is exactly what other tools do: pandapower / PSS®E / PowerFactory store lines as
sequence impedances or per-length R/X matrices directly and never reverse-synthesise
geometry, so they never add earth return to the positive sequence; EMTP/ATP keep the
Z1/Z0 split and only Z0 gets earth return. OpenDSS *does* apply its earth correction to
a 1-phase `r1/x1` line modelled as a single conductor with earth return — which is what
produced the "harmonic gap" we matched, but is an **artifact** for a balanced
positive-sequence equivalent.

We verify the split numerically: build a genuine 3-phase overhead geometry, run the full
Carson model, and decompose with the Fortescue transform
(`sequence.phase_to_sequence`). On that geometry, `X1(h)/(h·X1(f0)) → 0.9998` (∝ h, no
floor) while `X0(h)/(h·X0(f0)) → 0.86` (earth floor, sub-linear) and `R0/R1 ≈ 5` at
h = 25. See `tests/reference/test_carson_sequence.py` and the figure
`seq_xr_vs_harmonic.svg`.

## The model
For sequence / R-X-defined lines (the common case: IEEE-33, CIGRE LV), the corrected
harmonic impedance is

```
Z1(h) = R1 · m_skin(h)  +  j · X1 · (f / f0)
```

* `X1·(f/f0)` — geometric reactance scales linearly with frequency (constant `L1`). The
  explicit R/L/C assembly path already does this (`X(h)=2π f L`), with **no earth term**.
* `m_skin(h)` — the skin-effect resistance multiplier, the **same** Bessel `I0/I1`
  internal-resistance growth the Carson geometry path uses
  (`carson.internal_impedance`), fit so `m_skin(f0)=1`, with the **earth term dropped**.
  This is the one physical effect the naive "R const, X∝h" model was missing.

Two equivalent constructions (both in `geometry/sequence.py`):

1. **Direct** `positive_sequence_z(r1, x1, f0, freqs)` — the formula above. `X` is `∝ h`
   to floating point; differentiable in `R1`/`X1`; batched over lines and harmonics.
2. **Physical two-conductor go/return** `two_conductor_geometry` + `two_conductor_loop_z`
   — a `+I` go and `−I` return conductor pair; the Carson `[1,−1]` loop transform makes
   the large earth penetration-depth term **cancel analytically**, yielding a *physical*
   GMR (fixed at `0.7788·radius`) and a finite spacing `D = GMR·exp(X1/(2·f0·μ0))` for
   **any** `X1`. It agrees with the direct model (residual earth coupling ≲ 2 % to
   h ≈ 25) — i.e. the direct model is physically grounded, not an ad-hoc scaling.

The full Carson/Deri-with-earth-return model is **reserved** for genuinely
geometry-defined lines (`Line.conductor_geometry`) and for Z0 / ground-return paths,
where it is correct and remains bit-exact vs OpenDSS.

## How it is wired in
No schema change (`schemas/` is frozen). The positive-sequence model reuses the existing
`ResistanceFrequencyModel`:

* `apply_positive_sequence_harmonic_model(grid)` sets each R/X line's
  `resistance_frequency` to `AnalyticParam(law="carson_skin_multiplier",
  params={r1_ohm_per_m, f0_hz})`. The line keeps its explicit R/L/C (so `X(h)=X1·h`,
  no geometry, no earth floor).
* `assembly._resistance_multiplier` evaluates that law differentiably (lazy import of
  `sequence.skin_resistance_multiplier`) and now also supports `curve` multipliers
  (linear interpolation). The default `ConstantParam(value=1.0)` is unchanged, so all
  existing grids and oracle tests are untouched.

## Comparison to OpenDSS (honest scope)
`examples/evaluate_line_sequence_harmonics.py` (figure `feeder_h13.svg`) overlays, on
IEEE-33: the corrected positive-sequence model, the naive model, the single-conductor
Carson model (== OpenDSS, which we still match bit-exact on the SAME geometry), and the
live OpenDSS profile. The single-conductor earth correction shifts the h = 13 voltage
profile materially (≈ 0.0081 vs 0.0095 pu at the feeder end); the corrected model removes
that earth-floor artifact while keeping the skin-effect resistance rise (a smaller,
second-order effect on `|V|`). A *fully* independent OpenDSS cross-check of `Z1`/`Z0`
needs real 3-phase conductor coordinates rather than R/X synthesis; that
remains future work, but the Fortescue decomposition above already validates the physics
against the bit-exact Carson code.

## Tests & figures
* `tests/reference/test_carson_sequence.py` — Z1 vs Z0 on a 3-phase geometry (earth
  return only in Z0); `X1(h) ∝ h` to floating point; direct ≈ two-conductor loop;
  two-conductor synthesis stays physical where the single-conductor one fails; batching.
* `tests/differentiability/test_sequence_gradcheck.py` — gradcheck of `positive_sequence_z`
  (R1, X1), the skin multiplier, the go/return loop, and the gradient through a full
  assembly with the skin law active.
* `tests/gpu/test_device_parity.py` — CPU/CUDA parity of the positive-sequence model, its
  assembly, and (newly) the Carson geometry assembly path.
* `examples/evaluate_line_sequence_harmonics.py` — `seq_xr_vs_harmonic.svg` (R/X vs h,
  pos vs zero seq), `gmr_floor.svg` (single- vs two-conductor synthesis), `feeder_h13.svg`
  (feeder profile, corrected vs naive vs single-conductor Carson vs OpenDSS).
