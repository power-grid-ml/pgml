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

This is the standard symmetrical-components result, and it is what the major tools
implement (sources and the per-tool details are in *How different simulation tools model
this* below). In particular OpenDSS **does** carry an explicit Carson earth-return term
on its R/X line codes — the `Rg` / `Xg` parameters of a `LineCode`, default
`0.01805 + j·0.155081` Ω per 1000 ft at 60 Hz (100 Ω·m earth, user-overridable)
([LineCode docs](https://opendss.epri.com/LineCode1.html)). But `Rg`/`Xg` are a
**common-mode** (ground-loop) quantity: for a balanced 3-phase line they cancel in `Z1`
and surface only in `Z0`. They enter the *series* impedance a study actually sees only
when the line is modelled with a **single conductor / single phase** and an earth return
— which is exactly the single-conductor synthesis above (and the OpenDSS 1-phase
`LineGeometry` line we matched bit-exact). For a balanced positive-sequence equivalent
the earth floor is therefore an artifact, not a physical series-impedance term.

We verify the split two ways. (1) Build a genuine 3-phase overhead geometry, run the full
Carson model, and decompose with the Fortescue transform (`sequence.phase_to_sequence`):
`X1(h)/(h·X1(f0)) → 0.9998` (∝ h, no floor) while `X0(h)/(h·X0(f0)) → 0.86` (earth floor,
sub-linear) and `R0/R1 ≈ 5` at h = 25. (2) Drive a running OpenDSS with a native 3-phase
`R1/X1/R0/X0` line: its positive sequence comes back as `Z1(h) = R1 + j·X1·(f/f0)`
exactly (R1 constant, ratio 1.0000 at every harmonic), with the earth correction only in
`Z0(h)`; the same R/X as a 1-phase line instead carries the earth floor. Both are in
`tests/reference/test_carson_sequence.py`; the figure is `seq_xr_vs_harmonic.svg`.

## How different simulation tools model this
Earth return is, by construction, a **zero-sequence / ground-loop** quantity. Every tool
below keeps it out of the positive sequence; they differ only in how a line is *entered*
and how the reactance is frequency-scaled.

### OpenDSS
Three line-impedance paths, each with its own frequency behaviour:
1. **`LineGeometry`** (real conductor coordinates): full Carson/Deri recomputation at
   every frequency — earth return **and** skin effect. If `Geometry` is specified, all
   other impedance definitions are ignored
   ([Line docs](https://opendss.epri.com/Line.html)). The earth model is selectable
   (`earthmodel = Carson | Deri | FullCarson`,
   [Cable modeling](https://opendss.epri.com/CableModelinginOpenDSS.html)). This is the
   path `pgml.geometry` matches **bit-exact** (`references/opendss/carson.md`).
2. **`LineCode` / impedance-defined** (`R1 X1 R0 X0`, or `Rmatrix Xmatrix`): carries the
   explicit Carson earth-return terms `Rg`, `Xg`
   ([LineCode docs](https://opendss.epri.com/LineCode1.html), default
   `0.01805 + j·0.155081` Ω/1000 ft @ 60 Hz, overridable). **Skin effect is NOT applied
   to `R` here** — only `LineGeometry` lines get skin. At harmonics OpenDSS scales the
   reactance ∝ frequency and frequency-corrects `Rg`/`Xg`. Measured against a running
   OpenDSS (`tests/reference/test_carson_sequence.py`):
   - a **3-phase** R/X line → `Z1(h) = R1 + j·X1·(f/f0)` (R1 constant, earth cancels),
     the earth correction surfacing only in `Z0(h)`;
   - a **1-phase** R/X line → the earth term enters the single self-impedance, so `Z(h)`
     carries the earth floor (R rises, X sub-linear).
3. So even a "plain" R/X code is not pure `X∝h` at the *matrix* level (because of
   `Rg`/`Xg`), yet the **positive sequence a balanced study sees still is** `X∝h`,
   `R` const.

### pandapower / PowerFactory / PSS®E
Lines are entered as **sequence impedances** (or per-km positive/zero-sequence R/X)
directly; the reactance scales ~linearly with frequency for harmonic studies. Earth
return is a zero-sequence parameter and is never added to the positive sequence — the
textbook symmetrical-components convention (see the respective manuals).

### EMTP / ATP
Keep the explicit `Z1`/`Z0` split; only `Z0` carries the Carson earth return.

**Takeaway:** the *physically correct* positive-sequence harmonic series impedance — and
the one every tool above uses for the positive sequence — has **no earth floor**; it is
`X∝h` (+ skin on R if conductor data is known). The earth return belongs to `Z0`. For a
**balanced** study only `Z1` is excited, so the earth return never appears. For an
**unbalanced** study it does — see *Unbalanced / 4-wire studies* below.

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

## Unbalanced / 4-wire studies — the sequence-aware model
Most low-voltage grids are 4-wire (phase + neutral, often grounded) and are operated
**asymmetrically** (per-phase loads / generators / sources with their own spectra). There
the earth/neutral return **does** matter, but the trigger is *zero-sequence* current, not
unbalance per se:

* Decompose any injection into sequences. **Positive + negative** sequence currents sum to
  zero across the phases → no ground current → they see `Z1 = Z2`, **no earth return**.
* **Zero** sequence is the residual `I_a+I_b+I_c = 3·I_0` → it returns through earth/neutral
  → it sees `Z0`, which **carries the earth return**.

So earth return becomes mandatory exactly when the unbalanced study has a grounded/neutral
return path (a 4-wire grounded LV feeder with unbalanced loading); a 3-wire/delta
unbalanced load with no ground path produces no residual current and no earth return.

The right vehicle is the **full coupled `Z_abc(h)`**, not a per-phase earth floor:

```
Z_self(h)   = (Z0(h) + 2·Z1(h)) / 3        # earth return appears here…
Z_mutual(h) = (Z0(h) − Z1(h)) / 3          # …and here, coupling the phases
```

with each sequence frequency-corrected **separately** (`geometry/sequence.py`):

* `Z1(h)` — the earth-free positive-sequence model above (`positive_sequence_z`).
* `Z0(h) = ` conductor part (`X0∝h`, optional skin) `+ 3·(Re(f) − Re(f0))`, where
  `Re(f) = π²·f·10⁻⁷` Ω/m is **Carson's earth-return resistance** — geometry-independent,
  `∝ f`. This is the frequency-growing **damping** the positive sequence never sees, and it
  is what keeps an unbalanced study from over-predicting zero-sequence harmonics. `Re` is
  `≥ 0` and monotone, so `Z0(h)` can never go non-physical (unlike a single-conductor
  earth floor). `zero_sequence_harmonic_z`, `sequence_to_phase_z`, `sequence_aware_phase_z`.

What you must **not** do is model each phase as an independent single-conductor-with-earth
line (a diagonal `Z_abc` whose every diagonal carries the full earth floor): that ignores
the inter-phase mutual coupling **and** triple-counts the earth term — wrong for unbalanced
work, not just balanced.

**Scope / honesty.** The earth-return *resistance* (the dominant damping term) is universal
and robust. The earth-return *reactance* sub-linearity is **return-path dependent** — deep
earth (overhead, `De ≈ 658·√(ρ/f)` m) versus a nearby neutral/sheath a few cm away (LV
cable) — so it is NOT applied generically here (`X0` scales `∝h`); for a rigorous,
return-path-correct `Z0(h)` reactance, use the **geometry path** with the actual conductor
+ neutral coordinates (full Carson, bit-exact vs OpenDSS). `earth_resistance_coeff` is
exposed (default Carson `π²·10⁻⁷`) so the damping strength can be tuned or matched to a
reference tool, mirroring OpenDSS's user-settable `Rg`/`Xg`.

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

The **sequence-aware** model is opt-in via a `Line.tags` marker (also schema-free):
* `apply_sequence_aware_harmonic_model(grid)` tags each 3-phase R/X line
  `harmonic_line_model=sequence_aware` (with `seq_skin` / `seq_earth_coeff`).
* `assembly._stamp_sequence_aware_lines` decomposes the line's `Z_abc(f0)` into `Z1`/`Z0`,
  frequency-corrects each (lazy import of `sequence.sequence_aware_phase_z`), recombines to
  `Z_abc(h)`, and stamps it. Untagged lines are unaffected.

## Defaults are explicit and config-tracked (no hidden implicit model)
Modeling decisions are **deliberate, documented choices**, not silent implicit defaults —
the opacity that makes OpenDSS discrepancies hard to explain. Every default value and
default model choice lives in `src/pgml/config/defaults.yaml` (one ordered, self-describing
file), resolved with precedence **explicit > config > converter** (`pgml.config`). The
constants that used to be hard-coded (`gmr_over_radius = 0.7788`, the earth-return
coefficient `π²·1e-7`, conductor radius/heights, soil resistivity) now live there.

`apply_default_harmonic_model(grid)` is the **single deliberate entry point** that turns
the config defaults into per-line models: it reads `line.harmonic_model.three_phase`
(default `sequence_aware` — for 4-wire unbalanced LV studies) and `.single_phase`
(default `positive_sequence`), and applies them — but only to lines that do **not** already
carry an explicit model or a `conductor_geometry` (precedence 1 wins). Nothing is applied
silently at solve time; you call it (or set a per-line model) on purpose.

## Choosing the model in pgml — where to set it, and what it matches
You select the model per grid; the recommended path is the config-default dispatcher.
**What you compare against in OpenDSS decides whether they "agree" — the same R/X data
modelled as a 3-phase line vs. a 1-phase line gives different harmonic answers in OpenDSS
itself.**

| You have / want | How to set it in pgml | Harmonic line model | Matches OpenDSS… |
|---|---|---|---|
| **config defaults** (recommended) | `apply_default_harmonic_model(grid)` | per `line.harmonic_model.*`: 3-phase→`sequence_aware`, 1-phase→`positive_sequence` | 3-phase R/X with `Rg`/`Xg` (earth in `Z0`) |
| R/X feeder, raw `X∝h` (no skin/earth) | do nothing | `Z1(h) = R1 + j·X1·(f/f0)` | native **3-phase** `R1/X1` LineCode (`Z1`) |
| R/X feeder + physical skin on R | `apply_positive_sequence_harmonic_model(grid)` | `Z1(h) = R1·m_skin(h) + j·X1·(f/f0)` | 3-phase R/X **plus** a skin rise OpenDSS only adds for `LineGeometry` |
| **unbalanced 4-wire** R/X feeder (`Z1`+`Z0`) | `apply_sequence_aware_harmonic_model(grid)` | `Z_abc(h)`: `Z1` earth-free + `Z0` earth-damped | native 3-phase R/X with `Rg`/`Xg` (earth in `Z0`); reactance sub-linearity only via geometry |
| real 3-phase conductor coordinates | set `Line.conductor_geometry` | full Carson (earth in `Z0`, skin on R) | OpenDSS `LineGeometry` (bit-exact) |
| single-conductor / SWER / Carson-code check only | `synthesize_grid_geometry(grid)` | single-conductor + earth floor | OpenDSS **1-phase** `LineGeometry` line |

Concretely:
- **Existing R/X feeder (CIGRE LV, IEEE-33).** The pgml **default** (no call) already gives
  `X(h)=X1·h`, `R` const — *identical* to native OpenDSS modelling those same R/X values as
  a 3-phase line. Calling `apply_positive_sequence_harmonic_model(grid)` adds the
  physically-correct skin rise on `R1` (a small, extra-damping refinement on top — OpenDSS
  does not skin-correct R/X codes). `synthesize_grid_geometry(grid)` gives the
  single-conductor-with-earth-return model: only use it to validate the Carson code, not
  for representative harmonic magnitudes (it overstates the series reactance for balanced
  operation).
- **Unbalanced 4-wire R/X feeder.** Give the lines a full 3×3 `Z_abc(f0)` (so `Z0` is
  defined by the off-diagonal mutuals) and call `apply_sequence_aware_harmonic_model(grid)`.
  An unbalanced / zero-sequence current then sees the earth-return damping in `Z0`, while a
  balanced current still sees the earth-free `Z1`. This is the model an asymmetric LV study
  needs from sequence data; for a return-path-exact `Z0` reactance prefer geometry lines.
- **Custom grid with geometry-defined lines.** Put real conductor coordinates on
  `Line.conductor_geometry`; assembly auto-routes to the full Carson/Deri path (earth
  return + skin), which is bit-exact vs OpenDSS `LineGeometry` and correctly keeps earth
  return in `Z0` only. This is the **rigorous** path for unbalanced harmonic studies
  (correct neutral/earth return). Nothing else to set.

**So does the simplified R/X approach "significantly differ from OpenDSS"?** Only versus
OpenDSS modelling the lines as **1-phase / single-conductor with earth return** — that is
the comparison in `feeder_h13.svg` and it differs by ≈ 0.0081 vs 0.0095 pu at h = 13.
Versus native OpenDSS modelling the same R/X as a **3-phase** line (the standard way to
enter a balanced feeder), pgml's default **agrees** (`Z1(h) = R1 + j·X1·(f/f0)`). The
divergence is a property of the *reference setup*, not of the pgml model.

## Comparison to OpenDSS (honest scope)
`examples/evaluate_line_sequence_harmonics.py` (figure `feeder_h13.svg`) overlays, on
IEEE-33: the corrected positive-sequence model, the naive model, the single-conductor
Carson model (== OpenDSS **1-phase** `LineGeometry`, which we still match bit-exact on the
SAME geometry), and the live OpenDSS profile from that geometry. The single-conductor
earth correction shifts the h = 13 voltage profile materially (≈ 0.0081 vs 0.0095 pu at
the feeder end); the corrected model removes that earth-floor artifact while keeping the
skin-effect resistance rise (a smaller, second-order effect on `|V|`). Note this overlay
uses the 1-phase/single-conductor OpenDSS setup; native **3-phase** OpenDSS R/X gives the
naive curve (verified in `test_carson_sequence.py`), which the pgml default reproduces.

## Tests & figures
* `tests/reference/test_carson_sequence.py` — Z1 vs Z0 on a 3-phase geometry (earth
  return only in Z0); `X1(h) ∝ h` to floating point; direct ≈ two-conductor loop;
  two-conductor synthesis stays physical where the single-conductor one fails; batching.
  Plus two **native-OpenDSS** oracle checks: a 3-phase R/X line returns `Z1(h)=R1+jX1·(f/f0)`
  (matched by the pgml default) with earth only in `Z0`, while the same R/X as a 1-phase
  line carries the earth floor.
  Plus the **sequence-aware** checks: `Z0` gains a frequency-growing earth-return
  resistance the positive sequence does not; `Z_abc(h)` recombines back to exactly
  `(Z0, Z1, Z1)`; and a tagged 3-phase line, assembled at harmonics, recovers an earth-free
  `Z1` and a strongly damped `Z0`.
* `tests/differentiability/test_sequence_gradcheck.py` — gradcheck of `positive_sequence_z`
  (R1, X1), the skin multiplier, the go/return loop, `sequence_aware_phase_z` /
  `zero_sequence_harmonic_z` (R1/X1/R0/X0), and the gradient through both the skin-law and
  the sequence-aware assembly paths.
* `tests/gpu/test_device_parity.py` — CPU/CUDA parity of the positive-sequence model, the
  sequence-aware model, their assembly paths, and the Carson geometry assembly path.
* `examples/evaluate_line_sequence_harmonics.py` — `seq_xr_vs_harmonic.svg` (R/X vs h,
  pos vs zero seq), `gmr_floor.svg` (single- vs two-conductor synthesis), `feeder_h13.svg`
  (feeder profile, corrected vs naive vs single-conductor Carson vs OpenDSS).
