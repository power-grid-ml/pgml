# Harmonic line model

How pgml scales a line's impedance with frequency when only the line's sequence data
(`R1/X1/R0/X0`) is known. The geometric reactance scales with frequency, the conductor
resistance rises with skin effect, and the earth-return term belongs to the zero sequence.

## The problem

A line given as `R1 + jX1` can be reverse-fitted to a single conductor with earth return.
That fit reproduces the fundamental impedance exactly, but it does not extrapolate to
harmonics. A single overhead conductor with earth return has a self-reactance floor, the
Deri earth term of about 0.4 Ω/km at 50 Hz, and the floor exceeds the positive-sequence
`X1` of cables and low-reactance feeders. Reaching a small `X1` then drives the fitted
geometric mean radius past the conductor radius, which is not physical (about 270 m on the
CIGRE LV feeder, and 26 of the 33 IEEE-33 lines are affected), and the reactance can turn
negative at high orders. pgml warns when a synthesis lands there and records it in the
line's provenance. The harmonic magnitudes of such a feeder are still not representative.

## The physics

A line's phase impedance splits into three parts.

| part | frequency behaviour | appears in |
|---|---|---|
| conductor internal (skin effect, Bessel `I0/I1`) | resistance grows roughly with √f | `Z1` and `Z0` |
| geometric (Maxwell, `∝ ln(D/GMR)`) | reactance `∝ f` | `Z1` and `Z0` |
| earth return (Carson/Deri ground path) | sub-linear reactance, strongly rising `R0` | `Z0` only |

For a balanced positive-sequence current the three phase currents sum to zero. No net
current returns through the ground, so the earth-return terms cancel. This is the standard
symmetrical-components result. `Z1(h)` carries only the internal and geometric parts, so
`X1(h) = X1·(f/f0)` with a skin-effect rise on `R1` and no earth floor. `Z0(h)`, the
zero-sequence ground-return loop, carries the earth return and the floor.

Two independent checks confirm the split. A 3-phase overhead geometry run through the full
Carson model and decomposed with the Fortescue transform gives `X1(h)/(h·X1(f0)) → 0.9998`
(linear in `h`, no floor) while `X0(h)/(h·X0(f0)) → 0.86` (sub-linear) and `R0/R1 ≈ 5` at
`h = 25`. A running OpenDSS, fed a native 3-phase `R1/X1/R0/X0` line, returns
`Z1(h) = R1 + j·X1·(f/f0)` at every order with the earth correction only in `Z0(h)`; the
same data entered as a 1-phase line carries the earth floor instead.

```{figure} ../../_static/figures/seq_xr_vs_harmonic.svg
:alt: R and X versus harmonic order, positive vs zero sequence
:width: 95%

R and X versus harmonic order on a 3-phase geometry, decomposed into sequences. The
positive-sequence reactance is a straight line through the origin with no floor. The zero
sequence carries the sub-linear Carson earth-return reactance and a strongly rising
resistance.
```

## How different simulation tools model this

Earth return is a zero-sequence, ground-loop quantity. Every tool below keeps it out of the
positive sequence. They differ in how a line is entered and how the reactance is scaled
with frequency.

OpenDSS has three line-impedance paths. A `LineGeometry` (real conductor coordinates) is
recomputed at every frequency with the full Carson/Deri model, earth return and skin effect
included, and a specified geometry overrides every other impedance definition
([Line docs](https://opendss.epri.com/Line.html)); the earth model is selectable
(`earthmodel = Carson | Deri | FullCarson`). This is the path pgml's geometry model
reproduces, described in [Carson line constants](references/opendss/carson.md). An
impedance-defined line (`LineCode` with `R1 X1 R0 X0`, or `Rmatrix`/`Xmatrix`) instead
carries explicit Carson earth-return terms `Rg` and `Xg`, by default
`0.01805 + j·0.155081` Ω per 1000 ft at 60 Hz for 100 Ω·m earth and user-overridable
([LineCode docs](https://opendss.epri.com/LineCode1.html)); skin effect is not applied to
`R` on that path, and at harmonics OpenDSS scales the reactance with frequency and
frequency-corrects `Rg` and `Xg`. Because `Rg` and `Xg` are common-mode quantities they
cancel in `Z1` of a balanced 3-phase line and surface only in `Z0`. They do enter the
series impedance a study sees when the line is modelled as a single conductor with earth
return, which is exactly what the single-conductor synthesis above builds.

pandapower, PowerFactory and PSS®E take sequence impedances (or per-km positive and
zero-sequence R/X) directly and scale the reactance roughly linearly with frequency for
harmonic studies. Earth return is a zero-sequence parameter there and is never added to the
positive sequence. EMTP and ATP keep the same explicit `Z1`/`Z0` split, with the Carson
earth return in `Z0` only. So the positive-sequence harmonic impedance has no earth floor
in any of them. A balanced study excites only `Z1` and never sees the earth return; an
unbalanced study with a ground return path does, which is the case the sequence-aware model
below covers.

## The model

For a line defined by sequence or R/X data, the common case for published feeders such as
IEEE-33 and CIGRE LV, the harmonic impedance is

```
Z1(h) = R1 · m_skin(h)  +  j · X1 · (f / f0)
```

* `X1·(f/f0)` is the geometric reactance at constant inductance `L1`. The explicit R/L/C
  assembly path already produces this through `X(h) = 2π f L`, with no earth term.
* `m_skin(h)` is the skin-effect resistance multiplier, the same Bessel `I0/I1` internal
  resistance growth the Carson geometry path uses, normalised so that `m_skin(f0) = 1` and
  with the earth term dropped. This is the one physical effect a plain "R constant, X ∝ h"
  model misses.

Two details of that multiplier matter on a three-phase line, because the Bessel curve is
fitted through a DC resistance and the argument goes as `1/√Rdc`. The value it is fitted to is
the positive-sequence resistance, recovered from the line's own phase matrix as the mean
diagonal minus the mean off-diagonal, because a matrix expanded from sequence data carries
`R_self = (R0 + 2·R1)/3` on its diagonal. And the multiplier scales the conductor part only:
in Carson's equations the mutual resistance of a multi-phase line IS the earth-return term, so
the stamp splits the matrix as `R(h) = m(h)·(R − R_earth) + R_earth`, with `R_earth` the
off-diagonals and each diagonal entry set to that row's mean mutual. Skin effect is an
internal-conductor phenomenon and has no business scaling the earth path. A single-phase line
has no mutual, so its diagonal IS `R1` and neither detail applies.

Both details move the impedance noticeably. On the first IEEE-33 line a fit to the mean
diagonal returns twice `R1`, which understates `m(h)` by a third at order 25 (1.87 against
2.51), while scaling the whole matrix overstates `R0(h)` by up to a fifth. The two errors act
in opposite directions, which is why neither shows up as an outlier in an aggregate check. The
effect on a harmonic VOLTAGE is small on a reactance-dominated feeder, about a tenth of a
percent at order 13, so the correction matters for the impedance, and therefore for damping,
resonance sharpness and any loss or parameter-recovery study, rather than for the voltage
magnitude of such a case.

Two equivalent constructions are available. `positive_sequence_z` evaluates the formula
directly; it is linear in `h` to floating point, differentiable in `R1` and `X1`, and
batched over lines and harmonics. `two_conductor_loop_z` instead builds a physical `+I` go
conductor and `−I` return conductor, where the Carson `[1,−1]` loop transform cancels the
large earth penetration-depth term analytically. It keeps a physical geometric mean radius,
fixed at `0.7788·radius`, and finds a finite spacing `D = GMR·exp(X1/(2·f0·μ0))` for any
`X1`. The two agree to within the residual earth coupling, below about 2 % out to `h ≈ 25`,
which is what makes the direct model physically grounded rather than an ad-hoc scaling.

The full Carson/Deri model with earth return is reserved for genuinely geometry-defined
lines and for zero-sequence and ground-return paths. On the same conductor geometry it
agrees with OpenDSS to 4.8e-8 relative on `Z` and 2.1e-5 on `C`. Those two residuals are
the physical constants rather than the model, because pgml uses the SI values of `μ0` and
`e0` where OpenDSS truncates them. With the default conductor internal-inductance model
the agreement holds below 1 kHz, where both tools take the spacing term from the published
conductor GMR. OpenDSS moves that term to the physical radius outside 40 Hz to 1 kHz;
`line.geometry.internal_inductance: gmr_power_frequency` reproduces that rule and holds the
same agreement at every frequency.

## Unbalanced and 4-wire studies

Most low-voltage grids are 4-wire (phase plus a usually grounded neutral) and are operated
asymmetrically, with per-phase loads, generators and sources carrying their own spectra.
There the earth and neutral return does matter, but the trigger is zero-sequence current
rather than unbalance as such. Positive- and negative-sequence currents sum to zero across
the phases, produce no ground current, and see `Z1 = Z2` with no earth return. The residual
`I_a + I_b + I_c = 3·I_0` returns through earth or neutral and sees `Z0`. Earth return is
therefore mandatory exactly when an unbalanced study has a grounded or neutral return path.
A 3-wire or delta unbalanced load with no ground path produces no residual current and no
earth return.

The vehicle for that is the full coupled `Z_abc(h)` rather than a per-phase earth floor:

```
Z_self(h)   = (Z0(h) + 2·Z1(h)) / 3
Z_mutual(h) = (Z0(h) −   Z1(h)) / 3
```

Each sequence is frequency-corrected separately before recombining. `Z1(h)` is the
earth-free model above. `Z0(h)` is the conductor part (`X0 ∝ h`, optional skin) plus
`3·(Re(f) − Re(f0))`, where `Re(f) = π²·f·10⁻⁷` Ω/m is Carson's earth-return resistance,
geometry-independent and proportional to frequency. That is the frequency-growing damping
the positive sequence never sees, and it keeps an unbalanced study from over-predicting
zero-sequence harmonics. `Re` is non-negative and monotone, so `Z0(h)` can never become
non-physical the way a single-conductor earth floor can.

Modelling each phase as an independent single conductor with earth return, a diagonal
`Z_abc` whose every diagonal carries the full earth floor, is wrong twice over. It ignores
the inter-phase mutual coupling and it triple-counts the earth term.

Scope of the model. The earth-return resistance, the dominant damping term, is universal and
robust. The earth-return reactance sub-linearity depends on the return path. Deep earth
(`De ≈ 658·√(ρ/f)` m, overhead) and a neutral or sheath a few centimetres away (an LV
cable) behave differently, so pgml does not apply it generically and keeps `X0 ∝ h`. For a
return-path-correct `Z0(h)` reactance, use the geometry path with the actual conductor and
neutral coordinates. The earth-resistance coefficient is exposed (default Carson
`π²·10⁻⁷`) so the damping can be tuned or matched against a reference tool, mirroring the
user-settable `Rg`/`Xg` of OpenDSS.

The sub-linearity is available as an option, `line.earth_return.x0_frequency =
carson_sublinear`, which subtracts the Carson/Deri decay `1.5·μ0·f0·h·ln h` from `X0(h)`.
The soil resistivity cancels in that term, so it needs no extra data. It is not the
default because it can drive `X0(h)` negative at very high orders on a cable whose stored
`X0` is small, a property OpenDSS's own `Xg` correction shares. With the earth parameters
matched on both sides, the lumped `sequence_aware` impedance and OpenDSS's R/X-line
impedance agree to 1e-11 relative at every order up to 25, on all 32 lines of IEEE-33.

## The conductor's internal inductance above power frequency

A published GMR is measured at power frequency. It folds the conductor's internal inductance
into one equivalent radius: for a solid round conductor `GMR = e^(-1/4)·radius`, and the
reactance that adds, `(f·μ0)·ln(radius/GMR) = f·μ0/4`, is exactly `ω·μ0/(8π)`, the internal
reactance at uniform current density. Skin effect confines the current to the surface, so
the internal inductance decays and a fixed GMR over-states the reactance at harmonic
frequencies. `pgml.geometry.internal_reactance_ratio` returns that decay,
`g(f) = Im(Zint)/(f·μ0/4)`: for a 336 kcmil ACSR, `g = 0.97` at 250 Hz, `0.74` at 1 kHz and
`0.49` at 2.5 kHz. For a 1/0 ACSR the same numbers are `1.00`, `0.97` and `0.84`, so the
effect is a property of the conductor and not of the frequency alone.

`line.geometry.internal_inductance` selects how the geometry path handles it.

| value | model | use it for |
|---|---|---|
| `gmr` (default) | published GMR at every frequency | the default, and the only safe choice when a geometry was synthesized from R/X |
| `gmr_skin` | published GMR with its internal reactance scaled by `g(f)` | measured conductor data, harmonics above about 1 kHz |
| `gmr_power_frequency` | `gmr` while `40 Hz < f < 1 kHz`, `bessel` outside | reproducing OpenDSS at every frequency |
| `bessel` | physical radius plus the full `Im(Zint)` | a conductor known to be solid and round |

On a solid round conductor `gmr_skin` and `bessel` are the exact solution, and the default
is 0.24 % (median) below 1 kHz and 1.0 % above. `gmr_power_frequency` reproduces OpenDSS to
4.6e-8 relative from 20 Hz to 3 kHz, at the price of a discontinuity at each band edge. On
the published ACSR 1/0 of the OpenDSS line-constants example that discontinuity is 8.9 % of
`X`, because the rule replaces a measured `GMR/radius = 0.269` with the solid-round `0.7788`
in one step while only 3 % of that conductor's internal inductance has actually decayed.

Do not combine a radius-based model with a geometry produced by `synthesize_grid_geometry`.
The synthesis fits the GMR and leaves the radius at its default, so the radius carries no
information. Assembly warns when it sees that combination.

## Choosing the model

The model is the typed field `Line.harmonic_line_model`, one of `geometry`,
`sequence_aware`, `positive_sequence`, `naive`, or unset, with `Line.harmonic_skin_effect`
and `Line.earth_return` carrying its options. A converted grid already carries the
configured default, because the readers resolve it at conversion time and log which model
they applied. A three-phase R/X line becomes sequence-aware and a one- or two-phase line
positive-sequence, so a converted grid never reaches a harmonic solve as the naive model by
accident.

`apply_default_harmonic_model(grid)` applies the same defaults to a grid you assembled
yourself, and skips any line that already carries an explicit model or a conductor geometry.
Assembly resolves nothing: an unset model means the line is assembled from its stored
parameters as they are, and a line that is still unset when a harmonic assembly runs
produces a warning naming the call that resolves it. Every default value lives in one ordered,
self-describing defaults file inside the installed package, including the `0.7788` ratio of
geometric mean radius to conductor radius, the earth-return coefficient, the default
conductor radius and heights, and the soil resistivity.

What you compare against in OpenDSS decides whether the two agree. The same R/X data
modelled as a 3-phase line and as a 1-phase line gives different harmonic answers in
OpenDSS itself.

| You have or want | How to set it | Harmonic line model | Matches OpenDSS |
|---|---|---|---|
| the documented defaults | applied on import, or `apply_default_harmonic_model(grid)` | 3-phase sequence-aware, 1-phase positive-sequence | 3-phase R/X with `Rg`/`Xg` (earth in `Z0`) |
| R/X feeder, raw `X ∝ h`, no skin or earth | nothing to set | `Z1(h) = R1 + j·X1·(f/f0)` | native 3-phase `R1/X1` LineCode (`Z1`) |
| R/X feeder plus physical skin on R | `apply_positive_sequence_harmonic_model(grid)` | `Z1(h) = R1·m_skin(h) + j·X1·(f/f0)` | 3-phase R/X plus a skin rise OpenDSS applies only to geometry lines |
| unbalanced 4-wire R/X feeder (`Z1` and `Z0`) | `apply_sequence_aware_harmonic_model(grid)` | `Z_abc(h)` with earth-free `Z1` and earth-damped `Z0` | native 3-phase R/X with `Rg`/`Xg`; reactance sub-linearity only via geometry |
| real 3-phase conductor coordinates | set `Line.conductor_geometry` | full Carson (earth in `Z0`, skin on R) | `LineGeometry`, 4.8e-8 relative on `Z` below 1 kHz |
| single-conductor or SWER check | `synthesize_grid_geometry(grid)` | single conductor plus earth floor | 1-phase `LineGeometry` line |

For an unbalanced 4-wire feeder, give the lines a full 3×3 `Z_abc(f0)` so that `Z0` is
defined by the off-diagonal mutuals, then apply the sequence-aware model. The
single-conductor synthesis is there to validate the Carson code rather than to produce
representative harmonic magnitudes, because it overstates the series reactance for balanced
operation.

## Comparison to OpenDSS

The shipped example `run/examples/pgml/evaluate_line_sequence_harmonics.py` overlays four
line models on IEEE-33: the positive-sequence model, the naive model, the single-conductor
Carson model, and a live OpenDSS profile computed from the same geometry.

```{figure} ../../_static/figures/feeder_h13.svg
:alt: IEEE-33 h=13 voltage profile across line models
:width: 95%

IEEE-33 voltage profile at the 13th harmonic under four line models. The single-conductor
earth correction, which equals an OpenDSS 1-phase `LineGeometry`, separates visibly from
the other three. The positive-sequence model removes that earth-floor artifact and keeps
the smaller skin-effect resistance rise.
```

So does the simplified R/X approach differ significantly from OpenDSS? Only against OpenDSS
modelling the lines as 1-phase single conductors with earth return, the comparison shown
above, where the feeder-end voltage at `h = 13` is 0.00759 pu against 0.00894 pu. The
single-conductor profile and the live OpenDSS profile of the same geometry agree to the
printed digits. Against native OpenDSS modelling the same R/X as a 3-phase line, the
standard way to enter a balanced feeder, the pgml default agrees and reproduces
`Z1(h) = R1 + j·X1·(f/f0)`. The divergence is a property of the reference setup rather than
of the pgml model.
