# Carson line constants

The Carson/Deri line model OpenDSS uses, extracted from its line-constants source and
checked against a running OpenDSS on single-phase and three-phase-plus-neutral geometry
lines. This is the model pgml's geometry path implements. With the default conductor
internal-inductance model the two agree to 4.8e-8 relative on `Z` below 1 kHz; with
`line.geometry.internal_inductance: gmr_power_frequency`, which reproduces the 1 kHz rule
below, to 4.6e-8 relative from 20 Hz to 3 kHz.

OpenDSS recomputes line impedance at every frequency with an earth-return and skin-effect
model, for geometry-defined and R/X-defined lines alike. A plain "R constant, X proportional
to h" is therefore not what it does. A 0.5 + j0.5 Ω/km line comes back as 0.557 + j2.015 at
250 Hz rather than 0.5 + j2.5. The default earth model is Deri.

## Series impedance per unit length (Ω/m), DERI model
Constants: `mu0 = 12.56637e-7`, `Fw = 2π f`, `Lfactor = j·Fw·mu0/(2π)`.
Complex penetration: `Fme = sqrt(j·Fw·mu0/ρ)` (ρ = earth resistivity, default 100 Ω·m).

- Internal (skin effect, uses **Rdc**, round conductor):
  `α = (1+j)·sqrt(f·mu0/Rdc)`, `I0I1 = I0(α)/I1(α)` (=1 if |α|>35),
  `Zint = (1+j)·I0I1·sqrt(Rdc·f·mu0)/2`. This is the textbook internal impedance of a
  solid round conductor, `(k·ρ_c/(2π·a))·I0(k·a)/I1(k·a)`, rewritten in terms of `Rdc`;
  pgml reproduces it to 4.6e-16 relative against scipy.
- Self impedance, and the 1 kHz rule. While `40 Hz < f < 1 kHz` OpenDSS forces
  `Im(Zint) = 0` and takes the spacing term from the published GMR,
  `Z[i,i] = Re(Zint_i) + Lfactor·ln(1/GMR_i) + Ze(i,i)`; outside that band it keeps the
  full `Zint` and takes the spacing term from the physical radius,
  `Z[i,i] = Zint_i + Lfactor·ln(1/radius_i) + Ze(i,i)` (`LineConstants.pas`,
  `TLineConstants.Calc`; the same block is in `CNLineConstants.pas` and
  `TSLineConstants.pas`). The bounds are exclusive, so 40 Hz and 1000 Hz themselves are on
  the radius branch. Mutual terms and the capacitance never switch.
- The two branches are the same expression at low frequency. A published GMR is a
  power-frequency quantity that already carries the conductor's internal inductance: for a
  solid round conductor `GMR = e^(-1/4)·radius` and
  `(f·mu0)·ln(radius/GMR) = f·mu0/4 = ω·mu0/(8π)`, which is the `f → 0` limit of
  `Im(Zint)`. OpenDSS's own SIMPLECARSON and FULLCARSON earth models use that constant
  `ω·mu0/(8π)` at every frequency.
- pgml implements both branches and two more, selected by
  `line.geometry.internal_inductance`; see
  [Harmonic line model](../../harmonic-line-model.md).
- Mutual: `Z[i,j] = Lfactor·ln(1/Dij) + Ze(i,j)`, `Dij = |r_i − r_j|`.
- Earth return (Deri, method of images via complex depth):
  `hterm = (y_i + y_j) + 2/Fme`, `xterm = x_i − x_j`,
  `Ze = Lfactor·ln( sqrt(hterm² + xterm²) )`.
- Neutrals/shield wires eliminated by **Kron reduction** of Z (reduce=y).

## Capacitance (Maxwell potential coefficients)
`P[i,i] = ln(2·y_i / r_i)`, `P[i,j] = ln(Dij'/Dij)` with image distance
`Dij' = sqrt((x_i−x_j)² + (y_i+y_j)²)`; `C = (2π·e0)·inv(P)` then Kron-reduce P
before inverting for grounded neutrals. pgml's capacitance agrees with OpenDSS to 2.1e-5
relative, which is the ratio of the two `e0` constants: OpenDSS truncates it to
`8.854e-12` where pgml uses the SI value. The benchmark feeders used here carry no line
capacitance, so it does not affect their results. The series impedance, which is what
drives the harmonic behaviour, agrees to 4.8e-8.

## Bessel I0/I1 for complex argument (torch port)
Use the continued fraction `I1/I0 = 1/(2/z + 1/(4/z + 1/(6/z + …)))` (evaluate
bottom-up), differentiable and overflow-free for all z; `I0I1 = 1/(I1/I0)`,
clamped to 1 for |z|>35 (matches OpenDSS).
