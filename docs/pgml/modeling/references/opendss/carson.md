# Carson line constants

The Carson/Deri line model OpenDSS uses, extracted from its line-constants source and checked
bit-exact against a running OpenDSS, relative impedance error around 1e-13, for single-phase
and three-phase-plus-neutral geometry lines from 50 to 750 Hz. This is the model pgml's
geometry path implements.

OpenDSS recomputes line impedance at every frequency with an earth-return and skin-effect
model, for geometry-defined and R/X-defined lines alike. A plain "R constant, X proportional
to h" is therefore not what it does. A 0.5 + j0.5 Ω/km line comes back as 0.557 + j2.015 at
250 Hz rather than 0.5 + j2.5. The default earth model is Deri.

## Series impedance per unit length (Ω/m), DERI model
Constants: `mu0 = 12.56637e-7`, `Fw = 2π f`, `Lfactor = j·Fw·mu0/(2π)`.
Complex penetration: `Fme = sqrt(j·Fw·mu0/ρ)` (ρ = earth resistivity, default 100 Ω·m).

- Internal (skin effect, uses **Rdc**, round conductor):
  `α = (1+j)·sqrt(f·mu0/Rdc)`, `I0I1 = I0(α)/I1(α)` (=1 if |α|>35),
  `Zint = (1+j)·I0I1·sqrt(Rdc·f·mu0)/2`.
  In the 40–1000 Hz band OpenDSS **zeroes the internal reactance** (`Zi.im=0`);
  internal inductance is carried by GMR. So only `Re(Zint)` is used.
- Self: `Z[i,i] = Re(Zint_i) + Lfactor·ln(1/GMR_i) + Ze(i,i)`.
- Mutual: `Z[i,j] = Lfactor·ln(1/Dij) + Ze(i,j)`, `Dij = |r_i − r_j|`.
- Earth return (Deri, method of images via complex depth):
  `hterm = (y_i + y_j) + 2/Fme`, `xterm = x_i − x_j`,
  `Ze = Lfactor·ln( sqrt(hterm² + xterm²) )`.
- Neutrals/shield wires eliminated by **Kron reduction** of Z (reduce=y).

## Capacitance (Maxwell potential coefficients)
`P[i,i] = ln(2·y_i / r_i)`, `P[i,j] = ln(Dij'/Dij)` with image distance
`Dij' = sqrt((x_i−x_j)² + (y_i+y_j)²)`; `C = (2π·e0)·inv(P)` then Kron-reduce P
before inverting for grounded neutrals. OpenDSS uses a slightly different effective capacitance radius, so pgml's
capacitance is physically correct but not bit-exact against it. The benchmark feeders used
here carry no line capacitance, so it does not affect their results. The series impedance,
which is what drives the harmonic behaviour, is bit-exact.

## Bessel I0/I1 for complex argument (torch port)
Use the continued fraction `I1/I0 = 1/(2/z + 1/(4/z + 1/(6/z + …)))` (evaluate
bottom-up), differentiable and overflow-free for all z; `I0I1 = 1/(I1/I0)`,
clamped to 1 for |z|>35 (matches OpenDSS).
