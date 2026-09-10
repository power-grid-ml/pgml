# OpenDSS harmonic conventions

The conventions pgml's harmonic power flow reproduces, and the numbers they were checked
against. The spectrum phase convention was derived by running OpenDSS on a single-phase test
circuit and reading the injected currents and per-order bus voltages off monitors, rather than
from documentation.

## How the harmonic solve proceeds

1. A fundamental power flow is solved first, and every power-conversion element's fundamental
   current phasor `I1` is recorded, magnitude and angle.
2. For each spectrum order, every element carrying a spectrum becomes a harmonic current
   source. The network is linear per order, so `Y(h)·V(h) = I(h)` is solved once per order.
3. A voltage source is held at zero harmonic voltage, a short behind its Thévenin impedance,
   so it contributes only its Norton shunt unless it carries a spectrum of its own.
4. Network impedances scale with frequency, `X(h) = h·X1` and `B(h) = h·B1` with `R` fixed
   unless `XRConst=yes`. pgml's assembly matches this through `X = 2πfL` and `B = 2πfC` at
   `f = h·f₀`.

## Spectrum to harmonic current

For a device with fundamental current `I1 = |I1|∠a1` and a spectrum whose order `h` entry is
`(mag_h, ang_h)`, with `(mag_1, ang_1)` the declared fundamental entry:

```text
|I_h|    = (mag_h / mag_1) * |I1|
arg(I_h) = ang_h + h * (a1 - ang_1)        [degrees]
```

The spectrum is defined relative to its own declared fundamental angle. It is rotated so that
its fundamental aligns with the actual `a1`, and that base rotation is multiplied by `h` for
order `h`, which is what a fixed time shift of the waveform does. The magnitude uses the
fundamental current, not the power.

Measured on a 2 kW, 0.5 kvar load at 230 V with a spectrum at orders 1, 5 and 7, magnitudes
100, 20 and 14 %, angles 10, 30 and 55°, and `NeglectLoadY=yes`. The fundamental current came
out as `I1 = 9.23355 ∠ −15.0374°`.

| h | predicted `(mag_h/100)·|I1|` | OpenDSS magnitude | predicted angle | OpenDSS angle |
|---|---|---|---|---|
| 1 | 9.23355 | 9.23355 | −15.037 | −15.037 |
| 5 | 1.84671 | 1.84671 | −95.185 | −95.187 |
| 7 | 1.29270 | 1.29270 | −120.259 | −120.262 |

## The load shunt

`NeglectLoadY=yes` makes a load a pure current source, which is the model pgml uses by
default. With the OpenDSS default the spectrum current source sits in parallel with a shunt
admittance derived from the fundamental power and voltage, split between a series and a
parallel R-L branch by `%SeriesRL`, with a motor branch parameterised separately. pgml's
`HarmonicShuntModel` carries the same parameters, and requesting the shunt currently raises
because the split is not pinned against OpenDSS yet.

## A single-phase oracle

A source with `basekv=0.23` and `R1=X1=0.1 Ω` feeds a line with `R=X=0.5 Ω` and no
capacitance, which feeds a 2 kW, 0.5 kvar load with a spectrum at orders 1, 5 and 7 at 100, 20
and 14 % and zero angles, at 50 Hz. The fundamental current is `I1 = 9.23355 ∠ −15.0374°`.

| h | `NeglectLoadY=yes` | with the load shunt |
|---|---|---|
| 1 | 223.24690 ∠ −1.004° | 223.24690 ∠ −1.004° |
| 5 | 5.51315 ∠ −178.200° | 5.24266 ∠ 177.953° |
| 7 | 5.30916 ∠ 154.807° | 5.01599 ∠ 149.937° |

## Line impedance at harmonics

OpenDSS recomputes line impedance at every frequency with an earth-return and skin-effect
model, and it does so even for a sequence-defined line. At the fifth harmonic a 0.5 + j0.5 Ω
line comes back as 0.572 + j2.409 Ω rather than 0.5 + j2.5 Ω. The resistance rises and the
reactance scales sub-linearly.

Matching that needs the conductor-geometry path, where pgml's Carson/Deri model is bit-exact
against OpenDSS on the same geometry, see [Carson line constants](carson.md). A line given as
R and X is scaled analytically instead, and {doc}`../../harmonic-line-model` explains the
available models and which OpenDSS setup each one corresponds to. A further small contributor
to a voltage comparison is that `NeglectLoadY=yes` still leaves the constant-power load's
linearised admittance in OpenDSS's harmonic admittance matrix, which pgml omits.

The injection convention above is exact and is what the harmonic path validates against. A
voltage comparison is only as good as the line model both sides use.
