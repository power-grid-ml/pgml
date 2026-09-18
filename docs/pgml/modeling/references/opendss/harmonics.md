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
   so it contributes only its Norton shunt unless it carries a spectrum of its own. pgml
   expresses the same thing with a `NodeHarmonicSource` at the source's node rather than a
   spectrum on the source element.
4. Network impedances scale with frequency, `X(h) = h·X1` and `B(h) = h·B1` with `R` fixed
   unless `XRConst=yes`. pgml's assembly matches this through `X = 2πfL` and `B = 2πfC` at
   `f = h·f₀`. A transformer's `XRConst` is carried per element in `harmonic_xr_constant`,
   which the OpenDSS reader writes and the harmonic assembly consumes, so such a unit gets
   `R ∝ h`; `transformer.harmonic_resistance.law` sets the policy globally.

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

## The device Norton shunt (`NeglectLoadY`, `%SeriesRL`)

At orders `h > 1` an OpenDSS `Load` is a harmonic current source in parallel with a shunt
admittance derived from its fundamental operating point. `Load.pas`
(`TLoadObj.CalcYPrimMatrix`) splits that admittance between a series and a parallel R-L branch,
with `s = %SeriesRL/100`:

```text
Y_eq     = conj(P_ph + jQ_ph) / V_ph**2
Y_par(h) = (1 - s)*Re(Y_eq) + j*(1 - s)*Im(Y_eq)/h
Z_ser    = 1/(s*Y_eq),   Z_ser(h) = Re(Z_ser) + j*h*Im(Z_ser)
Y(h)     = Y_par(h) + 1/Z_ser(h)
```

`P_ph` and `Q_ph` are the specified power divided by the phase count, and `V_ph` the rated
voltage: `kV*1000` for a one-phase or delta load, `kV*1000/sqrt(3)` for a wye load of two or
three phases. The split is exact at the fundamental, `Y_par(1) + Y_ser(1) = Y_eq` for any `s`,
and only sets how the shunt rolls off with frequency. The parallel branch keeps its full
conductance at every order while the series branch's admittance falls roughly as `1/h`, so
`%SeriesRL=0` damps most and `%SeriesRL=100` least. `puXharm > 0` replaces the derived series
impedance by a fixed blocked-rotor reactance `X = kV**2*1000/(kVA*s)*puXharm` with
`Z_ser = X/XRharm + jX`.

`Set NeglectLoadY=Yes` replaces the whole shunt by `EPSILON = 1e-12 S`, a pure current source.
The measured `YPrim` under that option is `1.0e-12 + 0j`, or `2.0e-12` on a delta diagonal.
There is no residual load admittance.

pgml implements all three models: `load_shunt="opendss"`, the default, reproducing OpenDSS's
`NeglectLoadY=No` with `%SeriesRL=50`; `"motor"`; and `"none"`, which is `NeglectLoadY=Yes`.
The per-element admittance agrees with a live OpenDSS `Load`'s own `YPrim` to 4.7e-16 relative
for a one-phase wye, a three-phase wye and a three-phase delta load, at `%SeriesRL` 0, 50 and
100 and with the motor branch. Two conventions differ from a naive reading and are worth
repeating: the voltage is the rated one, and OpenDSS divides the susceptance of the parallel
branch by `h` and multiplies the series reactance by `h` whatever the sign of `Q`, because it
models both branches as R-L. For a leading load (`Q < 0`) that is a negative inductance. pgml's
shipped `appliance.harmonic_shunt.reactive_element: sign_aware` instead scales a leading
load as the capacitance it is, `h·B` in the parallel branch and `X/h` in the series branch,
which is also what the const-Z fold of the fundamental assembly does. The two laws are
identical for `Q ≥ 0`. For a 10 kW, −5 kvar load at 400 V and order 5 a live OpenDSS `YPrim`
reads 0.0366 + j0.0166 S, which `reactive_element: inductive` (selected by the `opendss`
preset) reproduces to rounding, while `sign_aware` gives 0.0699 + j0.0820 S. pgml derives `P`
and `Q` from the power the device actually draws at the converged fundamental solution rather
than from the specified power, which is identical for a constant-power device and 6e-5 to
1e-4 pu of nominal apart for a const-Z, const-I or ZIP one.

`HarmonicShuntModel` is the per-device override of the shunt; the default lives in
`appliance.harmonic_shunt.*`. A generation-sign device carries no shunt under the shipped
`appliance.harmonic_shunt.generation_model`, because the load expression would give it a
negative conductance; see [DER models](../../der-pv-storage.md).

## A single-phase oracle

A source with `basekv=0.23` and `R1=X1=0.1 Ω` feeds a line with `R=X=0.5 Ω` and no
capacitance, which feeds a 2 kW, 0.5 kvar load with a spectrum at orders 1, 5 and 7 at 100, 20
and 14 % and zero angles, at 50 Hz. The fundamental current is `I1 = 9.23355 ∠ −15.0374°`.

| h | `NeglectLoadY=yes` | with the load shunt |
|---|---|---|
| 1 | 223.24690 ∠ −1.004° | 223.24690 ∠ −1.004° |
| 5 | 5.51315 ∠ −178.200° | 5.24266 ∠ 177.953° |
| 7 | 5.30916 ∠ 154.807° | 5.01599 ∠ 149.937° |

Both columns are covered by `tests/reference/test_opendss_load_shunt.py`, which drives a live
OpenDSS engine at `%SeriesRL` 0, 50 and 100 and with the motor branch.

## Line impedance at harmonics

OpenDSS recomputes line impedance at every frequency with an earth-return and skin-effect
model, and it does so even for a sequence-defined line. At the fifth harmonic a 0.5 + j0.5 Ω
line comes back as 0.572 + j2.409 Ω rather than 0.5 + j2.5 Ω. The resistance rises and the
reactance scales sub-linearly.

Matching that needs the conductor-geometry path, where pgml's Carson/Deri model agrees with
OpenDSS to 4.8e-8 relative on the same geometry, see
[Carson line constants](carson.md). A line given as R and X is scaled analytically instead,
and {doc}`../../harmonic-line-model` explains the available models and which OpenDSS setup
each one corresponds to.

The injection convention above is exact and is what the harmonic path validates against. A
voltage comparison is only as good as the line model both sides use.
