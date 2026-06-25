# OpenDSS harmonic model — exact conventions (empirically verified)

The OpenDSS harmonic conventions that pgml's differentiable harmonic power flow reproduces and
validates against.
The spectrum phase convention below was derived EMPIRICALLY by running OpenDSS
(opendssdirect 0.9.4) on a 1-phase test circuit and reading the injected currents
and per-order bus voltages off monitors — not from docs, so it is trustworthy.

## How harmonic flow works (OpenDSS `Solve mode=harmonics`)
1. A base (fundamental) power flow is solved first. Each power-conversion element's
   FUNDAMENTAL current phasor `I1` is recorded (magnitude + angle).
2. For each spectrum order `h`, every element with a `Spectrum` becomes a HARMONIC
   CURRENT SOURCE injecting `I_h` (below). The network is LINEAR per harmonic:
   `Y(h) · V(h) = I(h)`, solved once per order.
3. `Vsource` is held at ZERO harmonic voltage (a short behind its Thévenin Z), i.e.
   it contributes only its Norton shunt `Y_s(h)` — unless it has its own spectrum.
4. Network impedances scale with frequency: `X(h) = h·X1`, `R` fixed (unless
   `XRConst=yes`), `B(h) = h·B1`. This matches our `assemble_network_ybus`
   (`X=2πfL`, `B=2πfC` at `f=h·f0`).

## Spectrum -> harmonic current injection (THE key convention)
For a device with fundamental current `I1 = |I1|∠a1` and a Spectrum whose order `h`
entry is `(mag_h, ang_h)` (mag in %, the fundamental entry is `(mag_1, ang_1)`):

    |I_h| = (mag_h / mag_1) · |I1|
    arg(I_h) = ang_h + h · (a1 − ang_1)              [degrees]

i.e. the spectrum is defined relative to its OWN declared fundamental angle `ang_1`;
it is rotated so its fundamental aligns with the actual `a1`, and that base rotation
`(a1 − ang_1)` is multiplied by `h` for order `h` (a fixed time-shift of the
waveform = `h·Δ` phase at harmonic `h`). Equivalently
`arg(I_h) = ang_h − h·ang_1 + h·a1`.

**Empirical check** (load 2 kW + 0.5 kvar at 230 V, spectrum
`h=[1,5,7] %mag=[100,20,14] angle=[10,30,55]`, `NeglectLoadY=yes`):
`I1 = 9.23355 ∠ −15.0374°`. Predicted vs OpenDSS injected current:
| h | pred \|I_h\| = (mag_h/100)·\|I1\| | OpenDSS \|I_h\| | pred ∠ = ang_h+h·(a1−10) | OpenDSS ∠ |
|---|---|---|---|---|
| 1 | 9.23355 | 9.23355 | −15.037 | −15.037 |
| 5 | 1.84671 | 1.84671 | −95.185 | −95.187 |
| 7 | 1.29270 | 1.29270 | −120.259 | −120.262 |
Exact to 3 decimals. `|I_h|` uses the fundamental CURRENT magnitude (not power).

## Load Norton shunt at harmonics (`NeglectLoadY`)
- `NeglectLoadY=yes` -> load is a PURE current source (no shunt). This is the clean
  first validation target.
- Default (`no`) -> the spectrum current source is in PARALLEL with a shunt
  admittance `Y_load(h)` derived from the fundamental P,Q and voltage, split between
  a SERIES R-L and a PARALLEL R-L branch by `%SeriesRL` (our
  `HarmonicShuntModel.series_rl_fraction`); a motor branch uses `puXharm`/`XRharm`.
  EXACT split formula is NOT yet pinned here — derive it (empirically from the
  `with_loadY` oracle below, or from the OpenDSS source) when implementing the shunt
  refinement. Our `HarmonicShuntModel` carries the needed params.

## Reference oracle (single-phase, `f0=50 Hz`)
Circuit: `Vsource` (basekv=0.23, Z1: R1=0.1, X1=0.1 Ω) → `Line.l1` (R=0.5, X=0.5 Ω,
C=0, length 1 m) → `Load.ld1` (kv=0.23, 2 kW, 0.5 kvar, model=1) with
`Spectrum h=[1,5,7] %mag=[100,20,14] angle=[0,0,0]`. `I1 = 9.23355 ∠ −15.0374°`.

Per-order load-bus voltage `V_ld(h)` (re/im → mag∠deg):
| h | NeglectLoadY=yes | with load shunt (default) |
|---|---|---|
| 1 | 223.24690 ∠ −1.004° | 223.24690 ∠ −1.004° |
| 5 | 5.51315 ∠ −178.200° | 5.24266 ∠ 177.953° |
| 7 | 5.30916 ∠ 154.807° | 5.01599 ∠ 149.937° |

The eventual test should build this SAME circuit in our schema (1-phase node, Line,
Source, Load with a `StaticSpectrum`), run our harmonic flow, and compare to OpenDSS
RUN LIVE (like the pandapower/pgm oracle tests) — start with the `NeglectLoadY=yes`
(pure current-source) case, then add the shunt.

## KNOWN discrepancy: OpenDSS harmonic LINE impedance (Carson earth-return)
Verified empirically (forced `BuildYMatrix` at 250 Hz, read `SystemY`): OpenDSS's
line SERIES impedance at the 5th harmonic is `Z_line(250) = 0.572 + j2.409 Ω`, NOT
the simple `R + j·h·X = 0.5 + j2.5 Ω` — R RISES (0.5→0.572) and X scales
SUB-linearly (2.5→2.409). This is OpenDSS's frequency-dependent earth-return
(Carson) line model, applied even to sequence-defined (R1/X1) lines.

Our model (and `assemble_network_ybus`) uses the STANDARD simple harmonic line
scaling: `X(h)=2π·h·f0·L` (∝ h), `R` constant. This is a valid, common harmonic
model, but it does NOT match OpenDSS's Carson-corrected impedance — the per-order
bus voltages differ by ~2.5% (h=5) growing with `h`. EXACT OpenDSS parity needs the
geometry/Carson path (POSTPONED per the roadmap). So:
- The harmonic INJECTION convention (above) is OpenDSS-exact and is what we validate.
- The harmonic VOLTAGE oracle match is BALLPARK-only until Carson lands; the rigorous
  correctness test is an independent numpy reimplementation of the simple model.
Additionally, `NeglectLoadY=yes` still leaves a residual load Norton admittance in
OpenDSS's harmonic Y (the const-power load's linearised `Y=conj(S)/|V|²`), a second
small contributor to the voltage gap (our `include_load_shunt=False` omits it).

## Schema gaps
None blocking for the current-source model: `StaticSpectrum`/`HarmonicComponent`
(order, magnitude_pu = fraction of fundamental, phase_deg) carry the spectrum, and
`HarmonicShuntModel` carries the shunt params. NOTE for the differentiable path:
`HarmonicComponent` fields are plain floats (tensor-duality deferred); the
harmonic-flow API should accept a SCENARIO-OVERRIDABLE per-device harmonic-injection
argument (tensor-friendly), analogous to `operating_point` for P/Q, so scenarios can
vary harmonic injections differentiably without editing the stored Spectrum.
