# Harmonic emission of a device: load dependence and phase structure

How pgml turns a device's operating point into the harmonic currents it injects, and
which parts of that are calibrated against measurements. Read this before changing the
scenario recipes (`pgml.scenarios.presets`) or the device library
(`pgml.scenarios.config.default_device_classes`): both draw the SAME emission law from
`pgml.scenarios.emission`, so the Task-A snapshots and the Task-B composed sequences
carry one physics.

## 1. The proportional law and what is wrong with it

Every generating path in pgml expresses a device's harmonic current as a RATIO to its own
fundamental current, `I_h = r_h · I_1`, with a per-order ratio `r_h` (the `h_mag` draw of
a `ParameterSpec`, the `harmonic_magnitude` band of a device class, or a stored
`StaticSpectrum`). The solver then scales the ratio by the fundamental current the device
actually draws at the converged voltage (`pgml.solver.harmonic_flow`).

Measured devices do not behave like that. Across certified PV inverter test reports and
laboratory device racks the ratio at 10 % loading is **6–10× its value at rating**, with a
coefficient of variation over one device's own loading sweep of 0.8–1.0 against a directly
measured repeatability floor of 0.01–0.05. The mechanism is a load-independent emission
floor: while the fundamental grows 10×, the absolute harmonic current grows only
1.2–1.9×. The law that fits, and that transfers to held-out devices, is **complex
affine**:

```
I_h(λ) = A_h + B_h · λ          λ = |I_1| / |I_1 at rating|
```

`A_h` is the floor (present whenever the device is on), `B_h` the load-proportional part.
Two more measured effects follow from the same two complex numbers:

- the emission ANGLE rotates with loading — the phasor turns from `arg A_h` at low load
  toward `arg B_h` at rating (across-set-point spread 18–105° against 2.6–7.7°
  repeatability), which decides whether two devices at different operating points add or
  cancel;
- `|I_h|(λ)` is non-monotone: `A_h` and `B_h` are near anti-phase, so `|A_h + B_h λ|` has a
  cancellation null at `λ* = |A_h| / |B_h|`, inside the measured range for about half of
  the sweeps.

The consequence that matters for state estimation is the elasticity `d ln|I_h| / d ln λ`:
the proportional law has 1.0 everywhere, the measured devices 0.04–0.08 at 10 % load and
0.5–0.8 at rating. A harmonic channel is 12–23× less informative about loading at low load
than a proportional draw pretends, and a learner trained on proportional data has nothing
to learn about how a harmonic follows the fundamental.

## 2. The law as a correction: `pgml.scenarios.emission`

The generators keep drawing the RATED ratio `r_h` and multiply the proportional phasor by
the complex correction

```
c(λ) = z(λ) / (λ · z(1)),      z(λ) = f · e^{jδ} + (1 − f) · λ
```

(`affine_emission_correction`), where `f = |A_h| / (|A_h| + |B_h|)` is the floor share and
`δ = arg A_h − arg B_h` the floor angle. Properties the tests pin:

- `c(1) = 1` for any `f`, `δ` — the rated operating point never moves, so the calibrated
  emission envelopes (IEC 61000-3-2 fractions, the class bands) keep their meaning;
- `f = 0` gives exactly `1 + 0j` — the proportional law is recovered bit-for-bit;
- `|c(λ)| = f / λ + (1 − f)` at `δ = 0`; `f = 0.61` reproduces the measured
  `ratio(λ) = ratio_rated · (0.39 + 0.61 / λ)`;
- `arg c(λ)` is the affine law's own rotation with loading.

An explicit slope `s_h` [deg per unit loading] adds `s_h · (λ − 1)` to the angle
(`phase_slope_shift`) — zero at rating — for the rotation measured devices show beyond the
affine geometry.

Below `LOADING_FLOOR` (5 % of rating) the correction is evaluated at the floor. The
correction is a ratio to the fundamental, which the solver multiplies by the actual
`|I_1| ∝ λ`; the ratio grows as `1/λ` and the fundamental shrinks as `λ`, so the harmonic
current stays finite, but at `λ = 0` the product would be `0 · ∞`. Under the floor the
harmonic current therefore falls linearly with the fundamental to zero — a device that
draws nothing emits nothing — rather than holding the floor current of an idle device.

## 3. Where each recipe applies it

| path | loading `λ` | parameters | draw |
|---|---|---|---|
| **randomized snapshots** (`se_random_scenario_config`, Task A) | the device's OWN `load_scale` / `pv_scale` draw: drawn active power over the nameplate (mean of the per-phase ratios under a per-phase draw); `1` for a device no power spec varies | `ParameterSpec` fields `h_floor` (share, `[0, 1]`), `h_floor_phase` [deg], `h_slope` [deg per unit] | per device and order, like `h_mag` / `h_phase`; applied in `pgml.scenarios.sampler` after every spec has written |
| **composed sequences** (`se_coherent_scenario_config`, Task B) | each roster member's per-step loading (AR(1) around its class mean, clamped at `loading_min`) | `DeviceClassSpec.emission_floor`, `emission_floor_phase_deg`, `phase_slope_deg` | per member and order at roster build; applied in `pgml.scenarios.composition` per step |

The calibrated recipe (`SE_PRESET_VERSION = "3"`) draws the same ranges in both:
floor share `U(0.43, 0.73)`, floor angle `U(100°, 150°)`, slope `U(−25°, +25°)` per unit
loading — for loads and PV inverters alike. Passing `(0, 0)` for a range to
`se_random_scenario_config` emits no spec for it (it would be inert, and not consuming a
sampling dimension keeps every other draw bit-identical), so all three at `(0, 0)` is the
proportional recipe of preset version 2 exactly.

What a dataset records (`samples`, persisted beside the state): the raw draws under each
spec's name, the REALIZED post-reference, post-law magnitude and phase (`<spec>_mag`,
`<spec>_phase`) and the loading the law read (`<spec>_loading`), so the magnitude-to-loading
relation of a dataset can be audited without re-running the recipe.

## 4. Phase structure across a device's terminals

A three-phase device's harmonic on phase `b` is its phase-`a` waveform delayed by a third
of a cycle, so order `h` is rotated by `−h · 120°`: **h3 (and every triplen) is
zero-sequence** — the three phases in phase, which is what the Dyn transformer traps —
**h5 is negative-sequence, h7 positive-sequence**, and so on. pgml does not draw this per
phase; it falls out of how the solver forms the per-element angle
(`pgml.solver.harmonic_flow._harmonic_injections`):

```
|I_h^e| = (mag_h / mag_1) · |I_1^e|
arg I_h^e = ang_h + h · (arg I_1^e − ang_1)
```

per terminal element `e`, with `I_1^e` the fundamental current that element actually
carries. The drawn `(mag_h, ang_h)` of a device broadcasts to all its elements; the
`h · arg I_1^e` term supplies the sequence rotation from the fundamental's own `0 / −120 /
+120°`, and a per-phase unbalance of the fundamental (the recipe's `small_imbalance` draw)
propagates into the harmonic magnitudes of the individual phases. The two laws above are
applied to the device-level `(mag_h, ang_h)`, so the three phases of one device share the
floor, the floor angle and the slope and differ only by the fundamental each phase carries
— the cross-phase relationship a learner can exploit: one device, one law, three
sequence-consistent terminals. A per-phase spectrum (`spectrum_per_phase`, or a per-element
`harmonic_injection` list) overrides this for a genuinely asymmetric device.

## 5. Provenance of the numbers

Floor share, floor angle and slope ranges come from the device-nonlinearity study on 52
certified PV inverter workbooks (569 operating points across 53 devices after the loader
fix) and 18 lab-rack sweeps; the floor survives a headroom stratification against the
analysers' own reporting floor (the ratio inflation is 6.8× even ten times above it), and
the phase rotation is device-side (the background voltage angle moves only 2.6–9.1° across
the same set-points). The aggregate-level calibration of the composed path — the shipped
range gives the best match of the measured spline gain (+15.9 % against +14.5 % measured)
— is documented in the composition module. Neither the floor nor the phase rotation is a
property of the bench's supply: the upstream background is a separate, additive source
(`BackgroundHarmonicConfig`, see [error injection](error-injection.md)).
