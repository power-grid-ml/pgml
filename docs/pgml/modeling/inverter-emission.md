# Inverter harmonic emission

How {mod}`pgml.inverter` models the harmonic, interharmonic and supraharmonic behaviour of a
grid-following inverter, and how {func}`pgml.studies.emission_flow` spreads it through a grid.
The classic harmonic model of a converter, a fixed current spectrum scaled to the fundamental
current, is what {func}`pgml.solve_harmonic_flow` uses (see {doc}`der-pv-storage`); this page
describes the physical model that replaces it for the devices it is attached to.

## Scope at a glance

The detailed model is a frequency-domain approximation of a grid-following inverter, not
a dynamic simulation of its electronics. It keeps the effects that set how much emission a
unit produces at each frequency and how the unit loads the network there, linearised at the
solved operating point. What it covers, and how closely each part was found to agree with
time-domain references (details under {ref}`inverter-validation`):

| Effect | Modelled as | Agreement with references |
|---|---|---|
| Passive LCL and EMC filter | exact circuit, all frequencies | to rounding (circuit tools) |
| Closed-loop current control (PR or dq PI, delay, feedforward, capacitor-current damping) | continuous transfer functions, exact per frequency | 1e-4 (continuous); a digital controller adds 0.6 to 20 % |
| PLL (synchronous reference frame) | linearised, direct term solved, mirror term reported | 5e-4 three-phase; single-phase SOGI not modelled (+31 % near 175 Hz) |
| PWM carrier groups | naturally sampled sine-triangle, three topologies | 1e-11 analytic, 1 % in a full-unit simulation; space-vector modulation not covered |
| Dead time | averaged blanking error with the switching-ripple dead zone | 0.2 to 14 % imposed current; 1.4 to 1.8 times high at a few hundred watts |
| MPPT interharmonics, double-frequency ripple | linear DC-link loop, perturb and observe | 0.1 to 4 % |
| Operating state (night, standby, MPP window, DC-link voltage from irradiance) | discrete states and per-scenario inputs | — |
| Network spread with Norton models, coherent and random-phase sums | exact linear solve at any frequency | 1e-9 to 1e-13 (OpenDSS, ngspice) |

Not modelled, and where it matters:

| Not modelled | Matters for |
|---|---|
| Frequency coupling (PLL mirror frequency `2 f_1 - f`, background harmonics into switching sidebands) | three-phase units with fast PLLs below about 500 Hz; harmonic instability; reported as `y_mirror` |
| Space-vector / min-max modulation, regular sampling, overmodulation | single supraharmonic lines of three-phase units (group rms within 0.8 to 1.35) |
| PWM common-mode voltage through ground capacitance | earth-return supraharmonic currents |
| Measurement filters, dq decoupling, constant-power references in the controller | admittance near the control bandwidth (each 4 to 99 % in one compared unit) |
| Digital (sampled-data) controller, aliasing | admittance near half the sampling frequency |
| Nonlinear behaviour (current limits, choke saturation, PLL under heavy distortion) | large disturbances, faults |
| Other MPPT patterns, multi-stage units | interharmonic amplitudes |
| Interharmonic and supraharmonic emission of devices without a model | the network sees only their harmonic shunt there |

When to use which mode: the simple injection mode (a spectrum, optionally with a passive
impedance) is the faster choice when only integer harmonics at known operating points
matter and the device spectra are measured. The detailed mode is the choice when emission
must follow the operating state (irradiance, night), when interharmonics or supraharmonics
matter, when the devices interact with resonances (the admittance decides how much of a
background distortion a unit absorbs or amplifies), or when primary and secondary emission
must be separated.

Cost: measured on a 50-node feeder with 10 units (CPU, complex128, one scenario), the classic
harmonic flow over 40 orders took 40 ms, the detailed study up to 2 kHz 191 ms (159
frequencies) and up to 150 kHz 337 ms (306 frequencies). Per solved frequency the two cost
about the same on small grids. The detailed study builds dense per-frequency matrices, so the
ratio grows with network size and scenario batch (14 to 27 times at 200 nodes, 39 to 78 times
per scenario in a batch of 64); gradients add about 1.7 times a forward pass in both.

### Which effects matter for the spread

A sensitivity study on a 10-node 400 V feeder with households and PV (feeder-end voltages
and one unit's current, at 100 W/m², 1000 W/m² and at night) compared the effect of each
modelled feature with the spread between plausible products (Monte Carlo over the filter,
carrier, dead time, controller, PLL and MPPT parameters) and with the model's known errors:

- **Larger than the product spread**: the closed-loop control admittance (replacing it by
  the filter alone changes the feeder's THD increment about ten times more than the product
  spread does, and a unit's THD_I by two to three times); having a device admittance at all
  (2-9 kHz voltage +130 to +230 % with pure current sources); the LCL capacitor and the EMC
  stages, together with the households' appliance capacitance (9-150 kHz); dead time (the only
  source in 2-9 kHz); MPPT (the only interharmonic source).
- **Within the product spread**: PLL dynamics, the single-phase PLL form, the double-frequency
  ripple, cable capacitance, regular versus natural sampling, space-vector modulation of a single
  three-phase unit, the coherent instead of the random-phase sum.
- **Missing effects that would matter**: the structure of the current controller (measurement
  filters, decoupling, constant-power references): an uncertainty of ±50 % on the control part
  of the admittance moves the THD increment by several times the product spread. Above 2 kHz the
  product parameters (dead time, carrier frequency, capacitor sizes) dominate the result, so
  measured product data helps more there than further model depth. The neglected mirror coupling
  is 0.1 to 0.9 of the direct admittance between 100 and 500 Hz at rated power: it matters for
  stability assessments, not for these spread metrics.

## Why a fixed spectrum is not enough

A fixed relative spectrum gives a constant relative distortion at every power, no secondary
emission and no influence of the grid on the device. Measurements contradict all three.
Relative current distortion of PV inverters rises 20 to 100 times at very low power while the
absolute harmonic current of some units stays or grows again; the terminal impedance changes
with the operating power, as does the grid inductance at which the unit becomes unstable; and
in the supraharmonic range the emission divides between the grid and the neighbouring
devices. A model that explains this has two terms at every frequency: a source and an
admittance, both following the operating state.

## Simple and detailed mode

A device's harmonic behaviour can be described in two ways.

- **Simple injection mode** (default): a current spectrum scaled to the fundamental and an
  optional passive `harmonic_impedance`. Every solver uses it, including
  {func}`pgml.solve_harmonic_flow`.
- **Detailed mode**: the optional `inverter_model` block of a `Generator` or `Storage`
  ({class}`~pgml.schemas.grid_schema.InverterModel`) holds the physical description of the
  unit below. Only {func}`pgml.studies.emission_flow` reads it; the same model can also be
  passed per solve as a {class}`~pgml.inverter.GridFollowingInverter`, which takes
  precedence. The two harmonic descriptions are alternatives: `inverter_model` and
  `harmonic_impedance` are mutually exclusive.

```python
from pgml.schemas import (
    Generator, InverterDCLink, InverterLCLFilter, InverterModel, InverterMPPT, InverterPLL,
    InverterPWM, InverterSeriesStage, InverterShuntStage, InverterCurrentControl, Phase,
)

pv = Generator(
    id=7, node=2, phases=(Phase.A,), p_nom_w=4600.0,
    inverter_model=InverterModel(
        filter=InverterLCLFilter(l1_h=1.5e-3, c_f=4.7e-6, r_d_ohm=1.0, l2_h=0.5e-3),
        control=InverterCurrentControl(kind="pr", kp=6.0, resonant={1: 500.0, 3: 150.0},
                                       sampling_hz=32000.0),
        pwm=InverterPWM(topology="single_phase_unipolar", carrier_hz=16000.0),
        pll=InverterPLL(kp=0.5, ki=40.0),
        emc=[InverterSeriesStage(l_h=20e-6), InverterShuntStage(c_f=1e-6)],
        dead_time_s=1e-6,
        dc_link=InverterDCLink(c_dc_f=1100e-6, kp=0.05, ki=2.0),
        mppt=InverterMPPT(v_step_v=4.0, rate_hz=5.0),
        u_dc_v=400.0,
    ),
)
```

`u_dc_v` is the DC-link voltage of the default operating state; a per-scenario state
({class}`~pgml.inverter.InverterState`) passed to the study overrides it.

## Generic units and unknown products

The internal design of a commercial inverter (filter values, controller structure and
gains) is rarely published, and a user of pgml will usually not know it. pgml therefore
offers generic units designed from their rating
({class}`~pgml.inverter.GenericInverterDesign`, {func}`~pgml.inverter.generic_single_phase_inverter`,
{func}`~pgml.inverter.generic_three_phase_inverter`). They describe no specific product:

- the LCL filter follows the design rules of Liserre, Blaabjerg and Hansen (IEEE TIA 41(5),
  2005): converter-side inductance from a current-ripple target, filter capacitor from its
  reactive power (at most 5 % of the rated power), damping resistor one third of the
  capacitor impedance at the resonance;
- the current controller (stationary-frame PR for single-phase, dq PI for three-phase,
  with grid-voltage feedforward) is tuned for a crossover at a fraction of the sampling
  frequency, the PLL for a damping ratio of 0.707, the DC-link loop for a set bandwidth;
- the capacitor-current damping gain is chosen per design: with the digital delay, a
  grid-current-controlled LCL filter is stable without active damping only for a resonance
  above about a sixth of the sampling frequency, and the damping turns into a negative
  resistance above it, so among a set of gains the stable one with the largest
  current-loop margin is taken (`GenericInverterDesign.report` gives the verdict).

{data}`~pgml.inverter.GENERIC_PARAMETER_RANGES` lists a plausible range for every design
knob with its basis, marking which ranges come from the literature and which are design
assumptions. Every physical parameter of every block also accepts a tensor with a scenario
batch, so a whole population of designs is one model and one batched study; structural
choices (topology, carrier and sampling frequency, MPPT rate) stay scalar because they fix
the frequency grid. Sampling a population over these ranges is the caller's choice: the
sensitivity results above show that the product parameters dominate the spread above 2 kHz
and that the control part of the admittance is the largest uncertainty below.

```python
import torch
from pgml.inverter import GenericInverterDesign

knobs = {"ripple_fraction": torch.tensor([0.1, 0.25, 0.4]),
         "dead_time_s": torch.tensor([0.5e-6, 1e-6, 3e-6])}
design = GenericInverterDesign(4600.0, phases=1, **knobs)
fleet = design.build()               # one model, batch of three designs
print(design.report()["stable"])     # tensor([True, True, True])
```

## The Norton equivalent

At each frequency a unit is

$$ i(f) = I_N(f;\mathrm{op}) - Y(f;\mathrm{op})\, v(f) $$

at its terminals, with `i` the current leaving the device per connection element and `v` the
terminal voltage. {meth}`~pgml.inverter.GridFollowingInverter.norton` builds both terms from
the physical blocks of the unit at its solved fundamental state:

| Block | Class | Enters |
|---|---|---|
| LCL output filter | {class}`~pgml.inverter.LCLFilter` | the plant of the current loop, the passive admittance |
| EMC stages (series chokes, X and Y capacitors) | {class}`~pgml.inverter.SeriesStage`, {class}`~pgml.inverter.ShuntStage` | outside the loop, carried as two-port stages |
| current controller, delay, feedforward, active damping | {class}`~pgml.inverter.CurrentController` | `Y` within the control bandwidth, the transfer of every source |
| PLL | {class}`~pgml.inverter.PhaseLockedLoop` | the operating-point dependent part of `Y` near the fundamental |
| PWM modulator | {class}`~pgml.inverter.PWMModulator` | supraharmonic sources (carrier groups) |
| dead time | `GridFollowingInverter.dead_time_s` | low-order harmonic sources |
| DC-link control, MPPT | {class}`~pgml.inverter.DCLinkControl`, {class}`~pgml.inverter.PerturbObserveMPPT` | interharmonic sources, the ripple harmonic |
| operating state | {class}`~pgml.inverter.InverterState`, {class}`~pgml.inverter.PVArray` | which terms exist, the DC-link voltage |

### Admittance: control and filter

The bridge voltage follows the controller, `v_b = G_del [G_c (i_ref - i) - k_c i_C + H_ff v]`,
with the digital delay `G_del = exp(-1.5 T_s s)`, a proportional-resonant controller in the
stationary frame or a PI controller in the synchronous frame, optional capacitor-current
damping and grid-voltage feedforward. Solving this together with the LCL circuit gives the
closed loop

$$ i = G_\mathrm{ref}\, i_\mathrm{ref} + G_d\, d - Y\, v , $$

where `d` is any additive bridge-voltage disturbance. For an L filter this is Wang, Harnefors
and Blaabjerg's `Y = Y_o / (1 + T)` with `T = G_del G_c Y_p` (IEEE TPEL 33(2), 2018, eqs.
6 to 8); pgml solves the general circuit exactly per frequency. The synchronous-reference-frame
PLL is linearised as `Δθ = H_PLL(s) ΔV_q` with `H_PLL = H_PI / (s + H_PI V_1d)` (same paper,
eq. 16). It perturbs the current reference by `j Δθ I_1` and, for dq-frame control, the
modulating voltage by `j Δθ V_c1`. Both terms are proportional to the operating point. This
is how the admittance near the fundamental follows the power.

The PLL makes the dq-frame model asymmetric, which couples a vector at `f` to its mirror at
`2 f_1 - f`. pgml keeps the direct (decoupled) term in the solve and reports the mirror
coefficient (`NortonEquivalent.y_mirror`) as a diagnostic of what is neglected. Above half
the controller's sampling frequency the control has no authority and `Y` is the passive filter.

Three-phase units are modelled in symmetrical components. The positive sequence is the space
vector at `+f`, the negative sequence the conjugate of the vector at `-f`, and a three-wire
unit has no zero-sequence inner path, only the ground capacitance of its EMC filter. The
element admittance `A diag(Y0, Y+, Y-) A^-1` is not symmetric when `Y+ ≠ Y-`, which is the
case for dq-frame control with a PLL. A single-phase unit takes the mean of the two
space-vector evaluations. This mean equals the ordinary transfer function whenever the model
has no frequency coupling.

`NortonEquivalent.admittance_passive` is the same admittance with the bridge shorted and the
control removed, so the contributions of the control and of the filter can be separated. The
study option `emission_flow(admittance="passive" | "none")` propagates the same sources with
the filter alone or as pure current sources.

### Sources

**PWM residues.** A naturally sampled sine-triangle leg has the double Fourier series with
Bessel weights `J_n(m π M / 2)` (Holmes and Lipo). The bipolar and unipolar single-phase
bridges and the three-phase bridge are combinations of legs. The unipolar first emission band
reproduces Klatt et al.'s value (IET GTD 13(24), 2019, fig. 3: about 136 V RMS at 230 V, 400 V
DC link, 1 kHz carrier) to 0.02 V. The residues depend on the modulation index and the DC-link
voltage only, so their power dependence comes from the DC link, as measured by Klatt et al.
2016. An overmodulated bridge (`M > 1`) is refused: the series does not hold there.

**Dead time.** During the blanking time `t_d` a leg's voltage follows the current, which
lowers the average leg voltage by `(t_d / T_s) U_dc` for a positive current (Mohan, Undeland
and Robbins, *Power Electronics*, sec. 8-5). The error is a square wave locked to the current,
with odd harmonics in a single-phase bridge and orders `6k ± 1` in a three-wire three-phase
unit, and its amplitude does not depend on the current. This is the mechanism that keeps the
absolute low-order harmonic current roughly constant while the fundamental falls. With
`dead_time_ripple="local"` the error vanishes where the current lies inside half the local
switching ripple. A leg commutes at the extremes of the ripple, so the two blanking errors of
a period cancel there; this matters at low power.

**MPPT interharmonics.** Perturb-and-observe MPPT cycles the DC-link reference through
`{0, +v, 0, -v}`. The DC-link loop passes its odd Fourier lines `n f_MPPT / 4` into the
current amplitude, which the PLL angle turns into lines at `f_1 ± n f_MPPT / 4`
(Sangwongwanich et al., IEEE TPEL 33(10), 2018, eqs. 1 to 7). Lower sidebands below 0 Hz are
folded. The amplitudes do not depend on the power, so they dominate the relative distortion at
low power.

**Double-frequency ripple.** A single-phase unit's power pulsates at `2 f_1`. The DC-link
ripple passes the voltage controller into the current amplitude and produces a third harmonic
proportional to the power, unless a notch at `2 f_1` blocks it.

**Locked and unlocked.** Dead-time and ripple harmonics follow the unit's own fundamental and
carry a phase in the grid's reference (`source_locked`). PWM residues follow free-running
carriers and MPPT lines independent clocks, so they have no common phase between units
(`source_unlocked`).

### Operating state

| State | `bridge_on` | `connected` | Model |
|---|---|---|---|
| night, standby, disconnected | no | no | EMC stages on the grid side of the relay (`emc_on_grid_side`) |
| bridge idle, relay closed | no | yes | passive LCL and EMC filter |
| MPPT, curtailed, reactive power at night | yes | yes | full model at the solved `P, Q` |

The DC-link voltage is an input of the state: fixed for a two-stage unit, the MPP voltage of
the array for a single-stage one ({meth}`~pgml.inverter.PVArray.mpp`). A single-stage unit whose
MPP voltage falls below the peak bridge voltage it needs cannot run and must be modelled as
not switching. Every field accepts a scenario batch, so an irradiance sweep from night to full
sun is one call.

## The emission study

{func}`pgml.studies.emission_flow` solves the network on a frequency grid that holds every
model line (integer harmonics, MPPT interharmonics, PWM residues) plus any requested
frequency:

1. the fundamental power flow gives each unit's terminal voltage and current;
2. each modelled unit evaluates its Norton equivalent at that state;
3. the network matrix is the harmonic flow's own `Y(f)` at every frequency (lines, source
   Norton shunt, load harmonic shunts, explicit DER impedances), plus each unit's `Y(f)`;
4. the right-hand side holds the Norton currents and, at integer orders, the other devices'
   spectra and the background distortion exactly as the harmonic flow forms them.

The units enter the network only at their terminals. The network without them is
factored once per frequency and solved for one column per unit terminal; the unit
admittances then act as a low-rank (Woodbury) update on that terminal impedance matrix.
The result equals stamping the admittances into `Y(f)` (agreement about 1e-15 relative),
and a batch of unit parameters (irradiance, DC voltage, design knobs) only repeats the
small terminal system, not the network factorization.

Without inverter models the study reproduces {func}`pgml.solve_harmonic_flow` at the integer
orders. The result carries:

- `v`, the coherent solution with every source at its stated phase, and `v_rms`, the expected
  magnitude when the unlocked components of different units have independent uniform phases,
  `E|V|² = |V_locked|² + Σ_d |V_unlocked,d|²`;
- per unit, the terminal current and its primary part, the current driven by the unit's own
  sources alone (Bollen and Rönnberg's definition); the remainder is secondary emission;
- per unit, the stability margin `σ_min(I + Z_g Y)` per frequency, with `Z_g` the rest of the
  system seen from the unit, the per-frequency reading of Sun's `Z_g / Z_o` criterion. It
  flags a frequency where the steady state may not exist; it is not a Nyquist test.

Everything is differentiable in the tensor-valued grid, filter, controller and state
parameters, and runs on CPU and CUDA.

(inverter-validation)=

## Validation

Each block was compared with an independent reference: a time-domain switching simulation,
a second circuit or power-system tool, or a published value. The numbers are maximum errors
over the stated range, at complex128.

| Block | Reference | Agreement |
|---|---|---|
| PWM residues, three topologies, M 0.3 to 0.95, 4 groups | brute-force switching instants, Fourier integral edge by edge | 1e-11 relative magnitude, 1.2e-9° phase |
| PWM residues | ngspice switched H bridge and three-phase bridge | 9e-4 relative (time-step limited), 0.04° |
| first unipolar band | Klatt et al. 2019, fig. 3 | 136.02 V against "about 136 V" |
| dead time, hard sign, leg and full bridge | switching simulation with blanking | 2e-4 relative, 0.7° |
| dead time, hard sign, three-phase | same, common mode removed | 1e-2 relative at h = 25 |
| dead time, ripple dead zone (default), bipolar, unipolar, three-wire, currents 0.3 to 30 times the half ripple, two parameter sets | imposed-current switching simulation with the bridge's own ripple | 0.2 to 14 % normalised error over orders 2 to 25 (hard sign: up to 2.4 to 10) |
| MPPT reference lines | Sangwongwanich et al. 2018, fig. 17(a) | 10.804 V (exact series) |
| MPPT current lines, 2 to 20 Hz rates | averaged time-domain DC-link model; python-control linear loop | 0.12 % magnitude, 0.04° on the dominant lines; 4.1 %, 4.4° against a model with instantaneous single-phase power, which the linear model leaves out |
| DC-link closed loop and ripple path | independent implementation of eq. 4; averaged time-domain DC-link model | 5e-16; ripple harmonic 0.12 % magnitude, 1.3° |
| closed-loop admittance, PR control, five variants (grid and converter feedback, active damping, feedforward, L filter) | python-control interconnection with a high-order Padé delay | 8.5e-8 (the reference's own convergence) |
| same | averaged time-domain simulation, frequency-sweep injection, continuous control | 1e-4; a digital controller adds 0.6 to 4 % (discretisation) |
| dq-frame control with SRF-PLL, direct and mirror terms | independent implementation of Wang et al.'s dq transfer matrices | 2.3e-14 and 1.6e-11 |
| same | averaged three-phase time-domain simulation with an SRF-PLL | 0.4 to 0.7 % direct term (continuous); a digital controller adds 12 to 20 % |
| network solve with the Norton models, 168 frequencies, three feeder variants | OpenDSS (compensation theorem for the non-symmetric admittance) and ngspice AC analysis | 2.3e-9 (OpenDSS, set by its own fundamental solve), 4e-13 (ngspice) |
| random-phase aggregation | 2000-draw Monte Carlo | within 2.8 standard errors |
| three-phase dq-controlled converter with SRF-PLL, L filter, direct and mirror terms, 76 Hz to 2.5 kHz | VeraGrid EMT (averaged bridge, 5 µs trapezoidal), injection sweep with the controller structure reduced to pgml's | 5.2e-4 direct, 0.004°; 1.1e-3 mirror |
| carrier-group current lines above the control band | same EMT, switched bridge | 0.977 to 1.016 of V_b / (Z_f + Z_g) |
| complete single-phase unit (4.6 kW and 0.5 kW, stiff and weak supply): supraharmonic groups at 32 and 64 kHz | switched time-domain simulation (exact edges, 1 µs dead time, regularly sampled unipolar PWM, digital PR control, SOGI-PLL) | group RMS ratio 0.999 to 1.010 |
| same: admittance 425 Hz to 25 kHz, dead time off | small-signal injection in the same simulation | 0.3 to 3.9 % magnitude, 0.6° |
| same: dead-time harmonics, orders 5 to 21, rated power | same | 0.4 to 5 % magnitude; THD_I 1.86 % against 1.82 % |

Wang et al.'s Table II gains reproduce the paper's stability trends (dq control stable at a
20 Hz PLL bandwidth, lightly damped at 175 Hz, unstable at 330 Hz; stationary-frame control
stable throughout) when read as a damping ratio 0.707 design, but not the paper's exact
intersection frequencies; the 330 Hz instability appears only in the coupled model with the
mirror term, not in the decoupled admittance alone. The model's controllers are continuous:
the sampled-data effects of a digital controller (the 0.6 to 20 % above) are not modelled.

The switched simulation of the complete unit also shows where the model is not yet good
enough. At 0.5 kW the measured THD_I is 6.25 %; the ripple dead zone predicts 8.9 % and the
hard sign 17 % (at 1 kW: 4.2 % measured, 7.5 % with the dead zone). The dead-time harmonics
are locked to the fundamental current, while the distorted current crosses zero 1.7° (4.6 kW)
to 3.4° (0.5 kW) earlier, so their phase drifts by about 2° per order (18° at the 9th, 78° at
the 39th); magnitudes are unaffected, a phasor sum of many units is. The single-phase unit's
PLL is the SRF form averaged over the two space-vector evaluations; against a SOGI-PLL it
overstates |Y| at 175 Hz by 31 % at rated power, and the mirror coupling it neglects measures
10 to 19 % there. With the dead time on, the small-signal response at low power depends on
the injection amplitude (the blanking error is nonlinear in the current).

The tests under `tests/inverter/` pin the closed forms of every block.

## Limits

- **Dead time at low power and in phase.** See the end-to-end comparison above: low-order
  emission at a few hundred watts is overstated by about 1.4 to 1.8 times, and the phase of
  higher dead-time orders drifts.
- **Single-phase PLL.** No SOGI model; the SRF form stands in for it.
- **Decoupled.** The solve keeps the same-frequency admittance. Mirror-frequency coupling from
  the PLL (`f` and `2 f_1 - f`), the coupling of background harmonics into the switching
  sidebands, and measured frequency coupling matrices are not solved; `y_mirror` reports the
  size of the first.
- **Sinusoidal modulation only.** Three-phase units with min-max (space-vector
  equivalent) zero-sequence injection have different carrier-group lines: against a
  switched simulation with that modulation, single lines differed by a median of 78 % and
  group rms by 0.80 to 1.35. The differential spectrum of such a modulator is not
  modelled.
- **Controller structure.** The current controller has no filters on its measurements, no
  dq cross-coupling decoupling and no constant-power reference (`i_ref ∝ 1 / |v|`); a
  controller with them has a different admittance near the control bandwidth (each changed
  the admittance of one compared unit by 4 to 99 %).
- **Natural sampling.** Regularly sampled PWM has the same line frequencies with slightly
  different amplitudes and adds baseband harmonics; overmodulation is refused, not modelled.
- **Differential mode only.** The PWM common-mode voltage of a three-wire unit, which drives
  current through the EMC ground capacitance, is not a source.
- **One MPPT pattern.** The three-level steady state of perturb and observe; other trackers
  and two-stage units with an MPPT on the boost stage need their own amplitudes.
- **Continuous control.** The controller and the delay are continuous-time; the
  sampled-data behaviour of a digital controller (aliasing, the zero-order hold) is not
  modelled.
- **Linearised at the operating point.** Saturation of filter chokes, current limits and the
  nonlinearity of the PLL under large distortion are outside the model.
- **Ideal inner tracking of reference lines.** MPPT and ripple lines are the grid current
  of the DC-link loop model (whose inner loop is first order) and are not passed through the
  closed current loop again; at 100 to 500 Hz that loop's reference gain is about 1.1 in
  magnitude for a typical single-phase unit.
- **Background source at integer orders only.** A voltage-kind `NodeHarmonicSource` (its
  Thevenin shunt and EMF) acts at the integer orders, as in the harmonic flow; at
  interharmonic and supraharmonic frequencies the network has no background source.
- **Integer orders for other devices.** Loads and generators without an inverter model emit
  only at integer orders, from their spectra; at other frequencies they are their harmonic
  shunt.
