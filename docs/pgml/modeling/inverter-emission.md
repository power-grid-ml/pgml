# Inverter harmonic emission

How {mod}`pgml.inverter` models the harmonic, interharmonic and supraharmonic behaviour of a
grid-following inverter, and how {func}`pgml.studies.emission_flow` spreads it through a grid.
The classic harmonic model of a converter, a fixed current spectrum scaled to the fundamental
current, is what {func}`pgml.solve_harmonic_flow` uses (see {doc}`der-pv-storage`); this page
describes the physical model that replaces it for the devices it is attached to.

## Why a fixed spectrum is not enough

A fixed relative spectrum gives a constant relative distortion at every power, no secondary
emission and no influence of the grid on the device. Measurements contradict all three.
Relative current distortion of PV inverters rises 20 to 100 times at very low power while the
absolute harmonic current of some units stays or grows again; the terminal impedance changes
with the operating power, as does the grid inductance at which the unit becomes unstable; and
in the supraharmonic range the emission divides between the grid and the neighbouring
devices. A model that explains this has two terms at every frequency: a source and an
admittance, both following the operating state.

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
| dead time, three-phase | same, common mode removed | 1e-2 relative at h = 25 |
| MPPT reference lines | Sangwongwanich et al. 2018, fig. 17(a) | 10.804 V (exact series) |
| DC-link closed loop and ripple path | independent implementation of eq. 4; averaged time-domain DC-link model | 5e-16; ripple harmonic 0.12 % magnitude, 1.3° |
| network solve with the Norton models, 168 frequencies, three feeder variants | OpenDSS (compensation theorem for the non-symmetric admittance) and ngspice AC analysis | 2.3e-9 (OpenDSS, set by its own fundamental solve), 4e-13 (ngspice) |
| random-phase aggregation | 2000-draw Monte Carlo | within 2.8 standard errors |

The closed-loop admittance with and without the PLL is checked against a python-control
interconnection of the same blocks, against an independent dq-frame implementation of Wang et
al.'s transfer matrices, and against an averaged time-domain simulation with frequency-sweep
injection; the end-to-end model of a single-phase unit is checked against a switched
time-domain simulation of the full unit. The tests under `tests/inverter/` pin the closed forms
of every block.

## Limits

- **Decoupled.** The solve keeps the same-frequency admittance. Mirror-frequency coupling from
  the PLL (`f` and `2 f_1 - f`), the coupling of background harmonics into the switching
  sidebands, and measured frequency coupling matrices are not solved; `y_mirror` reports the
  size of the first.
- **Natural sampling.** Regularly sampled PWM has the same line frequencies with slightly
  different amplitudes and adds baseband harmonics; overmodulation is refused, not modelled.
- **Differential mode only.** The PWM common-mode voltage of a three-wire unit, which drives
  current through the EMC ground capacitance, is not a source.
- **One MPPT pattern.** The three-level steady state of perturb and observe; other trackers
  and two-stage units with an MPPT on the boost stage need their own amplitudes.
- **Linearised at the operating point.** Saturation of filter chokes, current limits and the
  nonlinearity of the PLL under large distortion are outside the model.
- **Integer orders for other devices.** Loads and generators without an inverter model emit
  only at integer orders, from their spectra; at other frequencies they are their harmonic
  shunt.
