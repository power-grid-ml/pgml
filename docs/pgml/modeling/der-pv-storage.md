# DER: PV inverters, generators and storage

How pgml represents distributed energy resources, what rides the autograd tape and what does
not, and how the model lines up with pandapower, OpenDSS and power-grid-model.

## The five questions a DER model has to answer

1. Bus behaviour. Is the device a fixed P/Q injection, a voltage-regulating source with free
   reactive power, or the slack. Most DER on a distribution feeder are the first.
2. Inverter control law. A grid-following inverter does not hold P and Q constant. It follows
   a characteristic, constant power factor, `cosφ(P)`, Volt-VAr `Q(V)`, Volt-Watt `P(V)`, or
   a combination, subject to a capability limit.
3. Active-power source. A PV array's available power comes from irradiance and temperature
   through maximum-power-point tracking. A battery's comes from a dispatch decision.
4. Output impedance. A grid-following inverter behaves as a current source at the
   fundamental and injects harmonics as a current source. A synchronous machine sits behind
   its sub-transient reactance. This matters mostly at harmonic orders.
5. State and time coupling. A battery carries state of charge, and dispatch couples time
   steps through `SoC[t+1] = SoC[t] + η·P·Δt`.

Questions 1 to 4 are piecewise smooth functions of local quantities, so they belong on the
differentiable path. Question 5 is stateful and rule-driven, and in every tool surveyed it is
integrated outside the per-snapshot solve.

## What pgml does

One injection appliance covers every kind of device. `Generator` and `Storage` carry an
optional `control` block, and which control and harmonic model is set is what makes a device a
PV inverter, a wind plant or a genset. `consumer_type` is a closed taxonomy used as a
categorical feature and never drives the physics.

The control laws are a discriminated union over a common base that carries the capability
circle `s_rated_va` and a `smoothing` half-width.

| Control | Law |
|---|---|
| `ConstantPowerFactorControl` | `Q = P·tanφ` |
| `ConstantReactivePowerControl` | `Q` fixed |
| `PowerFactorWattControl` | `cosφ(P)`, the VDE-AR-N 4105 shape |
| `VoltVarControl` | `Q(|V|)` from a characteristic |
| `VoltWattControl` | `P(|V|)` curtailment above a voltage threshold |
| `VoltVarVoltWattControl` | Both together |

Characteristics use the generic tensor-capable `Characteristic`, with `x_values`, `y_values`
and linear or cubic interpolation, so curve points are differentiable parameters like any
other.

The capability limit applies with watt priority. Active power is clipped to the rating first,
because an oversized array cannot exceed the inverter rating through `P` alone, then the
reactive magnitude is bounded by the remaining headroom on the circle.

## Why the control law is not an outer loop

`device_current_injections` computes each element's effective power from a base power and a
ZIP voltage factor. A control law makes that base power voltage dependent, so instead of a
constant `S0` the device contributes
`S0 = P_ctrl(|V_term|, P_avail) + j·Q_ctrl(|V_term|, P)`. Everything downstream is unchanged.

The nonlinear solve shares one residual with its backward pass,
`F(V) = Y_eff·V + I_device(V) − I_slack`, and the implicit-function-theorem backward
differentiates that residual once at the converged voltage. The new `∂Q/∂|V|` and `∂P/∂|V|`
terms therefore appear in the Jacobian automatically. No new adjoint code, and no unrolled
iteration. Forward Newton uses the same Jacobian, so its behaviour near the loadability limit
is preserved.

pandapower and OpenDSS both realise a control law as an outer loop that re-solves the power
flow until the setpoints stop moving. Folding the law into the residual instead makes it part
of the implicit function `V*(θ)`, so the gradient of any output with respect to a curve slope,
a rating or an irradiance value is available directly. That is what makes gradient-based
tuning of inverter settings, or recovery of an unknown `Q(V)` slope from measurements,
possible.

## Kinks on a differentiable path

Volt-VAr and Volt-Watt curves, deadbands and capability clamps have corners. The forward pass
evaluates the exact piecewise-linear curve and the hard clamp, which is what matches the
reference tools at the operating point. For gradients each corner has a smooth variant, a
blended breakpoint and a soft saturation, controlled by the `smoothing` half-width. A width of
zero recovers the hard curve, and a positive width makes the map continuously differentiable
so the Jacobian is well defined. This is the same approach the rest of the library takes when
it guards a division by zero. Keep the forward correct and keep the gradient finite.

Genuinely discrete switches stay discrete. An inverter trip, or a cut-in and cut-out
threshold with hysteresis, is resolved upstream into whether the device injects at all, rather
than becoming a branch inside the solve.

## Storage, state of charge and dispatch

The snapshot solve sees a signed P/Q injection and nothing else, which is what pandapower and
OpenDSS do as well. `Storage` carries `p_nom_w` signed so that a positive value discharges,
plus the energy-state fields `energy_capacity_wh`, `soc`, `soc_min`, `soc_max`,
`efficiency_charge`, `efficiency_discharge` and `p_rated_w`. Those fields are inert in the
solve.

Dispatch and state of charge live in `pgml.scenarios`. A resolver maps a profile, a rule or a
price signal together with the present state of charge to an active power, then advances
`SoC[t+1] = SoC[t] + η·P[t]·Δt` with separate charge and discharge efficiencies, following
the OpenDSS equations. The rule is ordinary Python control flow and never touches the tape.
The resolved power is a tensor, so the solve differentiates with respect to the setpoint
value rather than with respect to the rule that produced it. The state-of-charge recurrence is
itself linear and could be kept on the tape if a gradient through time were ever needed.

Maximum-power-point tracking has the same shape. Irradiance and temperature go through the
array and efficiency curves to an active setpoint, resolved per scenario, differentiable with
respect to irradiance when a sensitivity is wanted.

## The voltage-regulating bus

pgml has no PV bus. The slack is the `Source`, and every other appliance is an injection. The
real fix is a regulation mode that replaces a terminal's power-balance row with
`|V_term| − V_set = 0` and lets reactive power float, with a smooth saturation at the reactive
limit. That stays inside the same residual and Jacobian, so it would remain differentiable and
batched, but it needs a solver change and a schema field to carry the setpoint.

What exists today is an opt-in approximation in the pandapower reader. A Volt-VAr
characteristic centred on the generator's own voltage setpoint and saturating at its reactive
limits reproduces PV-bus behaviour in the limit of an infinite slope. The bus settles where the
droop's reactive output balances the network, which is off the setpoint by
`Q / (slope · Q_base)` per unit, and the reactive limit comes from the capability bound instead
of a discrete bus-type switch. `gen_mode=GenMode.VOLT_VAR_APPROX` enables it, with a
steepness parameter whose default is one full reactive base per 0.002 pu of voltage. Because
this is an ordinary inverter control it rides the same backward pass.

The limit of the approximation is conditioning rather than steady-state fidelity. Outside the
band of width `1/slope` the derivative `dQ/d|V|` is exactly zero, so an iterate that starts far
from the setpoint sees no voltage-control feedback. Measured against a live pandapower solve
with reactive limits enforced, the per-bus voltage deviation falls as `1/slope`, from 4.3e-2 pu
at slope 5 to 2.3e-4 pu at slope 2000 on `case57`. On `case118` the usable steepness caps
around 5, and on `case39` every steepness converges silently onto the collapsed low-voltage
branch, 0.49 pu off with all nine machines pinned at their reactive limit. That branch is a
genuine second solution of the approximated system, which the exact regulation row would
exclude by construction. The approximation is therefore useful on small and moderately loaded
networks and for sensitivity studies, and it is not a faithful way to import transmission
benchmarks. Check the converged voltage profile against the source network's own result.

## Harmonics follow the control state

Each device's harmonic injection is derived from its fundamental current, with
`|I_h| = (mag_h/mag_1)·|I₁|` and `∠I_h = ang_h + h·(∠I₁ − ang_1)`. A control law changes the
fundamental operating point, which changes `I₁`, which rescales the harmonic current sources.
The spectrum therefore tracks the control state with no extra machinery, and it stays
differentiable through `I₁`.

Two refinements remain open. The inverter's output impedance at harmonic orders is the Norton
shunt of `HarmonicShuntModel`, and requesting it currently raises, so a grid-following
inverter does not yet present its damping at harmonics. Modern PWM inverters also emit little
at low orders while producing content between and above the integer orders, which the schema
can carry as data but the integer-order solve does not represent.

## Cross-tool comparison

| Capability | pandapower | OpenDSS | power-grid-model | pgml |
|---|---|---|---|---|
| DER element | `sgen`, a PQ injection | `PVSystem` | `sym_gen`, a PQ injection | `Generator` |
| Voltage-regulating bus | `gen` | `Generator model=3` | none | none, a droop approximation on import |
| Reactive limit | `enforce_q_lims`, switching PV to PQ | `min/maxkvar` | none | capability circle |
| Constant power factor | through a controller | `pf` | none | `ConstantPowerFactorControl` |
| Volt-VAr, Volt-Watt | `CharacteristicControl` | `InvControl`, `ExpControl` | none | control union |
| Storage | `storage`, PQ with external SoC | `Storage` with `StorageController` | none | `Storage` with external dispatch |
| Harmonics | no | yes, current-source Norton | no | yes, current-source Norton |
| Control coupling | outer loop around the solve | control-iteration loop | none | inside the residual |
| Differentiable | no | no | no | yes |

The pattern the other tools share is that a control or dispatch layer resolves the operating
point and the solve then sees a fixed injection. pgml keeps that split for stateful dispatch
and moves the equation-defined part inside the residual.

A few per-tool details are worth knowing when comparing results.

- pandapower's `sgen` is a constant-PQ injection in generator convention. Its short-circuit
  fields do not affect the load flow, and its `type` column is a free label.
- pandapower's `storage` is a constant-PQ element in load convention, and its state of charge
  is documented as not updated by the power flow.
- OpenDSS computes a `PVSystem`'s active power as `P_DC = irradiance·Pmpp·f(T)` followed by an
  efficiency curve, bounds P and Q by the inverter rating, and gates the inverter with
  hysteresis thresholds.
- OpenDSS `Generator` selects its behaviour with a `model` integer, where 1 is constant P and
  Q, 3 is a voltage-regulating bus and 7 is a current-limited inverter.
- power-grid-model has no harmonics, no storage and no voltage control. Its only regulator is
  a discrete transformer tap.

## Validation

Inverter control is checked against pandapower's `CharacteristicControl` for `Q(V)`, which
reaches the same equilibrium to about 1e-10 pu, and against OpenDSS `InvControl` in its
Volt-VAr and Volt-Watt modes. The differentiability gate runs a float64 gradient check of the
solved voltage with respect to a curve slope, the capability limit and the active setpoint
through the smoothed variants, and the device parity gate repeats every injection term on CPU
and CUDA.

## Sources

- pandapower documentation for `sgen`, `gen`, `storage`, the control loop and
  `CharacteristicControl`.
- OpenDSS documentation from EPRI for `PVSystem`, `Generator`, `Storage`,
  `StorageController`, `InvControl`, `ExpControl`, and the harmonic flow and load pages, plus
  the EPRI technical note on smart-inverter function modelling.
- power-grid-model component reference.
- Interconnection standards that define the characteristics, IEEE 1547-2018, EN 50549-1 and
  -2, and VDE-AR-N 4105.
- The spectrum and phase convention is recorded in
  [OpenDSS harmonics](references/opendss/harmonics.md).
