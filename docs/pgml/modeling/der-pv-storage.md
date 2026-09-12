# DER: PV inverters, generators and storage

How pgml represents distributed energy resources, what rides the autograd tape and what does
not, and how the model lines up with pandapower, OpenDSS and power-grid-model.

## The five questions a DER model has to answer

1. Bus behaviour. Is the device a fixed P/Q injection, a voltage-regulating source with free
   reactive power, or the slack. Most DER on a distribution feeder are the first. pgml models
   all three: a PQ injection (`Load`, `Generator`, `Storage`), the slack (`Source`) and a PV
   terminal (`Generator.voltage_regulation`).
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
differentiates that residual once at the converged voltage. At a voltage-regulating terminal
the residual instead carries a `|V|`-regulation row, with the reactive power eliminated
analytically so the Jacobian keeps its size. The new `∂Q/∂|V|` and `∂P/∂|V|` terms therefore
appear in the Jacobian automatically. No new adjoint code, and no unrolled
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

Dispatch and state of charge live in {mod}`pgml.dispatch`.
{func}`~pgml.dispatch.integrate_soc` realizes a requested power sequence under the state-of-
charge reserve, the capacity and the power rating, advancing
`SoC[t+1] = SoC[t] + η·P[t]·Δt` with separate charge and discharge efficiencies, following the
OpenDSS equations. The dispatch rule itself is the caller's ordinary Python control flow and
never touches the tape. The realized power is a tensor, so the solve differentiates with
respect to the setpoint value rather than with respect to the rule that produced it, and the
state-of-charge recurrence is itself differentiable if a gradient through time is wanted.

Maximum-power-point tracking has the same shape. Irradiance and temperature go through the
array and efficiency curves to an active setpoint, resolved per scenario, differentiable with
respect to irradiance when a sensitivity is wanted.

## The voltage-regulating (PV) bus

A `Generator` carrying a `VoltageRegulation` block is a PV terminal. Its active power is the
nameplate or operating-point value, its terminal voltage magnitude is held at `v_set_pu`, and
its reactive power is whatever that takes, bounded by `q_min_var` and `q_max_var`. It is the
model behind pandapower `net.gen`, power-grid-model's `voltage_regulator` and OpenDSS
`Generator model=3`, and it is what makes the MATPOWER transmission benchmarks importable.

### The residual row pair

The solver's real residual carries the complex nodal current mismatch
`F_c = Y_eff·V + I_device(V) − I_slack`. At a regulating terminal's row the two real equations
become the power-form pair

```
g = conj(V) · F_c / v0                      (v0 = the node's line-to-neutral nominal)
real half:  Re(g)                           active power balance   [A]
imag half:  (|V_reg|² − V_set²) / (2·V_set) the voltage setpoint   [V]
```

The generator's reactive current is `−conj(jQ)/conj(V)`, and `conj(V)` times that is `+jQ`,
purely imaginary. The active half is therefore independent of the reactive power, and the
imaginary half is the only place it appears, so replacing the imaginary half frees the reactive
power exactly. It is recovered from the converged solution as `Q = Q_pinned − Im(conj(V)·F_c)`,
summed over the unit's phases, and reported in `PowerFlowResult.regulation`.

Eliminating the reactive power analytically, rather than carrying it as an extra unknown, keeps
the state at `[Re V; Im V]`. The `[2N, 2N]` implicit-function-theorem Jacobian, the adjoint and
the batching are unchanged, and `dV/dv_set` flows through the same backward pass as every other
parameter. The replacement row is quadratic in `V`, with no `abs` and no `sqrt`, so it is
smooth and its Jacobian entries are O(1), the same scale as the ideal-slack pinning rows.

### What is regulated

`regulated="positive_sequence"`, the default, holds the positive-sequence magnitude `|V1|`.
That is balanced regulation, the standard for a machine or a three-phase inverter. A
three-phase unit then has one reactive power split equally over its phases, so the imaginary
halves of its other two phases carry the equal-split conditions, which are reactive-power-free,
while the first carries the setpoint. `regulated="per_phase"` instead holds every phase
magnitude at the setpoint with its own free reactive power, while the reactive capability stays
a machine total. On a balanced terminal the two agree exactly.

### Reactive limits

Limits are enforced the standard way, by switching the terminal's bus type between rounds of
the solve. A unit whose required reactive power leaves its band is re-solved as a plain PQ
injection pinned at the violated limit, and released when its terminal voltage crosses the
setpoint from the other side. A hysteresis band
(`appliance.generator.q_limit_hysteresis_*` in `pgml.defaults`) keeps solver noise from cycling
the decision. `solve_power_flow(enforce_q_limits=False)` solves every terminal unbounded, which
is what pandapower's `runpp` does by default.

The switching decision is off-tape, being a comparison of converged values, while the residual
at the resolved active set is on-tape, so the adjoint is exact for the solved configuration.
The gradient flows through `v_set_pu` at a regulating terminal and through the binding limit at
a pinned one. A smooth complementarity or saturation formulation would make the gradient
continuous across the switching boundary, at the price of satisfying `Q = q_max` only to the
smoothing width, and the reactive power at the limit is exactly the quantity a comparison
against another tool checks.

### Solver and scope

A grid with a regulating terminal is always solved by Newton: the current-injection fixed point
updates the voltage from a current injection, and a regulated row has no injection to form.
`method="current_injection"` logs the switch. Such a grid also gets a second Newton warm start,
the balanced nominal profile with every regulated row at its setpoint, which is tried first
when the constant-impedance seed collapses below half nominal somewhere, and as a fallback
otherwise.

The row pair is formed for a WYE terminal returning to ground. A DELTA machine raises, because
its reactive current is shared between two node rows, so only the circulating total is
observable and not the split. A WYE machine returning through its node's neutral row raises as
well, so set `return_path="ground"`. So do a positive-sequence setpoint on a two-phase
terminal, a regulating generator on a `Source`'s node, and two regulating generators on one
node. Regulation is a fundamental-frequency concept: at harmonic orders the machine stays the
Norton current source described below.

### Validation

Against `pp.runpp` on the MATPOWER benchmarks as published, the converged voltages agree to
4.4e-16 pu (case9), 6.2e-12 pu (case14), 8.9e-12 pu (case30), 1.8e-15 pu (case39) and
3.1e-15 pu (case57), with case14 and case30 at pandapower's own mismatch tolerance, and the
generator reactive powers to 7.9e-9 Mvar. With reactive limits enforced in both tools, the same
generators switch to PQ at the same reactive power: case39 one of nine, to 1.3e-10 Mvar;
case118 six of 53, to 7.3e-12 Mvar. case118 and case300 differ by 6.7e-3 and 1.0e-2 pu as
published, entirely because of the magnetizing-branch placement described in
{doc}`transformer`; with `i0_percent` zeroed in both tools they agree to 6.7e-16 and 3.0e-14 pu.
Against OpenDSS `Generator model=3` on a two-bus feeder: 6.6e-10 pu in voltage and
4.2e-4 kvar in reactive power while regulating, with 4 Newton iterations against OpenDSS's 108
of its own, and 1.7e-13 pu and 1.3e-8 kvar with the machine pinned at a reactive limit.

### The droop approximation

The earlier Volt-VAr approximation remains available as
`pgml.convert.pandapower.to_grid(net, gen_mode=GenMode.VOLT_VAR_APPROX)`. A Volt-VAr
characteristic centred on the generator's own voltage setpoint and saturating at its reactive
limits reproduces PV-bus behaviour in the limit of an infinite slope, and because it is an
ordinary inverter control it rides the same backward pass. It holds the voltage only to
`Q/(slope·Q_base)` per unit, and outside a `1/slope`-wide band it carries no voltage-control
feedback at all: on `case39` it converges silently onto the collapsed low-voltage branch at
every steepness. Use it to model a real droop-controlled DER, not to import a transmission
benchmark.

## Harmonics follow the control state

Each device's harmonic injection is derived from its fundamental current, with
`|I_h| = (mag_h/mag_1)·|I₁|` and `∠I_h = ang_h + h·(∠I₁ − ang_1)`. A control law changes the
fundamental operating point, which changes `I₁`, which rescales the harmonic current sources.
The spectrum therefore tracks the control state with no extra machinery, and it stays
differentiable through `I₁`.

A `Load` also carries the OpenDSS device Norton shunt in parallel with its current sources
(`load_shunt`, default `appliance.harmonic_shunt.model`), which is the dominant damping term at
a feeder parallel resonance. A generation-sign device does not: the load expression
`conj(S_eff)/V_rated²` has a negative conductance when `S` is negative, so applying it to an
inverter would feed harmonic energy into the network instead of damping it, and no physical
inverter or machine does that. The policy is the documented default
`appliance.harmonic_shunt.generation_model`, shipped as `none`, whose `load_style` setting
applies the load expression anyway and reproduces the negative-kW `Load` idiom an OpenDSS
export uses. Naming the `motor` model on a device grants it the blocked-rotor branch, and
`harmonic_model.neglect_shunt = true` always wins. One warning per assembly names how many
generation devices were left as pure current sources.

Measured against a live OpenDSS on a feeder whose only injecting device is a distorting
inverter, orders 3 to 25: the shipped default agrees with OpenDSS's `NeglectLoadY=Yes` to
1.05e-13 pu of nominal at 20 kV and 7.90e-13 pu on a 400 V cable feeder, and `load_style`
agrees with OpenDSS's negative-kW `Load` shunt to the same precision. The difference between
the two models on those feeders is 3.2e-7 and 4.9e-5 pu of nominal, and 0.9 % relative on the
node's THD. The choice is about the sign of the term rather than its present size.

What a real grid-following inverter presents is a filter impedance, and a synchronous machine
its subtransient reactance (OpenDSS `%R`/`%X`, `Xdpp`), neither of which is the
operating-point admittance the device shunt models and neither of which is a field of this
schema. A measured filter impedance is open work. Modern PWM inverters also emit little at low
orders while producing content between and above the integer orders, which the schema can carry
as data but the integer-order solve does not represent.

## Cross-tool comparison

| Capability | pandapower | OpenDSS | power-grid-model | pgml |
|---|---|---|---|---|
| DER element | `sgen`, a PQ injection | `PVSystem` | `sym_gen`, a PQ injection | `Generator` |
| Voltage-regulating bus | `gen` | `Generator model=3` | `voltage_regulator` | `Generator.voltage_regulation`, an exact residual row pair |
| Reactive limit | `enforce_q_lims`, switching PV to PQ | `min/maxkvar` | `q_min`/`q_max`, declared | PV to PQ switching, or the capability circle for a droop-controlled unit |
| Constant power factor | through a controller | `pf` | none | `ConstantPowerFactorControl` |
| Volt-VAr, Volt-Watt | `CharacteristicControl` | `InvControl`, `ExpControl` | none | control union |
| Storage | `storage`, PQ with external SoC | `Storage` with `StorageController` | none | `Storage` with external dispatch |
| Harmonics | no | yes, current source plus an operating-point shunt | no | yes, the same model, and no shunt on a generation device |
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
- power-grid-model has no harmonics and no storage. Its `voltage_regulator` component makes an
  existing generator or load a PV terminal, with optional reactive bounds. The pgml reader does
  not map it yet. Its only other regulator is a discrete transformer tap.

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
