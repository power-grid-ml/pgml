# DER modelling: PV systems, generators, and storage across tools

How pandapower, OpenDSS and power-grid-model model distributed-energy-resource (DER)
behaviour — PV inverters, synchronous/asynchronous generators, and battery storage —
contrasted with pgml's current model, followed by a differentiability-aware proposal for
extending pgml. The motivating gap: pgml has a single generic `Generator` appliance (a
fixed P/Q injection) and **no storage component**, so it cannot express the behaviour that
actually distinguishes these devices — inverter Volt-VAr / Volt-Watt control, MPPT,
reactive-power limits, and state-of-charge-driven dispatch.

This started as a design reference; the core of section 4 now ships.

**Implementation status.** Sections 4.1–4.4 and 4.6 (control→harmonic coupling) are
implemented and validated. The schema carries the `InverterControl` union + `Characteristic`
on `Generator`/`Storage`, the `Storage` appliance, and the `InjectionAppliance` base; the
differentiable control law lives in `assembly/_control.py` (folded into
`device_current_injections`, rides the IFT); storage SoC/dispatch is `scenarios/storage.py`.
Validated vs pandapower `CharacteristicControl` Q(V) (same equilibrium to ~1e-10 pu) and
OpenDSS `InvControl` VOLTVAR/VOLTWATT. Still open (optional): the voltage-regulating PV bus
(§4.5) and the harmonic load Norton shunt (§4.6, tracked as open work in `src/pgml/STATUS.md`).

---

## 1. The modelling questions

A "PV system", a "generator" and a "battery" differ along axes that a power-flow /
harmonic engine must represent explicitly. The libraries differ mainly in *how many* of
these axes they expose:

1. **Fundamental bus behaviour.** Is the device a fixed P/Q injection (a PQ node), a
   voltage-regulating source (a PV node, free Q to hold |V|), or the slack? Most DER on a
   distribution feeder are PQ; only large synchronous machines / the grid equivalent
   regulate voltage.
2. **Inverter control law.** A modern grid-following inverter does not hold P/Q constant —
   it follows an *autonomous* characteristic: constant power factor, power-factor-vs-power
   `cosφ(P)`, Volt-VAr `Q(V)`, Volt-Watt `P(V)`, or a combination, subject to a kVA /
   reactive-capability limit. These are **equation-defined** functions of the local voltage
   and the available active power.
3. **Active-power source.** A PV array's available power is set by irradiance and
   temperature through MPP tracking (`P = irradiance·Pmpp·f(T)`); a battery's is set by a
   dispatch decision; a synchronous genset's by a governor/schedule. The first is a smooth
   exogenous curve; the last is a **rule / schedule**.
4. **Internal / output impedance.** A grid-following PV inverter behaves as a near-ideal
   current source (high output impedance) at the fundamental and injects harmonics as a
   current source; a synchronous machine sits behind its sub-transient reactance `Xd"`.
   This matters mostly at harmonic frequencies (the Norton shunt).
5. **State and time-coupling.** A battery carries state of charge; dispatch couples
   timesteps (`SoC[t+1] = SoC[t] + η·P·Δt`) and may be governed by rules, price signals,
   profiles, or human behaviour — **not** by a closed-form equation of the present voltage.

Axes 1–4 are (piecewise) smooth functions and are candidates for the differentiable path.
Axis 5 is stateful/rule-based and is, in every tool surveyed, integrated **outside** the
per-snapshot solve.

---

## 2. Cross-library comparison

### 2.1 pandapower

pandapower is a NumPy/pandas + Newton-Raphson balanced (and three-sequence) load-flow
engine; **no harmonics**, **no autograd**.

- **`sgen` (static generator)** — the PV/wind/DER element. Modelled as a **constant-PQ
  injection in generator convention** (positive `p_mw` = injection); the connected bus
  stays a PQ bus. `scaling` multiplies P and Q; `sn_mva`, `k`, `rx`, `current_source`,
  `generator_type` affect only short-circuit (IEC 60909), not load flow. The `type` column
  accepts `'pv'` as a free label with no special treatment. [pp-create-sgen][pp-sgen-doc]
- **`gen` (voltage-controlled generator)** — a **PV bus**: fixed P, regulated `vm_pu`, free
  Q. `enforce_q_lims=True` adds the standard **PV→PQ switching loop**: a generator whose Q
  exceeds `min/max_q_mvar` is clamped, its bus converted to PQ, and the system re-solved.
  `ext_grid` is the slack (REF) bus. [pp-gen-doc][pp-qlims] pgml's pandapower converter
  DROPS this table by default and can, on request, approximate it with a steep Volt-VAr
  droop — see §4.5.
- **`asymmetric_sgen`** — per-phase P/Q; summed to a balanced injection in `runpp`, handled
  per phase in `runpp_3ph`. [pp-asym]
- **`storage`** — a **constant-PQ element at each snapshot**, in *load* convention
  (`p_mw > 0` = charging/consuming). `soc_percent`, `max_e_mwh`, `min_e_mwh` are stored but
  **never read by `runpp`** — the docs state SoC is not updated by the power flow. SoC
  integration is the caller's responsibility via the time-series module. [pp-storage]
- **Control loop.** `run_control` is an **outer fixed-point loop around `runpp`**: each
  controller's `control_step` writes setpoints, `runpp` solves, controllers re-check
  convergence, repeat until all converge (≤ `max_iter`, default 30). `ConstControl` writes
  profile values (1-step convergence); `CharacteristicControl` maps an input column (e.g.
  `res_bus.vm_pu`) through a piecewise-linear `Characteristic` to an output column (e.g.
  `sgen.q_mvar`), iterating until `|Δoutput| < tol`. This is exactly how a `Q(V)` / `cosφ(P)`
  law is realised. [pp-runcontrol][pp-characteristic]
- **Built-in DER curves.** Volt-VAr / Volt-Watt laws are assembled from the generic
  `CharacteristicControl` + `Characteristic`; pandapower 3.x additionally ships a
  dedicated `DERController` family (`QModelQV`, `QModelCosphiP`, PQV capability
  areas) — verify behaviour against the installed version before relying on it.
  [pp-characteristic]

### 2.2 OpenDSS

OpenDSS is the harmonic ground truth (phase-domain, current-injection). DER are Power
Conversion Elements with a fundamental dispatch model **and** a harmonic Norton model, and
control is realised by separate Control Elements iterating with the power-flow solve.

- **`PVSystem`** — combined PV array + inverter. Active power from MPP tracking:
  `P_DC = irradiance · Pmpp · P-TCurve(Temperature)`, then `P_AC = EffCurve(P_DC/kVA)·P_DC`.
  Inverter rating `kVA` bounds P and Q (capability curve). Reactive behaviour set by `pf`
  (constant power factor) or `kvar` (constant kvar). `%Cutin`/`%Cutout` gate the inverter
  on/off with **hysteresis**; `Vminpu`/`Vmaxpu` revert it to a constant-admittance model
  outside the band. Time variation via `daily`/`yearly`/`duty` LoadShapes (irradiance) and
  TShapes (temperature). [dss-pvsystem][dss-pvarray][dss-pvinv]
- **`Generator`** — the `model` integer selects the fundamental behaviour:
  `1` constant P,Q (default); `2` constant Z; `3` **constant P,|V| = PV bus** with
  `min/maxkvar` limits; `4` constant P, fixed Q; `5` constant P, fixed reactance;
  `6` user DLL; `7` **current-limited constant P,Q** (grid-following inverter / wind: limits
  terminal current to ≈1 pu below `Vminpu`). [dss-generator][dss-model]
- **`Storage`** — explicit `State` ∈ {IDLING, CHARGING, DISCHARGING}; nameplate `kWhrated`,
  `kWrated`, `%stored`, `%reserve`; one-way `%EffCharge`/`%EffDischarge` (round-trip ≈81%
  by default); `%IdlingkW`. SoC evolves over a QSTS solve
  (`E[t+Δt] = E[t] + P_ch·η_ch·Δt` charging, `E[t+Δt] = E[t] − P_dch·Δt/η_dch`
  discharging); a single snapshot is a **fixed P/Q injection** with no SoC update.
  `dispmode` ∈ {DEFAULT, FOLLOW, EXTERNAL, LOADLEVEL, PRICE}, and the **`StorageController`**
  element dispatches a fleet to a monitored quantity (PeakShave, Follow, Support, Time,
  Price, …). [dss-storage][dss-operation][dss-storagecontroller]
- **`InvControl` / `ExpControl`** — the inverter control elements. `InvControl` modes:
  `VOLTVAR` (piecewise-linear `Q(V)` XYcurve), `VOLTWATT` (`P(V)` limit), `DYNAMICREACCURR`
  (deadband reactive current), `WATTPF`, `WATTVAR`, and `CombiMode` `VV_VW` / `VV_DRC`.
  `ExpControl` is a continuous **exponential Volt-VAr** with an adaptive voltage reference
  (`VregTau`), modelling IEEE-1547 autonomous voltage regulation (no deadband). Both run in
  OpenDSS's **control-iteration loop**: solve power flow → read voltages → evaluate curve →
  push new P/Q setpoint → re-solve, up to `MaxControlIter`. Step damping (`deltaQ_Factor`,
  `deltaP_Factor`), averaging windows, and rate-of-change limits (`LPF`, `RISEFALL`)
  stabilise the loop. [dss-invcontrol][dss-vvfunc][dss-expcontrol][epri-smartinv]
- **Harmonics.** Each PCElement injects a harmonic **current source** scaled from the
  fundamental current and its `spectrum`: `|I_h| = (mag_h/mag_1)·|I₁|`,
  `∠I_h = ang_h + h·(∠I₁ − ang_1)`. Generators/PVSystem/Storage convert to a Thévenin
  behind an internal reactance (`Xd"` for machines; `%X` ≈ a few tens of % of kVA for the
  inverter) → Norton equivalent for the nodal solve. A well-designed UL-1741 inverter emits
  little at orders 3–13; significant content usually originates downstream. Loads add a
  Norton shunt split between series and parallel R-L (`%SeriesRL`, default 50%) unless
  `NeglectLoadY=yes` (pure current source). [dss-harmflow][dss-harmload][repo-harmonics]

### 2.3 power-grid-model (secondary data point)

Fundamental-frequency only; **no harmonics, no storage, no voltage control.**
`sym_gen`/`asym_gen` are injection components (a generator is a load with reversed sign)
parameterised by `LoadGenType` ∈ {const_power, const_impedance, const_current} — the same
ZIP-style voltage dependence pgml uses. `source` is the slack/Thévenin equivalent
(`u_ref`, `sk`, `rx_ratio`, `z01_ratio`). The only voltage regulator is
`transformer_tap_regulator` (discrete tap stepping); there is no PV bus, no Q-limit, no
inverter control, and no battery component — storage must be faked as a signed `sym_gen`
with no SoC. [pgm-components]

### 2.4 pgml (current state)

- **Elements.** One `Generator` appliance = a P/Q injection with a `LoadModel`
  (const_power / const_impedance / const_current / ZIP), connection (WYE/DELTA),
  per-phase split, and an optional harmonic `spectrum` / `spectrum_per_phase` +
  `HarmonicShuntModel` (Norton series/parallel R-L split, motor reactance). `consumer_type`
  (`"pv"`, `"battery"`, …) is a **free string used only as an ML categorical** — it does not
  change the physics. There is **no `Storage` component, no inverter control law, no
  reactive-power limit, and no PV (voltage-regulating) bus** (the slack is the `Source`).
  [grid_schema.py: `Generator`, `Load`, `HarmonicShuntModel`, `LoadModel`]
- **Fundamental solve.** const-P / ZIP via a current-injection fixed point or Newton, with
  **implicit-function-theorem (IFT) gradients** at the converged `V*`. The nodal balance is
  `Y_eff·V = I_slack − I_device(V)`; the residual the solver and the IFT backward share is
  `F_c(V) = Y_eff·V + I_device(V) − I_slack`. Devices enter only through `I_device(V)`:
  per element `S_eff(V) = S0·(z·r² + i·r + p)` with `r = |V_term|/|V0|`,
  `i_elem = conj(S_eff)/conj(V_term)`. [solver/power_flow.py:471–491; assembly/ybus.py:1461–1671]
- **Harmonics.** `solve_harmonic_flow` derives each device's fundamental current `I₁` from
  `V*` and injects per-order current sources using the OpenDSS-exact convention above;
  differentiable and batched. The inverter "internal impedance" at harmonics is the
  `HarmonicShuntModel` Norton shunt (matching OpenDSS), currently off by default
  (`include_load_shunt=False`, pure current source). [solver/harmonic_flow.py]
- **Differentiability.** Gradients flow grid params → Y → solve → outputs; the IFT backward
  builds `J = dR/dV` by autograd-differentiating **one** residual evaluation at `V*`. This
  is the lever for everything below: any extra smooth, V-dependent term added to
  `I_device(V)` is picked up by both the forward Newton **and** the IFT backward with no new
  adjoint code.

### 2.5 Summary

| Capability | pandapower | OpenDSS | power-grid-model | pgml (today) |
|---|---|---|---|---|
| PV/DER element | `sgen` (PQ) | `PVSystem` | `sym_gen` (PQ) | `Generator` (PQ) |
| PV (voltage-regulating) bus | `gen` | `Generator model=3` | — | — (slack only; a `net.gen` import is approximated by a Volt-VAr droop, §4.5) |
| Reactive-power limit | `enforce_q_lims` (PV→PQ) | `min/maxkvar` | — | — |
| Constant power factor | via control | `pf` | — | — |
| Volt-VAr `Q(V)` | `CharacteristicControl` | `InvControl VOLTVAR` / `ExpControl` | — | — |
| Volt-Watt `P(V)` | `CharacteristicControl` | `InvControl VOLTWATT` | — | — |
| MPPT (P from irradiance/T) | external profile | `PVSystem` P-T/Eff curves | external | external setpoint |
| Storage element | `storage` (PQ, ext. SoC) | `Storage` + `StorageController` | — | — |
| SoC in the solve | no (external) | QSTS only (external to snapshot) | — | — |
| Harmonics | no | yes (current-source Norton) | no | yes (current-source Norton) |
| Control ↔ solve coupling | outer loop around `runpp` | control-iteration loop | — | (would be inside the IFT residual) |
| Differentiable | no | no | no | **yes (IFT)** |

**The pattern every tool follows:** the device's *operating point* (P, Q, on/off, SoC) is
resolved by a control/dispatch layer that **iterates with** or sits **outside** the
power-flow solve; the solve itself sees a P/Q (or admittance) injection. pgml's distinctive
opportunity is that an *equation-defined* control law need not be an outer loop — it can be
folded **into** the residual `I_device(V)`, where the existing IFT machinery differentiates
it for free.

---

## 3. Behaviour taxonomy by differentiability

The user's framing is the right one: equation-defined behaviour is differentiable; rule- or
state-defined behaviour is not. Concretely:

### 3.1 Equation-defined → differentiable (fold into `I_device(V)`)

These are (piecewise) smooth maps from the local terminal voltage `V_term` and exogenous
setpoints to an injection `S(V)`. They enter the residual and are covered by the IFT with no
new backward code; the only requirement for `gradcheck` is C¹ smoothness (see §4.3 for
kinks).

- **MPPT active power.** `P = irradiance·Pmpp·f_PT(T)·η(P/kVA)` — a product of curves in
  *exogenous* inputs (irradiance, temperature). Differentiable w.r.t. those inputs and the
  ratings; for the solve it is simply the active setpoint `S0.real`.
- **Constant power factor / constant Q.** `Q = P·tanφ` or `Q = const`. Trivially smooth.
- **`cosφ(P)`.** Power-factor-vs-active-power characteristic (VDE-AR-N 4105). Smooth in `P`.
- **Volt-VAr `Q(V)`.** Reactive injection as a (piecewise-linear) function of `|V_term|`,
  clamped to the reactive capability. Smooth except at curve breakpoints and the clamp.
- **Volt-Watt `P(V)`.** Active curtailment above a voltage threshold — the user's "modern
  p(V) control". Same smoothness profile.
- **ZIP / constant-current limiting.** Already in pgml (`LoadModel`); OpenDSS `model=7`
  current limiting is a smooth saturation of `|I|`.
- **Synchronous PV bus + Q-limit.** Holding `|V|` with free Q, then saturating Q at a limit,
  is the const-|V| constraint plus a smooth clamp; pgml has no PV bus today, but the IFT
  residual can carry a `|V|`-regulation row (see §4.5).

### 3.2 State- or rule-defined → not differentiable as an equation (resolve off-tape)

These have memory, discrete switches, or exogenous logic; there is no closed-form `f(V)` to
differentiate, and forcing one is wrong.

- **State of charge.** A *recurrence* `SoC[t+1] = SoC[t] + η·P[t]·Δt`, not a function of the
  present voltage. The recurrence itself is differentiable (it is linear), but it is a
  time-series coupling, not part of any single snapshot.
- **Dispatch rules.** "Discharge when load > threshold", peak-shaving, price triggers,
  EXTERNAL/LOADLEVEL modes — `if/else` on aggregate state. Discontinuous; no useful gradient.
- **Deadbands, hysteresis, cut-in/cut-out, on/off.** `%Cutin`/`%Cutout`, Volt-VAr deadband,
  inverter trip at `Vmin/maxpu` — piecewise-constant switches.
- **Human behaviour / occupancy / stochastic profiles.** Sampled, not computed.

The robust treatment (and what pandapower/OpenDSS/pgm all do in effect): a **dispatch /
control resolution layer** produces a concrete operating-point `(P, Q)` per scenario/timestep,
which becomes pgml's `operating_point` (or `harmonic_injection`) input. Gradients then flow
w.r.t. the *resolved setpoint value* (a tensor — fully differentiable), **not** w.r.t. the
rule that produced it. For ML use cases that need a differentiable policy, the rule is
replaced by a *learned* differentiable surrogate (a small NN producing setpoints), which is
an ML-layer choice, not a physics equation.

---

## 4. Recommendations for pgml

Guiding principle, consistent with every surveyed tool and with pgml's architecture:

> **Separate operating-point *resolution* from the physics *solve*.** Smooth, voltage-local
> control laws fold into `I_device(V)` and ride the existing IFT. Stateful/rule-based
> dispatch is resolved upstream (in `scenarios` / an external service) into a setpoint that
> the solver consumes as data. The solver core never branches on a rule.

This keeps the two hard constraints intact: the differentiable path stays an
autograd-friendly residual; the non-differentiable logic never touches the tape.

### 4.1 Element taxonomy: extend `Generator`, add `Storage`

The behavioural differences (PV vs synchronous vs induction vs battery) are not separate
*topologies* — every one is a single-terminal P/Q injection with (a) an inverter/machine
control law, (b) a harmonic source/impedance, and (c) optionally state. So:

- **Keep one injection appliance (`Generator`)** and attach an optional **`control` block**
  (the inverter/machine law) alongside the existing `harmonic_model`. The device *kind* (PV,
  wind, CHP, synchronous, induction) is captured by which control + harmonic model is set,
  not by a new class per device. Promote `consumer_type` from a free string to a closed
  taxonomy used as an ML feature (it still must not drive physics implicitly).
- **Add a dedicated `Storage` appliance** (recommended over overloading `Generator`): a
  bidirectional injection with `p_setpoint` (signed: + discharge / − charge or the reverse,
  pick one convention and document it), `s_rated_va`, `kwh_rated`, `soc`, `eff_charge`,
  `eff_discharge`, and a `dispatch_ref` (external profile/rule id). At the snapshot it is a
  signed P/Q injection identical to a `Generator`; the extra fields are **inert in the
  solve** (exactly as pandapower `storage.soc_percent` is) and consumed only by the
  time-series layer. A separate type is clearer for ML stratification and for the SoC/dispatch
  service than a sign-flipped generator.

These require **schema changes**. Reuse existing machinery: `CurveParam` for characteristics, `FrequencyParam`
for any frequency dependence, the float/tensor duality so every new numeric field is
gradient-capable.

### 4.2 Inverter control block (the differentiable core)

Model the control law as a discriminated union (mirroring `Spectrum` / `FrequencyParam`),
each variant reusing `CurveParam` for its characteristic:

- `ConstantPowerFactor(pf)` / `ConstantReactivePower(q)`
- `CosPhiOfP(curve)` — `cosφ(P)`
- `VoltVar(q_v_curve, q_base)` — `Q(|V|)`, `q_base` ∈ {available-VAr, kVA rating}
- `VoltWatt(p_v_curve)` — `P(|V|)` curtailment
- `CombinedVoltVarVoltWatt(...)`
- an optional capability limit (kVA circle / PQ area) applied as a smooth clamp, with
  watt priority: the active power itself is clipped to the rating first (an oversized
  source cannot exceed the inverter VA rating through `P` alone), then `|Q|` is bounded
  by the remaining circle headroom

**Where it slots in.** `device_current_injections` already computes `S_eff(V_term)` from a
fixed `S0` and a ZIP voltage factor (`assembly/ybus.py:1648–1667`). A control law makes the
*base* power voltage-dependent: instead of a constant `S0`, compute
`S0 = P_ctrl(|V_term|, P_avail) + j·Q_ctrl(|V_term|, P)`, where `P_ctrl`/`Q_ctrl` evaluate
the curves. Everything downstream (`s_eff`, `i_elem = conj(s_eff)/conj(vt)`, the `Mᵀ`
scatter) is unchanged. Because the IFT backward differentiates the residual
`F_c(V) = Y_eff·V + I_device(V) − I_slack` **once at `V*`** via autograd
(`solver/power_flow.py:478–491`), the new `∂Q/∂|V|`, `∂P/∂|V|` terms appear in `J = dR/dV`
automatically — **no new adjoint, no iteration unrolling.** Forward Newton already uses that
same `J`, so convergence near the nose is preserved.

This is strictly better than the pandapower/OpenDSS outer control loop for the
differentiable use case: the control law becomes part of the implicit function `V*(θ)`, so
gradients of any output (voltages, currents, THD, loss) w.r.t. the curve parameters,
ratings, and irradiance flow directly — enabling gradient-based tuning of inverter settings,
parameter recovery of an unknown `Q(V)` slope, and physics-guided ML targets.

### 4.3 Handling non-smooth pieces on the differentiable path

Volt-VAr/Volt-Watt curves, deadbands, and capability clamps have kinks. Two-track approach:

- **Forward (accuracy):** evaluate the exact piecewise-linear curve (`CurveParam` linear
  interpolation) and hard clamps — matches OpenDSS/pandapower bit-for-bit at the operating
  point.
- **Backward (gradients):** for `gradcheck` (float64) and stable training, provide a
  **smooth variant** of each kink: a soft deadband / smooth breakpoint
  (`tanh`/`softplus`-blended segments or `CurveParam`'s cubic interpolation), and a smooth
  saturation (`s·tanh(x/s)` or `softplus`-based) for kVA / Q-limit clamps. At a curve
  breakpoint the exact map is C⁰ with a subgradient; the smoothing makes it C¹ so the IFT
  Jacobian is well defined. Document the smoothing width as a model parameter (→ 0 recovers
  the hard curve). This mirrors how pgml already guards `0/0` divides with `torch.where`:
  keep the forward correct, keep the gradient finite.
- **Discrete switches that must stay hard** (inverter trip, cut-in/cut-out): treat as
  **scenario state**, not an in-solve branch — resolve on/off upstream into whether the
  device injects at all (§4.4). A straight-through or sigmoid-gated relaxation is available
  if a gradient through the gate is genuinely needed for ML, but it is opt-in, not the
  default physics.

### 4.4 Storage, SoC, and rule-based dispatch (off-tape, by design)

Follow pandapower/OpenDSS exactly: **the snapshot solve sees a fixed signed P/Q injection;
SoC and dispatch live in the time-series layer.**

- **Snapshot.** `Storage` contributes `S0 = ±(P + jQ)` to `I_device(V)` like a generator —
  fully differentiable w.r.t. the (possibly tensor) setpoint value.
- **SoC / dispatch.** Implement in `scenarios` (the natural home for time-series and the
  batching layer). A dispatch resolver maps `(profile | rule | price |
  human-behaviour sample, SoC[t])` → `P[t]`, then advances `SoC[t+1] = SoC[t] + η·P[t]·Δt`
  with the charge/discharge efficiency split (OpenDSS's equations). The **rule** is ordinary
  Python control flow (no autograd); the **resulting `P[t]` tensor** is what the solver
  differentiates. If a gradient *through time* is ever needed (e.g. learning a dispatch
  policy end-to-end), the SoC recurrence is linear and can be kept on the tape while the
  decision rule is replaced by a differentiable policy — but the default is rule-resolved,
  off-tape dispatch, which matches every reference tool and keeps the core clean.
- **MPPT** is the same shape: irradiance/temperature → `Pmpp` curve → active setpoint,
  resolved per scenario; differentiable w.r.t. irradiance if a sensitivity is wanted, else a
  plain data input.

### 4.5 The PV (voltage-regulating) bus: an approximation today, the real fix later

If voltage-regulating DER/machines are needed (OpenDSS `model=3`, pandapower `gen`), the
real fix is a `|V|`-regulation mode: replace that terminal's power-balance row in the real
residual with `|V_term| − V_set = 0` and let Q be the free variable, with a smooth
Q-saturation for the `min/max_q` limit (the smooth analogue of pandapower's PV→PQ
switching). This stays inside the same `[2N,2N]` IFT residual/Jacobian, so it remains
differentiable and batched, but it needs both a solver change and a schema field to carry
`V_set`. Lower priority than §4.2 for distribution-feeder DER, which are overwhelmingly
PQ/inverter-curve controlled.

**What exists today: a droop approximation, opt-in, in the pandapower converter.** A
Volt-VAr characteristic centred on the generator's `vm_pu` and saturating at its reactive
limits reproduces PV-bus behaviour in the limit of an infinite slope — the bus is held
where the droop's reactive output balances the network, i.e. off the setpoint by
`Q / (slope · Q_base)` per unit, and the reactive limit is enforced by the capability
bound instead of by a discrete bus-type switch.
`pgml.convert.pandapower.to_grid(net, gen_mode=GenMode.VOLT_VAR_APPROX)` builds exactly
that (`gen_volt_var_slope_pu` is the steepness; default 500, i.e. one full reactive base
per 0.002 pu of voltage). Because the law is an ordinary inverter control it enters
`I_device(V)` and is differentiated by the same IFT backward — no new machinery.

The approximation's limit is **conditioning, not steady-state fidelity**. Outside the
`1/slope`-wide band `dQ/d|V|` is exactly zero, so an iterate that starts far from the
setpoint sees no voltage-control feedback: on heavily loaded transmission benchmarks
(`case39`, `case118`) the const-Z warm start is far enough out that Newton lands on the
collapsed low-voltage branch — a genuine second solution of the *approximated* system that
the exact `|V| − V_set = 0` row would exclude by construction. Measured against a live
`pp.runpp(..., enforce_q_lims=True)`, the per-bus |V| deviation falls as `1/slope`
(`case57`: 4.3e-2 pu at slope 5 → 2.3e-4 pu at slope 2000), but the usable steepness caps
at ~5 on `case118`, and on `case39` every steepness converges *silently* onto the collapsed
branch (0.49 pu off, all nine machines pinned at their reactive limit). The approximation
is therefore good for small and moderately loaded networks and for differentiable
sensitivity studies, and NOT a faithful way to import transmission benchmarks — those need
the residual-row fix above (and `shunt` conversion). Always sanity-check the converged
voltage profile against the source network's `res_bus`. Full record:
`src/pgml/convert/pandapower/CONTEXT.md`.

### 4.6 Harmonics coupling (already most of the way there)

pgml's harmonic model derives each device's injection from its **fundamental** current `I₁`.
A control law changes the fundamental operating point → changes `I₁` → automatically rescales
the harmonic current sources (`|I_h| = (mag_h/mag_1)|I₁|`). So once §4.2 lands, the harmonic
spectrum tracks the control state with no extra work — and it stays differentiable through
`I₁`. Two refinements (both tracked as open work in `src/pgml/STATUS.md`):

- The PV inverter "internal impedance" the user notes is the harmonic **Norton shunt**
  (`HarmonicShuntModel`, OpenDSS `%X`); finishing `include_load_shunt=True` lets a
  grid-following inverter present its high output impedance / damping at harmonics.
- Modern PWM inverters emit little at low orders but can produce **inter/supraharmonics**;
  the existing `Spectrum` (non-integer orders allowed) covers this as data.

### 4.7 Validation targets

- Fundamental control vs **pandapower** `CharacteristicControl` (`Q(V)`, `cosφ(P)`) and
  `enforce_q_lims` (Q-limit), on IEEE-33 / CIGRE LV.
- Inverter control + harmonics vs **OpenDSS** `InvControl` (VOLTVAR, VOLTWATT, VV_VW) and
  `ExpControl`, and `Generator model=3/7`, `Storage` + `StorageController` for dispatch.
- Differentiability gate: float64 `gradcheck` of `V*` (and THD) w.r.t. the `Q(V)`/`P(V)`
  curve slope, the kVA limit, and irradiance through the IFT path (smoothed variants).
- GPU device/dtype parity for every new injection term.

### 4.8 Suggested phasing

1. Inverter control block on `Generator` — constant-PF, `cosφ(P)`, `Q(V)`, `P(V)` folded
   into `I_device(V)` with smooth-backward variants (§4.2–4.3). Highest value, smallest core
   change, fully differentiable. *(shipped)*
2. `Storage` appliance as a signed PQ injection (snapshot only) + SoC/dispatch resolver in
   `scenarios` (§4.4). *(shipped)*
3. PV-bus regulation mode (§4.5) and harmonic Norton-shunt completion (§4.6) as demand
   arises. *(open — a Volt-VAr droop approximation of pandapower's `gen` ships in the
   converter meanwhile)*

Each step is gated by the differentiability + GPU tests and a reference comparison before it
is considered done.

---

## References

pandapower
- [pp-create-sgen] `pandapower.create.create_sgen`, source `pandapower/create.py`; bus
  injection `pandapower/build_bus.py::_calc_pq_elements_and_add_on_ppc`.
- [pp-sgen-doc] pandapower elements — static generator:
  https://pandapower.readthedocs.io/en/latest/elements/sgen.html
- [pp-gen-doc] pandapower elements — generator:
  https://pandapower.readthedocs.io/en/latest/elements/gen.html
- [pp-qlims] `pandapower/pf/run_newton_raphson_pf.py::_run_ac_pf_with_qlims_enforced`;
  `runpp(enforce_q_lims=...)`.
- [pp-asym] `pandapower.create.create_asymmetric_sgen`; `pandapower/pf/runpp_3ph.py`.
- [pp-storage] pandapower elements — storage (SoC not updated by power flow):
  https://pandapower.readthedocs.io/en/latest/elements/storage.html
- [pp-runcontrol] pandapower control loop:
  https://pandapower.readthedocs.io/en/latest/control/run.html
- [pp-characteristic] `pandapower.control.controller.CharacteristicControl` +
  `pandapower.control.util.characteristic.Characteristic`:
  https://pandapower.readthedocs.io/en/latest/control/controller.html

OpenDSS (EPRI manual, DSS-Extensions, source)
- [dss-pvsystem] https://opendss.epri.com/PVSystem.html
- [dss-pvarray] https://opendss.epri.com/PVarrayproperties.html
- [dss-pvinv] https://opendss.epri.com/PVinverterproperties.html ;
  capability curve https://opendss.epri.com/InverterCapabilityCurve.html ;
  format ref https://dss-extensions.org/dss-format/PVSystem.html
- [dss-generator] https://opendss.epri.com/Generator.html
- [dss-model] https://opendss.epri.com/Model.html
- [dss-storage] https://opendss.epri.com/Storage.html
- [dss-operation] https://opendss.epri.com/Operation.html
- [dss-storagecontroller] https://opendss.epri.com/StorageController.html ;
  dispatch modes https://opendss.epri.com/DispatchModes1.html
- [dss-invcontrol] https://opendss.epri.com/InvControl.html ;
  batch/property ref https://dss-extensions.org/dss_capi/classdss_1_1obj_1_1InvControlBatch.html
- [dss-vvfunc] https://opendss.epri.com/Propertiesofsmartinvertervolt-va.html ;
  calculation https://opendss.epri.com/Calculationofthesmartinverterfun.html
- [dss-expcontrol] https://opendss.epri.com/ExpControl.html
- [epri-smartinv] EPRI TechNote 3002002271, "Smart Inverter Function Modeling in OpenDSS".
- [dss-harmflow] https://opendss.epri.com/HarmonicFlowAnalysis.html
- [dss-harmload] https://opendss.epri.com/HarmonicsLoadModeling.html
- source: `Source/PCElements/{PVsystem,generator,Storage}.pas`,
  https://github.com/tshort/OpenDSS

power-grid-model
- [pgm-components] LF Energy power-grid-model component reference:
  https://power-grid-model.readthedocs.io/en/stable/user_manual/components.html

Interconnection standards defining the control characteristics (Volt-VAr, Volt-Watt,
`cosφ(P)`, reactive capability): IEEE 1547-2018; EN 50549-1/-2; VDE-AR-N 4105.

This repository
- [repo-harmonics] [OpenDSS harmonics](references/opendss/harmonics.md) (empirically verified
  spectrum/phase convention and Norton shunt) and the [OpenDSS brief](references/opendss/index.md).
- pgml model: `src/pgml/schemas/grid_schema.py` (`Generator`, `Storage`, `InverterControl`,
  `Characteristic`, `Load`, `HarmonicShuntModel`, `LoadModel`);
  `src/pgml/assembly/_control.py` (control laws); `src/pgml/assembly/ybus.py::device_current_injections`;
  `src/pgml/solver/power_flow.py` (IFT residual); `src/pgml/solver/harmonic_flow.py`;
  `src/pgml/scenarios/storage.py` (SoC / dispatch).
- Open work: `src/pgml/STATUS.md` (frequency-dependent device models — incl. the harmonic
  load Norton shunt — and the optional DER/storage extensions).
