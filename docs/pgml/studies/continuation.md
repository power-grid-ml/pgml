# Voltage stability: P-V and Q-V curves

{func}`pgml.studies.pv_curve` answers how far the loading of a grid can grow before the
fundamental power flow has no solution, and which bus collapses first.
{func}`pgml.studies.qv_curve` answers how much reactive power one bus needs, or can
absorb, at each voltage. Both solve the phase-domain fundamental power flow of pgml, so
unbalanced grids, single-phase and ZIP loads, inverter control laws, voltage-regulating
generators with reactive limits and fused zero-impedance branches are all handled the same
way as in {func}`~pgml.solver.solve_power_flow`.

```python
from pgml.grids import ieee33_geometry_grid
from pgml.studies import pv_curve, qv_curve

grid, _ = ieee33_geometry_grid()
curve = pv_curve(grid)                        # every load scaled by λ, λ = 1 is the base
print(f"λ* = {curve.lambda_max:.3f} at node {curve.critical_node} "
      f"phase {curve.critical_phase}")
print(f"P at the nose = {curve.p_total_w[curve.nose_index] / 1e6:.2f} MW")

qv = qv_curve(grid, node_id=18, v_range_pu=(0.3, 1.1), steps=33)
print(f"reactive margin {qv.q_margin_var / 1e6:.2f} Mvar at {qv.v_at_min_pu:.2f} pu")
```

```text
λ* = 3.623 at node 18 phase a
P at the nose = 13.46 MW
reactive margin 2.72 Mvar at 0.51 pu
```

## The P-V curve

`λ` multiplies the complex power of the selected appliances at the solved voltage: every
in-service load by default, every generator with `direction="generation"`, or any set of
loads, generators and storage units given by `selector` (ids or a predicate). Every
component of a ZIP law scales with it. `λ = 1` is the base operating point, which is the
grid's nameplate values or `config.operating_point`. The other appliances stay at their
base values.

A plain sweep of `λ` cannot pass the nose of the curve: the Jacobian of the power flow
becomes singular there, and every solve beyond it fails. `pv_curve` uses a
predictor-corrector continuation instead, which follows the solution curve through the nose
onto its lower branch:

- **Coordinates.** The state is `x = [Re V; Im V]`, measured in per unit of each row's
  line-to-neutral rated voltage, and `λ` is measured relative to `max(1, |λ|)`. The arc
  length combines both, so the fold has the same geometry whether the nose lies at `λ* = 2`
  or at `λ* = 700`.
- **Predictor.** The unit tangent of the curve, oriented along the previous one.
- **Corrector.** Newton on the pseudo-arc-length system `R(x, λ) = 0`,
  `tᵀ(z − z_pred) = 0`. Its Jacobian stays regular at the nose. The residual, the per-unit
  convergence test and the equilibration are those of `solve_power_flow`.
- **Step control.** The step grows or shrinks with the corrector's iteration count. A failed
  correction, an over-long one, or a tangent that turns by more than 0.25 rad within the
  step halves it.
- **Nose.** The sign change of the tangent's `λ` component. The fold is located by bisection
  on the arc length and inserted as a curve point (`nose_index`, `lambda_max`).
- **Reactive limits.** When a regulating generator reaches a reactive limit, the switch is
  located on the arc, the generator becomes a fixed-Q injection and the point is solved
  again (`events`). If `λ` can only decrease after the switch, the nose is limit-induced
  (`nose_kind="limit_induced"`) rather than a fold of the Jacobian.

The trace starts with an ordinary power flow at `lambda_start`, which defaults to 1. When
the base itself does not solve, start at `lambda_start=0`, with the selected appliances
switched off. The trace ends at the first of these:

- the nose, with `stop_at_nose=True`;
- `lambda_max`, reached before a nose;
- the lowest phase voltage falling below `v_stop_pu`, 0.3 pu by default, on either branch;
- `λ` returning below `lambda_start` on the lower branch;
- `max_steps`;
- a corrector failure at `step_min`, or no solution right after a reactive-limit switch
  (`"switch_failed"`). These two stops, and `max_steps`, report `converged=False`.

`critical_node` and `critical_phase` name the row with the lowest voltage at the nose.
Phases are reported as the schema value of {class}`~pgml.schemas.grid_schema.Phase`
(`"a"`, `"b"`, `"c"`, `"n"`) in `phases`, `critical_phase` and `QVCurve.phases`.
{func}`pgml.solver.loadability_limit` gives a quicker estimate of the same limit: it bisects
on whether a plain Newton solve converges, which gives a lower bound on `λ*`, and it
localizes the collapse through the singular vectors of the Jacobian.

## The Q-V curve

A fictitious source at the bus holds every non-neutral phase at a voltage setpoint (the
phase-to-ground magnitude) with its own free reactive power and zero active power. It is
the per-phase regulating terminal of the solver with unbounded reactive limits. The
setpoint sweeps `steps` values over `v_range_pu`, starting at the one nearest the bus's base
voltage and walking outward, each solve warm-started from its neighbour. Fixing the voltage
removes the fold, so each point is a regular power flow, even on the unstable side of the
curve.

`q_var` is the reactive power the source injects, with positive meaning support delivered to
the grid. The minimum of the curve is refined by a parabola through the lowest sample and its
two neighbours. `q_margin_var = −min(q_var)` is the reactive load the bus can take before
its voltage collapses. When it is negative, the bus needs that much support to operate at
all. `minimum_in_range=False` warns that the minimum lies at the end of the converged range.
Setpoints without a solution are reported with `converged=False` and `NaN`. One cause is a
load that cannot be carried at that voltage. Another is a voltage the source cannot reach
through a resistive line.

## Results and serialization

{class}`~pgml.studies.PVCurve` and {class}`~pgml.studies.QVCurve` hold host numpy arrays.
`to_dict()` returns plain lists with the units in the key names (`lambdas`, `p_total_w`,
`v_pu` with its row index `node_ids` and `phases`, `q_var`, `q_phase_var`, `q_margin_var`),
with `None` in place of a missing value, and `from_dict()` restores the object; it refuses a
dict whose `"study"` key names another study. `PVCurve.voltage_pu(node_id, phase)` returns
one node's voltage along the curve; `phase` is a `Phase` or its letter, in either case.

## Validation

All measurements are at complex128 (`tests/studies/test_continuation.py`).

| Check | Reference | Measured agreement | Tolerance in the test |
|---|---|---|---|
| Nose of a two-bus feeder with a constant-power load, `tan φ ∈ {0, 0.5, −0.3}`, lossless and with `R/X = 0.6` | `P_max = E² / (2(R + X tan φ + \|Z\| sec φ))` | relative error of `λ*` ≤ 7.8e-14 | 1e-6 |
| Every curve point on both branches | the load the solved voltage carries, from the closed-form quadratic (the quartic residual for the capacitive load) | ≤ 3.5e-14 of the base load (quartic residual ≤ 7.5e-16 of `E⁴`) | 1e-9 |
| Reverse power flow, `direction="generation"` | `P_max = E² / (2(\|Z\| − R))` | 1.2e-13 | 1e-6 |
| Reactive-limit switch of a regulating generator | `cos δ = 1 − q_max X / E²` | 1.1e-7 (arc-length bisection to 1e-6) | 1e-5 |
| Nose after that switch | nose of the fixed-Q curve | 1.2e-14 | 1e-6 |
| Limit-induced nose (`q_max > E²/(2X)`) | `λ*` at the switch | 1.4e-7 | 1e-5 |
| Unbalanced, mutually coupled three-phase feeder with single-phase, ZIP and generating appliances and a fused switch, 9 points on the upper branch below `0.97 λ*` | `solve_power_flow(method="newton")` at the same `λ` | ≤ 2.9e-15 pu | 1e-8 pu |
| Q-V curve of the lossless two-bus feeder, 61 setpoints | `Q(V) = Q_L + V²/X − √((EV/X)² − P²)` | relative error ≤ 2.3e-14 | 1e-8 |
| Q-V minimum | `V² = X²P²/E² + E²/4` | `V` within 5.0e-5 pu, margin within 9.0e-7 relative | 2e-3 pu, 1e-4 |
| complex64, two-bus nose | closed form | 1.2e-7 | 1e-3 |

The accuracy of the Q-V minimum follows the setpoint spacing. With the parabola refinement it
was 2.0e-4 pu and 4.4e-6 relative at 31 setpoints, and 1.2e-5 pu and 2.1e-7 relative at 121.
On the IEEE 33-bus example above, {func}`~pgml.solver.loadability_limit` with
`bisect_tol=1e-4` reports 3.6227, just below the `λ* = 3.6230` of the continuation, as a
lower bound should.

## Cost and limits

- Each corrector iteration builds the dense real Jacobian `[2N, 2N + 1]` by automatic
  differentiation and factors it, so memory grows with `N²` for `N` node-phase rows.
  Measured on a CPU with PyTorch 2.13, one Jacobian took 0.04 s at 204 rows and 0.12 s at
  606 rows, and the whole IEEE 33-bus curve above took 2.6 s.
- One grid and one operating point per call. The studies are not batched over scenarios.
- Not differentiable: the studies run detached and return numpy arrays.
- The Q-V source holds phase-to-ground magnitudes. On a node with a neutral conductor these
  differ from the line-to-neutral magnitudes by the neutral-point voltage.
