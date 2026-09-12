# Why differentiable

A harmonic power flow is a complex linear solve per harmonic order, and the fundamental
load flow is a fixed point with a known Jacobian. Both have a cheap adjoint. `pgml` keeps
the whole chain on one autograd tape, so the derivative of any scalar you compute from a
result, with respect to every grid parameter that entered it, costs about one extra solve.

That one property covers four jobs that otherwise need four different tools.

| Question | What the gradient does |
|---|---|
| What are my line parameters really? | Fits them to measurements |
| Which load moves this voltage? | Returns every sensitivity from one backward pass |
| How do I clear this violation? | Points at the setpoint change that removes it |
| Which asset should I reinforce? | Ranks candidates by the benefit of changing them |

## How gradients get in and out

Every physical field of a grid accepts a `torch.Tensor` in place of a float. Assign one and
it becomes an ordinary autograd leaf. Nothing else changes, the same `Grid` still assembles,
solves and serialises.

```python
p = torch.tensor(float(load.p_nom_w), dtype=torch.float64, requires_grad=True)
load.p_nom_w = p
```

The results come back as tensors on the tape. `state.node_voltages()`, `state.voltage()`,
`state.branch_currents()`, `state.branch_flows()` and `state.thd()` are all differentiable.
Call `.backward()` on any scalar built from them.

A finite-difference sweep costs one solve per parameter. One backward pass costs one solve
in total and returns the whole gradient, which is what makes the last two examples below
practical on a grid with hundreds of lines.

The four examples run on the CIGRE LV benchmark network, 44 nodes across three feeders.
They need the `convert` extra, they run on a laptop CPU, and each finishes in a few
seconds.

## 1. Line parameters from noisy voltage measurements

Cable data is often wrong. Age, temperature and undocumented replacements move the real
resistance away from the datasheet value. With voltage measurements at the nodes, the
resistance is a parameter to fit rather than a number to trust.

The example takes the first cable section of each feeder, gives it a true resistance that
differs from the datasheet, and generates measurements with 0.1 % noise. It then starts from
the datasheet value and lets the gradient find the correction factors.

```python
import torch
import pgml
from pgml.assembly import base_voltage_per_row
from pgml.grids import cigre_lv_full_grid
from pgml.schemas.grid_schema import Line

torch.manual_seed(0)
grid, _ = cigre_lv_full_grid()
config = pgml.SimulationConfig(calculation="power_flow")
base = base_voltage_per_row(grid)            # per-unit reference, one entry per node row
lines = {b.id: b for b in grid.branches if isinstance(b, Line)}
unknown = [45, 62, 63]                       # first cable section of each of the 3 feeders
datasheet = torch.tensor(
    [float(lines[i].series_resistance_ohm_per_m[0][0]) for i in unknown],
    dtype=torch.float64,
)

def set_factors(factor):
    for k, i in enumerate(unknown):
        lines[i].series_resistance_ohm_per_m = (datasheet[k] * factor[k]).reshape(1, 1)

def voltages_pu():
    return pgml.simulate(grid, config).node_voltages().abs() / base

true_factor = torch.tensor([1.30, 0.85, 1.15], dtype=torch.float64)
set_factors(true_factor)
measured = voltages_pu().detach()
measured = measured * (1.0 + 0.001 * torch.randn_like(measured))   # 0.1 % meter noise

factor = torch.ones(3, dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([factor], lr=0.08)
for step in range(60):
    set_factors(factor)
    loss = ((voltages_pu() - measured) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

print("true factors     ", [round(float(x), 3) for x in true_factor])
print("recovered factors", [round(float(x), 3) for x in factor.detach()])
print(f"residual          {loss.item():.1e} pu^2, noise floor 1.0e-06 pu^2")
```

```text
true factors      [1.3, 0.85, 1.15]
recovered factors [1.307, 0.845, 1.175]
residual          1.1e-06 pu^2, noise floor 1.0e-06 pu^2
```

Sixty solves recover three resistances to about 2 %. The residual sits at the measurement
noise floor, which is where the fit should stop.

Accuracy is bounded by information, not by the optimiser. These three sections carry the
whole current of their feeder, so their voltage drop is large compared with the noise. A
short lateral section serving one house produces a drop smaller than 0.1 % of the nominal
voltage, and its resistance is then not identifiable from a single snapshot. Fitting such a
section needs either quieter measurements or several operating points.

## 2. Every load sensitivity from one backward pass

Planning questions often read as "which load is holding this node down". The gradient of one
node voltage with respect to the active power of every load answers all of them at once.

```python
import torch
import pgml
from pgml.grids import cigre_lv_full_grid
from pgml.schemas.grid_schema import Load, Phase

grid, _ = cigre_lv_full_grid()
config = pgml.SimulationConfig(calculation="power_flow")
loads = [a for a in grid.appliances if isinstance(a, Load)]

p = torch.tensor([float(a.p_nom_w) for a in loads], dtype=torch.float64, requires_grad=True)
for load, p_i in zip(loads, p):
    load.p_nom_w = p_i                       # every load power is now one tensor's entry

watched = 44                                 # last node of the residential feeder
v_base = float(next(n.u_rated_v for n in grid.nodes if n.id == watched))
v_pu = pgml.simulate(grid, config).voltage(watched, Phase.A).abs()[0] / v_base
v_pu.backward()                              # one backward pass for all 15 sensitivities

print(f"voltage at node {watched}: {v_pu:.4f} pu")
print("largest sensitivities, pu per kW of extra load:")
order = torch.argsort(p.grad)
for k in order[:4]:
    print(f"  load at node {loads[k].node:>2}   {1e3 * p.grad[k]:+.5f}")

k = int(order[0])                            # check the largest against a finite difference
step = 1000.0
for i, load in enumerate(loads):
    load.p_nom_w = float(p.detach()[i]) + (step if i == k else 0.0)
v2 = pgml.simulate(grid, config).voltage(watched, Phase.A).abs()[0] / v_base
print(f"finite difference at node {loads[k].node}: {1e3 * (v2 - v_pu.detach()) / step:+.5f}")
```

```text
voltage at node 44: 0.9234 pu
largest sensitivities, pu per kW of extra load:
  load at node 44   -0.00130
  load at node 43   -0.00079
  load at node 41   -0.00049
  load at node 42   -0.00048
finite difference at node 44: -0.00130
```

The gradient agrees with the finite difference to the digits printed. The ranking is the
electrical distance along the feeder, which is the result you would expect, obtained here
without any sweep. The cost stays one backward pass whether the grid has 15 loads or 15000.

Because the sensitivities are exact derivatives at the operating point, they are valid for
small changes. A load step large enough to move the operating point needs a new solve.

## 3. Curtailment that removes an overvoltage

A sunny midday with rooftop PV at 150 % of the feeder peak load pushes the network above
1.05 pu. The question is which inverters to curtail, and by how little.

The violation is written as a differentiable penalty. Its gradient with respect to the
curtailment of each unit says how much each one contributes, so the descent naturally
concentrates on the inverters that matter and stops as soon as the network is inside limits.

```python
import torch
import pgml
from pgml.assembly import base_voltage_per_row
from pgml.grids import add_pv_systems, cigre_lv_full_grid
from pgml.schemas.grid_schema import Generator, Load

grid, _ = cigre_lv_full_grid()
config = pgml.SimulationConfig(calculation="power_flow")
base = base_voltage_per_row(grid)
add_pv_systems(grid, fraction=1.0)

for load in [a for a in grid.appliances if isinstance(a, Load)]:
    load.p_nom_w = 0.2 * float(load.p_nom_w)        # midday, light load
    load.q_nom_var = 0.2 * float(load.q_nom_var)
pv = [a for a in grid.appliances if isinstance(a, Generator)]
for unit in pv:
    unit.p_nom_w = 3.0 * float(unit.p_nom_w)        # 150 % PV penetration
rated = torch.tensor([float(u.p_nom_w) for u in pv], dtype=torch.float64)

def voltages_pu(curtail):
    for unit, c, r in zip(pv, curtail, rated):
        unit.p_nom_w = (1.0 - c) * r
    return pgml.simulate(grid, config).node_voltages().abs() / base

LIMIT = 1.05
curtail = torch.zeros(len(pv), dtype=torch.float64, requires_grad=True)
print(f"{len(pv)} PV units, {rated.sum() / 1e3:.0f} kW rated")
print(f"highest voltage before: {voltages_pu(curtail.detach()).max():.4f} pu")

opt = torch.optim.Adam([curtail], lr=0.02)
for step in range(60):
    excess = torch.relu(voltages_pu(curtail) - LIMIT)
    if float(excess.detach().max()) == 0.0:
        break
    opt.zero_grad()
    (excess**2).sum().backward()
    opt.step()
    with torch.no_grad():
        curtail.clamp_(0.0, 1.0)

lost = curtail.detach() * rated
print(f"highest voltage after:  {voltages_pu(curtail.detach()).max():.4f} pu, {step} steps")
print(f"curtailed: {lost.sum() / 1e3:.0f} kW of {rated.sum() / 1e3:.0f} kW")
for k in torch.argsort(lost, descending=True)[:3]:
    print(f"  PV at node {pv[k].node:>2}: {100 * curtail.detach()[k]:.0f} % curtailed")
```

```text
15 PV units, 1030 kW rated
highest voltage before: 1.0716 pu
highest voltage after:  1.0498 pu, 17 steps
curtailed: 227 kW of 1030 kW
  PV at node  3: 26 % curtailed
  PV at node 25: 24 % curtailed
  PV at node 17: 28 % curtailed
```

Seventeen solves bring the network back inside the limit at 22 % curtailment. Nothing tells
the optimiser which units are electrically close to the violated nodes. The gradient of the
power flow carries that information.

This is a demonstration of the mechanism, not a dispatch product. A real curtailment scheme
adds fairness between units, reactive power, inverter capability curves and a cost term.
Each of those is another differentiable term in the same objective.

## 4. Which line to reinforce

Reinforcing a feeder means replacing a cable with a larger cross section, which lowers its
impedance. Ranking candidates by trial and error costs one solve per candidate. The gradient
of a stress metric with respect to a per-line impedance scale ranks all of them at once.

```python
import torch
import pgml
from pgml.assembly import base_voltage_per_row
from pgml.grids import cigre_lv_full_grid
from pgml.schemas.grid_schema import Line

grid, _ = cigre_lv_full_grid()
config = pgml.SimulationConfig(calculation="power_flow")
base = base_voltage_per_row(grid)
lines = [b for b in grid.branches if isinstance(b, Line)]
r0 = torch.tensor([float(b.series_resistance_ohm_per_m[0][0]) for b in lines], dtype=torch.float64)
l0 = torch.tensor([float(b.series_inductance_h_per_m[0][0]) for b in lines], dtype=torch.float64)

def stress(scale):
    """Summed squared undervoltage below 0.95 pu, with every line impedance scaled."""
    for k, line in enumerate(lines):
        line.series_resistance_ohm_per_m = (r0[k] * scale[k]).reshape(1, 1)
        line.series_inductance_h_per_m = (l0[k] * scale[k]).reshape(1, 1)
    v = pgml.simulate(grid, config).node_voltages().abs() / base
    return (torch.relu(0.95 - v) ** 2).sum()

scale = torch.ones(len(lines), dtype=torch.float64, requires_grad=True)
j = stress(scale)
j.backward()                                  # one pass ranks all 37 lines

print(f"voltage stress {j:.5f} over {len(lines)} lines")
rank = torch.argsort(scale.grad, descending=True)
for k in rank[:3]:
    print(f"  line {lines[k].id} ({lines[k].from_node}->{lines[k].to_node})  dJ/dscale {scale.grad[k]:+.5f}")

for label, k in (("top ranked", int(rank[0])), ("10th ranked", int(rank[9]))):
    trial = torch.ones(len(lines), dtype=torch.float64)
    trial[k] = 0.5                            # halve that line's impedance and re-solve
    print(f"  {label} line {lines[k].id} halved -> stress {stress(trial):.5f}")
```

```text
voltage stress 0.01581 over 37 lines
  line 63 (25->26)  dJ/dscale +0.01292
  line 64 (26->27)  dJ/dscale +0.01292
  line 71 (27->34)  dJ/dscale +0.00596
  top ranked line 63 halved -> stress 0.01039
  10th ranked line 76 halved -> stress 0.01483
```

The first two lines are in series and carry the same current, so their gradients are equal to
the digits printed and the order between them is a tie-break.

The two head sections of the loaded feeder come out on top, ahead of the lateral that feeds
the worst node. Halving the impedance of the top-ranked line removes 34 % of the stress.
Halving the tenth-ranked line removes 6 %. The ranking from a single backward pass matches
what 37 separate re-solves would have told you.

The derivative is local, so it ranks candidates rather than sizing them. Use it to shortlist,
then re-solve the shortlist with the real conductor data.

## What else carries gradients

The same tape reaches further than the two parameter groups used above.

- Conductor geometry. A line with `conductor_geometry` gets its impedance from the
  Carson/Deri model, and coordinates, radius and earth resistivity are differentiable inputs.
- Transformer tap ratio, source impedance and shunt admittance.
- Harmonic injections. The magnitude and phase of a device spectrum can be overridden per
  solve with tensors, which is how harmonic source estimation is set up.
- Switch states. `branch_states` scales branch admittances continuously, so a topology
  decision has a gradient as well.

Two implementation notes. The fundamental-frequency solve is nonlinear, and its backward
pass uses the implicit function theorem rather than unrolling the iteration, so gradient cost
does not grow with iteration count. Several products of one solve cost one back-substitution
each after the first, because the adjoint factorization is cached on the autograd node. The
default dtype is complex128, which is also the dtype the gradient checks run in. complex64
halves memory and is meant for throughput once a study is calibrated; `precision="mixed"`
usually serves better, because it factors at single precision while keeping the complex128
residual, so the answer stays at double-precision accuracy. See
{doc}`modeling/solver-performance`.

For the API behind these examples see {doc}`public-api`. For what the model does and does
not represent see {doc}`concepts` and {doc}`modeling/index`.
