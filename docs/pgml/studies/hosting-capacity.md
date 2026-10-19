# Hosting capacity

How much new generation (a PV plant) or new demand (an EV charger, a heat pump) can each
bus take before a limit is violated? {func}`~pgml.studies.hosting_capacity` answers this for
every candidate bus in one call.

```python
from pgml.grids import synthetic_feeder
from pgml.studies import HostingLimits, hosting_capacity

grid = synthetic_feeder(25, n_feeders=3)             # 20 kV, three radial chains
result = hosting_capacity(grid, limits=HostingLimits(v_max_pu=1.03))
for node_id in (1, 10, 22):                          # along the first chain
    entry = result.result_for(node_id)
    print(node_id, f"{entry['p_max_w'] / 1e6:6.2f} MW", entry["binding"]["kind"])
```

```text
1 250.37 MW voltage
10  62.59 MW voltage
22  31.30 MW voltage
```

The synthetic feeder declares no line ampacities, so only the voltage band applies here.

## What is searched

For each candidate bus the study adds one new device, a WYE-connected generator or load, and
finds the largest active power `p_max_w` at which every limit holds. Every other bus stays
at the base operating point. The candidates do not interact: the result for one bus assumes
no new device anywhere else.

The search is a bisection on the device's power, run for all candidates in lockstep. Each
step solves one batch of power flows in which scenario `k` carries candidate `k`'s trial
power and nothing else, so the cost is about 15 to 30 batched solves whatever the number
of candidates. A scenario that does not converge counts as a violation: beyond the
loadability nose there is no operating point.

| Option | Default | Meaning |
|---|---|---|
| `candidates` | every bus without a source | Bus ids to test |
| `direction` | `"generation"` | `"generation"` or `"load"` |
| `base_operating_point` | worst case | What the other devices do, see below |
| `phases` | `"balanced"` | All phase conductors with an equal split, or `"a"`, `"b"`, `"c"` for a single-phase device |
| `power_factor` | `1.0` | Positive: reactive power flows with the active power; negative: against it (an under-excited generator) |
| `p_max_w` | none | Search cap; without one the bracket doubles from 1 kW |
| `tol_w` | 0.1 % | Bisection tolerance, by default relative to each candidate's bracket |
| `harmonic_emission` | none | Spectrum of the new device, harmonic calculation only |
| `config` | fundamental power flow | A {class}`~pgml.SimulationConfig` for slack, symmetry, convergence control and harmonic orders |

`p_max_w` is a power at which every limit was verified; the limit itself lies within
`tol_w` above it. When `p_max_w` reaches the cap without a violation, the binding entry is
`{"kind": "none"}`. A bus behind an ideal slack, for example one tied to the source bus by a
closed switch, never reaches a voltage limit.

## The base operating point

The base case decides the answer, so it is an explicit option:

- `None` or `"worst_case"`: for generation every load draws nothing and existing
  generators run at nameplate (the midday, low-demand case of a PV study); for demand every
  generator injects nothing and the loads draw nameplate. Storage keeps its nameplate
  setpoint in both.
- `"nameplate"`: every device at nameplate.
- A mapping `{appliance_id: {"p_w": ..., "q_var": ...}}`: those devices overridden on top of
  nameplate (per-phase keys `p_per_phase_w`, `q_per_phase_var` work too).

If the base case already violates a limit, every candidate reports `p_max_w = 0` with that
limit as binding, and `base_case_feasible` is `False`.

## Limits

{class}`~pgml.studies.HostingLimits` holds the limits; `None` switches one off.

| Limit | Default | Checked on |
|---|---|---|
| `v_min_pu`, `v_max_pu` | 0.90, 1.10 | Fundamental phase-to-neutral voltage of every phase conductor, per unit of the node's nominal line-to-neutral voltage |
| `line_loading` | 1.0 | Conductor current of every rated line, RMS over the solved orders, both terminals, neutral included |
| `transformer_loading` | 1.0 | Per-phase current times the terminal's nominal line-to-neutral voltage, against a third of `s_rated_va`, both terminals |
| `vuf_max` | off | Voltage unbalance factor at three-phase nodes |
| `thd_max` | off | Voltage THD of every phase conductor; needs a harmonic `config` |

The 0.90 to 1.10 pu band is the EN 50160 supply band. Connection rules for new generation
are often stricter: VDE-AR-N 4105 limits the voltage rise caused by generation in a
low-voltage grid to 3 %, which `v_max_pu=1.03` expresses for a grid at 1.0 pu.

Ratings come from the grid through {func}`~pgml.studies.branch_ratings`: a line's ampacity
from the `max_i_a` of the catalog type its `type_ref` names, a transformer's rating from its
own `s_rated_va` or its catalog type. A branch without a rating is not checked. Branch
currents are read through the solve's bus-fusion map, so ideal switches and jumpers are
handled exactly.

## The result

{class}`~pgml.studies.HostingCapacity` holds, per candidate, `p_max_w`, the binding limit
(`{"kind": "voltage", "limit", "node_id", "phase"}`, `{"kind": "line", "branch_id"}`,
`{"kind": "transformer", "branch_id"}`, `{"kind": "vuf", "node_id"}`,
`{"kind": "thd", "node_id", "phase"}`, `{"kind": "convergence"}` or `{"kind": "none"}`), the
margin of every limit at `p_max_w`, and the number of solves. `to_dict()` writes plain lists
with `None` for a margin that does not apply.

The binding limit is the one exceeded most, relative to its own value, at the smallest
violating power. Elements tied within working precision resolve to the first in node or
branch order.

## Accuracy and assumptions

The tests compare against closed forms and against a serial search with
{func}`~pgml.simulate`:

| Case | Measured |
|---|---|
| Voltage rise on a two-bus line, unity and 0.95 under-excited | within 0.9 W of the exact quadratic, `tol_w` 1 W |
| New demand to the 0.90 pu band | within 0.9 W of the exact quadratic |
| Line ampacity binding before voltage | within 0.1 % of the closed form |
| Switched 400 V feeder, line and voltage limits | equal to a serial `simulate` bisection, within `tol_w` 50 W |
| complex64 against complex128 | same powers within one `tol_w` |

The bisection assumes the feasible powers form one interval starting at zero. This holds
when every limit tightens as the device's power grows, which is the case for voltage rise
or drop and for branch current. The study runs without an autograd graph and does not modify the
grid passed in.
