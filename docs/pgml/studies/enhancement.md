# Enhancement sensitivities

Which cable should be reinforced first, which transformer is too small, which tap would help,
and where would new demand or new generation hurt the most?
{func}`~pgml.studies.enhancement_sensitivities` answers all of these at once. It solves one
power flow at a base operating point, measures how close every element is to its limit with
a smooth stress metric, and differentiates that metric with one backward pass with respect to
every candidate enhancement. A finite-difference ranking would need one power flow per
candidate. The gradient needs about one more linear solve in total, whatever the number of
candidates.

```python
from pgml.grids import cigre_lv_full_grid
from pgml.studies import enhancement_sensitivities

grid, _ = cigre_lv_full_grid()                       # 44 nodes, three feeders
result = enhancement_sensitivities(grid)             # nameplate loads, default limits

print(f"stress {result.stress:.3f}:", {k: round(v, 3) for k, v in result.stress_by_kind.items() if v})
for e in result.stressed[:3]:
    where = e.get("branch_id", e.get("node_id"))
    print(f"  {e['kind']:11s} {where:>3}  value {e['value']:.3f}  stress {e['stress']:.3f}")
for name, ascending in (
    ("line_impedance_elasticity", True),
    ("transformer_rating_elasticity", True),
    ("transformer_tap_gradient", False),
    ("d_stress_d_p_per_kw", False),
):
    ranked = result.ranking(name, 3, ascending=ascending)
    print(f"{name:30s}", ", ".join(f"{i}: {v:+.3f}" for i, v in ranked))
```

```text
stress 1.170: {'voltage': 1.045, 'transformer': 0.125}
  voltage      36  value 0.912  stress 0.149
  voltage      37  value 0.912  stress 0.149
  transformer  84  value 0.853  stress 0.069
line_impedance_elasticity      71: -4.415, 64: -4.334, 63: -4.334
transformer_rating_elasticity  84: -2.240, 82: -2.002, 83: -0.000
transformer_tap_gradient       84: +310.149, 82: +52.355, 83: +0.000
d_stress_d_p_per_kw            36: +0.268, 37: +0.268, 35: +0.216
```

The CIGRE grid is a per-phase equivalent: each voltage entry in `stressed` is the stress of
one conductor and counts three times in the voltage stress, and `d_stress_d_p_per_kw` is per
kW of three-phase power.

The grid is stressed by low voltages at the end of the residential feeder and by two
transformers above 80 % of their rating. Line 71 is the line whose reinforcement relieves the
most stress, transformer 84 the one whose rating matters most. Raising the off-nominal ratio
of transformer 84 lowers its secondary voltage and adds stress. New demand hurts most at the
feeder end (nodes 36 and 37). The example needs the `convert` extra for the CIGRE grid.

## The stress metric

Each limit of {class}`~pgml.studies.HostingLimits` becomes a utilisation `u` per element,
`0` at the reference and `1` exactly at the limit:

| Element | Utilisation |
|---|---|
| Phase conductor, voltage band | `(v - 1) / (v_max_pu - 1)` above 1 pu, `(1 - v) / (1 - v_min_pu)` below |
| Rated line | Largest phase-conductor current over both terminals per `line_loading * max_i_a` |
| Rated line, neutral (`neutral_loading` set) | Neutral current per `neutral_loading * max_i_a` |
| Rated transformer | Per-phase current times the terminal's nominal line-to-neutral voltage, per `transformer_loading` of the rated power per phase, larger terminal |
| Three-phase node (`vuf_max` set) | `VUF / vuf_max` |
| Phase conductor (`thd_max` set, harmonic config) | `THD / thd_max` |

These are the quantities {func}`~pgml.studies.hosting_capacity` checks, computed by the same
code, so the two studies agree on what is near a limit. Voltages are the phase-to-neutral
voltages of the phase conductors.

The stress of one element is the squared excess of its utilisation over the onset `c`
(`stress_onset`, default 0.8), rescaled to reach 1 at the limit:

```text
s = max(0, (u - c) / (1 - c))^2,      S = sum of s over every element
```

A voltage or THD row of a per-phase equivalent (`metadata.tags["phase_mode"] ==
"single_phase_equiv"`) stands for the three phase conductors of the system and counts three
times in `S`, so `S` and every sensitivity are the same for a network stored as a per-phase
equivalent, a three-phase build or an as-built phase-domain grid. (An untagged one-phase grid
counts each row once: it may be a genuine single-phase circuit.)

`S` is continuously differentiable. It is zero, and so is every gradient, while every element
stays below `c` of its limit. The onset therefore lies below 1, so an element approaching its
limit already contributes a gradient. `S = 1` means, for example, one element exactly at its
limit and every other one below the onset. `result.stressed` lists every element with a
positive term, largest first, with its value, its utilisation and its stress.

## The sensitivities

Every enhancement is an elasticity `dS / d ln k` of an upgrade factor `k` at `k = 1`.
**Negative means the upgrade relieves stress.** To first order, an upgrade by 10 %
(`ln 1.1 ≈ 0.095`) changes the stress by about `0.095 × elasticity`.

| Field | Upgrade | Applies to |
|---|---|---|
| `line_impedance_elasticity` | Series impedance `Z / k` (R and L together) | Every in-service line; `NaN` for a line modelled by its conductor geometry |
| `line_impedance_elasticity_per_km` | The same, divided by the line length | The benefit per km of cable to replace |
| `line_rating_elasticity` | Ampacity `k * max_i_a` | Rated lines; always `<= 0` |
| `transformer_impedance_elasticity` | Leakage impedance `Z / k` | Every in-service transformer |
| `transformer_rating_elasticity` | Rated power `k * s_rated_va` | Rated transformers; always `<= 0` |
| `transformer_tap_gradient` | `dS / dn` of the off-nominal ratio magnitude `n` | Every in-service transformer |
| `d_stress_d_p_per_kw` | `dS / dP` per kW of new constant-power demand (system power) | Every bus without a source |
| `d_stress_d_q_per_kvar` | `dS / dQ` per kvar of new reactive demand (inductive positive, system power) | Every bus without a source |

The injections use the load convention: positive `P` is new demand, so new generation changes
the stress by the negative of the reported value. The new device is wye-connected over every
phase conductor of the bus with the power split equally; `P` and `Q` are its three-phase
system power in every representation (one kW on a per-phase equivalent is a third of a kW on
its stored phase). Its sensitivities are taken on zero-power probe loads added to a copy of
the grid, which leave the base case unchanged.

`result.ranking(name, k, ascending=True)` returns `(id, value)` pairs sorted with the most
negative first and `NaN` left out; `ascending=False` reverses the order (the bus where new
demand hurts most).

A rating elasticity acts through the loading terms only, so it is zero while an element is
below the onset, and `NaN` when the element has no rating or its loading limit is disabled.

At a nominal tap the two terminals of a transformer without a core branch carry the same
loading, but the from-side loading falls with the tap while the to-side one does not. The
loading, the larger of the two, has a kink there. Terminals that agree to the tie tolerance
of the hosting search (`1e-7` relative in complex128) therefore count with their mean, which
makes the reported tap gradient the central derivative instead of the slope of whichever
terminal rounding ranks larger.

## The base operating point

`base_operating_point` takes the values of {func}`~pgml.studies.hosting_capacity`:
`"nameplate"` (the default here), `"worst_case"` with `direction="generation"` (loads off,
generation at nameplate) or `direction="load"` (generation off, loads at nameplate), or a
mapping `{appliance_id: {"p_w", "q_var", ...}}` on top of nameplate, in the grid's stored
units (on a per-phase equivalent a third of the device's three-phase power). A base that does not
converge raises {class}`~pgml.errors.ConvergenceError`: there is no operating point to
differentiate at. A grid whose loads are stored as individual peaks usually needs a mapping
with diversified powers.

| Option | Default | Meaning |
|---|---|---|
| `limits` | {class}`~pgml.studies.HostingLimits` defaults | The limits the stress is built from; a voltage bound must lie on its side of 1 pu |
| `base_operating_point` | `"nameplate"` | See above |
| `direction` | `"generation"` | Which worst case `"worst_case"` builds |
| `stress_onset` | `0.8` | Fraction of each limit where its stress starts |
| `config` | fundamental power flow | A {class}`~pgml.SimulationConfig`; a harmonic one makes loading RMS over the orders and enables `thd_max` |
| `dtype`, `device` | complex128, CPU | Working precision and device of the solve and the backward pass |

## What the gradient is good for

The derivative is local. It ranks candidates at the base operating point; it does not size an
upgrade. The stress is far from linear once an element leaves the stressed range: halving the
impedance of the top line in the example of {doc}`../differentiability` removes 87 % of the
voltage stress, while the first-order estimate `ln 2 × elasticity` would predict more than
all of it. Shortlist with the gradient, then re-solve the shortlist with the real upgrade.

## Accuracy and cost

The tests compare every sensitivity with central differences of re-solved grids on a
transformer-fed 400 V feeder (complex128, solve tolerance 1e-13 pu). The measured agreement is
1.3e-8 relative or better for line and transformer impedance, ampacity, rated power, tap and
new active and reactive power at every bus. A float64 `gradcheck` covers the fundamental and
the harmonic calculation and the neutral and unbalance terms.

The backward pass differentiates the power flow through the implicit function theorem at the
converged state. Its cost is one adjoint solve with the dense state Jacobian, independent of
the number of candidates and of the iteration count. Measured on a laptop CPU (Intel Core
i5-1345U, 10 threads, PyTorch 2.13, complex128, 2026-11-07; the range spans repeated calls
in two processes):

| Grid | Unknowns (real) | Candidates | Forward | Backward |
|---|---|---|---|---|
| CIGRE LV, 44 nodes, single-phase equivalent | 88 | 37 lines, 3 transformers, 43 buses | 0.04 to 0.5 s | 0.02 to 0.04 s |
| `synthetic_feeder(660)`, three-phase | 3960 | 659 lines, 659 buses | 0.2 to 1.6 s | 4 to 13 s |

On the larger grid the backward is dominated by forming the dense `[2N, 2N]` state Jacobian
and factoring it: 3.0 s and 2.2 s of a profiled 7.5 s backward pass. A finite-difference ranking of the same candidates would
need more than 1300 power flows.

The study does not modify the grid passed in. `to_dict()` writes plain lists with `None` for
`NaN`, and `from_dict()` restores the result.
