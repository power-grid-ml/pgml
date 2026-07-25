# Examples

The `run/examples/` directory contains end-to-end scripts. Each is self-contained and produces
paper-ready (SVG/PNG, ≥300 DPI) and interactive (Plotly HTML) figures under an output
directory. Run any example with:

```bash
pixi run -e cpu python run/examples/<script>.py [out_dir]
```

The figures below are produced by these scripts; they are the evidence behind the
[validation claims](index.md) and the
[modeling decisions](modeling/index.md).

---

## Comparing libraries: IEEE 33-bus

**File:** `run/examples/evaluate_ieee33.py`

Regenerates the full IEEE 33-bus evaluation, comparing pgml's differentiable solver to
**pandapower** and **OpenDSS**. The assembled Y-bus matches both reference tools to
floating-point precision, and the load-flow voltage profile matches pandapower.

```{figure} ../_static/figures/ybus_difference.svg
:alt: Difference between pgml and the reference Y-bus
:width: 80%

\|ΔY\| between pgml's assembled admittance and the closest reference (near floating-point
zero).
```

```{figure} ../_static/figures/voltage_profile.svg
:alt: pgml vs pandapower voltage profile
:width: 85%

Nonlinear power-flow voltage profile (per-unit vs distance), pgml versus pandapower.
```

Outputs (default `evaluation_output/`):

| File | Description |
|------|-------------|
| `ybus_heatmaps.svg` | pgml full Y vs pandapower network Y vs OpenDSS SystemY |
| `ybus_difference.svg` | \|ΔY\| of the two most-similar versions (near floating-point zero) |
| `voltage_profile.svg` | Nonlinear power flow vs pandapower (pu vs distance) |
| `voltage_error.svg` | Per-node \|Δpu\| bar chart |
| `harmonic_h5.svg` | h=5 magnitude/angle profile, pgml vs numpy oracle |
| `harmonic_3d.html` | Interactive 3D (h=3,5,7,9), pgml vs numpy oracle |
| `harmonic_models_interactive.html` | Config-default vs naive line model, toggleable |
| `harmonic_default_vs_naive_h5.svg` | Pairwise h=5 comparison |
| `grid_voltage_map.svg` | Topology coloured by voltage pu |

```{literalinclude} ../../run/examples/pgml/evaluate_ieee33.py
:language: python
:lines: 1-30
:caption: run/examples/pgml/evaluate_ieee33.py (header)
```

---

## Comparing modeling approaches: harmonic line models

Different ways of modeling a line's frequency dependence give materially different harmonic
results. Two examples make the differences explicit.

### Carson geometry vs OpenDSS

**File:** `run/examples/pgml/evaluate_harmonics_carson.py`

Synthesizes single-conductor Carson geometry that reproduces each line's R/X, runs the pgml
harmonic flow, and compares to OpenDSS's line model on the *same* geometry — validating that
the Carson/Deri path is **bit-exact** with OpenDSS.

### Naive vs sequence-aware

**File:** `run/examples/pgml/evaluate_line_sequence_harmonics.py`

Contrasts the naive "X ∝ h" scaling, the corrected positive-sequence model, the
single-conductor Carson model, and a live OpenDSS profile. See the
[harmonic line model](modeling/harmonic-line-model.md) page for the physics.

```{figure} ../_static/figures/harmonic_default_vs_naive_h5.svg
:alt: Config-default vs naive harmonic line model at h=5
:width: 85%

IEEE-33 at the 5th harmonic: the config-default line model versus the naive "X ∝ h" model.
```

```{literalinclude} ../../run/examples/pgml/evaluate_harmonics_carson.py
:language: python
:lines: 1-25
:caption: run/examples/pgml/evaluate_harmonics_carson.py (header)
```

---

## Scenarios: batched harmonic studies

**Files:** `run/examples/pgml/scenario_randomized.py`, `run/examples/pgml/scenario_node_injection_sweep.py`

Reproducible quasi-Monte-Carlo / cartesian sampling produces batches of operating points,
solved together, for studying how harmonic disturbances spread across a feeder and for
generating machine-learning training data.

```{figure} ../_static/figures/spread_h11.svg
:alt: Spread of the 11th-harmonic voltage across a scenario batch
:width: 85%

Distribution of the 11th-harmonic node voltages across a sampled scenario batch on CIGRE
LV.
```

---

## Solver behaviour and scale

**Files:** `run/examples/pgml/loadability_continuation.py`,
`run/examples/pgml/current_injection_convergence.py`, `run/examples/pgml/benchmark_speed.py`

The power-flow solvers expose actionable diagnostics near the loadability limit, and the
batched solve amortizes well on a GPU.

```{figure} ../_static/figures/pv_nose.svg
:alt: PV-nose loadability curve
:width: 80%

The loadability "nose" curve: voltage versus loading, with the continuation locating the
margin, the critical bus, and the limiting load.
```

```{figure} ../_static/figures/throughput_vs_batch.svg
:alt: Throughput versus batch size
:width: 80%

Scenario throughput versus batch size — the dense batched solve amortizes per-scenario cost
as the batch grows.
```

### Sparse vs dense factorization

**File:** `run/examples/pgml/benchmark_sparse.py`

Sweeps synthetic radial MV feeders (`pgml.grids.synthetic_feeder`) across system sizes and
times both `solve_power_flow` factorization backends — the batched dense `torch` LU and the
scipy SuperLU sparse factorization — for factorization time, back-substitution time (single
RHS and a batched scenario), and end-to-end wall time. On a CUDA host the dense rows are also
measured on the GPU, since the sparse-CPU advantage must be checked against the dense-GPU
baseline rather than assumed. The observed CPU crossover calibrates the row-count threshold
behind `linear_solver="auto"` (see the "Solve performance" section of
{doc}`api/solver`).

```{literalinclude} ../../run/examples/pgml/benchmark_sparse.py
:language: python
:lines: 1-30
:caption: run/examples/pgml/benchmark_sparse.py (header)
```
