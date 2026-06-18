# Examples

The `examples/` directory contains end-to-end demonstration scripts.  Each
script is self-contained and produces paper-ready (SVG/PNG, ≥300 DPI) and
interactive (Plotly HTML) figures under an output directory.

Run any example with:

```bash
pixi run --environment cpu python examples/<script>.py [out_dir]
```

---

## IEEE 33-bus evaluation

**File:** `examples/evaluate_ieee33.py`

Regenerates the full IEEE 33-bus evaluation figure set comparing our
differentiable solver to pandapower and OpenDSS.  Outputs (default
`evaluation_output/`):

| File | Description |
|------|-------------|
| `ybus_heatmaps.svg` | Our full Y vs pandapower network Y vs OpenDSS SystemY |
| `ybus_difference.svg` | \|ΔY\| of the two most-similar versions (near floating-point zero) |
| `voltage_profile.svg` | Nonlinear power flow vs pandapower (pu vs distance) |
| `voltage_error.svg` | Per-node \|Δpu\| bar chart |
| `harmonic_h5.svg` | h=5 magnitude/angle profile, pgml vs numpy oracle |
| `harmonic_3d.html` | Interactive 3D (h=3,5,7,9), pgml vs numpy oracle |
| `harmonic_models_interactive.html` | Config-default vs naive line model, toggleable |
| `harmonic_default_vs_naive_h5.svg` | Pairwise h=5 comparison |
| `grid_voltage_map.svg` | Topology coloured by voltage pu |

The harmonic figures attach a typical 6-pulse converter spectrum to a few loads.

```{literalinclude} ../examples/evaluate_ieee33.py
:language: python
:lines: 1-30
:caption: examples/evaluate_ieee33.py (header)
```

---

## IEEE 33-bus + CIGRE LV harmonic evaluation with Carson geometry

**File:** `examples/evaluate_harmonics_carson.py`

Synthesizes single-conductor Carson geometry that reproduces each line's R/X,
runs the pgml harmonic flow, and compares to OpenDSS's line model (same
synthesised geometry, same converged injection).  Validates that our
Carson/Deri path is bit-exact with OpenDSS.

Outputs per feeder (default `evaluation_output/carson/<feeder>/`):

| File | Description |
|------|-------------|
| `ybus_h5.svg` | pgml Y(5f₀) vs OpenDSS SystemY(5f₀) |
| `ybus_h5_diff.svg` | \|ΔY\| (near floating-point zero) |
| `harmonic_h5.svg` | h=5 magnitude/angle profile |
| `harmonic_3d.html` | Interactive 3D (h=5,7,11,13) |
| `models_h7.svg` | h=7: Carson, config-default, positive-seq, naive |
| `carson_vs_default_h7.svg` | Pairwise: Carson vs config-default |

```{literalinclude} ../examples/evaluate_harmonics_carson.py
:language: python
:lines: 1-25
:caption: examples/evaluate_harmonics_carson.py (header)
```

---

## Positive-sequence-aware harmonic line model

**File:** `examples/evaluate_line_sequence_harmonics.py`

Visualises the difference between single-conductor R/X geometry synthesis
(wrong for positive-sequence feeders) and the corrected sequence-aware model.
Demonstrates why zero-sequence carries the earth-return floor while positive
sequence has no such floor.

```{literalinclude} ../examples/evaluate_line_sequence_harmonics.py
:language: python
:lines: 1-20
:caption: examples/evaluate_line_sequence_harmonics.py (header)
```
