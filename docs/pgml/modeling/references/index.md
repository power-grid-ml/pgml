# Reference-library notes

pgml is validated against three established power-system tools, and converts grids from
each. These briefs distil the modeling facts that matter for that conversion and
validation — they are *not* full documentation of those tools, only the parts pgml depends
on.

- **OpenDSS** is the **harmonic ground truth**: pgml's assembled Y-bus and per-order
  voltages are compared against it.
- **pandapower** and **power-grid-model** are **fundamental-frequency load-flow oracles**
  (no harmonics): pgml's voltages and flows at 50/60 Hz are compared against them.

See [cross-tool conventions](../conventions.md) for how each tool's conventions map onto
pgml's canonical internal form at the converter boundary.

```{toctree}
:maxdepth: 1

opendss/index
opendss/carson
opendss/harmonics
pandapower/index
power-grid-model/index
```
