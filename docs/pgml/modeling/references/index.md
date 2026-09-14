# Reference tools

pgml reads grids from three established power-system tools and validates its results against
them. These briefs distil only the facts that matter for that reading and validation. They are
not documentation of those tools.

- OpenDSS is the harmonic reference. The assembled admittance matrix and the per-order
  voltages are compared against it.
- pandapower and power-grid-model are fundamental-frequency load-flow references. They model
  no harmonics, so the comparison is at 50 or 60 Hz.

How each tool's conventions map onto pgml's internal form is in
{doc}`../conventions`.

```{toctree}
:maxdepth: 1

opendss/index
opendss/carson
opendss/harmonics
pandapower/index
power-grid-model/index
```
