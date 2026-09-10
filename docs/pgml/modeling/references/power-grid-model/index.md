# power-grid-model

A load-flow and state-estimation reference from LF Energy. It models no harmonics and exports
no admittance matrix, so it serves as a fundamental-frequency cross-check only. Its C++ core is
not differentiable.

## Data model
- A dict of numpy STRUCTURED ARRAYS, one key per component type; each array
  element is one component. SI units, no prefixes (volts, watts, ohms).
- Components: `node` (u_rated), `line`, `transformer`, `sym_load`/`asym_load`,
  `source`, `shunt`, `sym_gen`/`asym_gen`. `generic_branch`: Y_series=1/(r+jx),
  Y_shunt=g+jb, ratio N=k*e^(j theta), which is the same pi form plus a complex tap, symmetric only.
- Symmetry is per-component AND per-calculation: sym calc averages asym loads;
  an asymmetric calculation splits symmetric loads equally over the phases. pgml follows the same rule.
- Transformer: u1,u2,sn,uk,pk,i0,p0, winding_from/winding_to (connection enums),
  clock (0-12 vector group), tap_side/pos/min/max/nom/size; zero-seq derived from
  winding connections + clock.

## Output (per component)
- node: u, u_pu, u_angle, p, q (+ per-phase in asym).
- branch: loading, p_from/q_from/i_from/s_from, p_to/q_to/i_to/s_to.
- injection: p, q, i, s (generator reference direction).

## Extracting ground truth
```python
from power_grid_model import PowerGridModel, CalculationType
# build input_data dict of structured arrays, then:
model = PowerGridModel(input_data)
out = model.calculate_power_flow(symmetric=False)  # asym -> per-phase output
# compare out["node"]["u"], out["line"]["i_from"], ... against the pgml result
```
Map the schema ids onto power-grid-model ids explicitly, and mind the SI units and the
per-phase array layout of an asymmetric result.
