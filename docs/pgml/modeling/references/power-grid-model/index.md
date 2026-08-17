# Reference brief: power-grid-model (LF Energy)

Use this as the distilled context; consult the installed package for specifics
(`pip install power-grid-model`). It is a LOAD-FLOW + state-estimation oracle,
NOT a harmonic oracle and NOT a Y-matrix oracle (no harmonics on its roadmap; no
public Ybus export). Its C++ core is not differentiable — we reuse its DATA MODEL
and use it to cross-check fundamental-frequency results only.

## Data model
- A dict of numpy STRUCTURED ARRAYS, one key per component type; each array
  element is one component. SI units, no prefixes (volts, watts, ohms).
- Components: `node` (u_rated), `line`, `transformer`, `sym_load`/`asym_load`,
  `source`, `shunt`, `sym_gen`/`asym_gen`. `generic_branch`: Y_series=1/(r+jx),
  Y_shunt=g+jb, ratio N=k*e^(j theta) (== our pi + complex tap; symmetric only).
- Symmetry is per-component AND per-calculation: sym calc averages asym loads;
  asym calc splits sym loads equally over phases. (We adopt this rule.)
- Transformer: u1,u2,sn,uk,pk,i0,p0, winding_from/winding_to (connection enums),
  clock (0-12 vector group), tap_side/pos/min/max/nom/size; zero-seq derived from
  winding connections + clock.

## Output (per component)
- node: u, u_pu, u_angle, p, q (+ per-phase in asym).
- branch: loading, p_from/q_from/i_from/s_from, p_to/q_to/i_to/s_to.
- injection: p, q, i, s (generator reference direction).

## Ground-truth extraction (for tests)
```python
from power_grid_model import PowerGridModel, CalculationType
# build input_data dict of structured arrays, then:
model = PowerGridModel(input_data)
out = model.calculate_power_flow(symmetric=False)  # asym -> per-phase output
# compare out["node"]["u"], out["line"]["i_from"], ... to our results
```
Map our schema ids to pgm ids explicitly; mind the SI units and the asym per-phase
array layout.
