# Planning studies

{mod}`pgml.studies` answers planning questions about a grid rather than solving one operating
point. Each study takes a {class}`~pgml.schemas.grid_schema.Grid` (and optionally a
{class}`~pgml.simulation.SimulationConfig`) and returns a frozen result object. The API
reference is {mod}`pgml.studies`.

- {doc}`impedance-scan`: the driving-point and transfer impedance a bus sees over
  frequency, and its parallel and series resonances.
- {doc}`continuation`: the P-V curve through its nose (maximum loadability) by continuation
  power flow, and the Q-V curve and reactive power margin of a bus.

## Shared conventions

| | Impedance scan | P-V / Q-V curves |
|---|---|---|
| Entry function | {func}`~pgml.studies.impedance_scan` | {func}`~pgml.studies.pv_curve`, {func}`~pgml.studies.qv_curve` |
| Result arrays | CPU torch tensors | host numpy arrays |
| Differentiable | yes: every impedance, w.r.t. tensor-valued grid parameters | no (runs detached) |
| Computation | `device` / `dtype` arguments | `device` / `dtype` arguments |

- **Inputs are never modified.** Operating-point adjustments are made on a copy of the grid.
- **Units are SI** (ohm, W, var, Hz), voltages in per unit of each row's line-to-neutral
  rated voltage where a key ends in `_pu`.
- **Serialization.** Every result has `to_dict()`, JSON-ready with plain lists, ids and the
  unit in each key name (`z_positive_re_ohm`, `p_total_w`, `q_margin_var`), and
  `from_dict()`, its inverse. The top-level dict carries a `"study"` key naming the entry
  function. Complex values are split into `*_re_<unit>` / `*_im_<unit>`; a value that
  would be `NaN` is `None`.
- **Device and dtype.** The computation runs on the requested `device` in complex64 or
  complex128; the result is returned on the host.

```{toctree}
:hidden:

impedance-scan
continuation
```
