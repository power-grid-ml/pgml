# Planning studies

{mod}`pgml.studies` answers planning questions about a grid rather than solving one operating
point. Each study takes a {class}`~pgml.schemas.grid_schema.Grid` (and optionally a
{class}`~pgml.simulation.SimulationConfig`) and returns a frozen result object. The API
reference is {mod}`pgml.studies`.

- {doc}`impedance-scan`: the driving-point and transfer impedance a bus sees over
  frequency, and its parallel and series resonances.
- {doc}`continuation`: the P-V curve through its nose (maximum loadability) by continuation
  power flow, and the Q-V curve and reactive power margin of a bus.
- {doc}`hosting-capacity`: how much new generation or demand each bus can take before a
  voltage, loading, unbalance or distortion limit is violated, for every candidate bus in
  one batched search.

## Shared conventions

| | Impedance scan | P-V / Q-V curves | Hosting capacity |
|---|---|---|---|
| Entry function | {func}`~pgml.studies.impedance_scan` | {func}`~pgml.studies.pv_curve`, {func}`~pgml.studies.qv_curve` | {func}`~pgml.studies.hosting_capacity` |
| Result arrays | CPU torch tensors | host numpy arrays | host numpy arrays |
| Differentiable | yes: every impedance, with respect to tensor-valued grid parameters | no (runs detached) | no (feasibility is boolean) |

- **Inputs are never modified.** Operating-point adjustments and added devices are made on
  a copy of the grid.
- **Units are SI** (ohm, W, var, Hz); voltages are in per unit of each row's
  line-to-neutral rated voltage where a key ends in `_pu`.
- **Serialization.** Every result has `to_dict()`, JSON-ready with plain lists, ids and the
  unit in each key name (`z_positive_re_ohm`, `p_total_w`, `q_margin_var`, `p_max_w`), and
  `from_dict()`, its inverse. The top-level dict carries a `"study"` key naming the entry
  function (`"impedance_scan"`, `"pv_curve"`, `"qv_curve"`, `"hosting_capacity"`), and
  `from_dict()` raises {class}`~pgml.errors.InputError` when it names another one. Complex
  values are split into `*_re_<unit>` / `*_im_<unit>`; a value that would be `NaN` is
  `None`.
- **Device and dtype.** Every study takes `device` and `dtype` (`torch.complex64` or
  `torch.complex128`, or the names `"complex64"` / `"complex128"`) and runs its assembly and
  solves there; the result is returned on the host.
- **Phases** are the schema values of {class}`~pgml.schemas.grid_schema.Phase`: `"a"`,
  `"b"`, `"c"`, `"n"`. Functions that take a phase accept a `Phase` too.
- **Load or generation.** {func}`~pgml.studies.pv_curve` and
  {func}`~pgml.studies.hosting_capacity` select the side with the same keyword,
  `direction="load"` or `direction="generation"`, and report it as `direction`.
- **Differentiable outputs** are only the impedance-scan tensors. The voltage-stability
  curves and the hosting capacity are decided by continuation events and feasibility
  checks and run without an autograd graph.

```{toctree}
:hidden:

impedance-scan
continuation
hosting-capacity
```
