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
- {doc}`enhancement`: which upgrade relieves the most limit stress (line and transformer
  impedance and rating, transformer tap, new power at each bus), ranked from one power flow
  and one backward pass.

## Shared conventions

| | Impedance scan | P-V / Q-V curves | Hosting capacity | Enhancement sensitivities |
|---|---|---|---|---|
| Entry function | {func}`~pgml.studies.impedance_scan` | {func}`~pgml.studies.pv_curve`, {func}`~pgml.studies.qv_curve` | {func}`~pgml.studies.hosting_capacity` | {func}`~pgml.studies.enhancement_sensitivities` |
| Result arrays | CPU torch tensors | host numpy arrays | host numpy arrays | host numpy arrays |
| Differentiable | yes: every impedance, with respect to tensor-valued grid parameters | no (runs detached) | no (feasibility is boolean) | its results are gradients, from one backward pass |

- **Inputs are never modified.** Operating-point adjustments and added devices are made on
  a copy of the grid.
- **Units are SI** (ohm, W, var, Hz); voltages are in per unit of each row's
  line-to-neutral rated voltage where a key ends in `_pu`.
- **Powers are system powers.** Every power a study takes or reports in W or var
  (`p_max_w`, `tol_w`, `p_cap_w`, `p_total_w`, `q_var`, `q_margin_var`, a transformer's
  `s_rated_va` from {func}`~pgml.studies.branch_ratings`) and every sensitivity per kW or
  kvar is the three-phase total, whatever representation the grid is stored in
  (`metadata.tags["phase_mode"]`, see {doc}`../modeling/conventions`). A per-phase
  equivalent stores a third of every power; the studies scale through
  {func}`pgml.representation.system_totals`, so the same network gives the same numbers as a
  per-phase equivalent, a three-phase build or an as-built phase-domain grid. Voltages stay
  per unit, currents stay amperes per conductor, loadings stay fractions.
- **Setpoints stay stored.** A `base_operating_point` mapping and an `operating_point`
  (`{appliance_id: {"p_w", "q_var", ...}}`) set appliance powers and are in the grid's
  stored units, like the appliance fields they replace: on a per-phase equivalent a third of
  the device's three-phase power.
- **Legacy grids** (`phase_mode = "legacy_ll"`, written by older converters: one-phase
  nodes at the line-to-line voltage with three-phase totals) are read as stored. Their per-unit voltages and powers agree with the
  other representations, but their branch currents are √3 too large, so line loading does
  not; migrate them with {func}`pgml.representation.to_per_phase_equivalent`. An untagged
  grid is read as the schema defines it (a one-phase node is a single-phase circuit).
- **Serialization.** Every result has `to_dict()`, JSON-ready with plain lists, ids and the
  unit in each key name (`z_positive_re_ohm`, `p_total_w`, `q_margin_var`, `p_max_w`), and
  `from_dict()`, its inverse. The top-level dict carries a `"study"` key naming the entry
  function (`"impedance_scan"`, `"pv_curve"`, `"qv_curve"`, `"hosting_capacity"`,
  `"enhancement_sensitivities"`), and
  `from_dict()` raises {class}`~pgml.errors.InputError` when it names another one. Complex
  values are split into `*_re_<unit>` / `*_im_<unit>`; a value that would be `NaN` is
  `None`.
- **Device and dtype.** Every study takes `device` and `dtype` (`torch.complex64` or
  `torch.complex128`, or the names `"complex64"` / `"complex128"`) and runs its assembly and
  solves there; the result is returned on the host.
- **Phases** are the schema values of {class}`~pgml.schemas.grid_schema.Phase`: `"a"`,
  `"b"`, `"c"`, `"n"`. Functions that take a phase accept a `Phase` too.
- **Load or generation.** {func}`~pgml.studies.pv_curve`,
  {func}`~pgml.studies.hosting_capacity` and {func}`~pgml.studies.enhancement_sensitivities`
  (for its worst-case base) select the side with the same keyword, `direction="load"` or
  `direction="generation"`, and report it as `direction`.
- **Base operating point.** {func}`~pgml.studies.hosting_capacity` and
  {func}`~pgml.studies.enhancement_sensitivities` share `base_operating_point`
  (`"worst_case"`, `"nameplate"` or a mapping of appliance powers) and the
  {class}`~pgml.studies.HostingLimits`, judged by the same code.
- **Differentiable outputs** are only the impedance-scan tensors. The voltage-stability
  curves and the hosting capacity are decided by continuation events and feasibility
  checks and run without an autograd graph. The enhancement sensitivities are gradients
  themselves, computed inside the study and returned as host arrays.

```{toctree}
:hidden:

impedance-scan
continuation
hosting-capacity
enhancement
```
