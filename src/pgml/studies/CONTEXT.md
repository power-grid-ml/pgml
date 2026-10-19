# studies — grid-planning studies

Analysis studies that answer a planning question about a grid by running batched solves:
each takes a `Grid`, returns a frozen result dataclass of host arrays with `to_dict()` /
`from_dict()` (plain lists, ids, units in key names, `NaN` as `None`). User guide:
`docs/pgml/studies/`. Signatures: docstrings / API_INDEX.md.

## Codemap

- `limits.py` — `branch_ratings` (line ampacity from the catalog type's `max_i_a`,
  transformer `s_rated_va` from the component, else its type; every branch listed) and
  `HostingLimits` (voltage band, line / transformer loading, optional VUF and THD).
- `hosting.py` — `hosting_capacity` and `HostingCapacity`.

## Decisions an edit must not undo

- `branch_ratings` is the same rule pgd applies when it reports loading; a change here is a
  cross-package contract change.
- Hosting capacity is a lockstep bisection: each step is ONE batched solve, scenario `k` =
  candidate `k` at its trial power. The candidates are inert zero-nameplate devices added
  to a COPY of the grid; the base case and trial powers reach the solver through
  `batch_from_values` (the grid is never mutated).
- The solve goes through the solver directly (`prepare_power_flow` once +
  `solve_power_flow(system=...)`, or `solve_harmonic_flow`) so every `SimulationConfig`
  field applies; `criticality="never"` because non-convergence is an expected verdict.
- Branch currents come from `SolvedState.branch_currents()` built on the batch, which passes
  the solve's fusion map and nodal injection to `assembly.branch_currents`: grids with ideal
  switches must work (test grid `tests/studies/grids.py::lv_feeder_with_switch`).
- Non-convergence counts as a violation. The reported power is the largest VERIFIED
  feasible one; the binding limit is the largest relative excess at the smallest violating
  power, ties within working precision resolved to the first element in row / branch order.
- Not differentiable (feasibility is boolean); the search runs under `torch.no_grad()`.

## Tests

`tests/studies/` (closed forms, serial `simulate` reference, every limit kind, round trip),
`tests/gpu/test_hosting_parity.py`.
