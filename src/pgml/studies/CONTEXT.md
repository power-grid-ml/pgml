# studies — grid-planning studies

Analysis studies that answer a planning question about a grid instead of solving one
operating point. Each study is one module with one entry function taking a `Grid` (plus an
optional `SimulationConfig`) and one frozen result dataclass per output.

Signatures: docstrings / API_INDEX.md. User guide: `docs/pgml/studies/`.

## Codemap

- `impedance.py` — `impedance_scan` / `ImpedanceScan` / `Resonance`: `Z(f) = Y(f)^-1`
  restricted to the requested buses' rows, per-bus phase matrices, positive- and
  zero-sequence impedance, optional transfer matrix, resonance detection.
- `continuation.py` — `pv_curve` / `PVCurve` (pseudo-arc-length continuation through the
  nose) and `qv_curve` / `QVCurve` (fictitious voltage-controlled source swept over a
  setpoint range). The residual, Jacobian inputs, fixed-`λ` solves and reactive-limit
  switching come from `pgml.solver._continuation`; the predictor, corrector, step control
  and event location (nose, reactive-limit switch, `λ` cap) live here.
- `limits.py` — `branch_ratings` (line ampacity from the catalog type's `max_i_a`,
  transformer `s_rated_va` from the component, else its type; every branch listed) and
  `HostingLimits` (voltage band, line / transformer loading, optional VUF and THD).
- `hosting.py` — `hosting_capacity` and `HostingCapacity`.
- `_common.py` — private helpers every study uses: `resolve_complex_dtype` (the `dtype`
  argument) and `check_study` (the `"study"` key check of `from_dict`).

## Shared conventions

- Inputs are never modified: operating-point adjustments and added devices are made on a
  copy. `device` / `dtype` apply to every tensor operation; every study takes `dtype` as
  `str | torch.dtype` (complex64 or complex128) through `_common.resolve_complex_dtype`.
- Phases in results and phase arguments are the schema value `Phase.value` (`"a"`, `"b"`,
  `"c"`, `"n"`); an argument also accepts a `Phase`, letters case-insensitively.
- The load / generation side is ONE keyword in every study that has one:
  `direction: Literal["load", "generation"]` (`pv_curve`, `hosting_capacity`), echoed by
  the result field and the `to_dict` key `direction`. (The `"kind"` key of a hosting
  `binding` entry names the limit kind, not the side.)
- Per-phase arrays name the quantity first, the unit last: `q_phase_var`.
- Every result has `to_dict()` (JSON-ready: plain lists, ids, units in the key names,
  `None` where JSON would need `NaN`) and `from_dict()`, its inverse. Every top-level
  `to_dict()` carries a `"study"` key naming the entry function (`"impedance_scan"`,
  `"pv_curve"`, `"qv_curve"`, `"hosting_capacity"`); nested records (`Resonance`,
  `HostingLimits`) do not. Every top-level `from_dict()` checks that key first
  (`_common.check_study`) and raises `InputError` naming expected and found.
- Complex values serialize as `*_re_<unit>` / `*_im_<unit>` lists.
- Containers follow differentiability: `ImpedanceScan` holds CPU tensors so impedances stay
  on the autograd tape (`to_dict` and resonance detection read values off it); the
  voltage-stability and hosting-capacity results are host numpy arrays because those
  studies are not differentiable (continuation events and feasibility verdicts; they run
  detached).

## Impedance scan decisions an edit must not undo

- The default network is the harmonic power flow's own system matrix at every scan
  frequency (`pgml.solver.harmonic_flow._harmonic_admittance`, the helper behind
  `assemble_harmonic_ybus`, which accepts non-integer orders): passive network, source
  Norton shunt, explicit DER `harmonic_impedance`, device harmonic shunt. `load_shunt` /
  `load_shunt_basis` resolve explicit > `config` > `appliance.harmonic_shunt.*`, so a scan
  matches `simulate` for the same config. `"const_z"` keeps the `assemble_ybus` fold as an
  explicit option. Pinned by `tests/studies/test_impedance_scan_flow_consistency.py`
  (scan == block of `assemble_harmonic_system`'s `Y(h)^-1`).
- The scan has no fundamental solution: device shunts are evaluated at the rated terminal
  voltage (`v1=None`), exact for constant-power devices.
- The source is ALWAYS its Norton impedance (an ideal slack shorts the scan); a
  zero-impedance source is refused, not regularised.
- Generation: the harmonic shunt models follow `appliance.harmonic_shunt.generation_model`;
  `fold_generation` belongs to `"const_z"` only (open by default, matching that policy:
  the const-Z image of an injecting device has a negative conductance) and raises with any
  other model.
- Solve the requested columns (`lu_factor_system` + `solve_factored`, R right-hand sides per
  frequency), never a full inverse. Frequencies chunk against
  `solver.harmonic.system_budget_mb`; only the per-bus diagonal blocks are kept unless
  `transfer=True` (the full `[F, R, R]` block is as large as `Y` when every bus is scanned).
  A tracked scan keeps every chunk on the graph (the budget bounds untracked scans only).
- Numerical singularity is a WARNING from a conditioning lower bound read off the solved
  columns (`max_j ||A^-1 e_j||_1` of the equilibrated `A`), not `estimate_condition`: that
  costs a rebuilt dense equilibrated matrix and measured 27 % of a single-bus sparse scan.
- Under bus fusion the requested rows map through `FusionMap.row_to_reduced`; fused buses
  report identical impedances.
- Resonances need a relative topographic prominence (`min_prominence`, default 0.05):
  neighbour-only extremum tests report rounding noise (54 spurious resonances on a flat
  100-point curve at 1e-15 noise). Plateaus collapse to one extremum; `1/|Z|^2` is formed
  only around peaks (no divide-by-zero on a lossless dip).
- Resonance refinement interpolates `1/|Z|^2` (peaks) and `|Z|^2` (dips), which are
  quadratic near a lightly damped resonance; `ln|Z|` was measured 20x less accurate.

## Continuation invariants

- `λ` multiplies the selected appliances' complex power at the solved voltage; `λ = 1` is
  the base operating point. A scaled generator pinned at a reactive limit keeps its
  reactive power at the limit (split between the scaled and the unscaled plan).
- The arc-length metric is `(x / V_base, λ / max(1, |λ|))`; `λ_ref` is re-based along the
  trace. An RMS-weighted voltage or an absolute `λ` makes the fold numerically sharp on
  stiff grids (nose at large `λ`), where the corrector then fails at any practical step.
- The corrector uses the solver's `_PuConvergence`; a corrector ending inside the floor band
  is accepted (complex64 near the nose), exactly as the solver accepts a stalled iterate.
- After a reactive-limit switch the continuation direction is checked by a trial step: a
  direction that would switch straight back is reversed, which is what turns a switch on
  the lower branch of the limited curve into a limit-induced nose.

## Hosting capacity decisions an edit must not undo

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

- Impedance scan: `tests/studies/test_impedance_scan.py` (closed forms, fusion, chunking,
  sparse backend, resonance prominence, contracts),
  `tests/studies/test_impedance_scan_flow_consistency.py`,
  `tests/studies/test_impedance_scan_opendss.py`
  (`-m opendss`, SystemY inverse), `tests/differentiability/test_impedance_scan_gradcheck.py`,
  `tests/gpu/test_impedance_scan_parity.py`.
- Continuation: `tests/studies/test_continuation.py` (closed forms on a two-bus feeder,
  plain Newton solves on an unbalanced feeder, serialization, inputs),
  `tests/gpu/test_studies_continuation_parity.py`.
- Hosting capacity: `tests/studies/test_hosting.py` (closed forms, serial `simulate`
  reference, every limit kind, round trip), `tests/studies/test_limits.py`, shared test
  grids in `tests/studies/grids.py`, `tests/gpu/test_hosting_parity.py`.
