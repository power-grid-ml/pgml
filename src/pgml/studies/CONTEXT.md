# studies

Planning studies on one grid: analyses that answer a planning question with many solves
rather than one operating point. Each study is a function taking a `Grid` (plus an optional
`SimulationConfig`) and returning a frozen result dataclass of host numpy arrays with
`to_dict()` / `from_dict()` (plain lists, ids, units in key names, `None` for a missing
value). Studies run detached (not differentiable) and never mutate their inputs.

Signatures: docstrings / API_INDEX.md. User guide: `docs/pgml/studies/`.

## Codemap

- `continuation.py` — `pv_curve` / `PVCurve` (pseudo-arc-length continuation through the
  nose) and `qv_curve` / `QVCurve` (fictitious voltage-controlled source swept over a
  setpoint range). The residual, Jacobian inputs, fixed-`λ` solves and reactive-limit
  switching come from `pgml.solver._continuation`; the predictor, corrector, step control
  and event location (nose, reactive-limit switch, `λ` cap) live here.

## Invariants (continuation)

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

## Tests

`tests/studies/test_continuation.py` (closed forms on a two-bus feeder, plain Newton solves
on an unbalanced feeder, serialization, inputs); GPU parity in
`tests/gpu/test_studies_continuation_parity.py`.
