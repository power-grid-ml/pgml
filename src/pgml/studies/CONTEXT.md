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

## Shared conventions

- Inputs are never modified: operating-point adjustments and added devices are made on a
  copy. `device` / `dtype` (complex64 or complex128) apply to every tensor operation.
- Every result has `to_dict()` (JSON-ready: plain lists, ids, units in the key names,
  `None` where JSON would need `NaN`) and `from_dict()`, its inverse. Every top-level
  `to_dict()` carries a `"study"` key naming the entry function (`"impedance_scan"`,
  `"pv_curve"`, `"qv_curve"`).
- Complex values serialize as `*_re_<unit>` / `*_im_<unit>` lists.
- Containers follow differentiability: `ImpedanceScan` holds CPU tensors so impedances stay
  on the autograd tape (`to_dict` and resonance detection read values off it); the
  voltage-stability results are host numpy arrays because the continuation is not
  differentiable (it runs detached).

## Impedance scan decisions an edit must not undo

- The network is `assemble_ybus` (linear model). The source is ALWAYS its Norton impedance
  (an ideal slack shorts the scan); a zero-impedance source is refused, not regularised.
- Generation-sign devices are open circuits by default (`fold_generation=False`), matching
  the harmonic device-shunt default `appliance.harmonic_shunt.generation_model: none`:
  their const-Z image has a negative conductance.
- Solve the requested columns (`lu_factor_system` + `solve_factored`, R right-hand sides per
  frequency), never a full inverse. Frequencies chunk against
  `solver.harmonic.system_budget_mb`; only the per-bus diagonal blocks are kept unless
  `transfer=True` (the full `[F, R, R]` block is as large as `Y` when every bus is scanned).
- Under bus fusion the requested rows map through `FusionMap.row_to_reduced`; fused buses
  report identical impedances.
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

## Tests

- Impedance scan: `tests/studies/test_impedance_scan.py` (closed forms, fusion, chunking,
  sparse backend, contracts), `tests/studies/test_impedance_scan_opendss.py`
  (`-m opendss`, SystemY inverse), `tests/differentiability/test_impedance_scan_gradcheck.py`,
  `tests/gpu/test_impedance_scan_parity.py`.
- Continuation: `tests/studies/test_continuation.py` (closed forms on a two-bus feeder,
  plain Newton solves on an unbalanced feeder, serialization, inputs),
  `tests/gpu/test_studies_continuation_parity.py`.
