# studies — grid-planning studies

Analysis studies that answer a planning question about a grid instead of solving one
operating point. Each study is one module with one entry function and one frozen result
dataclass (`to_dict()` JSON-ready with units in the key names, `from_dict()` its inverse).
Signatures: docstrings / API_INDEX.md. User guide: `docs/pgml/studies/`.

## Codemap

- `impedance.py` — `impedance_scan` / `ImpedanceScan` / `Resonance`: `Z(f) = Y(f)^-1`
  restricted to the requested buses' rows, per-bus phase matrices, positive- and
  zero-sequence impedance, optional transfer matrix, resonance detection.

## Conventions

- Results hold CPU tensors (not numpy) so impedances stay on the autograd tape; `to_dict`
  and resonance detection read values off it.
- Complex values serialize as `*_re_<unit>` / `*_im_<unit>` lists; undefined entries are
  `None` (JSON has no NaN), `NaN` in the tensors.
- Inputs are never modified: operating-point adjustments are made on a copy.

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

## Tests

`tests/studies/test_impedance_scan.py` (closed forms, fusion, chunking, sparse backend,
contracts), `tests/studies/test_impedance_scan_opendss.py` (`-m opendss`, SystemY inverse),
`tests/differentiability/test_impedance_scan_gradcheck.py`,
`tests/gpu/test_impedance_scan_parity.py`.
