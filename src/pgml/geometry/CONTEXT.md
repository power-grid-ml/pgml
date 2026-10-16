# geometry — line constants and harmonic line models

Conductor geometry → per-frequency line impedance and capacitance (Carson with the Deri
complex-depth earth return, the OpenDSS line-constants model), plus the lumped harmonic
line models for lines given as R/X data and the tools that attach a model or a synthesized
geometry to a grid. The published account is `docs/pgml/modeling/harmonic-line-model.md`
(model choice and evidence) and `docs/pgml/modeling/references/opendss/carson.md`
(agreement with OpenDSS).

Signatures: docstrings / `API_INDEX.md`.

## Codemap

- `carson.py` — torch line constants: `series_impedance` (earth return + geometric
  reactance + skin-effect internal impedance), `internal_impedance`,
  `internal_reactance_ratio`, `potential_coefficients` (Maxwell images), `kron_reduce`
  (eliminates neutral/shield conductors, which come after the phases), `line_constants`
  (phase-reduced `Z[*B,H,P,P]` Ω/m, `C[*B,P,P]` F/m), `i0_over_i1` (Bessel ratio by
  continued fraction). Constants `MU0`, `E0`, `INTERNAL_INDUCTANCE_MODELS`.
- `sequence.py` — lumped models from sequence data: `positive_sequence_z`,
  `skin_resistance_multiplier`, `fit_equivalent_rdc`, `zero_sequence_harmonic_z`,
  `carson_earth_resistance`, `x0_sublinear_deficit`, `sequence_aware_phase_z`, Fortescue
  helpers (`phase_to_sequence`, `sequence_impedances`, `sequence_to_phase_z`), and the
  two-conductor loop (`two_conductor_geometry`, `two_conductor_loop_z`) that cross-checks
  the positive-sequence model.
- `synthesis.py` — grid-level tools: `synthesize_line_geometry`,
  `synthesize_three_phase_geometry`, `synthesize_grid_geometry`, `strip_grid_geometry`,
  `apply_positive_sequence_harmonic_model`, `apply_sequence_aware_harmonic_model`,
  `positive_sequence_resistance_model`, `resolve_harmonic_line_models`,
  `apply_default_harmonic_model`.
- Schema side (`schemas/grid_schema.py`): `ConductorPlacement`, `LineGeometry`,
  `Line.conductor_geometry`, the typed selector `Line.harmonic_line_model`
  (`geometry | sequence_aware | positive_sequence | naive`), `Line.harmonic_skin_effect`,
  `Line.earth_return` (`EarthReturnModel`).
- Consumer: `assembly/ybus.py` (`_geometry_block_groups` batches geometry lines through
  `line_constants`, grouped by phase count, conductor count and resolved internal
  inductance; the lumped models go through `sequence.py`).

## Invariants

- Fully torch, differentiable, batched over lines and frequencies, device/dtype-following.
  Gradients flow conductor geometry / R1, X1, R0, X0 / every earth coefficient → Z, Yc →
  Y-bus. The frequency axis is an input, so the internal-inductance band mask carries no
  gradient; the `x0_nonnegative` clamp has zero gradient below its boundary.
- Units: Ω/m, F/m, S/m, Hz, metres. The module returns per-metre quantities; length is
  applied in assembly.
- A balanced positive-sequence current has no net earth current: the earth return lives in
  `Z0` only. `Z1(h)` is internal + geometric (`X1·h`, skin on `R1`), with no earth floor.
- Every lumped model is exact at the reference frequency `f0`. `skin_resistance_multiplier`
  returns ones for a request at `f0` alone (unless `freqs` / `f0` carry a gradient).
- In `zero_sequence_harmonic_z` only the phase conductor's share of `R0` (fitted to
  `phase_resistance = R1`) follows the skin curve; the return-path remainder is constant.
  `r0_includes_earth_return` says whether a stored `R0` already contains `3·Re(f0)`.
- Defaults come from `line.*` in `pgml.defaults` and are resolved at call / assembly time,
  so a preset applies without reimporting. Module constants (`INTERNAL_INDUCTANCE`,
  `POWER_FREQUENCY_BAND_HZ`, `CARSON_EARTH_R_PER_HZ`, `CARSON_EARTH_X_PER_HZ`,
  `X0_FREQUENCY`, `X0_EXPONENT`, `R0_INCLUDES_EARTH_RETURN`) are import-time snapshots for
  inspection only. Explicit arguments and per-line fields win.
- Assembly never resolves a line's harmonic model: `harmonic_line_model=None` uses the
  stored parameters as given. The converters call `resolve_harmonic_line_models` (and log
  the counts); hand-built grids call `apply_default_harmonic_model`.
- `strip_grid_geometry` drops the geometry and the harmonic model together; the schema
  rejects `harmonic_line_model="geometry"` without a geometry.
- A synthesized geometry's radius is a placeholder: use the `gmr` internal-inductance model
  with it (assembly warns otherwise). Synthesis is a validation vehicle (feeding the same
  geometry to pgml and OpenDSS), flagged `synth_unphysical` for low-X lines; in
  three-phase synthesis `R0` follows from the Carson earth return and is recorded in
  `provenance.extra.synth_r0_ohm_per_m`. Two-phase lines are not synthesized.

## Decisions

- Physical constants are SI; OpenDSS truncates `mu0` / `e0`. The remaining relative
  differences equal those constant ratios and are pinned as such.
- Internal-inductance models: `gmr` (default; power-frequency GMR at every frequency),
  `gmr_skin`, `gmr_power_frequency` (OpenDSS's 40–1000 Hz band rule, reproduces OpenDSS at
  every frequency, steps at the band edges), `bessel` (solid round conductor). All but
  `gmr` need a radius.
- Capacitance always uses the physical radius; OpenDSS's `capradius` is not read.
- The default X0 frequency law is `linear`; `carson_sublinear` is the overhead-line option.

## Tests

`tests/reference/test_carson_opendss.py`, `tests/reference/test_carson_internal_inductance.py`,
`tests/reference/test_carson_sequence.py`, `tests/reference/test_lumped_sequence_opendss.py`,
`tests/differentiability/test_carson_gradcheck.py`,
`tests/differentiability/test_sequence_gradcheck.py`, `tests/gpu/test_device_parity.py`.
</content>
