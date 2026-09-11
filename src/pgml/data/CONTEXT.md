# Interface ledger: data (shipped reference data — read-only, versioned with the code)

Files here are PACKAGE DATA: physical/standards tables and documented modeling defaults
that ship inside the wheel and are read via `importlib.resources` (never a filesystem
walk-up, never a required env var). They are not user run configuration — the
serializable run schema is `pgml.scenarios.config`; a downstream package that adds its own
run config follows the same convention.

## Files
- `defaults.yaml` — modeling default VALUES and default MODEL choices, ordered by
  component; each leaf is `{value, units, description}` so a default is a deliberate,
  documented decision. Loaded by `pgml.defaults`.
- `standards/en50160.yaml` — DIN EN 50160 per-order maximum harmonic voltage magnitudes
  (relative to the fundamental, per-unit), orders 1..49. Loaded by
  `pgml.scenarios.en50160`.

## Loaders + overrides
- `pgml.defaults` — `get/resolve/describe/units/defaults/reload`. Active source is
  `defaults.yaml`; `PGML_DEFAULTS=/path/to.yaml` (or `reload(path)`) overrides it.
- `pgml.scenarios.en50160` — `en50160_limits()/en50160_limit(order)`. Active file is the
  packaged `standards/en50160.yaml`; an explicit `path=` argument or `PGML_EN50160`
  overrides it.

Both env vars are OPTIONAL overrides (e.g. to ship a revised standard); nothing here
requires an env var to be set.

## defaults.yaml layout (ordered by component; extend as the library grows)
- `calculation.symmetry` — `auto | symmetric | asymmetric`; how an appliance's power is
  distributed across phases (pgml is always phase-domain, so this is operating-point
  resolution, not a network change). `auto` (default) = asymmetric iff any per-phase data
  is present (power-grid-model rule). Resolved by `pgml.assembly._symmetry.resolve_asymmetric`.
- `appliance.load.{default_connection, single_phase_connection}` — WYE/DELTA used for a
  Load/Generator with no explicit `connection` (multi- vs single-phase; both WYE by
  default). Resolved by `pgml.assembly._symmetry.resolve_connection`. See
  `docs/pgml/modeling/asymmetric.md`.
- `line.harmonic_model.{three_phase, single_phase, skin_effect}` — which
  frequency-dependent line model is written to `Line.harmonic_line_model` for an R/X line
  (default 3-phase = `sequence_aware` for 4-wire unbalanced studies; 1-/2-phase =
  `positive_sequence`). Resolved by the converters (`to_grid`, which logs it) and by
  `apply_default_harmonic_model(grid)` for hand-built grids; assembly itself never
  resolves a default. Per-line override: `Line.harmonic_line_model` /
  `Line.harmonic_skin_effect`.
- `line.earth_return.{resistivity_ohm_m, resistance_coeff_ohm_per_m_per_hz,
  reactance_coeff_ohm_per_m_per_hz, x0_frequency, x0_exponent}` — the lumped Carson earth
  path of the `sequence_aware` model (ρ; the `π²·1e-7` Ω/m/Hz resistance coefficient; the
  `μ0` reactance coefficient; whether `X0` scales linearly or with the Carson/Deri
  sub-linear decay; the `X0` exponent). Per-line override: `Line.earth_return`.
- `line.conductor.{gmr_over_radius, radius_m, height_overhead_m, height_cable_m,
  phase_spacing_m}` — R/X→geometry synthesis defaults (`gmr_over_radius = e^{-1/4} =
  0.7788`; `phase_spacing_m` seeds the equilateral 3-phase synthesis fit).
- `line.zero_sequence.{r0_over_r1, x0_over_x1, c0_over_c1, r0_includes_earth_return}` —
  zero/positive-sequence ratios used by the converter when only a positive-sequence
  impedance is given (the converter warns once per grid when it used them), plus whether
  a stored `R0` already contains the earth-return resistance at f0 (which decides how
  much of `R0` the skin multiplier scales).
- `source.{series_impedance_ohm, xr_ratio}` — default slack series impedance synthesis.
- `transformer.vector_group.{from, to, clock}` — winding connections + IEC clock assumed
  for a Transformer with no explicit `from_/to_connection` (default Dyn11). Resolved by
  `pgml.assembly._transformer.resolve_vector_group`; an explicit connection wins. See
  `docs/pgml/modeling/transformer.md`.

## Resolution precedence (highest first; `resolve(key, explicit, converted)`)
1. **explicit** — a value the user set on the component / grid (ALWAYS wins).
2. **defaults** — the value in `defaults.yaml`.
3. **converter** — inferred from a source library during conversion (last resort).

## Consumers (read these constants from `pgml.defaults` — do not re-hard-code)
- `assembly/_transformer.py`, `assembly/_symmetry.py`: vector-group + symmetry/connection.
- `geometry/sequence.py`: `_DEFAULT_GMR_OVER_RADIUS`, `CARSON_EARTH_R_PER_HZ`.
- `geometry/synthesis.py`: `_DEFAULT_HEIGHT`, `_DEFAULT_RADIUS`, `_DEFAULT_EARTH_RHO`,
  and the `apply_*` model/skin/earth-coeff defaults.
- `convert/_common.py`: zero-sequence ratios. `evaluation/oracles/grids.py`: source impedance.
