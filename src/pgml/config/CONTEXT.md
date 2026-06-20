# Interface ledger: config (modeling defaults — single source of truth)

Every default VALUE and default MODEL choice lives in `defaults.yaml`, each with a
value, units and a short description — so modeling decisions are deliberate and
documented, never hidden implicit constants scattered in code.

## Resolution precedence (highest first)
1. **explicit** — a value the user set on the component / grid (ALWAYS wins).
2. **config** — the default in `defaults.yaml`.
3. **converter** — inferred from a source library during conversion (e.g. an OpenDSS
   `LineGeometry`); the last-resort fallback when 1 and 2 do not apply.

`resolve(key, explicit=None, converted=None)` implements exactly this order.

## API (`pgml.config`)
- `get(key, default=_RAISE)` — value at a dotted `key`
  (e.g. `"line.earth_return.resistivity_ohm_m"`); raises `KeyError` if absent and no
  `default` given.
- `describe(key)` / `units(key)` — the documentation / units string at `key`.
- `resolve(key, explicit=None, converted=None)` — precedence resolution.
- `defaults()` — the parsed dict (cached). `reload(path=None)` — reload (tests /
  user override); `PGML_CONFIG=/path/to.yaml` overrides the packaged file.

## defaults.yaml layout (ordered by component; extend as the library grows)
- `calculation.symmetry` — `auto | symmetric | asymmetric`; how an appliance's power is
  distributed across phases (pgml is always phase-domain, so this is operating-point
  resolution, not a network change). `auto` (default) = asymmetric iff any per-phase data
  is present (power-grid-model rule). Resolved by `pgml.assembly._symmetry.resolve_asymmetric`.
- `appliance.load.{default_connection, single_phase_connection}` — WYE/DELTA used for a
  Load/Generator with no explicit `connection` (multi- vs single-phase; both WYE by
  default). Resolved by `pgml.assembly._symmetry.resolve_connection`. See
  `references/asymmetric_modeling.md`.
- `line.harmonic_model.{three_phase, single_phase, skin_effect}` — which
  frequency-dependent line model `apply_default_harmonic_model(grid)` applies to an R/X
  line (default 3-phase = `sequence_aware` for 4-wire unbalanced studies; 1-/2-phase =
  `positive_sequence`).
- `line.earth_return.{resistivity_ohm_m, resistance_coeff_ohm_per_m_per_hz}` — Carson
  earth path (ρ; the `π²·1e-7` Ω/m/Hz earth-return resistance coefficient).
- `line.conductor.{gmr_over_radius, radius_m, height_overhead_m, height_cable_m,
  phase_spacing_m}` — R/X→geometry synthesis defaults (`gmr_over_radius = e^{-1/4} =
  0.7788`; `phase_spacing_m` seeds the equilateral 3-phase synthesis fit).
- `transformer.vector_group.{from, to, clock}` — winding connections + IEC clock assumed
  for a Transformer with no explicit `from_/to_connection` (default Dyn11). Resolved by
  `pgml.assembly._transformer.resolve_vector_group`; an explicit connection wins. See
  `references/opendss/transformer.md`.

## Consumers (read these constants from the config — do not re-hard-code)
- `geometry/sequence.py`: `_DEFAULT_GMR_OVER_RADIUS`, `CARSON_EARTH_R_PER_HZ`.
- `geometry/synthesis.py`: `_DEFAULT_HEIGHT`, `_DEFAULT_RADIUS`, `_DEFAULT_EARTH_RHO`,
  and the `apply_*` model/skin/earth-coeff defaults; `apply_default_harmonic_model`
  dispatches on `line.harmonic_model.*`.
