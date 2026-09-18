# Interface ledger: geometry (Carson/Deri line constants — differentiable)

Conductor geometry -> per-frequency line impedance/admittance, the
"geometry -> impedance" path. Closes the harmonic line-impedance gap (OpenDSS applies
an earth-return + skin correction at every harmonic; naive `X∝h` is wrong). Model =
OpenDSS **DERI**; on the same geometry it agrees with OpenDSS to **4.6e-8 relative** on
`Z` and **2.1212e-5** on `C`, which is exactly the difference between the SI physical
constants used here and OpenDSS's truncated `mu0`/`e0`
(`docs/pgml/modeling/references/opendss/carson.md`). That holds below 1 kHz with the
default conductor internal-inductance model and at EVERY frequency with
`internal_inductance="gmr_power_frequency"` (see `carson.py` below).
Fully torch / autograd-safe / GPU-ready / batched over lines and H frequencies;
gradients flow conductor-geometry -> Z/Yc -> Y-bus -> solve -> outputs.

## carson.py (torch)
- `series_impedance(x, y, gmr, rdc, rho, freqs, *, radius=None,
  internal_inductance=None, power_frequency_band_hz=None)
  -> Z[*B, H, N, N]` (Ω/m): Deri earth return (complex penetration depth) + geometric
  reactance + skin-effect internal impedance (Bessel `I0/I1` via continued fraction
  `i0_over_i1`). The internal RESISTANCE is always present; `internal_inductance`
  selects the self-term spacing radius and the internal REACTANCE:
  - `"gmr"` (default): published GMR at every frequency, internal reactance dropped.
    The internal inductance stays at its power-frequency value; needs no `radius`.
  - `"gmr_skin"`: effective radius `radius*(gmr/radius)^g(f)` with `g` from
    `internal_reactance_ratio`. Continuous in f, exact at power frequency, and identical
    to `"bessel"` when `GMR = e^(-1/4)*radius`.
  - `"gmr_power_frequency"`: `"gmr"` inside `power_frequency_band_hz` (default
    `(40, 1000)` Hz, exclusive) and `"bessel"` outside — OpenDSS's rule
    (`LineConstants.pas`: `if (f < 1000.0) and (f > 40.0)`). Reproduces OpenDSS at every
    frequency; `Z(f)` steps at the band edges.
  - `"bessel"`: physical radius + the full `Im(Zint)` at every frequency, i.e. the
    first-principles solid round conductor.
  Every model except `"gmr"` needs `radius`; an unknown name raises `InputError`. The
  frequency axis is an INPUT, so the band mask carries no gradient and all four models
  are differentiable and GPU-safe.
- `internal_impedance(rdc, freqs) -> Zint[*B, H]` (Ω/m): the skin-effect internal
  impedance alone (Bessel `I0/I1`). Shared by `series_impedance` and `sequence.py`.
  Verified against the analytic solid-round form `(k*rho_c/(2*pi*a))*I0(ka)/I1(ka)`
  (scipy) to 1e-13 relative.
- `internal_reactance_ratio(rdc, freqs) -> g[*B, H]`: `Im(Zint)/(f*mu0/4)`, the internal
  inductance normalised by its uniform-current-density value (`g -> 1` at DC, `g -> 0`
  under full skin effect). The physical yardstick for how far a power-frequency GMR is
  off at a given harmonic.
- `potential_coefficients(x, y, radius) -> P[*B, N, N]` (Maxwell image method);
  `C = 2*pi*e0 * inv(P)`.
- `kron_reduce(M, n_phase)` eliminates neutral/shield conductors (>= n_phase).
- `line_constants(x, y, gmr, rdc, radius, rho, freqs, n_phase, *,
  internal_inductance=None, power_frequency_band_hz=None) -> (Z[*B,H,P,P] Ω/m,
  C[*B,P,P] F/m)` phase-reduced. Conductor arrays are `[*B, N]`, phases first.
  NOTE: capacitance differs from OpenDSS's by 2.1212e-5 relative (the `e0` constant
  ratio), always uses the physical radius, and OpenDSS's `capradius` option is not read;
  irrelevant for the c=0 standard feeders. Series Z agrees to 4.6e-8 relative (the `mu0`
  constant ratio).
- Module constants `INTERNAL_INDUCTANCE_MODELS`, `INTERNAL_INDUCTANCE` and
  `POWER_FREQUENCY_BAND_HZ` mirror `line.geometry.internal_inductance` and
  `line.geometry.power_frequency_band_hz` at import time for inspection. The public
  helpers and `assembly/ybus.py` resolve omitted arguments at call/assembly time, so a
  defaults override or modeling preset applies without reimporting; explicit arguments
  and per-line fields still win.

## sequence.py (positive-sequence harmonic model — NO earth floor)
The corrected R/X-line harmonic model. A balanced positive-sequence
current has no net ground current, so earth return CANCELS — `Z1` carries only
internal + geometric, earth return lives only in `Z0`. So `X1(h) = X1·h` (geometric ∝ f)
+ skin on `R1`, NO earth floor (the single-conductor synthesis floor that blew GMR up).
- `positive_sequence_z(r1, x1, f0, freqs, *, skin=True) -> Z1[*B, H]` (Ω/m): direct
  `R1·m_skin(h) + j·X1·(f/f0)`. `X` scales ∝ h to floating point; differentiable in R1/X1.
- `skin_resistance_multiplier(r1, f0, freqs) -> m[*B, H]` (`m(f0)=1`): Bessel `I0/I1`
  internal-resistance growth, earth term dropped. `fit_equivalent_rdc(r1, f0, freqs_ref)`.
  A request for the REFERENCE frequency alone returns ones directly: `m(f0)` is the exact
  constant 1 for every `R1` (numerator and denominator are the same expression) with an
  exactly zero derivative, and evaluating it anyway cost a twelve-step fit plus two
  forty-term continued fractions — which is what every FUNDAMENTAL assembly of a feeder
  with this line model was paying (4 ms of a 9.5 ms IEEE-33 solve, CPU, complex128). The
  shortcut is skipped while `freqs` or `f0` carries a gradient.
- `two_conductor_geometry(r1, x1, f0, *, radius_m, ...) -> dict` + `two_conductor_loop_z(geom,
  freqs) -> Z[H]`: a PHYSICAL go/return Carson loop (reuses `series_impedance` with
  `internal_inductance="gmr"` PINNED, matching `positive_sequence_z`'s strictly
  frequency-proportional reactance); earth cancels in the `[1,-1]` loop transform ->
  physical GMR/spacing for any X1, agrees with `positive_sequence_z` (residual ~earth
  coupling, ≲2% to h≈25).
- `phase_to_sequence(z_phase[*,3,3]) -> [*,3,3]` (Fortescue `A⁻¹ Z A`);
  `sequence_impedances(z) -> (Z0, Z1, Z2)` diagonal — shows 3-phase geometry keeps earth
  return only in `Z0`.

### Sequence-aware model (UNBALANCED / 4-wire: earth return lives in Z0)
For asymmetric studies the full coupled `Z_abc(h)` is needed: `Z1` earth-free, `Z0`
carrying the earth/neutral return (excited by zero-sequence/residual current).
- `carson_earth_resistance(freqs, *, coeff=π²·1e-7) -> Re[H]` (Ω/m): Carson earth-return
  resistance `Re(f)=coeff·f`, geometry-independent, ∝ f (the zero-seq damping).
- `zero_sequence_harmonic_z(r0, x0, f0, freqs, *, skin=True,
  earth_resistance_coeff=π²·1e-7, earth_reactance_coeff=μ0, x0_frequency=None, x0_nonnegative=None,
  x0_exponent=1.0, r0_includes_earth_return=False, phase_resistance=None) -> Z0[*B,H]`:
  `R0(h) = R_phase·m_skin(h) + (R0_cond − R_phase) + 3·(Re(f) − Re_offset)` with
  `R_phase = min(R1, R0_cond)` and the skin curve fitted to `phase_resistance = R1`
  (`sequence_aware_phase_z` passes it; without it the whole `R0_cond` is one fictitious
  conductor, which understates the rise: m = 1.07 instead of 1.63 at h = 25 for a
  150 mm² Al conductor with R0 = 4·R1), and
  `X0(h) = X0·h^p [− 1.5·kx·f0·h·ln h if carson_sublinear]`, exact at f0 for every
  option. `r0_includes_earth_return` chooses whether the stored `R0` already contains
  `3·Re(f0)` (real zero-sequence data; then the earth part is excluded from the skin
  multiplier) or is conductor-only (an `R0/R1`-ratio value; the earth return is added as
  an increment). `x0_frequency="carson_sublinear"` is the lumped Carson/Deri reactance
  decay (ρ-independent, reproduces OpenDSS's `Xg` correction) for an overhead line whose
  stored X0 contains the deep-earth term, with non-negative clamping; `"linear"` is the
  default and omits the correction (cables, ratio-derived X0).
  `x0_nonnegative=False` selects the unguarded reference law; the guard has zero
  gradient below its boundary.
  `x0_sublinear_deficit(x0, f0, freqs, *, earth_reactance_coeff, x0_exponent) -> bool[*B,H]`
  marks where the sub-linear law is negative (assembly turns it into one warning).
  Every coefficient may be a tensor
  (differentiable, batched over lines). Defaults come from `line.earth_return.*` /
  `line.zero_sequence.r0_includes_earth_return`; module constants
  `CARSON_EARTH_R_PER_HZ`, `CARSON_EARTH_X_PER_HZ`, `X0_FREQUENCY`, `X0_EXPONENT`,
  `R0_INCLUDES_EARTH_RETURN`.
- `sequence_to_phase_z(z1, z0) -> [*,H,3,3]`: inverse Fortescue, `Zself=(Z0+2Z1)/3`,
  `Zmutual=(Z0−Z1)/3` (balanced/transposed).
- `sequence_aware_phase_z(r1, x1, r0, x0, f0, freqs, *, skin, earth_resistance_coeff,
  earth_reactance_coeff, x0_frequency, x0_exponent, r0_includes_earth_return)
  -> Z_abc[*B,H,3,3]`: positive seq (earth-free) + damped zero seq, recombined. The
  zero-sequence keywords are those of `zero_sequence_harmonic_z`. Differentiable in
  R1/X1/R0/X0 and in every coefficient; batched.

## synthesis.py
- `synthesize_line_geometry(r1, x1, *, f0, phase, line_type, ...) -> LineGeometry` —
  single-conductor earth-return geometry reproducing `R1 + jX1` (Ω/m) at f0 (GMR sets
  reactance, Rdc the resistance via skin fixed-point); provenance records the synthesis.
  WARNS / flags `synth_unphysical` when X1 is below the earth floor (cables / low-X).
- `synthesize_three_phase_geometry(r1, x1, x0, *, f0, phases, line_type, ...) ->
  LineGeometry` — equilateral 3-conductor geometry reproducing the line's `Z1` (R1, X1)
  AND zero-sequence reactance `X0` at f0. GMR + spacing are Newton-fitted on the Carson
  forward (constant analytic Jacobian `coef·[[1,-1],[-2,-1]]`, `coef=f0·MU0`) to match
  (X1, X0); Rdc fits R1. `R0` is NOT a free target — it follows from the Carson earth
  return (`R0 ≈ R1 + 3·R_earth(f0)`), so a sequence dataset's assumed R0 is replaced by
  the geometry's physical value (recorded in `provenance.extra.synth_r0_ohm_per_m`).
  Reproduces X1/X0/R1 to ~1e-13 via pgml's own Carson. `synth_unphysical` flags
  GMR ≥ radius (low-X). Spacing seed: config `line.conductor.phase_spacing_m`.
- `synthesize_grid_geometry(grid, *, f0=None) -> grid` (in place) gives every R/X line a
  `conductor_geometry`: single-phase -> single-conductor; 3-phase ->
  `synthesize_three_phase_geometry` (2-phase skipped). For R/X feeders (IEEE-33,
  CIGRE LV) that ship no geometry; the SAME geometry is fed to pgml and OpenDSS for the
  harmonic comparison (apples-to-apples Carson, incl. the triplen / zero-sequence
  orders). Validation vehicle; non-physical for low-X.
- `strip_grid_geometry(grid) -> grid` (in place) — the inverse: drops every line's
  `conductor_geometry` AND its harmonic model, so the lines return to explicit R/L/C with
  an UNRESOLVED model (ready for `apply_default_harmonic_model`). Use it to compare the
  lumped models against the geometry model on the same feeder; a bare
  `line.conductor_geometry = None` now contradicts `harmonic_line_model="geometry"` and
  is rejected by the schema.
- `apply_positive_sequence_harmonic_model(grid, *, f0=None, skin=None) -> grid` (in place,
  RECOMMENDED for R/X feeders): sets `Line.harmonic_line_model="positive_sequence"`
  (and `harmonic_skin_effect` only for an explicit `skin`; `None` leaves it unset so
  `line.harmonic_model.skin_effect` resolves at assembly, like every other option of
  the lumped models, and a preset applied after conversion still acts); assembly then derives the skin multiplier from the line's OWN
  positive-sequence resistance (mean diagonal minus mean mutual), scales the CONDUCTOR
  part of the R matrix only (the mutual entries are the earth-return path) and gives
  `X(h)=X1·h`, NO geometry, NO earth floor. `positive_sequence_resistance_model(r1, *, f0)`
  still builds a `carson_skin_multiplier` `resistance_frequency` object for a line that
  carries a user-supplied law instead of a typed model; assembly's
  `_resistance_multiplier` evaluates that law differentiably (and `curve` multipliers via
  linear interp).
- `apply_sequence_aware_harmonic_model(grid, *, skin=None, earth_resistance_coeff=None)
  -> grid` (in place, for UNBALANCED 4-wire studies): sets
  `Line.harmonic_line_model="sequence_aware"` on each 3-phase R/X line (+
  `harmonic_skin_effect`, and `earth_return` when a coefficient is passed); assembly
  decomposes `Z_abc(f0)` -> `Z1`/`Z0`, frequency-corrects each
  (`sequence_aware_phase_z`), and stamps `Z_abc(h)`. Needs a full 3×3 R/L matrix
  (off-diagonals carry `Z0`). Other lines unaffected.
- `resolve_harmonic_line_models(grid, *, f0=None, model=None, skin=None) -> {model: count}`
  — resolve every UNRESOLVED R/X line from the modeling defaults (or from `model`), and
  report what was applied; `apply_default_harmonic_model(grid, *, f0=None, skin=None,
  model=None) -> grid` is the same thing returning the grid. The converters call the
  former at conversion time and log the counts, so only hand-built grids need the call.

## Schema (grid_schema.py)
- `ConductorPlacement(phase, x_m, y_m, gmr_m, radius_m, r_dc_ohm_per_m, is_neutral)` —
  physical fields tensor-capable (autograd through geometry).
- `LineGeometry(conductors, earth_resistivity_ohm_m=100, provenance,
  internal_inductance=None)`; explicit `gmr`/`gmr_skin`/`gmr_power_frequency`/`bessel`
  overrides the active default, and persists through grid JSON.
- `Line.conductor_geometry: Optional[LineGeometry]` — when set, assembly
  (`_geometry_block_groups` in `assembly/ybus.py`) uses Carson for Z(h)/Yc(h) instead of
  explicit R/L/C. Lines group by (n_phase, n_cond, resolved internal_inductance) and batch through `line_constants`.
- `Line.harmonic_line_model: Optional[Literal["geometry","sequence_aware",
  "positive_sequence","naive"]]`, `Line.harmonic_skin_effect: Optional[bool]` and
  `Line.earth_return: Optional[EarthReturnModel]` — the TYPED model selector (no free-text
  tags). `None` = unresolved: assembly uses the stored parameters as given, and the
  converters / `apply_default_harmonic_model` resolve it from `line.harmonic_model.*`.
  `EarthReturnModel(resistance_coeff_ohm_per_m_per_hz, reactance_coeff_ohm_per_m_per_hz,
  x0_frequency, x0_exponent, r0_includes_earth_return)` overrides the lumped earth path
  per line; every field is tensor-capable and part of the differentiable path.

## Validation
- `series_impedance`/`line_constants` vs OpenDSS geometry lines across 50–750 Hz:
  relZ 4.66e-8 (single conductor) / 4.81e-8 (3ph+neutral Kron) — the SI-vs-truncated
  `mu0` ratio (4.89e-8), i.e. the MODEL matches to floating point; relC 2.1212e-5 = the
  `e0` ratio exactly. Both are pinned as such (a test asserts `C`'s deviation EQUALS the
  constant ratio). At 1050 Hz the DEFAULT model steps to 1.2e-2 because OpenDSS leaves
  its GMR band at 1 kHz; `internal_inductance="gmr_power_frequency"` holds 4.6e-8 from
  20 Hz to 2.5 kHz on both geometries and across both band edges. Assembly geometry path
  vs `line_constants`: ~6e-16. Synthesis reproduces R1/X1 at f0 to ~1e-10. gradcheck
  passes w.r.t. Rdc, GMR, radius, height and rho for every internal-inductance model;
  CPU/CUDA parity for every model. Tests: `tests/reference/test_carson_opendss.py`,
  `tests/reference/test_carson_internal_inductance.py`,
  `tests/differentiability/test_carson_gradcheck.py`, `tests/gpu/test_device_parity.py`.
- Internal-inductance models, measured: on a solid round 150 mm² Al conductor (where the
  Bessel solution is exact) `gmr_skin` and `bessel` reproduce the first-principles `Z(f)`
  to 4e-16, while the default `gmr` is 0.24 % (median) off below 1 kHz and 1.0 % above.
  On the published ACSR of the OpenDSS line-constants example the OpenDSS rule steps
  0.05 % (336.4 kcmil, GMR/radius = 0.826) to 8.8 % (1/0, GMR/radius = 0.269) at 1 kHz,
  although only 3 % of that conductor's internal inductance has decayed there — the step
  is the GMR substitution, not skin effect. NEVER combine a radius-based model with a
  `synthesize_grid_geometry` geometry: its radius is a placeholder, and the CIGRE LV
  feeder's `Z` then moves by a factor of 20 above 1 kHz (assembly warns).
- Positive-sequence model: `Z1` from a genuine 3-phase Carson geometry scales ∝ h to
  ~1e-3 while `Z0` carries the earth floor (`X0(h)/(h·X0(f0))→0.88`, `R0/R1≈5`);
  `positive_sequence_z` X is ∝ h to floating point and agrees with the two-conductor
  Carson loop. Native-OpenDSS oracle: 3-phase R/X -> `Z1=R1+jX1·(f/f0)` (pgml default),
  1-phase -> earth floor.
- Sequence-aware (unbalanced) model: `Z0(h)` gains a frequency-growing earth-return
  resistance the positive sequence lacks; `Z_abc(h)` recombines to exactly `(Z0,Z1,Z1)`;
  a 3-phase line assembled at harmonics recovers an earth-free `Z1` and a damped `Z0`.
  Against OpenDSS's own R/X-line model with MATCHED earth parameters (`Rg = coeff·f0`,
  `Xg` from the reactance coefficient, `DefaultBaseFrequency = f0`) and `skin=False`, the
  lumped `Z_abc(h)` agrees to ~1e-10 relative at h = 1..25 for BOTH `x0_frequency` laws;
  with `skin=True` (a pgml refinement OpenDSS does not apply to an R/X line) the
  deviation is the skin rise alone (1.7 % at h=3 to 9.6 % at h=25 on `Z1` for an LV
  cable). gradcheck w.r.t. R1/X1/R0/X0, the earth coefficients and the X0 exponent, and
  through both assembly paths; CPU/CUDA parity. Tests:
  `tests/reference/test_carson_sequence.py`,
  `tests/reference/test_lumped_sequence_opendss.py`,
  `tests/differentiability/test_sequence_gradcheck.py`, `tests/gpu/test_device_parity.py`.
  Decision record: `docs/pgml/modeling/harmonic-line-model.md`.
