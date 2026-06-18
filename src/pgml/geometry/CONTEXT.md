# Interface ledger: geometry (Carson/Deri line constants — differentiable)

Conductor geometry -> per-frequency line impedance/admittance, the Phase-2
"geometry -> impedance" path. Closes the harmonic line-impedance gap (OpenDSS applies
an earth-return + skin correction at every harmonic; naive `X∝h` is wrong). Model =
OpenDSS **DERI**, verified **bit-exact** vs OpenDSS (`references/opendss/carson.md`).
Fully torch / autograd-safe / GPU-ready / batched over lines and H frequencies;
gradients flow conductor-geometry -> Z/Yc -> Y-bus -> solve -> outputs.

## carson.py (torch)
- `series_impedance(x, y, gmr, rdc, rho, freqs) -> Z[*B, H, N, N]` (Ω/m): Deri earth
  return (complex penetration depth) + GMR geometric reactance + skin-effect internal
  RESISTANCE (Bessel `I0/I1` via continued fraction `i0_over_i1`); internal reactance
  dropped in the 40–1000 Hz band (carried by GMR), matching OpenDSS.
- `internal_impedance(rdc, freqs) -> Zint[*B, H]` (Ω/m): the skin-effect internal
  impedance alone (Bessel `I0/I1`). Shared by `series_impedance` and `sequence.py`.
- `potential_coefficients(x, y, radius) -> P[*B, N, N]` (Maxwell image method);
  `C = 2*pi*e0 * inv(P)`.
- `kron_reduce(M, n_phase)` eliminates neutral/shield conductors (>= n_phase).
- `line_constants(x, y, gmr, rdc, radius, rho, freqs, n_phase) -> (Z[*B,H,P,P] Ω/m,
  C[*B,P,P] F/m)` phase-reduced. Conductor arrays are `[*B, N]`, phases first.
  NOTE: capacitance is physically correct but not bit-exact to OpenDSS (different
  capradius convention); irrelevant for the c=0 standard feeders. Series Z is exact.

## sequence.py (positive-sequence harmonic model — NO earth floor)
The corrected R/X-line harmonic model. A balanced positive-sequence
current has no net ground current, so earth return CANCELS — `Z1` carries only
internal + geometric, earth return lives only in `Z0`. So `X1(h) = X1·h` (geometric ∝ f)
+ skin on `R1`, NO earth floor (the single-conductor synthesis floor that blew GMR up).
- `positive_sequence_z(r1, x1, f0, freqs, *, skin=True) -> Z1[*B, H]` (Ω/m): direct
  `R1·m_skin(h) + j·X1·(f/f0)`. `X` scales ∝ h to floating point; differentiable in R1/X1.
- `skin_resistance_multiplier(r1, f0, freqs) -> m[*B, H]` (`m(f0)=1`): Bessel `I0/I1`
  internal-resistance growth, earth term dropped. `fit_equivalent_rdc(r1, f0, freqs_ref)`.
- `two_conductor_geometry(r1, x1, f0, *, radius_m, ...) -> dict` + `two_conductor_loop_z(geom,
  freqs) -> Z[H]`: a PHYSICAL go/return Carson loop (reuses `series_impedance`); earth
  cancels in the `[1,-1]` loop transform -> physical GMR/spacing for any X1, agrees with
  `positive_sequence_z` (residual ~earth coupling, ≲2% to h≈25).
- `phase_to_sequence(z_phase[*,3,3]) -> [*,3,3]` (Fortescue `A⁻¹ Z A`);
  `sequence_impedances(z) -> (Z0, Z1, Z2)` diagonal — shows 3-phase geometry keeps earth
  return only in `Z0`.

### Sequence-aware model (UNBALANCED / 4-wire: earth return lives in Z0)
For asymmetric studies the full coupled `Z_abc(h)` is needed: `Z1` earth-free, `Z0`
carrying the earth/neutral return (excited by zero-sequence/residual current).
- `carson_earth_resistance(freqs, *, coeff=π²·1e-7) -> Re[H]` (Ω/m): Carson earth-return
  resistance `Re(f)=coeff·f`, geometry-independent, ∝ f (the zero-seq damping).
- `zero_sequence_harmonic_z(r0, x0, f0, freqs, *, skin=True, earth_resistance_coeff=π²·1e-7)
  -> Z0[*B,H]`: conductor part (`X0∝h`+skin) `+ 3·(Re(f)−Re(f0))`; monotone, never
  non-physical; `coeff=0` -> pure conductor. (Earth REACTANCE sub-linearity is return-path
  dependent -> geometry path; `X0∝h` here.)
- `sequence_to_phase_z(z1, z0) -> [*,H,3,3]`: inverse Fortescue, `Zself=(Z0+2Z1)/3`,
  `Zmutual=(Z0−Z1)/3` (balanced/transposed).
- `sequence_aware_phase_z(r1, x1, r0, x0, f0, freqs, *, skin, earth_resistance_coeff)
  -> Z_abc[*B,H,3,3]`: positive seq (earth-free) + damped zero seq, recombined.
  Differentiable in R1/X1/R0/X0; batched.

## synthesis.py
- `synthesize_line_geometry(r1, x1, *, f0, phase, line_type, ...) -> LineGeometry` —
  single-conductor earth-return geometry reproducing `R1 + jX1` (Ω/m) at f0 (GMR sets
  reactance, Rdc the resistance via skin fixed-point); provenance records the synthesis.
  WARNS / flags `synth_unphysical` when X1 is below the earth floor (cables / low-X).
- `synthesize_grid_geometry(grid, *, f0=None) -> grid` (in place) gives every
  single-phase R/X line a `conductor_geometry`. For R/X feeders (IEEE-33, CIGRE LV)
  that ship no geometry; the SAME geometry is fed to pgml and OpenDSS for the harmonic
  comparison (apples-to-apples Carson). Validation vehicle; non-physical for low-X.
- `apply_positive_sequence_harmonic_model(grid, *, f0=None, skin=True) -> grid` (in place,
  RECOMMENDED for R/X feeders): sets each R/X line's `resistance_frequency` to the
  `carson_skin_multiplier` law; the explicit R/L path then gives `X(h)=X1·h` + skin on R,
  NO geometry, NO earth floor. `positive_sequence_resistance_model(r1, *, f0)` builds the
  model object. Assembly `_resistance_multiplier` evaluates the law differentiably (and
  also supports `curve` multipliers via linear interp).
- `apply_sequence_aware_harmonic_model(grid, *, skin=True, earth_resistance_coeff=None)
  -> grid` (in place, for UNBALANCED 4-wire studies): tags each 3-phase R/X line
  `harmonic_line_model=sequence_aware`; assembly `_stamp_sequence_aware_lines` decomposes
  `Z_abc(f0)` -> `Z1`/`Z0`, frequency-corrects each (`sequence_aware_phase_z`), and stamps
  `Z_abc(h)`. Needs a full 3×3 R/L matrix (off-diagonals carry `Z0`). Untagged lines and
  geometry lines unaffected.

## Schema (grid_schema.py)
- `ConductorPlacement(phase, x_m, y_m, gmr_m, radius_m, r_dc_ohm_per_m, is_neutral)` —
  physical fields tensor-capable (autograd through geometry).
- `LineGeometry(conductors, earth_resistivity_ohm_m=100, provenance)`.
- `Line.conductor_geometry: Optional[LineGeometry]` — when set, assembly
  (`_stamp_geometry_lines` in `assembly/ybus.py`) uses Carson for Z(h)/Yc(h) instead of
  explicit R/L/C. Lines group by (n_phase, n_cond) and batch through `line_constants`.

## Validation
- `series_impedance`/`line_constants` vs OpenDSS geometry lines: relZ ~1e-13 (single +
  3ph+neutral Kron) across 50–750 Hz. Assembly geometry path vs `line_constants`:
  ~6e-16. Synthesis reproduces R1/X1 at f0 to ~1e-10. gradcheck passes w.r.t. Rdc, GMR,
  height. Tests: `tests/reference/test_carson_opendss.py`,
  `tests/differentiability/test_carson_gradcheck.py`.
- Positive-sequence model: `Z1` from a genuine 3-phase Carson geometry scales ∝ h to
  ~1e-3 while `Z0` carries the earth floor (`X0(h)/(h·X0(f0))→0.88`, `R0/R1≈5`);
  `positive_sequence_z` X is ∝ h to floating point and agrees with the two-conductor
  Carson loop. Native-OpenDSS oracle: 3-phase R/X -> `Z1=R1+jX1·(f/f0)` (pgml default),
  1-phase -> earth floor.
- Sequence-aware (unbalanced) model: `Z0(h)` gains a frequency-growing earth-return
  resistance the positive sequence lacks; `Z_abc(h)` recombines to exactly `(Z0,Z1,Z1)`;
  a tagged 3-phase line assembled at harmonics recovers an earth-free `Z1` and a damped
  `Z0`. gradcheck w.r.t. R1/X1/R0/X0 and through both assembly paths; CPU/CUDA parity.
  Tests: `tests/reference/test_carson_sequence.py`,
  `tests/differentiability/test_sequence_gradcheck.py`, `tests/gpu/test_device_parity.py`.
  Decision record: `references/positive_sequence_harmonic_line_model.md`.
