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
- `potential_coefficients(x, y, radius) -> P[*B, N, N]` (Maxwell image method);
  `C = 2*pi*e0 * inv(P)`.
- `kron_reduce(M, n_phase)` eliminates neutral/shield conductors (>= n_phase).
- `line_constants(x, y, gmr, rdc, radius, rho, freqs, n_phase) -> (Z[*B,H,P,P] Ω/m,
  C[*B,P,P] F/m)` phase-reduced. Conductor arrays are `[*B, N]`, phases first.
  NOTE: capacitance is physically correct but not bit-exact to OpenDSS (different
  capradius convention); irrelevant for the c=0 standard feeders. Series Z is exact.

## synthesis.py
- `synthesize_line_geometry(r1, x1, *, f0, phase, line_type, ...) -> LineGeometry` —
  single-conductor earth-return geometry reproducing `R1 + jX1` (Ω/m) at f0 (GMR sets
  reactance, Rdc the resistance via skin fixed-point); provenance records the synthesis.
- `synthesize_grid_geometry(grid, *, f0=None) -> grid` (in place) gives every
  single-phase R/X line a `conductor_geometry`. For R/X feeders (IEEE-33, CIGRE LV)
  that ship no geometry; the SAME geometry is fed to pgml and OpenDSS for the harmonic
  comparison (apples-to-apples Carson).

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
