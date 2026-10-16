# convert.opendss

OpenDSS circuits to and from the pgml `Grid`. Inbound reads the live
`opendssdirect` engine (a circuit already loaded and solved, or `Calcvoltagebases`
called); outbound builds a live circuit for conformance checks.

Signatures: docstrings / API_INDEX.md. Cross-tool conventions (bases, transformer
referral, vector-group mechanics, source impedance, converter coverage):
`docs/pgml/modeling/conventions.md` and `docs/pgml/modeling/references/opendss/`.

## Codemap

- `__init__.py` — public surface: `to_grid`, `from_grid`, `OpenDSSExport`,
  `UnsupportedGridError`, `PhaseMode` (re-exported from `convert._common`).
- `converter.py` — `to_grid`. One pass per DSS element class, in order: buses
  (`YNodeOrder`), Vsource, Line, Transformer, Load, Capacitor, Reactor, Generator,
  PVSystem, Storage, InvControl (`_import_inv_controls`), then the generic
  dropped-element sweep over `Circuit.AllElementNames()`. Helpers that carry
  non-obvious rules, each documented in its own docstring:
  - `_parse_bus_connection` — Line terminals; `bus2` is parsed independently, so a
    phase-permuted line keeps its own `to_phases`.
  - `_parse_appliance_bus_connection` / `_resolve_wye_return_path` — Load,
    Generator, PVSystem, Storage terminals from `CktElement.NodeOrder()`.
  - `_parse_transformer_winding_bus` / `_cyclic_rotation_steps` — winding phases,
    grounding and clock rotation.
  - `_positive_sequence_scalar` — the single-phase-equivalent line reduction.
  - `_resolve_load_model` — `Loads.Model()` → `LoadModel` / `ZipCoefficients`.
  - `_grounded_shunt_phases` / `_shunt_phase_conductors` / `_build_shunt_appliance`
    — Capacitor and Reactor → `ShuntAppliance`.
  - `_generator_voltage_regulation` — `model=3` → `VoltageRegulation`.
  - `_der_harmonic_model` — native DER harmonic impedance and spectrum.
  - `_apply_line_earth_return` — `earth_return="opendss"`.
- `exporter.py` — `from_grid` → `OpenDSSExport` (live circuit + `ConversionReport`).
  The circuit itself is written by `pgml.evaluation.oracles.export_grid_to_opendss`;
  this module adds the matched earth-return parameters, the approximation guard for
  control laws / voltage regulation, and the report.
- Reverse transformer builder for oracle tests:
  `evaluation/oracles/opendss_oracle.py::_build_circuit_with_real_transformer`
  (with `_dss_leadlag_and_rotation`, `_dss_rotated_phase_suffix`). It is the exact
  inverse of the inbound clock and coil-referral rules below; change both together.

## Contract

- `to_grid` is pure with respect to pgml state: it only reads the active DSS engine.
  It returns `(Grid, id_map)`, or `(Grid, id_map, ConversionReport)` with
  `return_report=True`. The `Grid` is materialised (no `type_ref`).
- `id_map` keys: `bus`, `line`, `trafo`, `load`, `vsource`, `capacitor`, `reactor`,
  `generator`, `pvsystem`, `storage` (each lowercase DSS name → schema id), plus
  `slack_v_complex`: the first Vsource phasor `BasekV·pu·1000·e^{jθ}`, usable as
  `v_fixed` for an ideal slack without further scaling.
- Node ids follow `Circuit.YNodeOrder()` first appearance, so `node_phase_index`
  rows align with OpenDSS `SystemY` rows. Oracle tests depend on this order.
- `phase_mode`: `SINGLE_PHASE_EQUIV` (default) gives every node `phases=(A,)`, 1×1
  line values and `connection=None` loads (a delta load collapses with an INFO
  log); `THREE_PHASE` gives the real DSS phases (`Phase.N` for a neutral conductor),
  full n×n matrices and `connection` from `IsDelta()` (single-phase loads are
  always WYE).
- Nothing is dropped silently: each unhandled element class warns once with kind
  and count and is recorded in the report. Out-of-scope inputs raise
  `ConversionError` instead: three-winding transformers, differing per-winding kVA,
  non-zero neutral node or `Rneut`/`Xneut` earthing impedance, non-cyclic winding
  phase order, a DELTA `model=3` generator.

## Conventions an edit must keep

- Units: SI throughout. Line matrices: `RMatrix`/`XMatrix` Ω per length unit,
  `CMatrix` nF per length unit, converted with the length-unit code table
  `_DSS_UNIT_TO_METERS` (code 0 = total values, length 1 m); `L = X/(2πf₀)`, with
  `f₀ = Solution.Frequency()`.
- `u_rated_v` is line-to-line for every bus: `kVBase()·√3·1000` (`kVBase()` is
  always L-N). `u_ref_v = BasekV·pu·1000` with no phase-count branch: `basekv` is
  L-L for a ≥3-phase Vsource and used unscaled as the conductor EMF for a 1-phase
  one, and both formulas mirror what OpenDSS solves.
- Phase numbers: 1=A, 2=B, 3=C, 0=ground; an explicit 4th conductor is `Phase.N`.
- `Phases=` of a PC element counts phase conductors only; OpenDSS appends one
  return conductor for WYE. The converter reads `NodeOrder()` (OpenDSS's resolved
  assignment), never the raw bus string. Each WYE appliance carries its own
  `return_path` (`neutral` for an explicit `.4`, `ground` for a grounded element on
  a bus that has `Phase.N` from another element, else `auto`), so two elements on
  one four-wire bus can return differently.
- `First()`/`Next()` iterators skip disabled elements; no `Enabled()` check is
  needed (verified on opendssdirect 0.9.4).
- Line, single-phase equivalent: a native one-phase line keeps its scalar; a coupled
  multi-phase line uses `Z1 = Z_self − Z_mutual` (and `C1 = C_self − C_mutual`, the
  Maxwell off-diagonals are negative). For a line given by `r1/x1` this recovers
  `r1/x1` exactly.
- Transformer: winding 1 = HV/from, winding 2 = LV/to. Leakage
  `R = (%R₁+%R₂)/100·Z_base_LV`, `X = %XHL/100·Z_base_LV`, `Z_base_LV = kV_lv²·1000/kVA`,
  then referred to the TO coil: ×3 for a DELTA LV winding. Magnetizing: `%noloadloss`
  → G and `%imag` → B separately, each `/100·S/u_hv²`, referred to the HV terminal;
  no Pythagorean subtraction (unlike pandapower's `i0_percent`). Where it is stamped
  is `transformer.magnetizing_placement` (the OpenDSS preset selects `to_terminal`).
- Winding grounding follows the bus-string shape: no `(n+1)`-th conductor or an
  explicit `.0` is solid grounding; the `Rneut` getter reads `-1` ("not set") unless
  given, so it cannot decide grounding alone. An explicit non-zero neutral node is
  its own `YNodeOrder` row.
- Clock: `shift_deg = (base + 120·r_from − 120·r_to) mod 360`, `base` from `LeadLag`
  (30 Lag/ANSI, 330 Lead/Euro for Dy/Yd; 0 for Yy/Dd). Stored phases are always
  canonical `(A, B, C)`; the transformer stamp assumes winding position k = bus
  phase k. Clocks 2, 6, 10 and zigzag cannot be expressed in OpenDSS.
- Load models: 1 → `CONST_POWER` (returned as `(None, None)`, the schema default),
  2 → `CONST_IMPEDANCE`, 5 → `CONST_CURRENT`, 8 → `ZipCoefficients` from the first
  six `ZipV` values (a non-zero cutoff warns), 3/4/6/7 → `CONST_POWER` with a
  warning. `Vminpu`/`Vmaxpu`/`Vlowpu` bands are not modelled; conversion warns once
  per distinct band tuple.
- Shunts: Capacitor `Cuf` (µF per step) summed over active steps; for DELTA it is
  the per-leg value. Reactor `X` → `ShuntAppliance.inductance_h` (susceptance falls
  as 1/h); series R and X together convert to the f₀-equivalent parallel pair with a
  warning. Coupled `Rmatrix`/`Xmatrix` reactors are skipped with a warning. A DELTA
  bank under `SINGLE_PHASE_EQUIV` folds to `Y_wye = 3·Y_leg`.
- DER: Generator/PVSystem/Storage are read at their present solved kW/kvar (PV
  already derated; reported as `pvsystem.snapshot`). Storage `kW` is already
  discharge-positive. `model=3` `Vpu` is per unit of the machine's own `kV` (L-L for
  ≥2 phases) and is re-referred to the host node's base.
- Native DER harmonics (`der_harmonics=True`): machine = pure `Xdpp`; inverter
  `%R/%X` on device kV/kVA, a three-phase DELTA coil impedance is three times the
  star equivalent; the block selects `opendss_admittance` / `opendss_voltage`;
  spectra keep `%mag/100` and `angle_h − h·angle_1`; an empty spectrum keeps the
  shunt.
- A non-negligible Vsource `R1`/`X1` (> 1e-4 Ω) warns: it is unused under the
  default `slack="ideal"` and needs `slack="norton"`.

## Testing note

`opendssdirect` wraps one process-global engine and `Clear` does not reset
`DefaultBaseFrequency`; every DSS-driven test reasserts it after `Clear` (see
`_dss_clear` in `tests/reference/test_opendss_transformer.py`). For a `SystemY`
comparison build the circuit without loads: `SystemY` carries the Vsource Norton
shunt and voltage-dependent load shunts (`tests/reference/test_ieee33_opendss.py`).
