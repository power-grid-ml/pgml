# Interface ledger: convert.pgm (power-grid-model -> our schema)

Converts a power-grid-model ``input_data`` dict (structured numpy arrays) to a
schema ``Grid`` and an ``id_map``.  Pure function; no side effects.

## Not mapped yet: `voltage_regulator` (a PV terminal)

power-grid-model 1.13 added a `voltage_regulator` component that makes an existing
`sym_gen`/`asym_gen`/`sym_load`/`asym_load` a PV terminal: `regulated_object` (the
appliance id), `status`, `u_ref` (required; per unit of the regulated node's
`u_rated`, the same base as `source.u_ref`) and the optional `q_min`/`q_max`
(declared in the input schema, but pgm's own limit handling is still marked as
future work in its validation source). pgml's schema side is ready — the mapping is
`u_ref -> Generator.voltage_regulation.v_set_pu`, `q_min`/`q_max -> q_min_var`/
`q_max_var`, regulating the positive-sequence magnitude — but it is NOT implemented
here yet, because it cannot be validated without a working power-grid-model core in
the environment. pgm validates that every regulator and source on ONE node carries
the same `u_ref`, which matches pgml's refusal of a regulating generator on a Source
node and of two regulating generators on one node.

## Public API

```python
from pgml.convert.pgm import from_grid, to_grid

grid, id_map = to_grid(input_data, base_frequency_hz=60.0,
                        load_model=LoadModel.CONST_IMPEDANCE)

exported = from_grid(grid)  # exported.input_data plus Grid-id -> pgm-id maps
```

### Signature
```
to_grid(
    input_data: dict[str, np.ndarray],  # pgm structured arrays
    *,
    base_frequency_hz: float = 50.0,    # must match the network (pgm has no f0 field)
    load_model: LoadModel = LoadModel.CONST_IMPEDANCE,
    phase_mode: PhaseMode = PhaseMode.SINGLE_PHASE_EQUIV,  # see convert/CONTEXT.md
) -> tuple[Grid, dict[str, Any]]
```

Pure function. Converts a power-grid-model ``input_data`` dict to a schema
``Grid`` and an ``id_map``.  Handles: ``node``, ``line``, ``transformer``,
``sym_load``, ``asym_load``, ``source``, ``sym_gen``.  Unknown keys in
``input_data`` are silently ignored.  ``phase_mode`` (the shared
`convert._common.PhaseMode`) selects the positive-sequence single-phase
equivalent (default, bit-exact) vs a genuine abc expansion; see
`convert/CONTEXT.md` for the scaffold + abc/asymmetric details. pgm has NO load
connection field, so a converted ``asym_load`` is always ``WYE``. The converter
has NO runtime dependency on the ``power_grid_model`` package itself (it reads
``input_data`` as a plain numpy-structured-array format, including the raw pgm
``WindingType``/``BranchSide`` int values, mapped by a local table).

### Outbound signature
```
from_grid(grid: Grid, *, allow_approximation: bool = False) -> PgmExport
```

`PgmExport` carries `input_data`, `pgm_of_node`, `pgm_of_branch`,
`pgm_of_appliance`, and `reductions`. The supported balanced fundamental scope is nodes,
explicit R/L/C/G lines, switches, two-winding transformers, shunts, sources, loads,
generators and storage. Every node must use the same `(A,)` or `(A, B, C)` phase layout,
and branches and appliances must match it; partial-phase, mixed-layout and
neutral-conductor grids raise. Geometry lines, unresolved type references, generic branches,
PV terminals and source forms PGM cannot encode raise `UnsupportedGridError`. PGM has no
ZIP element; ZIP therefore raises by default and converts to constant power only with
`allow_approximation=True`. The same opt-in applies to unbalanced P/Q, inverter controls,
and any electrical term that must be dropped. Ideal sources and switches use the finite
stand-ins required by PGM and record them. Harmonic-only fields are outside this API and
are named in `reductions`; the input `Grid` is not mutated.

### id_map format
```python
{
    "node":           {pgm_node_id: Node.id, ...},
    "line":           {pgm_line_id: Line.id, ...},
    "link":           {pgm_link_id: Switch.id, ...},   # a perfect connection
    "transformer":    {pgm_transformer_id: Transformer.id, ...},
    "sym_load":       {pgm_load_id: Load.id, ...},
    "asym_load":      {pgm_load_id: Load.id, ...},  # THREE_PHASE only
    "sym_gen":        {pgm_gen_id: Generator.id, ...},
    "source":         {pgm_source_id: Source.id, ...},
    "slack_v_complex": complex,   # phasor V (line-to-line, V) for ideal-slack solve
    "load_types":     {pgm_load_id: int},  # original LoadGenType int value
}
```
- Keys are pgm integer ids (int) in the structured arrays.
- Only in-service elements are included (from_status/to_status for lines and
  transformers, status for loads/sources).
- ``slack_v_complex``: taken from the first in-service source; complex phasor in
  SI volts (line-to-line), ready to pass as ``v_fixed`` to ``solve_harmonic``.

### `link` — a perfect connection, converted to an ideal closed `Switch`

power-grid-model's `link` is a zero-impedance connection between two nodes, which its own
solver realises with a very large stand-in admittance (1e6 per unit). It converts to a
closed `Switch` with `resistance_ohm = inductance_h = 0`, whose terminal node-phase rows
the solve collapses exactly (`pgml.assembly.fusion_map`), so pgml carries no stand-in at
all. An out-of-service link (`from_status`/`to_status == 0`) is not converted. Measured
against power-grid-model 1.13 on a source–link–line–load feeder: node voltages agree to
3.5e-9 pu and the link's own current to 6.0e-9 relative, both residuals being the
reference's stand-in drop (7.0e-5 V across the link, which fusion makes exactly zero).
power-grid-model reports a symmetric calculation in three-phase quantities, so its link
current is the pgml single-phase-equivalent conductor current divided by `sqrt(3)`
(`tests/reference/test_switch_fusion_pgm.py`).

### Parameter conventions

| pgm field      | Our schema field                     | Notes                       |
|----------------|--------------------------------------|-----------------------------|
| ``node.u_rated``  | ``Node.u_rated_v``               | V, line-to-line             |
| ``line.r1``    | ``series_resistance_ohm_per_m``      | total Ohm, length_m=1       |
| ``line.x1``    | ``series_inductance_h_per_m``        | x1/(2*pi*f0), length_m=1    |
| ``line.c1``    | ``shunt_capacitance_f_per_m``        | total F, length_m=1         |
| ``line.tan1``  | ``shunt_conductance_s_per_m``        | tan*omega*C1, None if 0     |
| ``source.u_ref`` | ``Source.u_ref_v``               | u_ref * u_rated (V)         |
| ``source.u_ref_angle`` | ``Source.u_angle_deg``     | converted from radians       |
| ``source.sk`` + ``rx_ratio`` | ``resistance_ohm``, ``inductance_h`` | Z=V²/sk |
| ``sym_load.p_specified`` | ``Load.p_nom_w``            | W                           |
| ``sym_load.q_specified`` | ``Load.q_nom_var``          | VAr                         |

### Transformer field mapping (two-winding, vector-group aware)

Pinned against the INSTALLED power-grid-model 1.13.94 C++ source
(``transformer.hpp``/``branch.hpp``/``transformer_utils.hpp``, not just the
docs) — see `converter.py`'s module docstring for the full derivation and
`tests/reference/test_pgm_transformer.py` for the live-solve pins.

| pgm field | Our schema field | Notes |
|---|---|---|
| ``transformer.u1`` | ``Transformer.u_rated_from_v`` | V, UNTAPPED nameplate |
| ``transformer.u2`` | ``Transformer.u_rated_to_v`` | V, UNTAPPED nameplate |
| ``transformer.sn`` | ``Transformer.s_rated_va`` | VA |
| ``transformer.uk``, ``pk``, ``sn``, (effective) ``u2`` | ``series_resistance_ohm``/``series_inductance_h`` | `z_LL=abs(uk)*u2_eff²/sn`, `r_LL=pk*u2_eff²/sn²`; `u2_eff` = tap-adjusted to-side voltage (only moves when `tap_side` is the to-side). Stored TO-coil-referred: `x3` for a DELTA to-connection, unchanged otherwise, in BOTH phase modes (see below) |
| ``transformer.i0``, ``p0``, (effective) ``u2``/``u1`` | ``magnetizing_conductance_s``/``magnetizing_inductance_h`` | to-side-referred `(g,b)` from `i0`/`p0`/`u2_eff`, then referred to the FROM/HV terminal by `(u2_eff/u1_eff)²` (no coil factor — a line/no-load-test quantity, mirrors pandapower/OpenDSS) |
| ``transformer.winding_from``/``winding_to`` (``WindingType`` int) | ``from_connection``/``to_connection`` | `0=wye→WYE, 1=wye_n→WYE_GROUNDED, 2=delta→DELTA, 3=zigzag→ZIGZAG, 4=zigzag_n→ZIGZAG_GROUNDED`; unknown int → `ConversionError` |
| ``transformer.clock`` | ``tap.shift_deg`` | `shift_deg = (clock % 12) * 30`; VERIFIED positive clock ⇒ TO LAGS FROM, identical sign to pgml's own convention (no flip) — `TestPgmConventionPins.test_clock_sign_lv_lags_hv` |
| ``transformer.tap_side/pos/nom/min/max/size`` | ``tap.ratio_magnitude`` | `Δu = sign(tap_max-tap_min)·(tap_pos-tap_nom)·tap_size`; `tap_side==0` (from): `ratio_magnitude=(u1+Δu)/u1`; else (to, ALSO pgm's own default when `tap_side` is absent/invalid): `ratio_magnitude=u2/(u2+Δu)` — VERIFIED against a live solve for both sides, `TestPgmConventionPins.test_tap_{from,to}_side_*` |
| ``transformer.r_grounding_from/to``, ``x_grounding_from/to`` | ``from_grounding``/``to_grounding`` (`GroundingImpedance`) | pass-through; `None` when 0/absent (solid); a nonzero value raises `ModelingError` at ASSEMBLY (`resolve_vector_group`), not at conversion |

**The stored leakage is TO-coil-referred in both phase modes.**
`Transformer.series_resistance_ohm`/`series_inductance_h` carry the coil value
(`z_coil = 3·z_LL` for a DELTA to-winding, `z_coil = z_LL` for wye/zigzag) as
the schema documents, independent of `phase_mode`. Both assembly paths recover
the same line-to-line positive sequence from that value: the `THREE_PHASE`
winding-incidence stamp divides the delta factor back through `Mᵀ M`, and the
`SINGLE_PHASE_EQUIV` scalar pi applies its own `y_LL = 3·y_coil` referral in
`assembly.ybus` before `_scalar_tap_blocks`. Converters therefore never
mode-switch the stored value. Pinned in
`tests/reference/test_pgm_transformer.py::TestCoilFactorIsModeIndependent` and
`TestSymOracle::test_no_magnetizing_branch_is_machine_precision[YNd5]`
(machine precision in both modes).

**Two topology differences vs pgm (documented, not converter bugs; see
`tests/reference/test_pgm_transformer.py` module docstring for the full
derivation and achieved tolerances):**
- **Magnetizing-branch split.** pgm's `calc_param_y_sym` splits the (to-side)
  magnetizing admittance HALF onto `Y_tt` and HALF (reflected through the tap)
  onto `Y_ff`; pgml's frozen `assembly._transformer` stamps it as a simple,
  undivided shunt at the FROM/HV terminal only. Residual: ~1.5e-4 pu / ~1.2e-3
  deg for a realistic ~0.5% `i0` (recovers machine precision at `i0=p0=0`).
- **Zigzag zero-sequence VALUE.** pgm hardcodes a GROUNDED zigzag's own
  zero-sequence self-impedance as `0.1·Z1` (an empirical approximation);
  pgml's core stamps it at the full positive-sequence leakage, `Z0=Z1` (see
  `pgml.assembly._transformer`'s module docstring). Both tools agree a zigzag
  BLOCKS zero-sequence TRANSFER (machine precision on the non-zigzag bus even
  under imbalance); only the zigzag bus's OWN zero-sequence self-admittance
  differs, by very close to the expected 10x (`|V0_pgml|/|V0_pgm| ≈ 10` for
  the SAME injected zero-sequence current), producing up to ~6.8e-3 pu / 0.45°
  divergence on the zigzag-side bus under a ~25%-unbalanced load
  (`TestAsymOracle::test_ynzn5_unbalanced_zigzag_zero_sequence_gap`).

**Not read/modelled:** `uk_min`/`uk_max`/`pk_min`/`pk_max` (tap-dependent
short-circuit parameters — `uk`/`pk` treated as constant across the tap
range), `i0_zero_sequence`/`p0_zero_sequence` (no explicit magnetizing
zero-sequence override field in the schema), `three_winding_transformer` and
`transformer_tap_regulator` (still in the dropped-elements warning — no
3-winding or regulator support in the pgml core).

### Virtual length_m=1 convention
pgm lines store TOTAL (lumped) positive-sequence impedances without a length
field.  The converter sets ``length_m=1.0`` and uses the pgm total values as
"per-metre" values.  The assembly then computes:
    Z_total = r_per_m * length_m = r1 * 1 = r1 [Ohm]
which is numerically identical to the pgm value.

### Single-phase positive-sequence convention (default phase_mode)
Same as the pandapower converter:
- Every node: ``phases=(Phase.A,)``, ``u_rated_v = node.u_rated`` (line-to-line V).
- All loads: ``load_model=LoadModel.CONST_IMPEDANCE`` (caller-specified; default).
  This matches pgm ``LoadGenType.const_impedance`` so both sides solve the same
  linear system.

### Three-phase (abc) convention (phase_mode=THREE_PHASE)
See `convert/CONTEXT.md` for the shared behaviour. pgm-specific: line zero-sequence
from ``r0/x0/c0`` fields when present (NaN/absent -> config default); sources become
balanced 3-phase Thevenins; ``asym_load`` ``p_specified``/``q_specified`` (shape
(3,)) become ``p_nom_per_phase_w``/``q_nom_per_phase_var`` with ``connection=WYE``;
``sym_load`` stays a balanced total (``connection=None`` -> WYE from config).

### Validated on
- IEEE 33-bus Baran & Wu (built from pandapower ``case33bw()`` data), 60 Hz,
  33 buses, 32 in-service lines, 32 loads, 1 source.
  Oracle test: ``tests/reference/test_ieee33_pgm.py``.
  Achieved tolerance: node voltage magnitude atol ~1e-4 pu vs pgm sym power flow.
- Transformer: a 2-bus MV-source -> transformer -> LV-load circuit swept over
  Dyn1, Dyn5, YNd5, YNyn0, Yy6, YNzn5 (grounded zigzag) and an off-nominal tap
  on each side (sym, ``symmetric=True``), plus a balanced/unbalanced
  ``asym_load`` comparison (Dyn5, YNd5, YNzn5 balanced; Dyn5/YNyn0/YNzn5
  unbalanced, ``symmetric=False``) at 50 Hz.
  Oracle test: ``tests/reference/test_pgm_transformer.py`` (34 tests: 3 clock/tap
  convention pins, 6+6 sym vector-group cases with/without a magnetizing branch,
  2 off-nominal-tap cases, 2 delta-coil-factor unit checks, 6 asym cases, 6
  unsupported-model rejection checks).
  Achieved tolerance: sym, no magnetizing branch, atol < 1e-8 pu / 1e-6 deg
  (machine precision, ~1e-11 pu / 1e-9 deg in practice); sym, realistic
  magnetizing branch (i0=0.5%), atol < 5e-4 pu / 5e-3 deg (achieved ~1.5e-4 pu
  / 1.2e-3 deg — a documented magnetizing-branch topology residual, see above);
  asym balanced/Dyn5/YNyn0 unbalanced, atol < 1e-6 pu / 1e-4 deg (achieved
  ~3e-11 pu / 4e-9 deg); asym YNzn5 unbalanced (zigzag-side bus only), a
  documented bounded gap atol < 2e-2 pu / 2 deg (achieved ~6.8e-3 pu / 0.45
  deg — the zigzag zero-sequence VALUE difference above, not a converter bug).
