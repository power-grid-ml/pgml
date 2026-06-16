# Interface ledger: convert (external formats -> our schema)

One subpackage per source: `pandapower/`, `pgm/`, `opendss/`. Each exposes a pure
function producing a valid `grid_schema.Grid` (and, where relevant, the id map back
to the source so tests can align components).

Public API (all three IMPLEMENTED; per-source detail in each subpackage CONTEXT.md):
- [x] `convert.pandapower.to_grid(net) -> (Grid, id_map)` — this file, below.
- [x] `convert.pgm.to_grid(input_data, *, base_frequency_hz=50.0,
      load_model=LoadModel.CONST_IMPEDANCE) -> (Grid, id_map)` — see `pgm/CONTEXT.md`.
- [x] `convert.opendss.to_grid(dss_handle) -> (Grid, id_map)` — see `opendss/CONTEXT.md`.
Conventions: convert engineering units -> SI; record source convention in
Provenance; map sequence/nameplate inputs via the schema's input-convention DTOs;
never invent fields (schema has extra="forbid").

## `convert.pandapower.to_grid` — final signature and id_map format

```python
from pgml.convert.pandapower import to_grid

grid, id_map = to_grid(net)
```

### Signature
```
to_grid(net: pandapowerNet) -> tuple[Grid, dict[str, Any]]
```

Pure function. Converts a (materialised) pandapower network to a schema `Grid`
and an `id_map` dictionary.  Handles: `bus`, `line`, `load`, `ext_grid`.
Structured to extend to `trafo`/`shunt`/`gen`/`sgen` without redesign.

### id_map format
```python
{
    "bus":      {pp_bus_index: Node.id, ...},
    "line":     {pp_line_index: Line.id, ...},
    "load":     {pp_load_index: Load.id, ...},
    "ext_grid": {pp_extgrid_index: Source.id, ...},
    "slack_v_complex": complex,   # phasor V (line-to-line, V) for ideal-slack solve
}
```
- Keys are pandas integer indices (int) into the respective net.* DataFrames.
- Only in-service elements whose buses are also in-service are included.
- `slack_v_complex`: complex slack phasor = `vm_pu * vn_kv*1000 * exp(j*va_deg)`
  ready to pass directly as `v_fixed` to `solve_harmonic(..., v_fixed=...)`.

### Unit conversions (engineering -> SI)
| pandapower field    | SI field                       | factor        |
|---------------------|--------------------------------|---------------|
| `vn_kv` [kV]        | `Node.u_rated_v` [V]           | × 1000        |
| `length_km`         | `Line.length_m`                | × 1000        |
| `r_ohm_per_km`      | `series_resistance_ohm_per_m`  | / 1000        |
| `x_ohm_per_km`      | `series_inductance_h_per_m`    | / (1000 · 2πf₀) |
| `c_nf_per_km`       | `shunt_capacitance_f_per_m`    | × 1e-9 / 1000 = 1e-12 |
| `g_us_per_km`       | `shunt_conductance_s_per_m`    | × 1e-6 / 1000 = 1e-9  |
| `p_mw` [MW]         | `Load.p_nom_w` [W]             | × 1e6         |
| `q_mvar` [MVAr]     | `Load.q_nom_var` [VAr]         | × 1e6         |

### Single-phase positive-sequence convention
Every node: `phases=(Phase.A,)`, `u_rated_v = vn_kv * 1000` (line-to-line).
The assembly's `phase_voltage_magnitude` returns `u_rated_v` unchanged for
1-phase nodes, so const-Z shunt = `(P - jQ) / V_LL^2` — identical to
pandapower's const-Z reference.

### numpy 2.x compatibility
pandapower 2.14 uses removed numpy aliases. Apply before importing:
```python
import numpy as np
np.Inf = np.inf
np.in1d = np.isin
```

### Validated on
IEEE 33-bus Baran & Wu (`pandapower.networks.case33bw()`), 60 Hz, 33 buses,
32 in-service lines + 5 tie-lines (out of service), 32 loads, 1 slack.
Oracle test: `tests/reference/test_ieee33_pandapower.py`.

## IEEE 33-bus oracle status (Phase-1 load-flow gate)
All three reference oracles pass on the single-phase positive-sequence IEEE33:
- pandapower (results): node V within <1e-4 pu. `test_ieee33_pandapower.py`.
- OpenDSS (Y matrix, absolute siemens): off-diagonal ~2e-15 S, diagonal ~1e-10 S
  after accounting for load shunts. `test_ieee33_opendss.py`.
- power-grid-model (results, 2nd oracle): node V within ~2e-10 pu. `test_ieee33_pgm.py`.
Comparison is apples-to-apples: loads set to CONSTANT IMPEDANCE on the reference
side (pp `const_z_percent=100`, pgm `const_impedance`) so both solve the same
linear system; the source is an ideal slack (`solve_harmonic(fixed_rows, v_fixed)`).

## NONLINEAR power-flow oracle status (`solve_power_flow`, const-power)
- IEEE33, DEFAULT const-power `pp.runpp`: `solve_power_flow(slack="ideal")` matches
  `res_bus` to ~3.2e-9 pu / 1.3e-7 deg. `tests/reference/test_ieee33_power_flow_pandapower.py`.
- CIGRE LV (`create_cigre_network_lv`, 44 bus, 3 Dyn30 MV/LV trafos, bus-bus CBs):
  matches to ~1.1e-7 pu / 2.1e-6 deg. `tests/reference/test_cigre_lv_pandapower.py`.

## pandapower converter — element coverage (extended)
`to_grid` now handles `bus`, `line`, `load`, `ext_grid`, **`trafo`**, and **bus-bus
`switch`** (`et='b'`, modelled as near-ideal `Switch`, R=1e-4 Ω). `id_map` adds
`"trafo"` and `"switch"`. Transformer convention: leakage `y_se` referred to the LV
side with `tap.ratio_magnitude = n = vn_hv/vn_lv` and `tap.shift_deg` = the Dyn clock
shift (matches the assembly's off-nominal-tap pi stamp `Y_ff=y_se/|t|^2, Y_tt=y_se`).
Full vector-group / zero-sequence phase coupling is M2 (positive-sequence only now).

## Cross-converter conventions (single-phase positive-sequence equivalent)
- CANONICAL `u_rated_v` for a 1-phase positive-sequence node is LINE-TO-LINE
  (`vn_kv*1000`), validated against pandapower's const-Z reference
  (`y = conj(P+jQ)/V_LL^2`). pandapower and pgm converters follow this.
- KNOWN INCONSISTENCY (follow-up): the OpenDSS converter stores `u_rated_v` as
  LINE-TO-NEUTRAL (`kVBase = basekv/sqrt(3)` for 1-phase DSS buses). This does NOT
  affect the Y oracle (built load-free) but would give a factor-of-3 wrong const-Z
  load shunt if the DSS-converted Grid is used on the load-flow path. Reconcile to
  line-to-line before using `opendss.to_grid` for load flow.
- `base_frequency_hz` is read from the source (`net.f_hz`,
  `dss.Solution.Frequency()`); pgm has no f0 field so the caller passes it. For
  IEEE33 (no line charging) the absolute f0 cancels in `X=2πf·L`; it matters once
  C≠0 / for harmonics.
