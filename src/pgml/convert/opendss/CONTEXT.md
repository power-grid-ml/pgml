# Interface ledger: convert.opendss

Converts a live OpenDSS circuit (via `opendssdirect`) to our schema `Grid`.

## Public API

```python
from pgml.convert.opendss import to_grid

import opendssdirect as dss
dss.Text.Command("Redirect feeder.dss")
dss.Text.Command("Solve")
grid, id_map = to_grid(dss)
```

### Signature
```
to_grid(dss_handle: Any) -> tuple[Grid, dict[str, Any]]
```

Pure function (reads from the active OpenDSS engine state). The circuit must
already be loaded and solved (or `Calcvoltagebases` called) before calling.

### id_map format
```python
{
    "bus":     {dss_bus_name_lower: Node.id, ...},
    "line":    {dss_line_name_lower: Line.id, ...},
    "load":    {dss_load_name_lower: Load.id, ...},
    "vsource": {dss_vsrc_name_lower: Source.id, ...},
}
```
- Keys are lowercase DSS element/bus names.
- Node ids are assigned in `YNodeOrder` bus sequence (bus.phase pairs, in
  DSS's internal order), guaranteeing our `node_phase_index` rows align to
  the DSS Y-matrix rows/columns for oracle tests.

### Supported element types (IEEE 33-bus element set)
| DSS type | Schema type       | Notes                                      |
|----------|-------------------|--------------------------------------------|
| Line     | `Line`            | 1- or multi-phase; R/X/C from matrix API   |
| Vsource  | `Source`          | R1/X1 via text commands; near-zero Z if 0  |
| Load     | `Load`            | kW/kvar total; L-N kV stored in Node       |

Transformer/Capacitor/Reactor extension is structurally straightforward.

### Unit conversions (OpenDSS engineering -> SI)
| DSS field              | SI field                        | factor                       |
|------------------------|---------------------------------|------------------------------|
| `Lines.RMatrix()` [Ω/unit] * `Lines.Length()` [unit] | total R [Ω] / length_m | ÷ length_m |
| `Lines.XMatrix()` [Ω/unit] | series_inductance_h_per_m | ÷ (2πf₀ · length_m)    |
| `Lines.CMatrix()` [nF/unit] | shunt_capacitance_f_per_m | × 1e-9 ÷ length_m       |
| `Vsources.BasekV()*PU` [kV] | `Source.u_ref_v` [V]      | × 1000                   |
| `Vsource.r1` [Ω]       | `Source.resistance_ohm`         | direct (SI)                  |
| `Vsource.x1/(2πf₀)` [H] | `Source.inductance_h`         | direct (SI)                  |
| `Loads.kW()` [kW]      | `Load.p_nom_w` [W]              | × 1000                       |
| `Loads.kvar()` [kVAR]  | `Load.q_nom_var` [VAR]          | × 1000                       |

### Length unit codes (OpenDSS → metres)
| Code | Unit | Factor       |
|------|------|--------------|
| 0    | none | 1.0 (total values, length=1 by convention) |
| 1    | mi   | 1609.344     |
| 2    | kft  | 304.8        |
| 3    | km   | 1000.0       |
| 4    | m    | 1.0          |
| 5    | ft   | 0.3048       |
| 6    | in   | 0.0254       |
| 7    | cm   | 0.01         |

### Phase convention
- Each (bus, phase) in `YNodeOrder` becomes a `(Node.id, Phase.X)` slot.
- Phase numbers: 1=A, 2=B, 3=C, 0=N.
- Single-phase positive-sequence circuits: every node has `phases=(Phase.A,)`.
- `Node.u_rated_v` = `Bus.kVBase() * 1000` [V]. For 1-phase nodes this is the
  L-N voltage as reported by OpenDSS (kVBase of the bus from `Calcvoltagebases`).

### EE convention note (important for harmonics, Phase 3)
OpenDSS `SystemY` / `Export Y` **includes** the following shunt admittances on
the diagonal beyond the passive line pi-model:
1. **Vsource Norton shunt** `Y_s = (R1 + jX1)^-1` added to the source bus.
2. **Load shunt** (model-dependent):
   - model=1 (const-P): shunt evaluated at the current iteration voltage
     (approximately `conj(S)/|V_sol|^2`); changes each Newton iteration.
   - model=2 (const-Z): shunt `conj(S)/(kv*1000)^2` at rated voltage; static.
   - `NeglectLoadY=yes`: pure current source, no shunt in Y.
3. **No line G/C shunts** are included in `Export Y` if `c1=c0=0` (case33bw).

For the Y-bus oracle test: the cleanest comparison is to build the OpenDSS
circuit **without loads** (so only line series/shunt + Vsource Norton shunt
appear), then compare OpenDSS Y to our Y minus load shunts, accounting for the
Vsource shunt on the source bus diagonal. See `tests/reference/test_ieee33_opendss.py`.

### Validated on
IEEE 33-bus Baran & Wu circuit built from `pandapower.networks.case33bw()` data,
50 Hz (converted), 33 buses, 32 in-service lines, 1 Vsource, 32 loads.
Oracle test: `tests/reference/test_ieee33_opendss.py`.
