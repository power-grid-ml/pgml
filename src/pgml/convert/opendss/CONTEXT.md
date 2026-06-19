# Interface ledger: convert.opendss

Converts a live OpenDSS circuit (via `opendssdirect`) to our schema `Grid`.

## Public API

```python
from pgml.convert.opendss import to_grid, PhaseMode

import opendssdirect as dss
dss.Text.Command("Redirect feeder.dss")
dss.Text.Command("Solve")
grid, id_map = to_grid(dss)                                      # SINGLE_PHASE_EQUIV (default)
grid_3ph, id_map_3ph = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)
```

### Signature
```
to_grid(dss_handle: Any, *, phase_mode: PhaseMode = PhaseMode.SINGLE_PHASE_EQUIV) -> tuple[Grid, dict[str, Any]]
```

Pure function (reads from the active OpenDSS engine state). The circuit must
already be loaded and solved (or `Calcvoltagebases` called) before calling.

### `phase_mode` parameter

| Value                 | Node phases          | Line matrices      | Load `connection`           |
|-----------------------|----------------------|--------------------|-----------------------------|
| `SINGLE_PHASE_EQUIV`  | `(Phase.A,)` always  | 1×1 (diagonal [0][0]) | `None` (resolves from config) |
| `THREE_PHASE`         | Real DSS phases (incl. `Phase.N` for neutral buses) | Full n×n from `RMatrix()/XMatrix()/CMatrix()` | `WYE` or `DELTA` from `IsDelta()` |

`SINGLE_PHASE_EQUIV` is byte-identical to the historical converter output, except
for the `u_rated_v` fix below (which does not change the IEEE 33-bus numbers).

### id_map format
```python
{
    "bus":             {dss_bus_name_lower: Node.id, ...},
    "line":            {dss_line_name_lower: Line.id, ...},
    "load":            {dss_load_name_lower: Load.id, ...},
    "vsource":         {dss_vsrc_name_lower: Source.id, ...},
    "slack_v_complex": complex,  # slack phasor (V, L-L) from the first Vsource
}
```
- Keys are lowercase DSS element/bus names.
- Node ids are assigned in `YNodeOrder` bus sequence (bus.phase pairs, in
  DSS's internal order), guaranteeing our `node_phase_index` rows align to
  the DSS Y-matrix rows/columns for oracle tests.
- `"slack_v_complex"` is the complex voltage phasor of the first in-service
  Vsource (`BasekV * pu * 1000 * exp(j*angle_deg)`); pass it as `v_fixed` to
  `solve_harmonic(..., v_fixed=...)` for ideal-slack mode.

### Supported element types (IEEE 33-bus element set)
| DSS type | Schema type       | Notes                                      |
|----------|-------------------|--------------------------------------------|
| Line     | `Line`            | 1- or multi-phase; n×n R/X/C from matrix API |
| Vsource  | `Source`          | R1/X1 via text commands; `thevenin_from_z` |
| Load     | `Load`            | kW/kvar total; `IsDelta()` -> `connection` |

Transformer/Capacitor/Reactor extension is structurally straightforward.

### Unit conversions (OpenDSS engineering -> SI)
| DSS field              | SI field                        | factor                       |
|------------------------|---------------------------------|------------------------------|
| `Lines.RMatrix()` [Ω/unit] * `Lines.Length()` [unit] | total R [Ω] / length_m | ÷ length_m |
| `Lines.XMatrix()` [Ω/unit] | series_inductance_h_per_m | ÷ (2πf₀ · length_m)    |
| `Lines.CMatrix()` [nF/unit] | shunt_capacitance_f_per_m | × 1e-9 ÷ length_m       |
| `Vsources.BasekV()*PU` [kV] | `Source.u_ref_v` [V]      | × 1000                   |
| `Vsource.r1` [Ω]       | `Source.resistance_ohm`         | via `thevenin_from_z`        |
| `Vsource.x1/(2πf₀)` [H] | `Source.inductance_h`         | via `thevenin_from_z`        |
| `Loads.kW()` [kW]      | `Load.p_nom_w` [W]              | × 1000                       |
| `Loads.kvar()` [kVAR]  | `Load.q_nom_var` [VAR]          | × 1000                       |

### `u_rated_v` convention (fix applied)

OpenDSS `Bus.kVBase()` always returns `BasekV_LL / sqrt(3)` (line-to-neutral kV),
regardless of the number of phases on the circuit element. The pgml schema (like
pandapower and pgm) stores the **line-to-line** rated voltage so the const-Z load
shunt formula `y = conj(S) / u_rated_v^2` is consistent across all converters.

The converter applies: `u_rated_v = kVBase * sqrt(3) * 1000` (recovering `BasekV_LL`).

The old converter stored `kVBase * 1000` (L-N), which was a factor-of-sqrt(3)
error in the const-Z shunt when the DSS-converted Grid is used on the load-flow
path. The Y-bus oracle test (passive network, load-free) was unaffected by this
bug. The fix changes the IEEE 33-bus `u_rated_v` from 7309 V to 12660 V (matching
pandapower). All oracle tests remain green.

### Load connection capture (`THREE_PHASE` only)

`dss.Loads.IsDelta()` determines the connection:
- `False` → `WindingConnection.WYE` (phase-to-neutral / L-N)
- `True`  → `WindingConnection.DELTA` (phase-to-phase / L-L)

Single-phase loads (regardless of `IsDelta()`) always receive `WYE` because
OpenDSS single-phase loads are inherently two-conductor L-N elements.

Under `SINGLE_PHASE_EQUIV`, delta loads are collapsed to `phases=(A,)` with
`connection=None` and an INFO message is logged on `logging.getLogger("pgml")`.

### Phase convention
- Each (bus, phase) in `YNodeOrder` becomes a `(Node.id, Phase.X)` slot.
- Phase numbers: 1=A, 2=B, 3=C, 0=N.
- Under `THREE_PHASE`, multi-phase buses get their real phase tuple (e.g.
  `(Phase.A, Phase.B, Phase.C)` or `(Phase.A,)` for single-phase buses).
- Under `SINGLE_PHASE_EQUIV`, every node has `phases=(Phase.A,)`.

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
60 Hz, 33 buses, 32 in-service lines, 1 Vsource, 32 loads.
Oracle test: `tests/reference/test_ieee33_opendss.py`.
Phase-mode test: `tests/convert/test_opendss_phase_mode.py`.
