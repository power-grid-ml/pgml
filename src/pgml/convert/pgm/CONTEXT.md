# Interface ledger: convert.pgm (power-grid-model -> our schema)

Converts a power-grid-model ``input_data`` dict (structured numpy arrays) to a
schema ``Grid`` and an ``id_map``.  Pure function; no side effects.

## Public API

```python
from pgml.convert.pgm import to_grid

grid, id_map = to_grid(input_data, base_frequency_hz=60.0,
                        load_model=LoadModel.CONST_IMPEDANCE)
```

### Signature
```
to_grid(
    input_data: dict[str, np.ndarray],  # pgm structured arrays
    *,
    base_frequency_hz: float = 50.0,    # must match the network (pgm has no f0 field)
    load_model: LoadModel = LoadModel.CONST_IMPEDANCE,
) -> tuple[Grid, dict[str, Any]]
```

Pure function. Converts a power-grid-model ``input_data`` dict to a schema
``Grid`` and an ``id_map``.  Handles: ``node``, ``line``, ``sym_load``,
``source``.  Unknown keys in ``input_data`` are silently ignored.

### id_map format
```python
{
    "node":           {pgm_node_id: Node.id, ...},
    "line":           {pgm_line_id: Line.id, ...},
    "sym_load":       {pgm_load_id: Load.id, ...},
    "source":         {pgm_source_id: Source.id, ...},
    "slack_v_complex": complex,   # phasor V (line-to-line, V) for ideal-slack solve
    "load_types":     {pgm_load_id: int},  # original LoadGenType int value
}
```
- Keys are pgm integer ids (int) in the structured arrays.
- Only in-service elements are included (from_status/to_status for lines,
  status for loads/sources).
- ``slack_v_complex``: taken from the first in-service source; complex phasor in
  SI volts (line-to-line), ready to pass as ``v_fixed`` to ``solve_harmonic``.

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

### Virtual length_m=1 convention
pgm lines store TOTAL (lumped) positive-sequence impedances without a length
field.  The converter sets ``length_m=1.0`` and uses the pgm total values as
"per-metre" values.  The assembly then computes:
    Z_total = r_per_m * length_m = r1 * 1 = r1 [Ohm]
which is numerically identical to the pgm value.

### Single-phase positive-sequence convention
Same as the pandapower converter:
- Every node: ``phases=(Phase.A,)``, ``u_rated_v = node.u_rated`` (line-to-line V).
- All loads: ``load_model=LoadModel.CONST_IMPEDANCE`` (caller-specified; default).
  This matches pgm ``LoadGenType.const_impedance`` so both sides solve the same
  linear system.

### Validated on
IEEE 33-bus Baran & Wu (built from pandapower ``case33bw()`` data), 60 Hz,
33 buses, 32 in-service lines, 32 loads, 1 source.
Oracle test: ``tests/reference/test_ieee33_pgm.py``.
Achieved tolerance: node voltage magnitude atol ~1e-4 pu vs pgm sym power flow.
