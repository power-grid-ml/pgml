# Interface ledger: convert (external formats -> our schema)

One subpackage per source: `pandapower/`, `pgm/`, `opendss/`. Each exposes a pure
function producing a valid `grid_schema.Grid` (and, where relevant, the id map back
to the source so tests can align components).

Public API (all three IMPLEMENTED; per-source detail in each subpackage CONTEXT.md):
- [x] `convert.pandapower.to_grid(net, *, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
      -> (Grid, id_map)` — this file, below.
- [x] `convert.pgm.to_grid(input_data, *, base_frequency_hz=50.0,
      load_model=LoadModel.CONST_IMPEDANCE, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
      -> (Grid, id_map)` — see `pgm/CONTEXT.md`.
- [x] `convert.opendss.to_grid(dss_handle, *, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
      -> (Grid, id_map)` — see `opendss/CONTEXT.md`.
Conventions: convert engineering units -> SI; record source convention in
Provenance; map sequence/nameplate inputs via the schema's input-convention DTOs;
never invent fields (schema has extra="forbid").

## `convert._common` — the shared scaffold (library-agnostic)

The duplicated plumbing of every `to_grid` lives in `convert/_common.py`; each
per-library converter contains only the source-specific field reading and calls
these helpers so all sources stamp identical schema objects for the same physical
input. Designed to also serve the OpenDSS converter (native multi-phase + explicit
n x n line matrices). Every helper is unit-tested in `tests/convert/test_common.py`.

API:
- `IdCounter` — the single monotonic id allocator (one instance per conversion).
- `PhaseMode(str, Enum)` — `SINGLE_PHASE_EQUIV` (today's positive-sequence 1-phase
  equivalent, `phases=(Phase.A,)`, 1x1 line matrices) vs `THREE_PHASE` (genuine abc).
  Re-exported from `convert`, `convert.pandapower`, `convert.pgm`.
- `phases_for(mode, *, native=None) -> tuple[Phase, ...]` — the one place the phase
  tuple is decided: `SINGLE_PHASE_EQUIV -> (A,)`; `THREE_PHASE -> native` if given
  (OpenDSS passes real phases incl. `Phase.N`) else `(A, B, C)`.
- `zero_sequence_ratios() -> (r0/r1, x0/x1, c0/c1)` — read from `pgml.defaults`
  (`line.zero_sequence.*`).
- `sequence_to_phase_matrices(r1, x1, c1, *, r0=None, x0=None, c0=None, two_pi_f0,
  g0=0, g1=0) -> (R, L, C, G)` — sequence -> 3x3 phase matrices via
  `self=(Z0+2*Z1)/3`, `mutual=(Z0-Z1)/3` (applied to R, X, C, G); `L=X/two_pi_f0`.
  A balanced sequence input (Z0==Z1) yields a pure diagonal (decoupled) matrix; in
  general a symmetric circulant (off-diagonal == mutual). `r0/x0/c0` default to
  `r1*ratio` etc. from `zero_sequence_ratios()` when `None`.
- `single_phase_matrix(value) -> [[value]]` — the exact 1x1 wrapper used by the
  positive-sequence-equivalent line path (preserves `[[r1]]`/`[[l1]]`/`[[c1]]`).
- `thevenin_from_z(r_ohm, x_ohm, two_pi_f0) -> (R, L)` and
  `thevenin_from_sk(u_rated_v, sk_va, rx_ratio, two_pi_f0) -> (R, L)` — source
  Thevenin from an explicit impedance / from short-circuit power + R/X ratio (the
  latter moved verbatim from the pgm converter; same fallback floors).
- `make_metadata(name, description) -> GridMetadata`.
- Emit helpers (the single place the phase decision + per-phase mapping live):
  `build_node`, `build_load`, `build_generator` (generation-positive PQ, for
  pp `sgen` / pgm `sym_gen`), `warn_dropped_elements` (the loud-drop contract),
  `build_source`, `build_line_from_sequence`,
  `build_line_from_matrices`. `build_line_from_matrices` takes explicit n x n
  R/L/C(/G) matrices + a `phases` tuple — implemented and unit-tested for the
  OpenDSS converter (pp/pgm route through `build_line_from_sequence`).

**Zero-sequence assumption (THREE_PHASE line expansion).** When a positive-sequence
line is expanded to abc and the dataset has no native zero-sequence data, the
zero-sequence quantity defaults to `r1*(R0/R1)` etc. from config. An explicit native
`r0/x0/c0` always wins.

## `phase_mode` + asymmetric capture (pandapower / pgm / opendss)

`phase_mode=PhaseMode.SINGLE_PHASE_EQUIV` (the DEFAULT) reproduces the historical
positive-sequence single-phase-equivalent output BYTE-FOR-BYTE (same phases, 1x1
matrices, ids, id_map) — the bit-exact regression gate; the reference oracle suite
calls `to_grid(net)` with no `phase_mode` and stays green.

`phase_mode=PhaseMode.THREE_PHASE`:
- nodes/branches become `(A, B, C)`; lines use `sequence_to_phase_matrices`
  (pandapower: per-m `r1/x1/c1` from `r_ohm_per_km/x_ohm_per_km/c_nf_per_km`, with
  `r0/x0/c0` from `net.line` columns `r0_ohm_per_km/x0_ohm_per_km/c0_nf_per_km` IF
  present else config defaults; pgm: total `r1/x1/c1` and `r0/x0/c0` if present else
  defaults);
- sources become BALANCED 3-phase Thevenins (angles `u_angle / -120 / +120`,
  diagonal R/L); zero-seq source impedance = positive-seq (no short-circuit data read);
- standard balanced `net.load` / `sym_load`: `connection=None` (resolves to WYE from
  config), no per-phase split (the symmetric/auto calc splits the total equally);
- static generators: pandapower `sgen` and pgm `sym_gen` -> `Generator`
  (generation-positive nameplate; id_map buckets `"sgen"` / `"sym_gen"`); every other
  non-empty pgm component (`transformer`, `three_winding_transformer`, `shunt`,
  `asym_gen`, `link`, `transformer_tap_regulator`) triggers a `warn_dropped_elements`
  WARNING — nothing is dropped silently;
- ASYMMETRIC loads captured: pandapower `net.asymmetric_load` -> `connection=WYE`
  (`type=="wye"`) or `DELTA`, `p_nom_per_phase_w=(p_a,p_b,p_c)*1e6`,
  `q_nom_per_phase_var=(q_a,q_b,q_c)*1e6`; pgm `asym_load` -> `p_specified`/
  `q_specified` shape (3,) per-phase, `connection=WYE` ALWAYS (pgm has no load
  connection field). Both add an `"asymmetric_load"` / `"asym_load"` id_map bucket.
- Under `SINGLE_PHASE_EQUIV` an asymmetric load cannot be represented; its phases are
  summed to a balanced 1-phase total and an INFO is logged on `logging.getLogger("pgml")`.

`phase_mode` for opendss (added with the scaffold migration):
- `SINGLE_PHASE_EQUIV` (default): every node/branch `phases=(A,)`, 1×1 line matrices
  (the `[0][0]` entry of the DSS matrix).  Load `connection=None`.
- `THREE_PHASE`: nodes carry real DSS phases (incl. `Phase.N`); lines carry the full
  n×n matrices from `RMatrix()/XMatrix()/CMatrix()`; loads get `IsDelta()` connection
  (`WYE` or `DELTA`); sources become balanced Thevenins with per-phase angles.
  Single-phase loads land on their real bus-suffix phase (`.1`→`(A,)`) with `WYE`.
  A delta load under `SINGLE_PHASE_EQUIV` logs INFO on `logging.getLogger("pgml")`.
OpenDSS provides native multi-phase matrices directly, so `build_line_from_matrices`
is used (not `build_line_from_sequence`); no sequence assumption is made.

## `convert.pandapower.to_grid` — final signature and id_map format

```python
from pgml.convert.pandapower import to_grid

grid, id_map = to_grid(net)
```

### Signature
```
to_grid(net: pandapowerNet, *, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
    -> tuple[Grid, dict[str, Any]]
```

Pure function. Converts a (materialised) pandapower network to a schema `Grid`
and an `id_map` dictionary.  Handles: `bus`, `line`, `load`, `asymmetric_load`,
`ext_grid`, `trafo`, bus-bus `switch`, and `sgen` (-> `Generator`,
generation-positive). Every OTHER non-empty element table (`gen`, `shunt`,
`trafo3w`, `impedance`, `ward`, `xward`, `dcline`, `storage`, `motor`,
`asymmetric_sgen`) triggers a WARNING naming the kind and count — nothing is
dropped silently.

### id_map format
```python
{
    "bus":      {pp_bus_index: Node.id, ...},
    "line":     {pp_line_index: Line.id, ...},
    "load":     {pp_load_index: Load.id, ...},
    "sgen":     {pp_sgen_index: Generator.id, ...},
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
from pgml.convert.pandapower import ensure_numpy_compat

ensure_numpy_compat()
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
- CIGRE LV (`create_cigre_network_lv`, 44 bus, 3 Dyn1 20/0.4 kV MV/LV trafos, bus-bus CBs):
  matches to ~1.1e-7 pu / 2.1e-6 deg. `tests/reference/test_cigre_lv_pandapower.py`.

## pandapower converter — element coverage (extended)
`to_grid` now handles `bus`, `line`, `load`, `ext_grid`, **`trafo`**, and **bus-bus
`switch`** (`et='b'`, modelled as near-ideal `Switch`, R=1e-4 Ω). `id_map` adds
`"trafo"` and `"switch"`. Transformer convention: leakage `y_se` referred to the LV
coil; the NOMINAL ratio + vector-group shift come from `u_rated_from/to_v` +
`from_connection`/`to_connection`, so `tap = (ratio_magnitude=1.0, shift_deg=clock·30)`
is the OFF-NOMINAL tap + clock only. The CIGRE LV trafos are Dyn1
(`from_connection=DELTA`, `to_connection=WYE_GROUNDED`, `shift_degree=30`); assembly
builds the full vector-group winding-incidence primitive (delta blocks zero-sequence
/ triplen harmonics). See `docs/pgml/modeling/transformer.md`. pandapower tap-changer
positions (`tap_pos`/`tap_step`) are not read yet (off-nominal tap stays 1.0). The
OpenDSS converter does NOT emit transformers yet (DSS `Transformer` parsing is a
documented gap; CIGRE/IEEE feeders enter via pandapower).

## Cross-converter conventions (voltage base, slack, frequency)

> The authoritative cross-tool convention record (base voltage L-L/L-N, transformer
> reference side, vector-group/clock, power signs, SI/phase-domain, harmonic earth-return)
> for pgml vs pandapower / OpenDSS / power-grid-model is `docs/pgml/modeling/conventions.md`. Keep
> it in sync when a converter's convention handling changes.

- CANONICAL `u_rated_v` for any node is LINE-TO-LINE (`vn_kv*1000` / `u_rated*1`
  for nodes that are already L-L), validated against pandapower's const-Z reference
  (`y = conj(P+jQ)/V_LL^2`). All three converters follow this convention.
  OpenDSS `Bus.kVBase()` returns L-N (= `BasekV_LL/sqrt(3)`); the OpenDSS converter
  recovers L-L via `kVBase * sqrt(3) * 1000`. The old converter stored `kVBase * 1000`
  (L-N), which was a factor-of-sqrt(3) error in the const-Z shunt on the load-flow
  path. The Y-bus oracle test (passive, load-free) was unaffected; the fix was applied
  together with the phase_mode scaffold migration.
- VOLTAGE BASE vs WORKING VOLTAGE: `u_rated_v` is stored L-L, but every per-phase voltage
  the solver touches is LINE-TO-NEUTRAL. `assembly._params.phase_voltage_magnitude(u_rated_v,
  n)` is the single base helper — `u_rated_v/sqrt(3)` for a WYE element on a >=3-phase node,
  `u_rated_v` for DELTA / 1-phase — and feeds the const-Z/ZIP load, the harmonic source
  admittance, and the per-unit reporting in `pgml.evaluation`. Accordingly the slack/source
  EMF is L-N: under `THREE_PHASE` `build_source` divides the L-L `u_ref_v` by sqrt(3) for the
  balanced wye expansion (1-phase keeps it). `id_map["slack_v_complex"]` stays the L-L phasor
  (a single-phase ideal-slack `v_fixed` convenience). Pinning the L-L magnitude on each phase
  makes every 3-phase voltage sqrt(3) too high (~1.73 pu on the L-N base).
- `base_frequency_hz` is read from the source (`net.f_hz`,
  `dss.Solution.Frequency()`); pgm has no f0 field so the caller passes it. For
  IEEE33 (no line charging) the absolute f0 cancels in `X=2πf·L`; it matters once
  C≠0 / for harmonics.
