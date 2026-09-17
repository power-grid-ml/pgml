# Interface ledger: assembly (Y-bus)  (FROZEN rev 1)

Builds the complex nodal admittance tensor and current-injection vector from a
(materialised) Grid, per phase, per frequency, batched, differentiable,
device/dtype-honoring.

## Node–phase indexing (THE canonical layout — everything aligns to this)
The solve uses a **compact** ordering: one matrix row per *existing* (node, phase)
slot — NOT a padded (A,B,C,N) grid. This keeps Y non-singular and the solve dense
and clean. Ordering is deterministic:

  rows are assigned by iterating `grid.nodes` in list order, and within each node
  by that node's `phases` tuple order; row indices are consecutive from 0.

`N = sum(len(node.phases) for node in grid.nodes)`.

(The padded fixed-(A,B,C,N)-with-mask layout is a *downstream ML tensor*
materialisation concern, defined later in the result/ML adapter, NOT here.)

- `node_phase_index(grid) -> NodePhaseIndex` (frozen dataclass) — IMPLEMENTED (`index.py`):
  - `size: int`  (== N)
  - `row(node_id: int, phase: Phase) -> int`
  - `rows(node_id: int) -> list[int]`  (aligned to that node's `phases` order)
  - `node_id_of(row: int) -> int`, `phase_of(row: int) -> Phase`
  - tensors `node_ids: Tensor[int64][N]`, `phase_codes: Tensor[int64][N]` (a=0,b=1,c=2,n=3)
  - `rows_for_terminal(node_id, phases, *, device=None) -> Tensor[int64]` (vectorized scatter)
- `base_voltage_per_row(grid, *, device=None, dtype=torch.float64) -> Tensor[N]` — IMPLEMENTED
  (`index.py`): per-row line-to-neutral base voltage aligned to `node_phase_index` (the
  per-unit reference; `u_rated/sqrt(3)` for >=3-phase nodes, else `u_rated`, via the
  connection-aware `_params.phase_voltage_magnitude` convention). A fixed reference constant
  for per-unit reporting/normalization (NOT on the differentiable path); used by downstream
  ML metrics.

## Public API (IMPLEMENTED — final signatures)
Module: `pgml.assembly`
(`from pgml.assembly import assemble_ybus, build_injections, node_phase_index, NodePhaseIndex, YBus, branch_currents, BranchCurrent`).

- `assemble_ybus(grid, frequencies_hz, *, dtype=torch.complex128, device=None,
     operating_point=None, param_overrides=None, symmetry=None) -> YBus`
  - `symmetry`: `None`/`"auto"`/`"symmetric"`/`"asymmetric"` (`None`
    -> config `calculation.symmetry`). Resolved ONCE via `resolve_asymmetric` and
    logged ONCE via `log_modeling_summary`; `False` ignores per-phase data and
    splits each total equally over the phases (power-grid-model rule). Loads/gens are
    folded connection-aware (WYE/DELTA/neutral) — see the incidence model below.
  - `frequencies_hz`: 1-D real tensor/sequence of H absolute frequencies (Hz). f0 =
    `grid.base_frequency_hz`; harmonic order h = f/f0 (need not be integer).
  - `YBus` (frozen dataclass): `Y: Tensor` complex `[*batch, H, N, N]`,
    `index: NodePhaseIndex`, `frequencies_hz: Tensor[H]`.
  - Contains every PASSIVE / Norton-shunt contribution: line series+shunt, switch,
    generic branch, shunt reactor, transformer stamp, ShuntAppliance (WYE
    phase-to-ground diagonal, or DELTA cyclic phase-to-phase `M^T diag(G+jB) M` bank),
    source Thévenin shunt `Y_s`, and the linear const-Z load/gen shunt admittance.
  - Unbatched squeeze to `[N,N]` ONLY when `frequencies_hz` is a bare python scalar
    and H==1; a length-1 list/tensor keeps the `[H,N,N]` (== `[1,N,N]`) shape.
  - `param_overrides` (ADDED, optional — see assumption note): `{(kind, id, field):
    leaf Tensor}` injects gradient-bearing parameters (line R/L/C, source R/L,
    transformer R/L/tap, switch/generic-branch R/L) so `gradcheck` can perturb
    physical params without mutating the frozen schema. `None` (default) = read
    from the grid. Keys used by tests: `("line", id, "series_resistance_ohm_per_m"
    | "series_inductance_h_per_m" | "shunt_capacitance_f_per_m" |
    "shunt_conductance_s_per_m")`, `("source", id, "resistance_ohm" |
    "inductance_h")`, `("transformer", id, "series_resistance_ohm" |
    "series_inductance_h" | "tap_magnitude" | "zero_sequence_resistance_ohm" |
    "zero_sequence_inductance_h")` (tap.shift_deg selects the discrete
    vector-group clock, not a gradient leaf),
    `("switch", id, "resistance_ohm" | "inductance_h")`,
    `("generic_branch", id, "series_resistance_ohm" | "series_inductance_h")`.
- `build_injections(grid, frequencies_hz, index, *, dtype=torch.complex128,
     device=None, operating_point=None, param_overrides=None) -> Tensor` complex `[*batch, H, N]`
  - Source Norton current `i_s = Y_s @ V_th` (V_th = `u_ref∠u_angle`) at the source
    rows; harmonic current sources from spectra are not yet implemented (return 0
    contribution). Pure passive grids -> 0. Same scalar-frequency squeeze (`[N]`) rule as
    `assemble_ybus`.
- `branch_currents(grid, v, frequencies_hz, index, *, dtype=torch.complex128,
     device=None, param_overrides=None) -> list[BranchCurrent]` — IMPLEMENTED (`ybus.py`).
  Per-branch TERMINAL currents from solved node voltages. For every in-service
  `BranchBase` branch (Line — explicit/geometry/sequence-aware —, Transformer,
  Switch, GenericBranch, ShuntReactor) the currents come from the SAME primitive
  block `Yprim` the Y-bus STAMP scatters: `V_term = concat(V[from_rows], V[to_rows])`,
  `I_term = Yprim @ V_term`, `i_from = I_term[..., :Pf]`, `i_to = I_term[..., Pf:]`.
  Sign = positive INTO the branch terminal (matches `result_schema.BranchResult`). A
  single-terminal `ShuntReactor` -> `to_node=None`, `to_phases=()`, `i_from = Yprim @
  V[from_rows]`, `i_to` = empty `[*batch,H,0]`. `v` is `[*batch, H, N]` (`N==index.size`,
  `H==len(frequencies_hz)`); a missing batch dim is allowed and a missing H axis
  (`[*batch, N]`/`[N]`) is broadcast over H. Returns one `BranchCurrent` per branch in
  `grid.branches` order (open switches carry no admittance and are skipped). The
  `BranchCurrent` frozen dataclass: `branch_id:int, from_node:int, to_node:int|None,
  from_phases:tuple[Phase,...], to_phases:tuple[Phase,...], i_from:Tensor` complex
  `[*batch,H,Pf]`, `i_to:Tensor` complex `[*batch,H,Pt]`. KCL invariant (oracle-free,
  machine precision): scattering every `(i_from,i_to)` back to its node rows and summing
  equals `assemble_network_ybus(grid, freqs).Y @ V` at every row. `param_overrides` keys
  identical to the stamps (line/switch/generic-branch/transformer R/L/C/tap). Differentiable
  (grad flows grid params -> `i_from`/`i_to`), GPU/dtype-honoring, vectorized per KIND/group
  (no python loop over branches on the tape). Iterates the SAME branch-stamp registry
  (`_BRANCH_STAMPS`, see below) the assembly does: each `_*_block_groups` builder yields
  `(group, block, rows, cols)` consumed by BOTH the stamp (scatters) and `branch_currents`
  (matmuls) — assembly behaviour is BIT-IDENTICAL (oracle suite unchanged).
- `branch_stamp_blocks(grid, frequencies_hz, branch_ids, index, *,
     dtype=torch.complex128, device=None, param_overrides=None) -> list[BranchStampBlock]`
  — IMPLEMENTED (`ybus.py`). The primitive admittance block + global rows of each NAMED
  branch, from the SAME registry walk as the assembly (no stamp physics re-derived).
  `BranchStampBlock` (frozen dataclass): `branch_id:int`, `kind:str`, `block:Tensor`
  complex `[H,M,M]` (UNSCALED — exactly what a state of 1 stamps), `rows:Tensor` int64
  `[M]`, `single_terminal:bool`. Every requested branch is stamped regardless of its
  `in_service`/`closed` flags; unknown or unstamped ids raise `InputError`. The builders
  run on a shallow grid view holding only the requested branches, so the cost is
  O(len(branch_ids)), while `rows` index the FULL grid's `index`. This is the structural
  input to a LOW-RANK admittance update — a branch enters `Y` only as
  `Y[rows, rows] += block`, so scaling it by `s` changes `Y` by `(s−1)` times a rank-`≤M`
  term (consumer: `pgml.solver.lowrank.branch_state_terms`, the Woodbury switch-state
  sweep). Differentiable w.r.t. the branch parameters; device/dtype follow the arguments.
- `node_phase_index(grid) -> NodePhaseIndex` (above).

`operating_point` format (linear/const-Z assembly): `{appliance_id: {"p_w": float, "q_var": float}}` or
per-phase `{"p_per_phase_w": [...], "q_per_phase_var": [...]}`; default = nameplate.

## Branch-stamp registry (THE extension point for new branch / device models)
The branch KINDS the assembler knows live in ONE light registry, `_BRANCH_STAMPS`
(in `ybus.py`): an ordered list of `(kind, builder, single_terminal)` entries. Each
builder is a generator `_<kind>_block_groups(grid, f, index, cdt, rdt, device,
param_overrides)` that YIELDS `(group, block, rows, cols)` — `group` the source
branches, `block` the primitive admittance `[H,K,M,M]` (`M=2P` two-terminal series,
`M=P` single-terminal shunt), `rows==cols` the global node-row indices `[K,M]` the
block scatters into / gathers voltage from. Builders are registered in place with the
`@_branch_stamp(kind, *, single_terminal=False)` decorator at their definition.

THREE consumers iterate the registry, so the paths can never drift:
- `_stamp_network` (passive assembly, shared by `assemble_ybus` /
  `assemble_network_ybus`) scatters every yielded block (`scatter_blocks_into`).
  Scatter-add is order-independent, so the assembled Y is BIT-IDENTICAL regardless of
  registration order.
- `branch_currents` multiplies each block with the gathered terminal voltage. A
  `single_terminal` builder is emitted as `to_node=None` / empty `i_to` (no TO half);
  a two-terminal block is split `i_from = I_term[..., :P]`, `i_to = I_term[..., P:]`.
- `branch_stamp_blocks` hands each named branch's block + rows out unchanged (the
  low-rank update seam).

Registered kinds (BRANCH-scoped only): `line`, `switch`, `generic_branch`,
`shunt_reactor` (single-terminal), `transformer`. The registry is deliberately NOT a
plugin framework — it does not include the non-branch stamps (source Norton, const-Z
device shunts, ShuntAppliance, current injections), which stay as their own calls in
`assemble_ybus` / `assemble_network_ybus`.

To add a NEW branch kind: write a `_<kind>_block_groups` generator following the
shared signature/yield contract (group its branches, build the primitive `[H,K,M,M]`
block batched over branches+phases — no python loop over individual branches on the
tape — and emit `_series_terminal_indices` / `_shunt_node_indices` for `rows`), then
decorate it with `@_branch_stamp("<kind>", single_terminal=...)`. Y-bus assembly,
`branch_currents` and `branch_stamp_blocks` pick it up automatically — no edit to any
dispatch. Keep `rows == cols` (the symmetric scatter every registered stamp uses);
`branch_stamp_blocks` rejects an asymmetric mapping.

## Stamps (differentiable, vectorized — no Python loop over branches)
- Per harmonic, per phase. Frequency scaling: `X = 2*pi*f*L`, `B = 2*pi*f*C`
  (closed-form, inline vectorized torch).
- Series branch (Line/Switch/GenericBranch): primitive
  `[[Ys, -Ys], [-Ys, Ys]]` where `Ys = (R(f) + jX(f))^-1` (n×n matrix inverse via
  `torch.linalg.inv`); shunt `Y_sh = G + jB` split half to each terminal diagonal.
- LINE harmonic models (`_line_block_groups` dispatches on the typed
  `Line.harmonic_line_model`, grouping lines by model + skin flag + phase count):
  `conductor_geometry` -> the Carson/Deri `_geometry_block_groups`, which reads the
  conductor internal-inductance model from `LineGeometry.internal_inductance`, falling
  back to `line.geometry.internal_inductance` at
  ASSEMBLY time (so a defaults override applies without reimporting) and passes it to
  `line_constants`; `sequence_aware` ->
  `_sequence_aware_block_groups` (`Z_abc(f0)` -> `Z1`/`Z0`, each frequency-corrected,
  recombined; the per-line `earth_return` coefficients are stacked into tensors so a
  tensor coefficient keeps its gradient, while the DISCRETE options — skin flag,
  `x0_frequency`, `x0_nonnegative`, `r0_includes_earth_return` — form the batching key;
  all four resolve from the defaults at assembly when the line leaves them unset; the
  skin rise of `R0` is carried by the phase conductor's share `R1`; when
  `carson_sublinear` exhausts `X0(h)` inside the requested frequencies
  `_warn_x0_sublinear_deficit` logs ONE warning per affected line set with the line
  count, which costs one host read per assembly under that law only); everything else ->
  `_line_rx_block_groups`, whose resistance is
  `R(h) = m(h)·(R − R_earth) + R_earth` with `R_earth` the mutual entries (the
  earth-return path, which the skin multiplier must NOT scale) and `m(h)` either the
  line's own positive-sequence skin curve (`positive_sequence`), 1 (`naive`) or the
  `resistance_frequency` law (unresolved model). A line whose model is unresolved is
  assembled from its stored parameters and reported by `log_line_models`, which also
  WARNS (`log_synthesized_geometry_radius`) when a radius-based internal-inductance model
  meets a synthesized geometry whose radius is a placeholder (`synth_unphysical`).
- Single-terminal shunt (ShuntReactor / ShuntAppliance):
  `Y(h) = G + 1/(j·2πh f0 L) + j·2πh f0 C`; the inductive term only where
  `inductance_h` is set (grouped separately because it needs a matrix inverse).
- Source: Thévenin (`u_ref∠u_angle` behind per-phase `R + jX` matrix) -> Norton:
  `Y_s = Z_s(f)^-1` added to the source-node diagonal block; current handled by
  `build_injections`.
- Transformer (`_transformer_block_groups` + `_transformer.py`): VECTOR-GROUP winding-incidence
  primitive `Y_node = Nᵀ Y_winding N`. The winding-voltage primitive (leakage `y_se`
  referred to the TO/LV coil, coil turns ratio `τ`) is
  `Y_winding = [[(y/τ²)I, −(y/τ)I],[−(y/τ)I, y I]]`; the constant real incidence
  `N = blockdiag(N_hv, N_lv)` maps coil voltages to bus phase rows — `wye_grounded → I3`,
  `delta → M = [[1,-1,0],[0,1,-1],[-1,0,1]]` (or `Mᵀ`, clock-selected), ungrounded
  `wye → I − 11ᵀ/3`, `zigzag`/`zigzag_grounded` → the normalised limb difference
  `Z = (I − C)/√3` applied to the OTHER side's block (`Ñ_other = Z·N_other`), its own
  side keeping the plain star block. A delta or zigzag winding BLOCKS the zero sequence
  (`M·[1,1,1]=0`, `Z·[1,1,1]=0`, traps triplen harmonics) and contributes an intrinsic
  ±30° clock shift (delta also the √3 magnitude), so the NOMINAL ratio comes from
  `u_rated_from/to_v` + connections and `tap.ratio_magnitude` is the OFF-NOMINAL tap only
  (`tap.shift_deg = clock·30`). Magnetizing `y_m = G_m + j·(−1/(2π f L_m))` is added to
  the HV terminal diagonal directly, OUTSIDE the incidence transform (a documented
  placement deviation from OpenDSS's internal T — `docs/pgml/modeling/transformer.md`).
  - `from_connection`/`to_connection` resolve via `resolve_vector_group` (explicit, else
    config `transformer.vector_group.*`, default Dyn11). EVERY winding pairing except
    zigzag-zigzag is modelled, at EVERY clock of the pairing's parity: an odd number of
    delta/zigzag windings (Dy, Yd, Yz, Zy) admits odd clocks only, an even number (Yy, Dd,
    Dz, Zd) even clocks only, and a parity mismatch raises `ModelingError`. The winding
    orientation / cyclic permutation `C^m` / polarity combination that realises a
    requested clock is selected by matching the candidate's positive-sequence rotation
    against `clock·30°` (`_incidence_pair`); clock 6 is the reversed LV polarity
    `−N_lv`. A zigzag winding IS modelled (the limb-difference incidence above) but is
    EXPERIMENTAL — only power-grid-model can express the same unit, so constructing one
    logs a WARNING once per process; a finite
    `from_grounding`/`to_grounding` (non-solid neutral) raises rather than being silently
    ignored. `P==1` (single-phase / positive-sequence equivalent) folds the group into a
    complex scalar tap `t = (u_from/u_to)·tap_mag·e^{jθ}` on the textbook
    off-nominal-tap pi (with `k_ll = 3` when the TO winding is delta, the coil-vs-terminal
    referral) — reducing EXACTLY to the 3-phase positive sequence.
  - ZERO-SEQUENCE LEAKAGE VALUE: a 3-phase unit whose zero-sequence leakage differs from
    its positive-sequence one (an explicit `Transformer.zero_sequence`, or a non-unit
    `transformer.zero_sequence.{r0_over_r1,x0_over_x1}` default) carries a per-phase
    leakage MATRIX instead of a scalar — the symmetric-component split
    `Z_self=(Z0+2·Z1)/3`, `Z_mutual=(Z0−Z1)/3` (`sequence_leakage_matrices`), inverted as
    a matrix inside `winding_leakage_block`. The zero-sequence PATH stays pure topology,
    so a YNyn three-limb core and a grounded zigzag carry their true Z0 while a delta
    still blocks it. `Z0 == Z1` keeps the scalar stamp (the matrix form reproduces it to
    3e-16 relative, measured on a YNyn unit); the group key
    (`group_key(vg, p, sequence_aware)`) separates the two forms.
  - WINDING-RESISTANCE FREQUENCY LAW: `R(f) = R · m(f) · (f/f0 if the unit scales R with
    the order else 1)`, with `m(f)` the shared `ResistanceFrequencyModel` multiplier
    (`_resistance_multiplier`, the same helper the line path uses). WHICH units scale R is
    `transformer.harmonic_resistance.law` (`harmonic_resistance_law` /
    `resistance_scales_with_order`): `element` (default) = each transformer's own
    `harmonic_xr_constant` (OpenDSS's `XRConst`), `constant` / `xr_constant` force one law
    on every unit. Defaults to no change (X ∝ h at fixed R).
  - Differentiable w.r.t. R, L, the zero-sequence R0/L0 and the off-nominal tap
    magnitude; the discrete vector group / clock selects the constant `N`.
    `param_overrides` keys: `series_resistance_ohm`/`series_inductance_h`/`tap_magnitude`
    plus `zero_sequence_resistance_ohm`/`zero_sequence_inductance_h`; `tap_shift_deg` is
    not a continuous (gradient) leaf — it selects the clock.
- Build by scatter-add of primitive blocks into Y via `_scatter.scatter_blocks_into`
  (clone + `index_add_` along a flattened N*N axis with linear index `row*N+col`;
  accumulates duplicates; the scattered VALUES are differentiable, the indices are
  int64). Vectorized over branches of the same kind (grouped by phase count) and
  over phases — no python loop over individual branches on the tape.

## Implementation notes / linear-assembly simplifications
- `ShuntReactor` (a `BranchBase`) is stamped as a single-terminal shunt at its
  `from_node`/`from_phases` only.
- `ResistanceFrequencyModel` multiplier (`_resistance_multiplier`, shared by the line and
  transformer stamps): `constant`, `analytic` with `law="carson_skin_multiplier"` (the
  differentiable Bessel skin curve; any other analytic law falls back to `base_value`) and
  `curve` (piecewise-linear, constant extrapolation) are wired; an `equation` law is not
  implemented and yields 1.0. A multiplier scales the CONDUCTOR part of a multi-phase
  resistance matrix only. Setting a law together with a typed `harmonic_line_model` is
  rejected by the schema (the model derives its own curve).
- Line/transformer `type_ref` must already be MATERIALISED before assembly (the
  resolver is a separate component); assembly reads explicit params only.

## Linear (const-Z) load model (so the linear solve matches a const-Z oracle)
Loads/generators are converted to a constant shunt admittance from an operating
point: per phase `y = conj(P + jQ) / |U_nom|^2` (load sign +consumes; gen sign
inverts P,Q). `operating_point` defaults to nameplate `p_nom_w/q_nom_var`. The
const-power (nonlinear) successive-admittance iteration is implemented separately
(see the nonlinear network/device split below) and does not change this linear path.
Over FREQUENCY the fold keeps its conductance flat (a resistance) and scales its
susceptance as the equivalent reactive element — `B(f0)·h` where the device is
capacitive, `B(f0)/h` where it is inductive (`_const_z_frequency_scaling`, built from
`clamp` so `h = 1` is exact and P/Q stay differentiable). `assemble_network_ybus`
contains no device fold at all and is the harmonic path.

`_load_shunt.py` — the HARMONIC device shunt (a different model from the const-Z fold
above, and the one the harmonic solver uses). Internal API:
`HARMONIC_SHUNT_MODELS = ("none", "opendss", "motor")`,
`resolve_shunt_model_name(model) -> str` (`None` -> the modeling default
`appliance.harmonic_shunt.model`; an unknown name raises `InputError`),
`resolve_harmonic_shunt(appliance, model) -> ResolvedHarmonicShunt` (device override >
run-level model > defaults file; `"none"` at run level wins over every device, which is
OpenDSS's global `NeglectLoadY`),
`SHUNT_BASES = ("operating_point", "nameplate")` +
`resolve_shunt_basis(basis) -> str` (`None` -> the modeling default
`appliance.harmonic_shunt.basis`, shipped `operating_point`: the shunt is built from the
power the device draws in THIS scenario at the solved fundamental terminal voltage, which
is what OpenDSS's `YPrim` does with its Load's specified kW/kvar. `nameplate` builds it
from the device's stored P, Q at the rated terminal voltage, which keeps `Y(h)`
scenario-independent — one factorization per order for a whole batch — at a model error
measured in `solver/CONTEXT.md`),
`GENERATION_SHUNT_MODELS = ("none", "load_style")` +
`generation_shunt_is_neglected(appliance) -> bool` (the documented default
`appliance.harmonic_shunt.generation_model`, shipped `none`: a GENERATION-sign device
carries NO shunt, because `Y_eq = conj(S)/V_rated²` has a negative conductance for an
injecting device and would feed harmonic energy into the network. A stored
`harmonic_model` block does not override that — every grid written before the field was
consumed carries the former default block on every device — while naming the `motor`
model does, and so does setting the default to `load_style`, which is the expression
OpenDSS applies to the negative-kW `Load` idiom an export writes. The stamp logs one
WARNING per assembly naming how many generation devices were left as pure current
sources), and
`harmonic_shunt_element_admittance(s_elem, v_rated, h, series_rl, *, motor_x_pu,
motor_xr, motor_s_base, cdtype) -> [*batch, H, K, E]`, the per-element admittance
`(1−s)Re(Y_eq) + j(1−s)Im(Y_eq)/h + 1/(Re(Z_s) + j·h·Im(Z_s))` with
`Y_eq = conj(S)/V_rated²` and `Z_s = 1/(s·Y_eq)` (OpenDSS `Load.pas`
`CalcYPrimMatrix`, verified against a live `YPrim` to 4.7e-16 relative). NOTE the
parallel branch divides its susceptance by `h` whatever the sign of `Q` (OpenDSS models
it as R‖L), which is NOT the const-Z fold's sign-dependent law. The nodal stamp lives in
`pgml.solver.harmonic_flow._stamp_harmonic_load_shunt`, where the fundamental solution
that sets the operating point is available.

## Asymmetry: connection-aware load/gen modeling (DONE)
`_symmetry.py` (torch-free, PURE; runs on every assemble/solve + PF residual eval):
- `resolve_asymmetric(grid, operating_point=None, *, mode=None) -> bool` — resolves the
  config `calculation.symmetry` (`auto`/`symmetric`/`asymmetric`) to True==per-phase;
  `auto` => asymmetric iff any appliance `*_per_phase_*` or per-phase operating point.
  PURE (NO logging — it is called per residual eval; the single INFO emitter is
  `log_modeling_summary`).
- `resolve_connection(appliance) -> WindingConnection` — explicit `connection` else the
  config default (single- vs multi-phase). WYE_GROUNDED folds to WYE for a terminal.
- `log_modeling_summary(grid, *, asymmetric)` — INFO log of the FINAL modeling (neutral
  modeled iff a node carries `Phase.N`; WYE/DELTA mix; symmetry; the line harmonic models
  in use). Emitted ONCE per user entry point (`assemble_ybus` / `solve_power_flow` /
  `solve_harmonic_flow`).
- `log_line_models(grid)` — the line-model part of that summary, plus a WARNING naming the
  count, the first ten line ids and `apply_default_harmonic_model` when a line's
  `harmonic_line_model` is still unresolved (such a line is assembled from its stored
  parameters, i.e. the naive model above f0, where a three-phase lumped line would resolve
  to the sequence-aware model and its zero-sequence earth-return term). Reached once per
  solve: `assemble_ybus` and `solve_power_flow` each log the summary, and a harmonic study
  delegates to the fundamental solve's.
- `log_synthesized_geometry_radius(lines)` — WARNS when `line.geometry.internal_inductance`
  is anything but `gmr` while some line carries a synthesized geometry tagged
  `synth_unphysical`: those geometries keep the modeling-default radius as a placeholder,
  and every other internal-inductance model reads it. Called from `log_line_models`.

`_incidence.py` — the terminal incidence model wired into `_stamp_const_z_loads`,
`device_current_injections` and the harmonic device shunt. A constant real `M [n_elem, n_used]` maps used node-rows
to element ("terminal") voltages `V_term = M @ V_used`; nodal admittance block =
`Mᵀ diag(y_elem) M [n_used, n_used]`, nodal current = `Mᵀ i_elem`. Cases (n=len(phases)):
- WYE, node NO `Phase.N` (to ground): `M = I_n` (== the historical diagonal stamp,
  bit-exact — THE regression guarantee). V0 = L-N (`u_rated/√3` for ≥3-phase nodes).
- WYE, node HAS `Phase.N` (4-wire): `M = [I_n | -1]`; element k = V_phase_k − V_N; the
  N row receives `-Σ i_k` (Kirchhoff). used_rows = phase rows + the node's N row.
- DELTA, n==3: `M = [[1,-1,0],[0,1,-1],[-1,0,1]]` (circulant; element k between phase_k
  and phase_{(k+1)%3}; per-phase value k -> delta branch k). V0 = L-L (`u_rated`).
- DELTA, n!=3: raises `NotImplementedError` (open/2-phase delta not modeled).
- WYE return conductor override (`InjectionAppliance.return_path`): the 4-wire neutral
  decision is per-appliance. `"auto"` (default) = the node-level rule above (neutral iff
  the node carries `Phase.N`) — BYTE-IDENTICAL to the historical behaviour; `"ground"`
  pins the return to ground even on a `Phase.N`-carrying node (`M = I_n`); `"neutral"`
  requires `Phase.N` (raises `ModelingError` otherwise). `return_path` folds into
  `has_neutral_return` and hence the grouping key, so a grounded and a neutral-returning
  WYE appliance on the SAME node land in DIFFERENT groups with different `M`. A non-`"auto"`
  value on a DELTA appliance raises `ModelingError` (fail loud).
Public helpers: `group_appliances(appliances, node_map) -> list[IncidenceGroup]`
(groups by `(connection, n_phases, has_neutral_return)`; `has_neutral_return` now
incorporates `return_path`), `build_incidence(grp, rdt, device) -> M`,
`cyclic_delta_incidence(n, rdt, device) -> M [n,n]` (the general cyclic DELTA incidence —
`M[k,k]=1`, `M[k,(k+1)%n]=-1` — shared by the DELTA load `build_incidence` and the DELTA
`ShuntAppliance` stamp; `n==3` is the historical circulant), `used_rows(grp, index,
device) -> [K, n_used]`. `phase_voltage_magnitude`
gained a `line_to_line: bool` arg (DELTA ⇒ True; `n_phases>=3` covers 4-wire ABCN).
`resolve_operating_power` gained an `asymmetric: bool` arg (False ⇒ ignore per-phase,
split totals equally). Its symmetric branch returns n INDEPENDENT entries
(`[t/n for _ in range(n)]`, NOT `[t/n]*n` which would alias one leaf into all phases)
and totals via the autograd-safe `_tensor_sum` (graph-preserving for tensor leaves).
Vectorized per group (one `M`, blocks `[K, n_used, n_used]` scattered via `_scatter`);
differentiable (autograd flows through `y_elem`/`i_elem`, not the constant `M`); GPU/
dtype-honoring. Basis: `docs/pgml/modeling/asymmetric.md`.

NOTE (PF warm start): `solve_power_flow` uses a PHASE-AWARE balanced warm start (source
magnitude rotated by the row's positive-sequence phase angle; neutral rows start at
~0 V). A flat start would make DELTA element voltages identically 0 (V_a−V_b=0) and a
WYE-neutral element collide V_a with V_N, both giving 0/0 in the const-P current.

## Rules
- DIFFERENTIABLE + GPU (CLAUDE.md): every Y/I entry differentiable w.r.t. R,L,G,C,
  length, tap, source Z, and operating P,Q. No `.item()/.detach()/.numpy()`, no
  in-place on tracked tensors, no hard-coded device, no per-node/branch Python loop.
- Honor input `device`/`dtype`; complex128 for gradcheck, complex64 ok otherwise.
- Self-check: a tiny hand-built grid whose Y is verifiable in numpy, plus gradcheck
  of Y entries w.r.t. line R/L/C and source Z.

# =====================================================================
# Network/device split for NONLINEAR power flow (FROZEN — IMPLEMENTED)
# =====================================================================
The const-Z `assemble_ybus` above is the LINEAR-model assembler and STAYS (its
docstring now labels it "LINEAR (const-Z) = assemble_network_ybus + const-Z device
shunts + source Norton"; the IEEE33 oracle tests keep using it as a regression
suite — behaviour UNCHANGED). For nonlinear (const-P / ZIP) power flow, loads are
NOT baked into Y; split as implemented below.

- `assemble_network_ybus(grid, frequencies_hz, *, dtype=torch.complex128,
     device=None, param_overrides=None) -> YBus`  — IMPLEMENTED (`ybus.py`).
  The PASSIVE network only: lines, switches, generic branches, shunt reactors,
  transformers, and ShuntAppliance (a fixed linear shunt). **No loads/generators,
  no source Norton.** Constant, differentiable w.r.t. network params. Same compact
  index + `[*batch, H, N, N]` shapes and scalar-frequency squeeze as `assemble_ybus`.
  Shares the `_stamp_network` core with `assemble_ybus` (which adds source Norton +
  const-Z device shunts on top).

- `assemble_ybus(...)` — IMPLEMENTED, behaviour UNCHANGED, relabelled LINEAR ==
  `assemble_network_ybus` + const-Z device shunts (loads/gens at nominal V) +
  source Norton shunt. Kept for the linear path / regression oracle.

- `device_current_injections(grid, v, index, frequencies_hz, *,
     dtype=torch.complex128, device=None, operating_point=None,
     param_overrides=None, symmetry=None) -> Tensor`  complex `[*batch, H, N]`  — IMPLEMENTED.
  - `symmetry` (Increment 1): resolved to a bool internally (SILENTLY — no
    `log_modeling_summary` in the PF iteration). `solve_power_flow` resolves once and
    passes the resolved string through. Connection-aware: `V_term = M @ V_used`,
    element current `i_elem = conj(S_eff(V_term))/conj(V_term)`, nodal `I_used =
    M^T @ i_elem`. WYE-to-ground (`M=I`) is bit-identical to the historical per-phase
    form.
  Voltage-dependent nodal current ABSORBED by Load/Generator per `LoadModel`:
      S_eff(V) = S0 * ( z*(|Vt|/|V0|)^2 + i*(|Vt|/|V0|) + p ),  I_term = conj(S_eff)/conj(Vt)
  with S0 = sign*(P+jQ) (sign +1 load / -1 generator), V0 = NOMINAL line-to-neutral
  voltage (`u_rated`, /sqrt(3) for 3-phase nodes per the existing convention), Vt =
  the terminal voltage gathered from `v`. ZIP triples (z,i,p) taken PER power
  component (P and Q independently) from the load model: const_power=(0,0,1),
  const_impedance=(1,0,0), const_current=(0,1,0), `zip` from `ZipCoefficients`.
  Sign matches the const-Z fold: at z=1 `I_term == Y_devZ @ Vt`, so a const-Z ZIP
  solve reproduces the linear `assemble_ybus` system EXACTLY (verified, machine
  precision — a required regression link). `v` may be `[*batch, (H,) N]`; a missing
  H axis is broadcast. Batched, differentiable, device/dtype-honoring. Tensor P/Q
  (tensor duality) and per-phase split are handled autograd-safely (`_per_phase_
  power_tensor`); a per-load/scenario leading batch dim broadcasts over `v`.
  The nodal balance used by the solver is `Y_eff @ V = I_slack - I_device(V)`.
  `param_overrides` keys: `("load"|"generator", id, "p_nom_per_phase_w"
  |"q_nom_per_phase_var")` inject leaf per-phase P/Q.
  - INJECTION APPLIANCES: `device_current_injections` / `_stamp_const_z_loads` iterate
    every `InjectionAppliance` (Load, Generator, Storage), so `Storage` is a signed PQ
    injection identical to a Generator (sign −1; its `p_nom_w` is the signed setpoint,
    >0 = discharge/inject).
  - INVERTER CONTROL (`_control.py`, differentiable, GPU): a Generator/Storage with a
    `control` block follows a voltage-dependent (P, Q) law instead of the constant ZIP
    base. Controlled appliances take a SEPARATE per-element pass in
    `device_current_injections` (the control-free majority keeps the bit-exact stacked
    ZIP path). `resolve_injection_power(control, p_avail, v_pu)` returns (P, Q) from the
    control mode (constant-PF / cosφ(P) / Volt-VAr / Volt-Watt / combined), bounded by the
    `s_rated_va` capability circle via `smooth_clamp` (`smoothing`>0 = C¹ backward,
    `smoothing`=0 = hard). `evaluate_characteristic` is the differentiable piecewise
    (linear/cubic) curve lookup. Because the control enters `I_device(V)` and the IFT
    backward differentiates one residual eval at V*, gradients flow to the curve / rating
    with no new adjoint (gradcheck-verified). Control is honored at the FUNDAMENTAL solve;
    the linear const-Z `assemble_ybus` uses the base P/Q (control ignored). A stiff
    Volt-VAr/Volt-Watt loop needs `method="newton"` (the fixed point oscillates).

Verification (CPU): const-Z consistency (norton + ideal, 1ph + 3ph) exact to
1e-9..1e-10; gradcheck (float64) of the downstream power-flow V w.r.t. line R/L and
load P/Q passes; full suite green.

# =====================================================================
# rev 2 additions (branch-state masking + injection plan)
# =====================================================================
- Every public assembler gains `branch_states: Optional[dict] = None`
  (`{branch_id: float | 0-d | [*batch] tensor}`): a listed branch is ALWAYS
  stamped and its primitive block multiplied by the state (0 = open, 1 = in
  service, continuous in between, autograd-carrying), OVERRIDING the static
  `in_service`/`closed` flags. Applied centrally in `_stamp_network` /
  `branch_currents` via `_group_states`/`_masked_block`; a batched state
  promotes `Y` to `[*batch, H, N, N]` through the scatter broadcast, and
  `branch_currents` scales each branch's terminal currents by the same state
  (an open branch reports exactly 0 A). Builders take the extra
  `branch_states` argument for their in-service filter only.
- `build_injection_plan(grid, index, frequencies_hz, *, dtype, device,
  operating_point, param_overrides, symmetry) -> InjectionPlan` and
  `injections_from_plan(plan, v) -> Tensor` — the two halves of
  `device_current_injections` (which now composes them, byte-identical): the
  V-independent operating-point resolution (once per solve) and the pure-tensor
  per-iteration evaluation. The nonlinear solvers reuse one plan across all
  iterations; a plan built under `no_grad` is the detached fast path, one built
  on the tape stays differentiable. `v` with `H == 1`: only a bare `[1, N]` is
  read as carrying the H axis; any deeper `v` is `[*batch, N]`, so a trailing
  scenario dim of one (a `[B, 1]` operating point) is never mistaken for H.
  Two plan-reshaping helpers live next to them for consumers that evaluate the residual
  over a DIFFERENT scenario axis than the plan was built with (both in
  `assembly/ybus.py`, both autograd-safe, both no-ops for a scalar / broadcast plan):
  `flatten_plan_batch(plan, batch_shape)` collapses a multi-dimensional operating-point
  batch onto one axis, and `select_plan_batch(plan, rows, *, batch_size)` picks a subset of
  that one axis. The solver's gradient path uses the first to build a block-diagonal state
  Jacobian over a flattened batch and the second to build it in memory-budgeted CHUNKS;
  its criticality diagnostic uses the second to analyse one scenario of a batch.

# =====================================================================
# rev 3 additions (exact bus fusion of zero-impedance branches)
# =====================================================================
A branch whose series impedance is EXACTLY zero — a closed `Switch` with the schema's
zero R/L default, a bus coupler or jumper modelled as a zero-impedance `Line`, a
zero-length line, a zero-impedance `GenericBranch` — has no primitive admittance (the
nodal formulation inverts the series impedance). It is nevertheless an ordinary network
element: an ideal conductor, which constrains its two terminal voltages to be equal. That
constraint is expressed EXACTLY by collapsing the branch's terminal node-phase rows into
ONE row of the solved system (`V = P v`, `(Pᵀ Y P) v = Pᵀ I`), never by a small-impedance
stand-in and never by editing the grid.

- `fusion_map(grid, *, param_overrides=None, branch_states=None, zero=None) -> FusionMap
  | None` (`_fusion.py`) — union-find over the fused terminals PER PHASE (position k of
  `from_phases` is shorted to position k of `to_phases`, so a phase-rotating jumper is
  handled), then one reduced row per connected group. `None` when nothing fuses, which is
  the signal to every caller to keep the unreduced path. Reads the EFFECTIVE impedance: a
  `param_overrides` entry wins, so a substituted finite leaf keeps the branch stamped and
  a substituted zero fuses an ordinary branch. `zero` accepts an already-collected
  `zero_impedance_branches` list (the solve collects it once and feeds both the map and
  the `check_branch_impedances` gate, instead of walking the branch list per consumer).
- `FusionMap` (frozen dataclass): `size` (M), `index` (the REDUCED `NodePhaseIndex`, whose
  `(node, phase) -> row` map is MANY-TO-ONE), `full_index` (the grid's own N-row layout),
  `row_to_reduced [N]`, `representative [M]`, `fused_branch_ids`, `groups` (the full rows
  of each multi-row reduced row), `indeterminate_branch_ids`. Methods:
  `prolong(v_red) -> v_full` (gather; the adjoint of `restrict`), `restrict(x_full)`
  (index_add — the right reduction for an extensive quantity: the fused rows' current
  balances are summed), `sample(x_full)` (gather at the representatives — the right
  reduction for an intensive one: a voltage, a per-row base), `reduce_rows(rows)` (keeps a
  row SET a set: the ideal-slack `fixed_rows`, a block-diagonal partition),
  `duplicate_rows(rows)`, `node_groups()`, `describe()`, `to(device)`.
  THE KEY PROPERTY: because the reduced index is a many-to-one row map, every existing
  stamp / injection / incidence path that indexes through an index accumulates `Pᵀ Y P`
  and `Pᵀ I` DIRECTLY — no reduction matrix and no full-size matrix is ever materialised.
- `zero_impedance_branches(grid, *, param_overrides=None) -> list[ZeroImpedanceBranch]`
  — `(branch_id, component, detail, fusable)` per offending in-service branch. NOT
  fusable: a `Transformer` (its ratio and vector group relate the terminals by more than
  equality) and a zero-series branch that still carries a shunt admittance (fusing would
  drop that shunt). Both keep raising through `pgml.solver.check_branch_impedances`.
- `assemble_ybus` / `assemble_network_ybus` gain `fusion: FusionMap | bool | None = None`
  — `None` resolves the documented policy `branch.zero_impedance` (`fuse` default,
  `error` refuses), `False` refuses, a map uses that map (how a solve shares ONE map
  across the fundamental and every harmonic order), and the singleton `NO_FUSION`
  (`_fusion.py`) says the resolution has already run and produced no map, which is what a
  solve passes down so the assembler does not walk the branch list again to rediscover
  the empty answer (`resolve_fusion` takes the same value, plus `zero=`). When fusion
  applies, the returned `YBus.Y` is the reduced `[*batch, H, M, M]`, `YBus.index is
  fusion.index`, and the field `YBus.fusion` carries the map. The fused branches are
  dropped from a shallow grid view before stamping (`_unfused_view`), so no builder sees a
  singular impedance.
- `branch_currents(..., fusion=None, i_inj=None)` — a fused branch has no primitive block,
  so its terminal current comes from Kirchhoff's law at the fused rows instead: the defect
  `d = i_inj − Y_network_without_fused_branches · V` (assembled once in the FULL layout)
  is inverted per fused group by the structural maps `pinv(A_known)` of the group's
  oriented incidence, with the rows of a Source terminal left out (an ideal slack's own
  current is an output of the solve, not an input). `i_from = -i_to` exactly. `v`/`index`
  must be the FULL layout (what a result reports). `i_inj` is optional: without it the
  injection at the fused rows is taken as zero, which is exact for a fused node carrying
  no appliance and logs a WARNING when one does. A LOOP of ideal conductors (parallel
  fused branches) leaves a circulating current undetermined by the node voltages — the
  physics of ideal conductors; those branches are listed in
  `FusionMap.indeterminate_branch_ids`, the reported split is the minimum-norm one, and a
  WARNING names them once.
- `branch_stamp_blocks` REFUSES a zero-impedance branch by name (a low-rank update needs
  a stamped block), and `NodePhaseIndex` gains `has(node_id, phase) -> bool`.
- `fusion_map` RAISES (`ModelingError`) when a zero-impedance branch is listed in
  `branch_states` (the two mechanisms are mutually exclusive: a swept branch is reached by
  SCALING a stamped admittance, a fused branch has no stamp to scale), when an ideal
  branch joins two different rated voltages, and for the unfusable kinds above.
- Differentiable (the reduced assembly, the prolongation gather and the current recovery
  are linear operations on the tape; a fused branch's own impedance is structurally not a
  parameter of the solve, so its gradient is a finite ZERO), batched, device/dtype from
  the inputs. `tests/topology/test_fusion_map.py`,
  `tests/reference/test_switch_fusion_pandapower.py`,
  `tests/reference/test_switch_fusion_pgm.py`,
  `tests/differentiability/test_fusion_gradcheck.py`, `tests/gpu/test_fusion_parity.py`.

The default transformer magnetizing placement is `split`. Conformance uses
`pgml.defaults.use_preset` around conversion/assembly/solve to select the reference
placement and frequency law. Per-line geometry model selection also controls the
synthesized-radius warning.
