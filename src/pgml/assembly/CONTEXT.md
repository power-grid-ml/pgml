# Interface ledger: assembly (Y-bus)  (FROZEN rev 1 — orchestrator-pinned)

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

## Public API (IMPLEMENTED — final signatures)
Module: `pgml.assembly`
(`from pgml.assembly import assemble_ybus, build_injections, node_phase_index, NodePhaseIndex, YBus`).

- `assemble_ybus(grid, frequencies_hz, *, dtype=torch.complex128, device=None,
     operating_point=None, param_overrides=None, symmetry=None) -> YBus`
  - `symmetry` (Increment 1): `None`/`"auto"`/`"symmetric"`/`"asymmetric"` (`None`
    -> config `calculation.symmetry`). Resolved ONCE via `resolve_asymmetric` and
    logged ONCE via `log_modeling_summary`; `False` ignores per-phase data and
    splits each total equally over the phases (power-grid-model rule). Loads/gens are
    folded connection-aware (WYE/DELTA/neutral) — see the incidence model below.
  - `frequencies_hz`: 1-D real tensor/sequence of H absolute frequencies (Hz). f0 =
    `grid.base_frequency_hz`; harmonic order h = f/f0 (need not be integer).
  - `YBus` (frozen dataclass): `Y: Tensor` complex `[*batch, H, N, N]`,
    `index: NodePhaseIndex`, `frequencies_hz: Tensor[H]`.
  - Contains every PASSIVE / Norton-shunt contribution: line series+shunt, switch,
    generic branch, shunt reactor, transformer stamp, ShuntAppliance, source
    Thévenin shunt `Y_s`, and the M1 const-Z load/gen shunt admittance.
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
    "series_inductance_h" | "tap_magnitude")` (tap.shift_deg selects the discrete
    vector-group clock, not a gradient leaf),
    `("switch", id, "resistance_ohm" | "inductance_h")`,
    `("generic_branch", id, "series_resistance_ohm" | "series_inductance_h")`.
- `build_injections(grid, frequencies_hz, index, *, dtype=torch.complex128,
     device=None, operating_point=None, param_overrides=None) -> Tensor` complex `[*batch, H, N]`
  - Source Norton current `i_s = Y_s @ V_th` (V_th = `u_ref∠u_angle`) at the source
    rows; harmonic current sources from spectra are a later milestone. Pure passive
    grids -> 0. Same scalar-frequency squeeze (`[N]`) rule as `assemble_ybus`.
- `node_phase_index(grid) -> NodePhaseIndex` (above).

`operating_point` format (M1): `{appliance_id: {"p_w": float, "q_var": float}}` or
per-phase `{"p_per_phase_w": [...], "q_per_phase_var": [...]}`; default = nameplate.

## Stamps (differentiable, vectorized — no Python loop over branches)
- Per harmonic, per phase. Frequency scaling: `X = 2*pi*f*L`, `B = 2*pi*f*C`
  (compose from the `equations` registry laws).
- Series branch (Line/Switch/GenericBranch): primitive
  `[[Ys, -Ys], [-Ys, Ys]]` where `Ys = (R(f) + jX(f))^-1` (n×n matrix inverse via
  `torch.linalg.inv`); shunt `Y_sh = G + jB` split half to each terminal diagonal.
- Source: Thévenin (`u_ref∠u_angle` behind per-phase `R + jX` matrix) -> Norton:
  `Y_s = Z_s(f)^-1` added to the source-node diagonal block; current handled by
  `build_injections`.
- Transformer (`_stamp_transformers` + `_transformer.py`): VECTOR-GROUP winding-incidence
  primitive `Y_node = Nᵀ Y_winding N`. The winding-voltage primitive (leakage `y_se`
  referred to the TO/LV coil, coil turns ratio `τ`) is
  `Y_winding = [[(y/τ²)I, −(y/τ)I],[−(y/τ)I, y I]]`; the constant real incidence
  `N = blockdiag(N_hv, N_lv)` maps coil voltages to bus phase rows — `wye_grounded → I3`,
  `delta → M = [[1,-1,0],[0,1,-1],[-1,0,1]]` (or `Mᵀ`, clock-selected), ungrounded
  `wye → I − 11ᵀ/3`. A delta winding BLOCKS the zero sequence (`M·[1,1,1]=0`, traps
  triplen harmonics) and supplies the intrinsic √3 magnitude + ±30° clock shift, so the
  NOMINAL ratio comes from `u_rated_from/to_v` + connections and `tap.ratio_magnitude`
  is the OFF-NOMINAL tap only (`tap.shift_deg = clock·30`). Magnetizing
  `y_m = G_m + j·(−1/(2π f L_m))` is added to the HV terminal diagonal directly.
  - `from_connection`/`to_connection` resolve via `resolve_vector_group` (explicit, else
    config `transformer.vector_group.*`, default Dyn11). Supported clocks: Dyn → 1/11,
    wye-wye/delta-delta → 0/6 (others raise NotImplementedError); zigzag and non-solid
    `*_grounding` not modelled yet. `P==1` (single-phase / positive-sequence equivalent)
    folds the group into a complex scalar tap `t = (u_from/u_to)·tap_mag·e^{jθ}` and uses
    the textbook off-nominal-tap pi — reducing EXACTLY to the 3-phase positive sequence.
  - Differentiable w.r.t. R, L and the off-nominal tap magnitude; the discrete vector
    group / clock selects the constant `N`. `param_overrides` keys unchanged
    (`series_resistance_ohm`/`series_inductance_h`/`tap_magnitude`); `tap_shift_deg` is
    no longer a continuous (gradient) leaf — it selects the clock.
- Build by scatter-add of primitive blocks into Y via `_scatter.scatter_blocks_into`
  (clone + `index_add_` along a flattened N*N axis with linear index `row*N+col`;
  accumulates duplicates; the scattered VALUES are differentiable, the indices are
  int64). Vectorized over branches of the same kind (grouped by phase count) and
  over phases — no python loop over individual branches on the tape.

## Implementation notes / M1 simplifications
- `ShuntReactor` (a `BranchBase`) is stamped as a single-terminal shunt at its
  `from_node`/`from_phases` only.
- `ResistanceFrequencyModel` skin-effect multiplier: only `constant` is wired
  (analytic falls back to `base_value`); curve/equation laws are Phase-2.
- Line/transformer `type_ref` must already be MATERIALISED before assembly (the
  resolver is a separate component); assembly reads explicit params only.

## M1 load model (so the linear solve matches a const-Z oracle)
Loads/generators are converted to a **constant shunt admittance** from an operating
point: per phase `y = conj(P + jQ) / |U_nom|^2` (load sign +consumes; gen sign
inverts P,Q). `operating_point` defaults to nameplate `p_nom_w/q_nom_var`. The
const-power (nonlinear) successive-admittance iteration is a later milestone; note
it but do not implement in M1.

## Asymmetry: connection-aware load/gen modeling (Increment 1 — DONE)
`_symmetry.py` (torch-free, PURE; runs on every assemble/solve + PF residual eval):
- `resolve_asymmetric(grid, operating_point=None, *, mode=None) -> bool` — resolves the
  config `calculation.symmetry` (`auto`/`symmetric`/`asymmetric`) to True==per-phase;
  `auto` => asymmetric iff any appliance `*_per_phase_*` or per-phase operating point.
  PURE (NO logging — it is called per residual eval; the single INFO emitter is
  `log_modeling_summary`).
- `resolve_connection(appliance) -> WindingConnection` — explicit `connection` else the
  config default (single- vs multi-phase). WYE_GROUNDED folds to WYE for a terminal.
- `log_modeling_summary(grid, *, asymmetric)` — INFO log of the FINAL modeling (neutral
  modeled iff a node carries `Phase.N`; WYE/DELTA mix; symmetry). Emitted ONCE per
  user entry point (`assemble_ybus` / `solve_power_flow` / `solve_harmonic_flow`).

`_incidence.py` — the terminal incidence model wired into `_stamp_const_z_loads` +
`device_current_injections`. A constant real `M [n_elem, n_used]` maps used node-rows
to element ("terminal") voltages `V_term = M @ V_used`; nodal admittance block =
`Mᵀ diag(y_elem) M [n_used, n_used]`, nodal current = `Mᵀ i_elem`. Cases (n=len(phases)):
- WYE, node NO `Phase.N` (to ground): `M = I_n` (== the historical diagonal stamp,
  bit-exact — THE regression guarantee). V0 = L-N (`u_rated/√3` for ≥3-phase nodes).
- WYE, node HAS `Phase.N` (4-wire): `M = [I_n | -1]`; element k = V_phase_k − V_N; the
  N row receives `-Σ i_k` (Kirchhoff). used_rows = phase rows + the node's N row.
- DELTA, n==3: `M = [[1,-1,0],[0,1,-1],[-1,0,1]]` (circulant; element k between phase_k
  and phase_{(k+1)%3}; per-phase value k -> delta branch k). V0 = L-L (`u_rated`).
- DELTA, n!=3: raises `NotImplementedError` (open/2-phase delta not modeled).
Public helpers: `group_appliances(appliances, node_map) -> list[IncidenceGroup]`
(groups by `(connection, n_phases, has_neutral_return)`), `build_incidence(grp, rdt,
device) -> M`, `used_rows(grp, index, device) -> [K, n_used]`. `phase_voltage_magnitude`
gained a `line_to_line: bool` arg (DELTA ⇒ True; `n_phases>=3` covers 4-wire ABCN).
`resolve_operating_power` gained an `asymmetric: bool` arg (False ⇒ ignore per-phase,
split totals equally). Its symmetric branch returns n INDEPENDENT entries
(`[t/n for _ in range(n)]`, NOT `[t/n]*n` which would alias one leaf into all phases)
and totals via the autograd-safe `_tensor_sum` (graph-preserving for tensor leaves).
Vectorized per group (one `M`, blocks `[K, n_used, n_used]` scattered via `_scatter`);
differentiable (autograd flows through `y_elem`/`i_elem`, not the constant `M`); GPU/
dtype-honoring. Basis: `references/asymmetric_modeling.md`.

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
# Phase-2: network/device split for NONLINEAR power flow (FROZEN — IMPLEMENTED)
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

Verification (CPU): const-Z consistency (norton + ideal, 1ph + 3ph) exact to
1e-9..1e-10; gradcheck (float64) of the downstream power-flow V w.r.t. line R/L and
load P/Q passes; full suite green.
