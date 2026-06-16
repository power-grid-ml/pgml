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
     operating_point=None, param_overrides=None) -> YBus`
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
    "series_inductance_h" | "tap_magnitude" | "tap_shift_deg")`,
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
- Transformer (`_stamp_transformers`): IMPLEMENTED in-phase ratio + uniform phase
  shift leakage-pi with `ComplexTap` `t = ratio_magnitude * exp(j*shift_deg)`:
      Y_ff = y_se/|t|^2 + y_m,  Y_ft = -y_se/conj(t),  Y_tf = -y_se/t,  Y_tt = y_se
  with per-phase scalar leakage `y_se=(R+jX(f))^-1` and magnetizing shunt
  `y_m=G_m + j*(-1/(2*pi*f*L_m))` on the HV diagonal. Differentiable w.r.t. R, L,
  tap mag/shift. DEFERRED to M2 (marked in code): full vector-group phase coupling
  (Dyn/Yd connection matrices that mix phases), zero-sequence path from winding
  connection, neutral grounding impedance. M1 treats each phase as a diagonal
  coupled two-port; `from_connection`/`to_connection`/`zero_sequence`/`*_grounding`
  are NOT consumed yet.
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
     param_overrides=None) -> Tensor`  complex `[*batch, H, N]`  — IMPLEMENTED.
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
