# assembly — Y-bus and current injections

Builds the complex nodal admittance `Y(f)` and the injection vectors from a materialised
`Grid`: per phase, per frequency, batched, differentiable, on the caller's device/dtype.
Signatures: docstrings / API_INDEX.md. Physics: `docs/pgml/modeling/` (conventions,
asymmetric, transformer, der-pv-storage, references/opendss/harmonics).

## Codemap

- `index.py` — `NodePhaseIndex` / `node_phase_index` (the row layout) and
  `base_voltage_per_row` (per-row per-unit reference).
- `ybus.py` — the public assemblers and the branch-stamp registry: `assemble_ybus`
  (linear const-Z), `assemble_network_ybus` (passive network), `build_injections`
  (source Norton current), `device_current_injections` = `build_injection_plan` +
  `injections_from_plan` (ZIP device current), `flatten_plan_batch` / `select_plan_batch`
  (reshape a plan's scenario axis), `branch_currents`, `branch_stamp_blocks`; the
  `_<kind>_block_groups` builders for lines (explicit R/L/C, sequence-aware, conductor
  geometry), switches, generic branches, shunt reactors and transformers.
- `_stamps.py` — series / shunt / pi primitive blocks.
- `_scatter.py` — `scatter_blocks_into` and `PatternAccumulator` (dense or
  structural-pattern accumulation of blocks).
- `_structure.py` — `ybus_structure`, the topology-derived sparsity superset of `Y`.
- `_transformer.py` — vector-group winding incidence, clock selection, sequence-aware
  leakage, winding-resistance frequency law, magnetizing placement.
- `_incidence.py` — terminal incidence `M` of WYE / WYE-neutral / DELTA appliances.
- `_params.py` — `phase_voltage_magnitude`, const-Z shunt admittance,
  `resolve_operating_power`.
- `_symmetry.py` — symmetric/asymmetric resolution, connection defaults, the one-shot
  modeling-summary and line-model warnings.
- `_control.py` — inverter control laws (constant PF, cosφ(P), Volt-VAr, Volt-Watt),
  capability-circle clamp, characteristic lookup.
- `_load_shunt.py` — the harmonic device Norton shunt (OpenDSS `Load` model, motor model,
  basis and reactive-element laws, generation-device policy). The nodal stamp itself is in
  `solver/harmonic_flow.py`, where the solved fundamental is available.
- `_fusion.py` — exact bus fusion of zero-impedance branches: `fusion_map`, `FusionMap`,
  `zero_impedance_branches`, `resolve_fusion`, `NO_FUSION`, fused-branch current recovery.

## Row layout (everything aligns to it)

- Compact: one row per EXISTING (node, phase) slot, rows assigned by iterating
  `grid.nodes` in list order and, within a node, its `phases` tuple order. `N =
  sum(len(node.phases))`. No padded (A,B,C,N) grid here; a padded/masked layout is a
  downstream ML-tensor concern.
- Phase codes a=0, b=1, c=2, n=3.
- `base_voltage_per_row` is line-to-neutral (`u_rated/√3` on ≥3-phase nodes, else
  `u_rated`), a constant reference, not on the differentiable path.
- Under fusion the reduced `NodePhaseIndex` maps (node, phase) → row MANY-TO-ONE; every
  stamp indexing through it accumulates `PᵀYP` and `PᵀI` directly. No reduction matrix and
  no full-size matrix is materialised.

## Shapes and conventions

- `Y` is complex `[*batch, H, N, N]`; frequencies are absolute Hz, `h = f / f0` need not be
  integer. Squeeze to `[N, N]` / `[N]` only for a bare python scalar frequency; a length-1
  list or tensor keeps the H axis.
- `assemble_ybus` = `assemble_network_ybus` + const-Z device fold + source Norton shunt
  (the linear model). The harmonic path uses `assemble_network_ybus` plus device current
  sources and never the const-Z fold.
- Harmonic current sources from spectra are computed in `solver.harmonic_flow`, not here
  (scope boundary): `build_injections` carries only the source Norton current.
- Device sign: `S0 = sign·(P + jQ)`, +1 load (consumes), −1 generator/storage. A const-Z
  ZIP solve reproduces the `assemble_ybus` system exactly; the nodal balance is
  `Y_eff·V = I_slack − I_device(V)`.
- `V0` of a device element is line-to-neutral for WYE and line-to-line for DELTA
  (`phase_voltage_magnitude(line_to_line=…)`).
- Branch current sign: positive INTO the branch terminal (matches
  `result_schema.BranchResult`). KCL invariant: scattering every `(i_from, i_to)` back to
  its rows equals `assemble_network_ybus(...).Y @ V` at every row.
- `operating_point`: `{appliance_id: {"p_w", "q_var"}}` or per-phase `p_per_phase_w` /
  `q_per_phase_var`; default nameplate.
- Control power quantities (`s_rated_va`, `q_var`, `p_ref_w`) are DEVICE TOTALS, shared
  equally over the device's elements; each element evaluates its curve at its own terminal
  voltage. Converters never pre-divide.
- `param_overrides` `{(kind, id, field): leaf}` injects gradient-bearing parameters
  without editing the schema; the key list is in the `assemble_ybus` docstring. Discrete
  choices (vector-group clock, model names) are never leaves.
- `branch_states` `{branch_id: float | 0-d | [*batch]}` stamps a listed branch ALWAYS and
  scales its block by the state, overriding `in_service` / `default_closed`; `branch_currents` must
  receive the same states as the assembly the voltages came from.

## Branch-stamp registry (the extension point)

`_BRANCH_STAMPS` in `ybus.py` is an ordered list of `(kind, builder, single_terminal)`.
Each builder yields `(group, block [H,K,M,M], rows, cols)` with `rows == cols`. Three
consumers iterate it, so they cannot drift: the Y-bus stamp (scatter), `branch_currents`
(block · terminal voltage) and `branch_stamp_blocks` (the low-rank update seam consumed by
`solver.lowrank`). To add a branch kind: write `_<kind>_block_groups` (batched over branches
and phases), decorate it with `@_branch_stamp("<kind>", single_terminal=...)`; no dispatch
edit. Registered: `line`, `switch`, `generic_branch`, `shunt_reactor` (single-terminal),
`transformer`. Non-branch stamps (source Norton, const-Z devices, `ShuntAppliance`,
injections) are deliberately NOT in the registry.

`ybus_structure` reads the pattern off the incidence, not off an assembled matrix, so it is
a superset of `Y`'s nonzeros at every frequency, operating point, branch state and scenario
(out-of-service and open components included). A builder that writes outside its
components' node rows breaks it (`tests/topology/test_ybus_structure.py`). The pattern
accumulator raises on an off-pattern target instead of dropping it.

## Model decisions an edit must not undo

- Transformer: winding-incidence primitive `Y_node = Nᵀ Y_winding N`; the nominal ratio
  comes from `u_rated_from/to_v` + connections, `tap.ratio_magnitude` is the off-nominal
  tap only, `tap.shift_deg = clock·30`. Every pairing except zigzag–zigzag, at every clock
  of the pairing's parity; a parity mismatch or a finite (non-solid) neutral grounding
  raises. Zigzag is modelled but experimental (one warning per process). The zero-sequence
  PATH is topology, its VALUE is the unit's own Z0. Magnetizing shunt placement is
  `transformer.magnetizing_placement`, default `split` (half on each terminal); OpenDSS
  conformance selects `to_terminal` through the `opendss` preset. `P == 1` folds the group
  into a complex scalar tap that reduces exactly to the three-phase positive sequence.
- Lines: every path is a lumped pi. The skin multiplier scales the CONDUCTOR part of the
  resistance only, never the earth-return (mutual) part. A line with an unresolved
  `harmonic_line_model` is assembled from its stored parameters and warned about.
- Const-Z fold over frequency: conductance flat, susceptance `B·h` where capacitive, `B/h`
  where inductive. The harmonic device shunt's parallel branch divides by `h` whatever the
  sign of Q (OpenDSS R‖L) under the `inductive` reactive-element law; the shipped
  `sign_aware` law matches the const-Z fold.
- A generation-sign device carries NO harmonic Norton shunt by default
  (`appliance.harmonic_shunt.generation_model: none`): the load expression has a negative
  conductance for an injecting device. A stored `harmonic_model` block does not override
  this (older grids carry a default block on every device); naming the `motor` model or
  setting `load_style` does.
- Incidence: WYE to ground `M = I`; WYE on a `Phase.N` node `[I | −1]`; DELTA a cyclic
  difference (n == 3 only; other DELTA raises). `return_path` (`auto` / `ground` /
  `neutral`) is part of the grouping key.
- Zero-impedance branches are FUSED (exact), never replaced by a small impedance and never
  removed from the grid (`branch.zero_impedance: fuse`, `branch.switch_model: ideal`).
  Not fusable: transformers, and a zero-series branch with a shunt. Fusion and
  `branch_states` on the same branch are mutually exclusive (raises). A loop of ideal
  conductors leaves a circulating current undetermined: minimum-norm split, warned.
  A fused branch's own impedance has a gradient of exactly zero.
- `type_ref` must be materialised before assembly; no resolver exists in pgml.

## Invariants

- Differentiable w.r.t. R, L, G, C, length, tap magnitude, source Z, zero-sequence R0/L0,
  operating P/Q, control curves and ratings, and branch states. Gradients flow through
  `y_elem` / `i_elem`, never through the constant incidence matrices or index tensors.
- No python loop over individual branches, phases or scenarios on the tape (loops only over
  component KINDS / groups). Index tensors are int64 bookkeeping.
- `resolve_operating_power` returns independent per-phase entries (never `[t/n]*n`, which
  would alias one leaf across phases).
- `resolve_asymmetric` is pure (it runs per residual evaluation); `log_modeling_summary` is
  the single INFO emitter, once per user entry point (`assemble_ybus`,
  `solve_power_flow`; a harmonic study delegates to its fundamental solve).
- An `InjectionPlan` built under `no_grad` is the detached fast path; built on the tape it
  stays differentiable. With `H == 1`, only a bare `[1, N]` voltage is read as carrying the
  H axis.

## Tests that pin this module

`tests/topology/` (structure, pattern accumulator, fusion map), `tests/differentiability/`
(gradcheck incl. fusion), `tests/reference/test_switch_fusion_{pandapower,pgm}.py`,
`tests/gpu/test_fusion_parity.py`.

## Cross-package surface

`node_phase_index`, `NodePhaseIndex`, `base_voltage_per_row`, `phase_voltage_magnitude`,
`branch_currents` / `BranchCurrent`, `branch_stamp_blocks`, `assemble_network_ybus`,
`assemble_ybus`, `build_injections`, `device_current_injections` (consumed by pgl, pgd,
pgg). Consumers import only from `pgml.assembly`, never private modules.
