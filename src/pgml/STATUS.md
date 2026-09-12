# pgml — status & open work

The "where am I / where do I pick up" doc for the pgml package. For the suite map and the
package layout see the root `CONTEXT.md` and `src/pgml/CONTEXT.md`; for how to work here see
`CLAUDE.md`; for the published human docs see `docs/pgml/`.

## Status — what works (validated)

Phases 0–4 of the roadmap are done bar the large-scale forks below. The subpackage
`CONTEXT.md` files hold the interfaces; `docs/pgml/` holds the concepts and modeling
decisions. One entry per capability:

- **Public API** — `pgml.simulate(grid, config) -> SolvedState` (eager voltages + lazy
  branch currents/flows/spectra/THD), `simulate_serializable`, `SimulationConfig`, the
  `pgml.errors` hierarchy (`docs/pgml/public-api.md`).
- **Provenance** — `pgml.provenance.code_provenance()` stamps commit/dirty/versions into
  every persisted artifact (dataset `meta.json`, checkpoints, cluster run dirs), and
  `generation_provenance()` adds the ACTIVE EN 50160 / IEC 61000-3-2 tables with their
  content hashes (a `PGML_*` override replaces them silently). A generator with its own
  versioned inputs — a calibrated recipe, a device library — records them through
  `write_dataset(provenance=...)`, since those change the data without changing the config.
- **Load flow** — linear (const-Z) + nonlinear (const-P / full ZIP); current-injection
  fixed point AND Newton (matrix-free option); IFT gradients; `ConvergenceDiagnostics` +
  `loadability_limit` λ-ramp (with a `capped` flag when no limit is found inside the ramp,
  and a `ramp` choice between the load-only ramp — the default, the textbook continuation
  quantity — and the joint ramp of the whole operating point). Batched-
  robust: per-scenario `converged_mask`/`failed_states` instead of raising, and the
  criticality diagnostic analyses the hardest scenario of a batch. Validated vs
  pandapower & OpenDSS on IEEE-33 / CIGRE LV.
- **Convergence in PER UNIT** — the nonlinear solve converges on the largest nodal
  apparent-power mismatch over a power base (pandapower's and power-grid-model's
  criterion, default 1e-8 pu) AND the largest per-row voltage update over the node's
  line-to-neutral rated voltage (default 1e-8 pu), each capped by the precision floor of
  the dtype / factorization backend / low-rank amplification. Per-row normalisation makes
  one tolerance mean the same thing on every voltage level and independent of the row
  count, so iteration counts are comparable with the other tools (measured: the same
  deviations against pandapower and OpenDSS at 25–43 % fewer iterations than the former
  absolute volt criterion).
- **Equilibrated factorizations** — every linear system is factored in its diagonally
  equilibrated form `D_r A D_c` and the scaling is undone on the solution, so results,
  units and gradients are unchanged while the factored system is far better conditioned
  (measured 1-norm estimates: three-phase CIGRE LV's fundamental free block 5.5e4 → 2.1e3,
  a 2016-row network's 1.4e5 → 1.3e4, IEEE-33 at harmonic order 13 8.0e8 → 1.5e3). Default
  `solver.equilibration.mode = "symmetric"` (van der Sluis `d_i = |A_ii|^-1/2`, scale
  factors rounded to powers of two so the scaled matrix is exact in binary floating point);
  `"row_column"` is the two-sided variant, `equilibrate="off"` factors the matrix as
  assembled. It buys robustness rather than forward accuracy — on a 2016-row system a plain
  complex64 solve runs to the iteration cap unscaled and converges in 9 iterations
  equilibrated.
- **Mixed precision** — `precision="mixed"` factors at complex64 and keeps complex128
  accuracy: the linear solve by iterative refinement, the nonlinear solve by running its
  fixed point in residual-correction form (measured below 1e-9 pu against a complex128
  reference on IEEE-33, CIGRE LV three-phase and a 3600-row feeder, where a plain
  complex64 run is 4e-6 to 2e-3 pu off). A plain complex64 solve logs a one-time warning
  with the estimated condition number.
- **Harmonic flow** — `solve_harmonic_flow`: nonlinear fundamental + linear per-harmonic,
  OpenDSS-exact spectrum injection, vector-group transformers (Dyn traps triplen),
  connection-aware per-phase injection. Integer orders only — non-integer orders raise
  (see the time-domain note under open work).
- **Harmonic device shunt** — every Load carries the OpenDSS load Norton shunt in
  parallel with its current source (`load_shunt`, default
  `appliance.harmonic_shunt.model = opendss` ≡ `NeglectLoadY=no`, `%SeriesRL=50`;
  `none` = the pure current-source model, `motor` = a blocked-rotor series reactance;
  per-device override via `HarmonicShuntModel`). A GENERATION device (Generator, Storage)
  carries NO shunt by default (`appliance.harmonic_shunt.generation_model = none`, one
  WARNING per solve naming the devices): the expression's conductance is negative for an
  injecting device, i.e. it would feed harmonic energy into the network, and pgml's schema
  carries no inverter filter or subtransient impedance to put there instead. `load_style`
  applies it anyway, which is what OpenDSS computes for the negative-kW `Load` idiom an
  export writes (measured difference on a feeder whose only distorting device is a 400 kW
  inverter: 3.2e-7 pu of nominal at 20 kV, 4.9e-5 pu on a 400 V cable feeder, and 0.9 %
  of the node's THD). The load shunt is the dominant damping at a feeder parallel
  resonance: the undamped model overstates a 370 kvar resonance peak by 40–75 %
  and the three-phase CIGRE LV THD by 2.4 pp. Derived from the power the device draws at
  the converged fundamental, so it is on the autograd tape (gradcheck) and batched over
  devices, orders and scenarios. Validated against a live OpenDSS `YPrim` to 4.7e-16
  relative and end to end to 1.6e-12 pu of nominal on IEEE-33, 1.3e-9 on the
  Carson-geometry feeder and 4.5e-9 on three-phase CIGRE LV, at `%SeriesRL` 0/50/100 and
  with the motor branch. Because the shunt follows the scenario, so does `Y(h)`: a batch of
  B scenarios costs B factorizations per order, and the system is assembled and factored in
  chunks bounded by `solver.harmonic.system_budget_mb` (1024 scenarios of a 294-row grid at
  13 orders would otherwise ask for an 18 GB matrix). `load_shunt_basis="nameplate"` trades
  that for the nameplate load's shunt, which is scenario-independent: measured on CPU,
  complex128, batch 256, 13 orders — 66-73 -> 1513-2008 studies/s on the 294-row Kerber
  feeder (sparse) and 282-322 -> 4038-4076 on three-phase CIGRE LV (dense), i.e. the
  no-shunt throughput — at a harmonic-voltage error against a live OpenDSS carrying the
  scenario's own kW of 2.1e-4 pu at half load, 6.7e-4 pu at 1.5x load and up to 1.0e-2 pu
  on a parallel resonance, where the shunt IS the damping.
- **Voltage-regulating generators (PV terminals)** — a `Generator` with a
  `VoltageRegulation` block (setpoint in per unit of the node rating, reactive limits,
  positive-sequence or per-phase regulated magnitude) has its terminal's REACTIVE
  power-balance row replaced by `|V|² − V_set²`, with the reactive power eliminated
  analytically, so the `[2N,2N]` IFT Jacobian/adjoint is unchanged and `dV/dv_set` /
  `dV/dq_limit` are exact (`solver/_pv_bus.py`). Reactive limits are enforced by
  PV-to-PQ switching with hysteresis (one complete solve per round at a fixed active
  set; `enforce_q_limits`, default from `pgml.defaults`). Such a grid is solved by
  Newton (logged); batched per-scenario setpoints work. Converted from pandapower
  `net.gen` (the default `GenMode.VOLTAGE_REGULATING`) and OpenDSS
  `Generator model=3`. Validated against `pp.runpp` on the MATPOWER benchmarks as
  published — case9/39/57 to 1e-15 pu, case14/30 at pandapower's own 1e-10 mismatch
  floor, case118/case300 limited only by the transformer magnetizing-branch placement
  (1e-15 pu once `i0_percent` is zeroed in both tools) — and against OpenDSS
  `model=3` to 6.6e-10 pu / 4.2e-4 kvar (`docs/pgml/modeling/der-pv-storage.md` §4.5).
- **DER inverter control + storage** — `InverterControl` on `Generator`/`Storage`
  (constant PF, cosphi(P), Volt-VAr, Volt-Watt, combined; capability circle, C¹
  smoothing), differentiated through the IFT; `Storage` = signed injection, SoC/dispatch
  off-tape in `pgml.dispatch`. Validated vs pandapower and OpenDSS InvControl
  (`docs/pgml/modeling/der-pv-storage.md`).
- **Geometry → impedance** — differentiable Carson/Deri (earth return + skin + Maxwell C)
  with a selectable conductor internal-inductance model
  (`line.geometry.internal_inductance`: `gmr` (default) / `gmr_skin` /
  `gmr_power_frequency` / `bessel`). Agreement with OpenDSS is 4.6e-8 relative on `Z` (the
  SI-vs-truncated `mu0` constant) at every order incl. triplen — below 1 kHz with the
  default, and at EVERY frequency with `gmr_power_frequency`, which reproduces OpenDSS's
  own 40 Hz-1 kHz GMR band. Analytic sequence-based harmonic line models for R/X
  feeders, selected by the typed `Line.harmonic_line_model` and applied by the converters
  from the documented defaults (`docs/pgml/modeling/harmonic-line-model.md`).
- **Scenarios (the batch contract)** — a batch of per-component deltas on one grid:
  reproducible QMC/cartesian sampling with correlated and per-phase draws, IEC 61000-3-2
  device current-emission references (default; EN 50160 stays a voltage-shaped option), the
  affine load-dependent emission law, the excitation sweeps (perturbation, spectrum,
  per-node source) and the upstream `BackgroundHarmonicConfig`. `batch_from_values` builds a
  batch from tensors a caller already has, and `run_scenarios` accepts any `ScenarioSpec`
  (an object with `sample(grid)`), so a downstream generator plugs its own recipe in without
  this package knowing it. A batch declares its own step count `n_steps` and its
  batch-shared records, so the persistence layer no longer guesses either. Parquet
  persistence records the time axis, the config owner and the caller's own provenance
  stamps. `batched == loop`, differentiable through the batch.
  The state-estimation RECIPES (the device-class library, the statistical composition, the
  load profiles, the node-coherent fingerprint sampler, the two calibrated presets) are not
  part of this package: they are calibrated modeling choices of a study and live with it.
- **Scale / solver architecture** — factor-once-solve-many, the `InjectionPlan` fast path,
  prepared systems (`prepare_power_flow`), scenario chunk tiling, the CPU sparse (SuperLU)
  backend with backend-aware convergence floors, switch-state batching (`branch_states`
  admittance scaling, differentiable) with the optional Woodbury LOW-RANK update-solve
  (`branch_states_method="woodbury"` — one base factorization for the whole sweep,
  `pgml.solver.lowrank`), multi-grid disjoint-union batching
  (`pgml.multigrid.merge_grids`) plus its BLOCK-DIAGONAL factorization backend
  (`linear_solver="block"` + `MergedGrid.block_rows()` — factors each member's diagonal
  block, equal sizes stacked into one batched LU, so an ensemble costs `O(Σ n³)` instead
  of the union's `O((Σ N)³)`; the CUDA path, where dense is otherwise the only union
  option), and pre-solve connectivity checks
  (`ConnectivityError` / `on_disconnected="zero"`). Design + measurements:
  `docs/pgml/modeling/solver-performance.md`; benchmarks:
  `run/examples/pgml/benchmark_speed.py`, `run/examples/pgml/benchmark_sparse.py`,
  `run/examples/pgml/benchmark_woodbury.py`.
- **Zero-impedance branches (ideal closed switches, bus couplers, jumpers)** — solved by
  EXACT BUS FUSION: the branch's terminal node-phase rows are collapsed into one row of the
  solved system (`pgml.assembly.fusion_map`; the reduced row index makes every stamp
  accumulate `Pᵀ Y P` directly, so nothing is materialised and nothing is approximated),
  the result is reported on the original node ids, and the current through a fused branch
  follows from Kirchhoff's law at the fused node. The published idiom this unblocks: six
  UKGDS HV/MV workbooks define a line type with R = X = B = 0 and used to crash; they now
  solve and agree with pandapower (which merges the same buses) to ≤ 7e-14 pu. Against the
  near-ideal stand-in resistance it is both more accurate and better conditioned —
  CIGRE LV: 1.6e-14 pu vs 1.1e-7 pu against `pp.runpp`, `κ(Y_ff)` 910 vs 1.44e4; a 132 kV
  coupler: `κ` 648 vs 4.9e6, and two of the UKGDS nets need 200 instead of 10 iterations
  with the stand-in. Differentiable (a fused branch's own impedance is structurally not a
  parameter of the solve: its gradient is a finite zero), batched, CPU/CUDA parity 4.7e-14
  pu at 5370 rows. Policy: `branch.zero_impedance` (`fuse` default / `error`);
  `branch.switch_model` decides what a converter writes for a switch with no impedance
  data. A branch under a `branch_states` sweep is never fused (a swept branch needs a
  stamped admittance to scale) and is refused by name.
- **Convert** — pandapower / OpenDSS / power-grid-model → `Grid`, with per-terminal phase
  permutations, n_phases-aware neutrals, positive-sequence reduction (Z1 = Zself−Zmutual),
  pandapower `parallel` + line/trafo switches, ZIP load models, pandapower `gen`
  (voltage-regulating by default) and `shunt` (a capacitor bank or a reactor, by the sign
  of `q_mvar`), OpenDSS Capacitor/Reactor/Generator/PVSystem/Storage, the resolved
  harmonic line model on every converted line, and a dropped-element warning naming any
  unconverted kind. Coverage table: "Known modeling gaps" below.
- **Evaluation** — comparison plots + reference oracles (optional `oracles` extra),
  including the INDEPENDENT OpenDSS scenario oracle (`opendss_scenario_oracle`): full DSS
  export, snapshot + sequence dataset generation in `write_dataset` format with
  `meta["engine"]="opendss"`, matched + default assumption modes. Matched-mode parity is
  ~1e-8 pu on the feeder cases (~1e-6 on CIGRE LV 3-phase; the residual is the documented
  magnetizing-branch placement difference, `docs/pgml/modeling/transformer.md`).
- **Measurement instrumentation** — `MeasurementDevice` on the `Grid` (schema rev 0.0.3):
  node-anchored meters + CT channels, accuracy class, averaging intervals; inert metadata
  for downstream state-estimation and acquisition tooling to consume.
- **Infra** — PEP 621 packaging, `py.typed`, GitHub Actions (ruff + tests + strict docs),
  pytest markers (`gpu`/`opendss`/`slow`, registered in `pyproject.toml`).

## How to run

- **Use it**: `import pgml; pgml.simulate(grid, pgml.SimulationConfig(...))` — see
  `docs/pgml/public-api.md` (and the suite's getting-started guide on the published site).
- **Tests**: `pixi run -e cpu pytest -q` (diff gate `tests/differentiability`, GPU gate
  `tests/gpu`). **Lint**: `pixi run -e cpu ruff check src tests run`. **Docs**: see `CLAUDE.md`.
- **Examples**: `run/examples/pgml/` (each self-documenting — see `run/examples/README.md`).

## Open work — where to start

WHAT / WHY / WHERE / HOW. "⚠️ decision" = confirm the approach with the maintainer before a
large rework (a schema change needs the maintainer's sign-off first).

### A. Batching / scale → production GPU training-data generation  ⚠️ decision — main gap

**What.** Generate LARGE volumes of harmonic-flow training data on a GPU, reproducibly.
The sampling layer and the dense scale wins are done (see Status +
`docs/pgml/modeling/solver-performance.md`); what remains:

- **DONE — switch-state sweeps as a low-rank update.** A sweep over `S` switch states no
  longer costs `S` assemblies and factorizations: `branch_states_method="woodbury"`
  factors the sweep's BASE network once and reaches every state through a
  Sherman-Morrison-Woodbury update of that factorization (`pgml.solver.lowrank`,
  `O(N²k + k³)` per state with `k = Σ 2P` over the switched branches). Explicit opt-in;
  the default `"assemble"` path is unchanged. Measured on an i7-12700 (CPU, `auto`
  backend = SuperLU sparse, `S = 8` states, `run/examples/pgml/benchmark_woodbury.py`)
  for 1-4 switched 3-phase branches (`k = 6…24`): 3.2-3.5x at 600 rows, 4.3-6.8x at
  1200, ~5.8x at 2100, ~5.4x at 3000; still 1.7-4.4x at `k = 96`. The crossover is
  around `k ≈ N/3` (measured: 1.1x at `k = 192` and 0.2x at `k = 384` on 600 rows),
  beyond which assembling per state is cheaper. Voltages agree with the assemble path
  to ~1e-12 relative, and the memory profile changes from `[S, N, N]` to one base
  factorization plus `[S, k, k]`.
  The BASE omits every switched branch it can (opening one is a stable update, removing
  a near-ideal switch from the base is not — `pgml.solver.lowrank`).
- **GPU / memory micro-optimisations (benchmark-informed)**: measure `precision="mixed"`
  on CUDA, where the FP64 throughput ratio (1/64 on consumer cards) makes the
  single-precision factorization the dominant lever — the CPU measurement is only the
  lower bound of the win; re-measure the state-Jacobian memory budget
  (`solver.ift.jacobian_budget_mb`, 1 GiB, calibrated on a 12 GB card) on the device the
  campaign runs on (`run/examples/pgml/benchmark_ift_backward.py`);
  chunk-to-chunk warm starting for sorted/correlated scenario chunks;
  `torch.cuda.CUDAGraph` / `torch.compile` over the fixed-point iteration (static shapes
  per chunk — likely wins at small N).
- **Very large N / large unions — the designated GPU sparse-direct route: cuDSS via
  nvmath-python.** torch has no batched sparse direct solve, so today CUDA is always
  dense and a merged ensemble is bounded by union-sized dense memory. The intended
  backend is NVIDIA's cuDSS through nvmath-python's sparse direct-solver API: it
  supports complex matrices, batches uniformly over ONE sparsity pattern (exactly the
  scenario/ensemble case — the pattern is fixed, the values vary) and separates
  analysis / factorization / solve so the symbolic phase is paid once and reused across
  a whole run, matching the factor-once-solve-many design. It is Beta upstream — pin and
  re-validate before adopting. PREREQUISITE: COO/block-aware assembly, so the union's
  dense `[N, N]` `Y` is never materialised. The union-sized dense remainders today are
  (1) assembly itself (the stamps scatter into a dense accumulator), (2) the ideal-slack
  `Y_fs` coupling gather, and (3) the IFT backward's dense `[2N, 2N]` Jacobian (shared
  with Newton). `linear_solver="block"` already removes the union-sized FACTORIZATION for
  an ensemble; these three are what still bound it.
- **Matrix-free / factorization-reusing IFT backward.** The backward builds the real
  `[B, 2N, 2N]` state Jacobian by autograd, which still dominates the cost of a gradient
  even after the build was budgeted and the adjoint factorization cached (measured CPU,
  one process per point: IEEE-33 at batch 64, 773 ms against a 15 ms forward; a 1176-row
  ensemble at batch 1, 658 ms against 136 ms, and 2.26 s at batch 4). The adjoint system is
  the TRANSPOSE of the same operator the forward already factors, and its device term is
  block-diagonal per node-phase row, so the autograd build is avoidable altogether: see the
  design sketch in the solver ledger (`src/pgml/solver/CONTEXT.md`, "IFT backward"). This is
  the largest remaining differentiability cost.
- **Beyond direct-factorization scale**: a GPU-resident Krylov path (block-Jacobi /
  additive-Schwarz preconditioning, the shape GPU power-flow solvers built on iterative
  methods take) is the fork for networks too large to factor at all — secondary at LV
  sizes, where direct factorization wins.
- Resolved as won't-do (measured): a batch-native Newton forward — the block-diagonal
  Jacobian build was 4× slower than the per-scenario path at B=64/N=180; bulk batches
  belong to the current-injection method.

**Where.** `src/pgml/solver/`, `src/pgml/scenarios/`, `tests/gpu/`.

### B. Harmonic state estimation

Out of scope for this repository. Harmonic state-estimation models and training consume
pgml's public API from a downstream package.

### C. Frequency-dependent device models (shipped transformer loss curves)

**What.** Both halves of the former device-model gap are CLOSED. The harmonic device
shunt is implemented and carried by default on every LOAD (`load_shunt`, modeling default
`appliance.harmonic_shunt.model = opendss`; a generation device carries none, see
`appliance.harmonic_shunt.generation_model`): per element
`Y_eq = conj(S_eff)/V_rated²` split into a parallel and a series R-L branch, with the
`motor` blocked-rotor variant, validated against a live OpenDSS `YPrim` to 4.7e-16
relative and end to end to 1.6e-12 pu of nominal (6.8e-12 on a capacitor resonance at
order 6.9, 4.5e-9 on three-phase CIGRE LV). `resistance_frequency` and
`harmonic_xr_constant` are consumed by the transformer stamp
(`R(f) = R · m(f) · (f/f0 if harmonic_xr_constant else 1)`, validated against OpenDSS's
`XRConst`). What REMAINS is (i) that no eddy-current/stray-loss curve is SHIPPED, so a
transformer's default winding resistance is still constant with frequency (a user must
supply a measured `resistance_frequency` curve), and (ii) that a DER has no harmonic
output impedance at all: the load expression is anti-damping for an injecting device and
is therefore off by default, while the physical quantity — an inverter's filter impedance
or a machine's subtransient reactance (OpenDSS `%R`/`%X`, `Xdpp`) — has no schema field.
**How.** Ship a documented default transformer loss curve (IEC 60076-based or a published
measurement) with a builder that attaches it, and add a DER harmonic impedance field (a
per-device `R + jX` or a frequency curve) with its stamp. **Where.**
`src/pgml/data/standards/`, `assembly/ybus.py` (`_resistance_multiplier`),
`schemas/grid_schema.py` + `assembly/_load_shunt.py` for the DER impedance,
`tests/reference`.

### D. Smaller follow-ups (no decision needed)

- **The mismatch floor's `|Y|` pass on the one-shot path**: the per-row precision floor of
  the power-mismatch criterion needs `Σ_j |Y_ij| V_base,j`, one read of the whole
  admittance matrix per solve (measured inside a solve, complex128, CPU: 0.07 ms at 33 rows,
  1.8 ms at 294, 59 ms at 2400, i.e. up to 18 % of a single unbatched solve). A prepared system
  carries it (`PowerFlowSystem.row_abs_scale`), so a scenario run pays it once per grid;
  a caller that solves one grid repeatedly through the one-shot entry point pays it per
  call. Cheaper formulations were measured and do not pay: the bound `|Y| <= |Re| + |Im|`
  (no square root) is 8x faster at 294 rows but only 10 % faster at 2400, because at that
  size the pass is memory-bandwidth-bound rather than arithmetic-bound (isolated on a dense
  matrix of the same size: 1.06 ms against 0.12 ms at 294 rows, 41 ms against 37 ms at 2400). The remedy is
  either to document `prepare_power_flow` as the repeated-solve entry point or to derive
  the row scale from the branch stamps during assembly. WHERE:
  `src/pgml/solver/power_flow.py` (`_abs_row_scale`, `_PuConvergence`).
- **The prepared system's staleness guard costs a grid serialization per solve**:
  `solve_power_flow(system=...)` verifies `network_fingerprint(grid)`, which serializes
  every branch to JSON — measured 0.18 ms at 33 rows, 1.5 ms at 294 and 4.9 ms at 2400,
  which is 4 % to 11 % of a prepared solve (the path a scenario generator uses). The check
  is what turns a silently stale factorization into an error, so it should not simply be
  dropped; a cheaper witness (a mutation counter on the grid, or verifying once per
  `(system, grid)` pair and accepting mutation between calls) is a contract decision,
  and serializing the branch list in one pydantic call instead of per branch was measured
  NOT to be faster. WHERE: `src/pgml/solver/power_flow.py`, `src/pgml/topology.py`.
- **The Bessel skin multiplier above the fundamental**: the positive-sequence / sequence-
  aware line model fits an equivalent `Rdc` with twelve fixed-point steps, each evaluating
  a forty-term continued fraction, and then evaluates it once more per frequency. Measured
  complex128, CPU, 13 orders: 4.9 ms for 32 lines, 10 ms for 294, 49 ms for 2400, of which
  the fit is 3.2 ms at any size (it is complex-division bound). The fit converges to
  machine precision in about five steps for distribution resistances, so `iters` is three
  times what it needs, but reducing it moves the last bits of every harmonic impedance.
  A per-grid cache is not available either: the multiplier is differentiable in `R1`, so a
  parameter fit changes it on every call. WHERE: `src/pgml/geometry/sequence.py`
  (`fit_equivalent_rdc`), `src/pgml/geometry/carson.py` (`i0_over_i1`).
- **A harmonic study resolves the zero-impedance structure twice**: `solve_harmonic_flow`
  resolves it for the study and its inner `solve_power_flow` resolves it again, because
  `solve_power_flow` has no way to accept an already-resolved structure (`system=` is the
  only such channel and it also carries a factorization). A structure-only hand-off would
  save one branch walk per study (0.2 ms at 294 branches, 0.6 ms at 799). WHERE:
  `src/pgml/solver/harmonic_flow.py`, `src/pgml/solver/power_flow.py`.
- **Equilibrate the sparse backend without a dense pass**: the symmetric scaling writes one
  full `[N, N]` copy, which on the SuperLU path costs as much as the factorization itself
  (measured on a 1176-row feeder, complex128, CPU: 6.7 ms of scaling against a 7.0 ms
  factorization; on the dense path 6.7 ms against 29.9 ms). Scaling the CSC data instead is
  `O(nnz)`, i.e. free, but the factorization keeps the SCALED matrix as `y_mat` for the
  mixed-precision residual, the sparse adjoint and `estimate_condition`, so it would have to
  become lazy for the no-gradient case. WHERE: `src/pgml/solver/harmonic.py`
  (`_SciPySparseLU`, `FactoredSystem.y_mat`), `src/pgml/solver/equilibration.py`.
- **The anchored solve factors an unscaled SI matrix**: `AnchoredSystem` builds
  `Z = Y_ff^-1` with a plain `torch.linalg.solve`, so it is the one factorization in the
  package that does not go through the equilibration every other path uses. WHERE:
  `src/pgml/solver/harmonic.py`.
- **A low-rank harmonic shunt for a sparse device population**: with the shunt on the
  `operating_point` basis, `Y(h)` is per scenario and a batch costs one factorization per
  scenario and order. A Woodbury correction over the nameplate system would pay where the
  distorting devices sit on few rows (`k/N < 1/3`, the measured crossover of the
  switch-state path); on the benchmark grids `k/N` is 0.34 to 0.97 (every load node-phase),
  so it does not. The DER case is the one to revisit. WHERE:
  `src/pgml/solver/harmonic_flow.py`, `src/pgml/solver/lowrank.py`.
- **Per-scenario time anchors in the batch contract**: a sequence batch records ONE `[T]`
  `time_unix_s` vector for the whole batch, so a generator that wanted to stagger its
  scenarios over the day has nowhere to put a per-scenario anchor. Allowing a `[B, T]`
  `time_unix_s` is a contract change here (the shared-record split, the sidecar's
  `t0_unix_s`, and every downstream time-feature consumer). WHERE:
  `src/pgml/scenarios/sampler.py` (the record contract), `persistence.py` (the sidecar).
- **SolvedState mutation guard**: lazy accessors recompute from the referenced grid; the
  no-mutation-after-solve rule is currently a docstring contract only. Reuse the network
  fingerprint (the `PowerFlowSystem` guard mechanism) to detect post-solve grid mutation
  on lazy access and raise. WHERE: `src/pgml/simulation.py`.
- **Native OpenDSS end-to-end parity gate**: the parity test against OpenDSS's NATIVE
  harmonics mode on IEEE-33 (Carson geometry lines) and the 3-phase CIGRE LV (native Dyn
  transformers) lives with the paper's benchmark harness (the `pgml-paper` repository),
  because `pgml.evaluation.oracles.opendss_scenario_oracle` still refuses
  conductor-geometry lines. Teach the scenario oracle geometry lines (export the
  `LineGeometry` as an OpenDSS `LineGeometry`/`WireData` pair) and rebuild that gate on
  it here, self-contained. WHERE: `src/pgml/evaluation/oracles/opendss_scenario_oracle.py`,
  `tests/reference/`.
- **Legacy live oracle injection model**: `evaluation/oracles/opendss_oracle.py` stamps
  device harmonic injections on all host-node rows from phase-to-ground voltages and
  nameplate P/Q (documented in its docstring) — not connection-aware, unlike the solver
  and the scenario oracle. Fix it or deprecate those paths in favor of
  `opendss_scenario_oracle`. WHERE: `src/pgml/evaluation/oracles/`.
- **Convert (OpenDSS)**: not yet read: 3-winding units, `RegControl`, `XfmrCode`/
  frequency-correction curves, explicit non-solid neutral nodes, and load
  `Vminpu`/`Vmaxpu`/CVR (currently silently ignored — at minimum warn). WHERE:
  `src/pgml/convert/opendss/`.
- **Converter test gaps**: the OpenDSS SINGLE_PHASE_EQUIV positive-sequence reduction
  (`Z1 = Zself − Zmutual`, `_positive_sequence_scalar`) has no direct value-pinning test
  (only a shape check); pandapower line/trafo switches lack a minimal dedicated
  open-switch parity case (currently exercised only inside the large CIGRE MV /
  mv_oberrhein comparisons). WHERE: `tests/convert/`, `tests/reference/`.
- **DER (optional extensions)**: a REMOTE regulated bus (a machine holding a voltage at
  another node — pandapower has no column for it, OpenDSS `RegControl` does), a DELTA or
  neutral-returning regulating terminal (the row pair is formed for a grounded WYE
  terminal; the neutral case needs the generator's rows folded into the neutral row
  before the substitution), a scenarios `ParameterSpec(field="v_set")` so a setpoint
  sweep is writable from a scenario config, and carrying the SOLVED reactive power of a
  regulating machine into its harmonic injection scaling (today the nameplate value is
  used). WHERE: `src/pgml/solver/_pv_bus.py`, `src/pgml/scenarios/`,
  `src/pgml/solver/harmonic_flow.py`.
- **Storage dispatch (optional extensions)**: higher-level dispatch policies in
  `pgml.dispatch`, and a scenarios `Selector(component="storage")` to sample storage
  setpoints across a batch.
- **Harmonic flow**: batch-dim mismatch guard (operating_point vs harmonic_injection);
  vectorize the device×order python loop in `harmonic_flow._harmonic_injections`.
- **Criticality beyond one scenario**: the IFT-Jacobian criticality analysis describes the
  hardest scenario of a batch (named in the result). A per-scenario SVD for every element of
  a batch, or a sparse / matrix-free variant for states above `2N = 4000`, is open.
- **Per-node harmonic-source sweeps** (`run_node_injection_sweep`) loop one solve per
  node; batch the target-row index (`[B, P]` scatter) to lift the loop.
- **Transformer (assembly)**: non-solid neutral grounding (`GroundingImpedance`) and
  zigzag-zigzag pairings raise `ModelingError`. What remains open on the zero sequence is
  the MAGNETIZING branch (pandapower `mag0_percent`/`mag0_rx`, power-grid-model
  `i0_zero_sequence`/`p0_zero_sequence`) and the HV/LV split of the zero-sequence leakage
  inside a T (pandapower `si0_hv_partial`); the converter names both in a WARNING rather
  than dropping them silently.
- **Transformer (solver)**: a fully ungrounded secondary island has no absolute
  zero-sequence reference (line-to-line-correct, absolutely-undetermined voltages as
  load → 0); a reference injection / per-island pin would close it.
- **Geometry**: low-X R/X lines hit the GMR floor (flagged `synth_unphysical`); 2-phase
  lines are skipped by `synthesize_grid_geometry`; Carson `C` agrees with OpenDSS's to
  2.1212e-5 relative = the `e0` constant ratio (pgml uses the SI value), and OpenDSS's
  `capradius` option is not read (match it if a c≠0 feeder is added).
- **Internal-inductance model is global, not per line**: `line.geometry.internal_inductance`
  is a modeling default read at assembly time; a grid that mixes synthesized geometries
  (placeholder radius) with measured conductor data cannot choose per line. A
  `LineGeometry.internal_inductance` field would close it; until then assembly WARNS when a
  radius-based model meets a `synth_unphysical` geometry.
- **Stranded conductors**: `gmr_skin` and `bessel` take the internal-inductance decay from
  the equivalent SOLID round conductor. That is exact for a solid conductor and an
  approximation for ACSR/stranded ones (tubular / multi-layer internal impedance, steel
  core); measured spread on the published ACSR data in the OpenDSS line-constants example
  is under 2 % of `X` up to 2.5 kHz.
- **Continuation/Newton polish**: a true arc-length predictor-corrector (today's
  `loadability_limit` is a step-and-bisect on Newton FEASIBILITY, so its
  `breaking_lambda` is a lower bound on the nose — measured ~4 % below the closed-form
  nose of a two-bus feeder); a GMRES preconditioner near the nose; batched continuation.
  Iwamoto's optimal multiplier
  (Iwamoto & Tamura 1981, IEEE Trans. PAS-100:1736): the complex power-flow residual is
  EXACTLY quadratic in `(V, conj(V))`, so the second-order Taylor term is exact and the
  optimal Newton step length has a closed form from one extra residual evaluation per
  iteration — an ill-conditioned/near-nose robustness upgrade (larger convergence
  region, no divergence overshoot), not a throughput lever; adopt it with the
  continuation work, on both the plain Newton solve and the corrector.
- **Typing / mypy gate** (incremental): targeted annotations on the public API + a gate on
  the non-duck-typed modules; don't fight the deliberate `Any` of the float/tensor duality.
- **Deferred (no priority) — time-domain simulation.** The steady-state frequency-domain
  scope is deliberate (integer harmonic orders only; non-integer orders raise).
  Interharmonics, flicker and transients belong to a future differentiable TIME-DOMAIN
  companion path over the same Grid contract, which would subsume interharmonic support.

## Known modeling gaps (physics NOT currently modeled — keep this list honest)

Each entry states what the simulator deliberately (or currently) does NOT capture, so
results are never read as more physical than they are. Details live in `docs/pgml/modeling/`.

- **Transformer, frequency dependence.** `R(f) = R · m(f) · (f/f0 if
  harmonic_xr_constant else 1)`: the default is OpenDSS's `XRConst=No` (X ∝ h at constant
  R, so X/R grows with the order), and `harmonic_xr_constant=True` holds X/R constant
  (validated against OpenDSS's own `Yprim` at h = 1, 5, 13 to 1.25e-6 S, the residual
  being OpenDSS's anti-float shunt). `resistance_frequency` accepts a constant, the
  Carson skin law or a sampled curve, the same multiplier the line path uses. No
  eddy-current/stray-loss curve is shipped as a default, and no saturation/inrush
  (steady-state tool).
- **Transformer, magnetizing placement.** A documented modeling choice,
  `transformer.magnetizing_placement`: `from_terminal` (shipped default — the HV/from
  phase diagonal, so core loss is independent of loading), `to_terminal` (OpenDSS's own
  placement: it attaches the whole branch to its LAST winding's terminal, verified on a
  live `Yprim` difference) or `split` (power-grid-model's: half on each terminal). Measured
  against a live OpenDSS solve on a 500 kVA 20/0.4 kV unit (Dyn and YNyn, 0-500 kW load):
  `to_terminal` agrees to 3e-10…2e-9 pu, while the `from_terminal` default deviates by
  9.2e-5 pu at i0 = 0.1 %, 2.2e-4 pu at 0.5 % and 8.2e-4 pu at 2 %, and `split` by half of
  that. `split` reproduces power-grid-model to machine precision
  (`tests/reference/test_opendss_magnetizing_placement.py`,
  `tests/reference/test_pgm_transformer.py`). No zero-sequence-specific magnetizing branch
  exists for any placement.
- **Transformer, construction.** All winding pairings except zigzag-zigzag, at every clock
  of the pairing's parity; non-solid neutral grounding raises (fail loud). The ZIGZAG
  limb-domain incidence is EXPERIMENTAL: it reproduces the clock shift, the blocked
  zero-sequence transfer and the winding's own zero-sequence path, and agrees with
  power-grid-model on an unbalanced solve once the zero-sequence VALUE is carried, but no
  second reference tool can express the same unit as one two-winding element. Constructing
  one logs a WARNING once per process. The
  zero-sequence leakage VALUE is `Transformer.zero_sequence` when set (read from
  pandapower's `vk0_percent`/`vkr0_percent`), else the documented
  `transformer.zero_sequence.*` ratios (1.0 = Z0 = Z1, which is what OpenDSS and
  power-grid-model imply since neither has a zero-sequence leakage input). What is NOT
  modelled: a zero-sequence MAGNETIZING branch (pandapower `mag0_percent`/`mag0_rx`,
  power-grid-model `i0_zero_sequence`/`p0_zero_sequence` — the three-limb-core path
  through tank and air), the HV/LV split of the zero-sequence leakage inside a T
  (pandapower `si0_hv_partial`), and a neutral earthing impedance (`3*Z_N`; pandapower
  `xn_ohm`/`rn_ohm`, OpenDSS `Rneut`/`Xneut`). Each is named in a converter WARNING (or
  refused) rather than silently dropped.
- **Load harmonic behaviour.** Each Load/Generator/Storage is a harmonic current source
  in PARALLEL with the OpenDSS device shunt (`appliance.harmonic_shunt.model`, default
  `opendss` ≡ `NeglectLoadY=no` with `%SeriesRL=50`; `none` reproduces the pure
  current-source model and `motor` the blocked-rotor variant). The shunt is derived from
  the power the device REALLY draws at the converged fundamental, where OpenDSS uses the
  SPECIFIED power — identical for a constant-power device, and 6e-5 to 1e-4 pu of nominal
  apart for a const-Z / const-I / ZIP one (measured on IEEE-33, bounded by a test).
  pgml has no per-device rated voltage: the shunt's `V_rated` comes from the host node,
  while OpenDSS takes each Load's own `kV` property, so an imported circuit whose load
  `kV` differs from its node's nominal carries a correspondingly different shunt.
  A DER carries NO harmonic shunt by default
  (`appliance.harmonic_shunt.generation_model = none`). The load-style
  `conj(S)/V_rated²` would have a NEGATIVE conductance for an injecting device — the
  negative-load idiom an OpenDSS export writes, and anti-damping rather than damping —
  while the quantity a real inverter or machine presents is its filter or subtransient
  impedance (OpenDSS `%R`/`%X`, `Xdpp`), which this schema does not carry. `load_style`
  reaches the idiom for a comparison; a measured DER harmonic impedance needs its own
  schema field and is open work (§C).
- **Sources.** The per-phase Thevenin is sequence-aware: converters read the native
  zero-sequence data (OpenDSS `Vsource.R0`/`X0`, power-grid-model `source.z01_ratio`,
  pandapower `ext_grid.x0x_max`/`r0x0_max` with `s_sc_max_mva`/`rx_max`) into the
  symmetric-component self/mutual split; without it the documented
  `source.zero_sequence.*` ratios apply (1.0 = Z0 = Z1) and a WARNING names the element.
  NEGATIVE sequence is always `Z2 = Z1` (a passive upstream network); a rotating-machine
  source with `Z2 != Z1` would need the third circulant entry. pandapower's own
  `runpp_3ph` instead pins the positive sequence and puts the short-circuit impedance in
  the negative-sequence network, and scales its zero-sequence shunt by the IEC factor
  `c = 1.1` — both differences are quantified in
  `tests/reference/test_pandapower_source_zero_sequence.py`.
- **PV-terminal scope.** The regulated row pair is formed for a WYE terminal whose
  return is ground: a DELTA machine (its reactive current is shared between two node
  rows, so only the circulating total is observable), a WYE machine returning through
  its node's neutral row (use `return_path='ground'`), a positive-sequence setpoint on
  a 2-phase terminal, and a regulating generator on a Source's node all raise
  `ModelingError`. Two regulating machines on ONE node are not separable either (their
  summed reactive power is observable, the split is not) and raise; the pandapower
  converter merges such rows instead. Reactive limits bind by SWITCHING,
  so the solution is exact at the limit but the active set is piecewise constant in the
  parameters: at a switching boundary the gradient is one-sided. Regulation is a
  fundamental-frequency concept; at orders h>1 the machine stays a Norton current
  source, and its harmonic current is scaled from the NAMEPLATE reactive power, not the
  regulated one (`docs/pgml/modeling/der-pv-storage.md` §4.5).
- **ZIP loads sharing a bus with generation (cross-tool).** pandapower reduces a ZIP load's
  coefficients onto the BUS and applies them to that bus's NET injection
  (`_calc_pq_elements_and_add_on_ppc`), so a const-Z load and a generator on one bus cancel
  BEFORE the voltage-dependency is applied. pgml keeps the devices distinct, which is the
  physical model — a constant-impedance load and a constant-power inverter only cancel at
  nominal voltage. Measured on an LV bench where each inverter mirrors its bus's load:
  3.9e-3 pu at the affected buses (both tools agree to 5e-8 with the generation stopped). Not a converter defect; take it into account when a pandapower
  reference is used as ground truth for a grid with co-located ZIP load and generation.
- **Closed bus-bus switch with a NON-ZERO `z_ohm`.** The pandapower converter reads
  `switch.z_ohm` as the switch RESISTANCE (`Switch.resistance_ohm`, `inductance_h = 0`);
  pandapower itself splits that value across R and X at its `switch_rx_ratio` (default 2,
  so X = z/√5). Physically a closed contact is resistive, but the difference is not
  negligible on a low-reactance cable network: on an LV cable feeder (line X ≈ 1.4 mΩ) a
  1 mΩ switch contributed a 0.0125° node-angle divergence from pandapower until the
  reference run was given a purely resistive switch. Decide whether to reproduce
  pandapower's split. This affects only a switch that CARRIES a `z_ohm`; with
  `z_ohm = 0` (every pandapower built-in) the converted switch is ideal and the solve
  collapses its two rows exactly, which is pandapower's own treatment (see the capability
  list). WHERE: `src/pgml/convert/pandapower/converter.py` section 4 (bus-bus switches).
- **Line geometry (Carson/Deri).** No conductor temperature dependence, no sub-conductor
  bundling; transposition per the documented Deri assumptions.
- **Shunt reactor with a SERIES resistance.** The shunt primitive is the parallel form
  `G + 1/(j·2πh f0 L) + j·2πh f0 C`. An OpenDSS `Reactor` with `R = 0` (its default) and a
  pandapower inductive `net.shunt` map exactly at every order (both carry an
  `inductance_h`, so `|B(h)| = B/h`); an OpenDSS reactor with a series `R > 0` is converted
  as the equivalent parallel pair at f0 (exact there, and the converter warns), so its LOSS
  term stays flat where the series branch decays as `1/h²`. Add a series resistance to the
  inductive branch if a lossy reactor has to be harmonically exact.
- **Zero-sequence line impedance at harmonics — lumped R/L lines only.** Lines WITH
  `conductor_geometry` compute Z(h) from first principles (agreeing with OpenDSS to
  4.6e-8 relative at every order below 1 kHz with the default internal-inductance model,
  and at every order with `line.geometry.internal_inductance: gmr_power_frequency`). Lines WITHOUT geometry embed the earth return AT f0; extrapolating Z0 to h·f0
  is an ASSUMPTION in every tool. pgml's lumped `sequence_aware` model adds the Carson
  earth-return resistance `3·(Re(f) − Re(f0))` to R0 and scales X0 ∝ h; OpenDSS's R/X
  line does the same with its `Rg` and additionally bends X0 sub-linearly with `Xg`. Both
  engines agree to ~1e-10 relative when the earth parameters are MATCHED (see
  `tests/reference/test_lumped_sequence_opendss.py`), so the remaining divergence is a
  choice of parameters, not of formula: OpenDSS's DEFAULT `Rg`/`Xg` are the physical
  Carson values at 60 Hz in Ω per 1000 ft and are reinterpreted in the line's `units`, so
  on a metric line they are ≈3.28× smaller than pgml's physical default. The sub-linear
  X0 law is available as `line.earth_return.x0_frequency: carson_sublinear` (off by
  default: it can drive X0 negative above h ≈ 30 for a cable whose stored X0 is small,
  exactly as OpenDSS does). Supply conductor geometry when the zero-sequence earth return
  must be right (`docs/pgml/modeling/harmonic-line-model.md`).
- **Missing zero-sequence line data** is invented with global overhead-line ratios
  (R0/R1=4, X0/X1=3, C0/C1=0.5 — `data/defaults.yaml`); weak for cables. The converters
  now WARN once per grid when they used them, naming the ratios.
- **Spectra tables.** EN 50160 = supply-VOLTAGE compatibility levels (orders 26–49 a
  marked manual extension) — correct for source/background distortion, NOT device
  emission. Device current fingerprints default to IEC 61000-3-2 (per class A/B/C/D;
  `emission_class="auto"` maps consumer_type→class, an approximation — no lighting
  consumer_type yet for Class C).
- **complex64 on ill-conditioned grids.** The engine carries no per-unit normalisation, so
  an SI-unit system is badly SCALED: the 1-norm condition estimate of the fundamental
  system AS ASSEMBLED is 2.8e3 (IEEE-33), 1.7e4 (CIGRE LV single-phase-equivalent), 5.5e4
  (three-phase), 6.9e4 (mv_oberrhein) and 4.6e5 (a 3600-row synthetic feeder), and a stiff
  source or a near-ideal switch pushes it decades higher. The equilibration above removes
  most of that (and is what the reported estimate now describes), but not all of it: a
  low-voltage-only harmonic system stays at 1e7 because its conditioning is not a scaling
  artefact, and equilibration does not make single precision accurate — LU with partial
  pivoting is backward stable, so the complex64 error barely moves. A plain complex64 solve
  therefore still keeps only `7 − log10(cond)` digits (measured |ΔV| against complex128 at
  the fundamental, CPU, batches of 1 to 256: 2.3e-6 pu dense / 4.9e-6 pu sparse on
  IEEE-33, 2.7e-6 / 5.8e-6 on three-phase CIGRE LV, 3.3e-5 / 2.3e-5 on the 294-row Kerber
  Vorstadtnetz, 2.0e-5 / 2.5e-5 at 1176 rows, against 1e-14 to 1.2e-12 for
  `precision="mixed"` on the same runs), i.e. four to five digits of a per-unit voltage,
  which no correctness guard at 1e-6 pu can pass. It logs a one-time warning naming the
  estimate, and a solve that stops at that plateau reports it: the iteration reaches an
  exact fixed point of the rounded map or a small limit cycle, and a scenario whose cycle
  sits above the dtype's floor converges AT THE FLOOR with
  `diagnostics.floor_governed` set and a warning naming the level, rather than running to
  the iteration cap. The
  recipe: solve at complex128, or at complex128 with `precision="mixed"`
  (single-precision factorization refined against double-precision residuals — measured
  1.5–1.9x faster than complex128 on the DENSE path at 132–3600 rows, and no faster on the
  CPU SuperLU sparse path, where single precision does not speed the factorization up),
  and store complex64.
- **Scenario sampling runs on CPU** (`SobolEngine`), promoted to the solve device after —
  deliberate; the sampled tensors are tiny next to the solve.
- **Converter coverage.** Converted: pandapower `bus`/`line`/`load`/`asymmetric_load`/
  `trafo` (vector groups + taps, `tap_changer_type` honoured as pandapower does)/bus-bus
  + line/trafo `switch` (+`parallel`)/`ext_grid`/`sgen`/`gen` (an exact PV terminal by
  default)/`shunt`; OpenDSS Lines/2W-Transformers/Vsources/Loads (ZIP models)/Capacitor/
  Reactor/Generator (incl. `model=3`)/PVSystem/Storage; pgm `node`/`line`/`sym_load`/
  `asym_load`/`source`/`sym_gen`/`transformer`/`link` (an ideal closed switch, fused at
  solve time). NOT converted (a WARNING names any non-empty dropped kind): pandapower
  `trafo3w`, `impedance`, `ward`/`xward`, `dcline`, `storage`, `motor`,
  `asymmetric_sgen`; pgm `three_winding_transformer`, `shunt`, `asym_gen`,
  `transformer_tap_regulator`; OpenDSS items under D above.
  pandapower ideal phase-shifter taps raise, and a tap position whose
  `tap_changer_type` is unset is dropped with a WARNING (pandapower ignores it too).
  Neither the pandapower nor the pgm converter ever emits `Phase.N` (pandapower's
  `THREE_PHASE` mode is fixed `(A, B, C)`; pgm has no neutral phase at all) — a 4-wire
  grid with an explicit neutral conductor comes only from the OpenDSS converter's
  `THREE_PHASE` mode or a hand-built `Grid`.
  NOT YET mapped: power-grid-model's `voltage_regulator` component (1.13+; it makes a
  `sym_gen`/`asym_gen`/`sym_load`/`asym_load` a PV terminal through
  `regulated_object` + `u_ref`, with `q_min`/`q_max` declared but not yet enforced by
  pgm itself) — the schema side is ready, the mapping and its oracle run need an
  environment with a working power-grid-model core.

## Conventions a contributor must respect (full list + the package map: root `CONTEXT.md`)

- `schemas/` is FROZEN (a change needs the maintainer's sign-off); everything imports and conforms to it. Each
  subpackage's `CONTEXT.md` is its interface ledger — read before editing, update after a
  public-signature change.
- The two hard constraints: **differentiable** + **GPU-ready** (float64 `gradcheck` and the
  GPU device/dtype test must pass — see `CLAUDE.md`).
- Phase-domain, SI, store **L/C** not X/B; phasors as (real, imag); index by frequency;
  float/tensor duality (schema fields accept floats OR tensors, autograd flows through).
