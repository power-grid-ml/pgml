# Solver architecture: performance and structural checks

This page records how the nonlinear solve is organised for speed and robustness, and *why*
each choice was made — usually by measuring the alternatives rather than assuming. Where a
choice mirrors (or deliberately departs from) pandapower, power-grid-model, or OpenDSS, the
comparison is stated explicitly. The [reference-library notes](references/index.md) hold the
per-tool briefs.

One design rule shapes everything here: the iterative forward solve is **detached** from
autograd, and gradients come from the implicit function theorem at the converged solution
(one adjoint linear solve, one residual differentiation). Because no iteration is ever on
the tape, the forward is free to use whatever is fastest — sparse factorizations, threads,
precomputed plans — without any differentiability cost.

## Factor once, solve many

In the current-injection formulation, constant-power and ZIP loads enter the nodal balance
as **right-hand-side currents**, never as admittances:

$$Y_{\text{eff}}\,V_{k+1} = I_{\text{slack}} - I_{\text{device}}(V_k).$$

`Y_eff` is therefore constant across all fixed-point iterations *and* across a scenario
batch that varies only injections. pgml factors it once and back-substitutes every
iteration and every scenario as extra right-hand-side columns of that single factorization
— the dense matrix is never tiled across the batch.

The same reasoning extends across *calls*: everything operating-point-independent (the
node-phase index, `Y_eff`, the slack rows and reference, the factorization, the parameter
discovery) can be computed once and reused. `pgml.solver.prepare_power_flow` packages that
state, and `pgml.scenarios.run_scenarios` shares one prepared system across its whole
chunked batch. pandapower documents the same lesson as its `recycle` option (reusing the
Y-bus and internal structures across time-series steps, worth severalfold there);
power-grid-model's batch API likewise reuses the topology and factorization across
scenarios. Reuse never touches gradients: when parameter gradients are requested, the
implicit-function-theorem backward rebuilds its differentiable system from the parameter
leaves regardless of what the forward reused.

## Resolve the operating point once

Profiling a mid-size CPU solve showed that more than half the wall time was not linear
algebra at all — it was *python*: re-resolving nameplate powers, per-phase splits,
connection groups, and configuration defaults on **every** fixed-point iteration. The
evaluation of the device current is therefore split in two
(`pgml.assembly.build_injection_plan` / `injections_from_plan`): the voltage-independent
resolution runs once per solve into stacked tensors, and each iteration evaluates pure
tensor operations (gather, the ZIP law, scatter). Newton's line search and Jacobian, the
convergence diagnostics, and the loadability continuation reuse the same plan.

The general lesson transfers to any tensor-library solver: per-iteration cost is dominated
by object-model traversal long before the matrix work matters, so the resolution work must
be hoisted out of the loop.

A related trap is worth recording because it cost seconds per solve before it was found:
applying one *shared* `Y` to a scenario batch as a broadcast matrix-vector product makes
the backend re-read the $N \times N$ matrix once per scenario (memory-bandwidth-bound).
Folding the batch into the rows of a single matrix–matrix product reads the matrix once;
pgml's residuals do this whenever `Y` is shared across the batch.

## Dense or sparse — measured, per device

A power-grid `Y` has $O(N)$ non-zeros: every branch stamps a fixed-size block, and
distribution feeders are radial or lightly meshed. Sparse direct factorization is the
textbook answer — it is what the dedicated tools use throughout (power-grid-model's C++
core; pandapower via SciPy's SuperLU; OpenDSS via KLU). A dense factorization is $O(N^3)$
but has far smaller constants and maps perfectly onto batched BLAS, so below a few hundred
rows it simply wins.

pgml keeps **both** and selects by measurement (`lu_factor_system(backend=...)`, surfaced
as `solve_power_flow(linear_solver=...)`):

- **CPU, ≥ ~500 node-phase rows** → SciPy SuperLU (COLAMD ordering). At the measured
  crossover (~600 rows on a 12-core desktop CPU) the sparse factorization is already ~4×
  faster and a single back-substitution ~7×; at 4800 rows the factorization is ~6× and a
  single back-substitution ~30–50× faster, and the end-to-end nonlinear solve gains ~4×.
- **CPU, below the crossover, and CUDA always** → batched dense `torch.linalg.lu_factor` /
  `lu_solve`.

CUDA staying dense is a deliberate, benchmarked decision, not an omission. PyTorch has no
batched sparse direct solve, GPUs are built precisely for batched dense factorizations, and
any custom sparse GPU path (cuDSS/cuSOLVER) must first beat that baseline. On an
entry-level workstation GPU (RTX A2000) the question did not even arise at double
precision: **CPU-sparse beat GPU-dense at every size above ~300 rows** (its FP64 units run
at 1/32 of FP32 rate), while single-precision bulk data generation remains the GPU's
territory. `examples/pgml/benchmark_sparse.py` reproduces the comparison on any host; the
selection threshold is calibrated from it. The backends are also never mixed silently — a
tensor is not moved between devices to reach a faster backend.

Differentiability is preserved through the sparse path by the standard adjoint of a linear
solve: for $V = Y^{-1} I$, the backward needs only one solve with the conjugate-transposed
factors ($\lambda = Y^{-H}\,\bar V$, then $\bar Y = -\lambda V^{H}$) — the same
factorization serves forward and backward, with no re-factorization and no unrolling.

### Threaded back-substitution

SuperLU's multiple-right-hand-side solve processes columns sequentially in one thread.
power-grid-model reaches its batch throughput largely by solving scenarios on parallel
threads; pgml applies the same idea one level down: the back-substitution releases the GIL
and is deterministic under concurrent calls against one factorization, so large column
batches are split across up to eight threads (measured ~5× on a 256-scenario
back-substitution at 4800 rows). Chunked solves differ from the single call only at machine
epsilon — the same class of difference as any BLAS blocking change.

## Structural checks before numerics

A (node, phase) row with no galvanic path to a source makes the nodal system singular — or,
with constant-power loads, makes the fixed point diverge. Discovering that as a cryptic
factorization error (or as quiet non-convergence) wastes exactly the time a pre-solve graph
check costs microseconds to prevent. The reference tools all guard this structurally, with
different philosophies:

- **pandapower** (`runpp(check_connectivity=True)`) finds buses unsupplied from any
  `ext_grid` via `scipy.sparse.csgraph` and silently sets them out of service; their
  results are `NaN`.
- **power-grid-model** marks source-less islands as non-energized and reports 0 V with an
  `energized = 0` flag — no error.
- **OpenDSS** warns about isolated buses in its topology processor and otherwise surfaces
  KLU singular-matrix errors naming a node.

pgml's default is deliberately the loudest of the three: every solve entry runs
{func}`pgml.topology.connectivity_report` first, and a disconnected row raises
{class}`pgml.errors.ConnectivityError` naming the islands, the *specific* open switches or
out-of-service branches that would reconnect them, and the concrete fixes. The rationale is
the library's primary workload: unattended, ML-scale data generation. A pipeline that
silently writes `NaN` (or zeros the user did not ask for) into a training dataset is worse
than one that stops with instructions. Two escape hatches cover the other philosophies:
`on_disconnected="zero"` solves the energized sub-grid and reports 0 V on dead rows —
power-grid-model's convention — and `"ignore"` skips the check entirely.

The check works per **(node, phase) row**, not per bus: branches merge the rows they
actually couple, so a three-phase node fed on only one phase is caught — a case a bus-level
check cannot see and that still yields a singular matrix.

Beyond connectivity, non-convergence remains diagnosed *after* the fact — as in every
reference tool, because loadability-type divergence is not decidable structurally. pgml's
post-hoc instrumentation (per-scenario convergence masks, the `likely_cause` heuristic, the
Jacobian criticality analysis, and the `loadability_limit` continuation) is described with
the solver API.

## Switch states as admittance scaling

Varying which switches are open across a scenario batch changes the sparsity of `Y`, so it
cannot be an operating-point override. Two implementations were considered: assemble one
system per configuration (exact, but costs one assembly per configuration and forbids
batching), or assemble the **superset** once and scale each switchable branch's primitive
stamp by a per-scenario state in $[0, 1]$. pgml chose the scaling (`branch_states`), for
three reasons:

1. The stamps are already built vectorised and scatter-added, so a per-branch factor is a
   single broadcast multiply — one assembly yields the whole batch of matrices.
2. A state of exactly 0 reproduces the un-stamped matrix bit-for-bit, and 1 the closed
   branch, so the mask is not an approximation.
3. A *continuous* state is a differentiable parameter: gradients flow through the implicit
   function theorem to the state itself, which turns switching/reconfiguration into a
   quantity that gradient-based search can optimise — something none of the reference tools
   expose.

The state deliberately *overrides* the branch's static `in_service`/`closed` flags, so one
superset grid describes every configuration; genuinely adding a branch absent from the
superset still requires a new assembly. Because an open state can disconnect part of a
grid, the per-scenario topologies are connectivity-checked up front (vectorised over the
batch on a condensed component graph), and a disconnecting scenario is reported by index
rather than surfacing as a singular batch element.

## Ensembles of grids: the disjoint union

Solving many *different* grids at once (a generated population, a multi-feeder study)
needs no dedicated machinery at all, and that is the design: a disjoint union of grids —
ids remapped, nothing else changed — is itself a valid `Grid`. No branch crosses the
members, so its assembled admittance is exactly the block-diagonal
$Y = \mathrm{diag}(Y_1, \dots, Y_G)$, each member keeps its own source and slack rows, and
the sparse backend factors the union in roughly the sum of the members' costs.
{func}`pgml.multigrid.merge_grids` builds the union and translates member-local
identifiers, operating points, and switch states; the solved state splits back into
per-member views.

The alternative — padding every member to the largest size and batching dense — was
rejected for the reason padding always fails on heterogeneous data: a few large members
dominate the padded size for everyone (and a dense factorization's cost *cubes* with that
padded size). The union approach is the same trick PyTorch-Geometric's `Batch` uses for
variable-size graphs, applied to the physics: concatenate, remember the offsets, never pad.
Because the union shares the members' parameter objects, gradients from an ensemble solve
land on the original grids' own tensors.

## What deliberately stays sequential

Not everything batches profitably, and one negative result is recorded so it is not
re-attempted: Newton's forward for a *batched operating point* solves scenario by scenario.
A batch-native alternative (block-diagonal Jacobian assembled column-by-column, per-element
line search) was implemented and measured ~4× *slower* at 64 scenarios on a mid-size
feeder: the full batched Jacobian needs $O(B^2 (2N)^2)$ memory, and building its diagonal
blocks column-wise costs $2N$ full residual sweeps per Newton step, which loses to one
vectorised per-scenario Jacobian evaluation. The division of labour stands: the
current-injection fixed point is the bulk-batch solver; Newton is the hard-grid,
near-the-loadability-nose solver.

## Precision policy

`complex128` is the accuracy and gradient-checking dtype; `complex64` is the throughput
dtype for bulk data generation (and the reason the GPU is attractive there at all, given
workstation-class FP64 rates). One caveat is important enough to state as policy: a
physical feeder's `Y` in SI units mixes admittances across many decades (line shunts around
$10^{-8}\,\mathrm{S}$ against source stamps around $10^{5}\,\mathrm{S}$), producing
condition numbers of $10^6$–$10^9$. At single precision ($\varepsilon \approx 1.2\times
10^{-7}$) the error bound $\kappa\,\varepsilon$ can reach order one — and this is observed,
not theoretical: on such a grid, a `complex64` *representation* of the operator already
perturbs the solved voltages at the percent level, and different backends (CPU vs CUDA
factorizations) disagree visibly with each other. The policy that follows:

- solve in `complex128` wherever the result feeds anything precision-sensitive, and cast
  the *outputs* down — storage in `complex64` then costs only the final rounding;
- generate datasets at `complex128` and store them as `complex64`, rather than generating
  at `complex64` outright, whenever the grid's conditioning is unknown.

The learning package's physics decoder follows the same rule internally: its cached network
operators are assembled and solved in `complex128` regardless of the training dtype.
