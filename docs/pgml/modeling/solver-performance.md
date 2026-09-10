# Solver architecture, performance and structural checks

How the nonlinear solve is organised for speed and robustness, and what was measured to get
there. Where a choice mirrors or departs from pandapower, power-grid-model or OpenDSS, the
comparison is stated. The per-tool briefs are in the
[reference-library notes](references/index.md).

One rule shapes the rest. The iterative forward solve is detached from autograd, and gradients
come from the implicit function theorem at the converged solution, which costs one adjoint
linear solve and one residual differentiation. No iteration is ever on the tape, so the forward
path is free to use sparse factorizations, threads and precomputed plans at no cost in
differentiability.

## Factor once, solve many

In the current-injection formulation a constant-power or ZIP load enters the nodal balance as a
right-hand-side current rather than an admittance.

$$Y_{\text{eff}}\,V_{k+1} = I_{\text{slack}} - I_{\text{device}}(V_k)$$

`Y_eff` is therefore constant across the fixed-point iterations and across a scenario batch
that varies only injections. It is factored once, and every iteration and every scenario
becomes an extra right-hand-side column of that one factorization. The dense matrix is never
tiled across the batch.

The same reasoning extends across calls. The node-phase index, `Y_eff`, the slack rows and
reference, the factorization and the parameter discovery are all operating-point independent.
`pgml.solver.prepare_power_flow` packages that state and `pgml.scenarios.run_scenarios` shares
one prepared system across its whole chunked batch. pandapower documents the same lesson as its
`recycle` option, and power-grid-model's batch API reuses topology and factorization across
scenarios. Reuse never touches gradients, because the backward pass rebuilds its
differentiable system from the parameter leaves whatever the forward reused.

## Resolve the operating point once

Profiling a mid-size CPU solve put more than half the wall time outside the linear algebra. It
went into re-resolving nameplate powers, per-phase splits, connection groups and configuration
defaults on every fixed-point iteration. Evaluating the device current is therefore split in
two. `pgml.assembly.build_injection_plan` does the voltage-independent resolution once per
solve into stacked tensors, and `injections_from_plan` evaluates pure tensor operations per
iteration. Newton's line search and Jacobian, the convergence diagnostics and the loadability
continuation reuse the same plan.

The lesson transfers to any tensor-library solver. Per-iteration cost is dominated by
object-model traversal long before the matrix work matters, so resolution has to be hoisted out
of the loop.

A related trap cost seconds per solve before it was found. Applying one shared `Y` to a
scenario batch as a broadcast matrix-vector product makes the backend re-read the whole
$N \times N$ matrix once per scenario, which is memory-bandwidth bound. Folding the batch into
the rows of a single matrix-matrix product reads the matrix once, which is what the residuals
do whenever `Y` is shared across the batch.

## Dense or sparse, measured per device

A power-grid admittance matrix has $O(N)$ non-zeros, since every branch stamps a fixed-size
block and distribution feeders are radial or lightly meshed. Sparse direct factorization is the
textbook answer and the one the dedicated tools use, power-grid-model in its C++ core,
pandapower through SuperLU, OpenDSS through KLU. A dense factorization is $O(N^3)$ but has far
smaller constants and maps onto batched BLAS, so below a few hundred rows it wins outright.

Both are kept, and `solve_power_flow(linear_solver=...)` selects between them.

- CPU at roughly 500 node-phase rows and above uses SuperLU with COLAMD ordering. At the
  measured crossover near 600 rows on a 12-core desktop CPU the sparse factorization is already
  about 4 times faster and a single back-substitution about 7 times. At 4800 rows the
  factorization is about 6 times faster, a single back-substitution 30 to 50 times, and the
  end-to-end nonlinear solve about 4 times.
- CPU below the crossover, and CUDA always, use the batched dense `torch.linalg.lu_factor` and
  `lu_solve`.

A third backend, `"block"`, is available for a system that is structurally block diagonal. See
the ensemble section below.

CUDA staying dense is a benchmarked decision. PyTorch has no batched sparse direct solve, GPUs
are built for batched dense factorizations, and a custom sparse GPU path would have to beat
that baseline first. On an entry-level workstation GPU the question did not arise at double
precision, where CPU-sparse beat GPU-dense at every size above about 300 rows, because that
card's FP64 units run at a thirty-second of its FP32 rate. Single-precision bulk data
generation remains the GPU's territory. The shipped `benchmark_sparse.py` example reproduces
the comparison on any host, and the selection threshold is calibrated from it. Backends are
never mixed silently, and no tensor is moved between devices to reach a faster backend.

Differentiability survives the sparse path through the standard adjoint of a linear solve. For
$V = Y^{-1} I$ the backward needs one solve with the conjugate-transposed factors,
$\lambda = Y^{-H}\bar V$ and then $\bar Y = -\lambda V^{H}$, so one factorization serves
forward and backward with no refactorization and no unrolling.

SuperLU's multiple-right-hand-side solve walks its columns sequentially in one thread.
power-grid-model reaches its batch throughput largely by solving scenarios on parallel threads,
and pgml applies the same idea one level down. The back-substitution releases the GIL and is
deterministic under concurrent calls against one factorization, so a large column batch is
split across up to eight threads, measured at about 5 times on a 256-scenario
back-substitution at 4800 rows. A chunked solve differs from a single call only at machine
epsilon, the same class of difference as any change in BLAS blocking.

## Structural checks before numerics

A node-phase row with no galvanic path to a source makes the nodal system singular, and with
constant-power loads it makes the fixed point diverge. A pre-solve graph check costs
microseconds and prevents a cryptic factorization error or quiet non-convergence. The reference
tools all guard this, with different philosophies.

| Tool | Behaviour |
|---|---|
| pandapower | Finds buses unsupplied from any external grid and sets them out of service. Their results are `NaN` |
| power-grid-model | Marks source-less islands as non-energized, reports 0 V with an `energized` flag, no error |
| OpenDSS | Warns about isolated buses in its topology processor, otherwise surfaces a singular-matrix error naming a node |

pgml's default is the loudest of the three. Every solve entry runs
{func}`pgml.topology.connectivity_report` first, and a disconnected row raises
{class}`pgml.errors.ConnectivityError` naming the islands, the specific open switches or
out-of-service branches that would reconnect them, and the concrete fixes. The reason is the
primary workload, which is unattended data generation. A pipeline that silently writes `NaN`, or
zeros nobody asked for, into a training dataset is worse than one that stops with instructions.
Two escape hatches cover the other philosophies. `on_disconnected="zero"` solves the energized
sub-grid and reports 0 V on dead rows, which is power-grid-model's convention, and `"ignore"`
skips the check.

The check works per node-phase row rather than per bus. Branches merge the rows they actually
couple, so a three-phase node fed on one phase only is caught, which a bus-level check cannot
see and which still yields a singular matrix.

Non-convergence stays a post-hoc diagnosis, as in every reference tool, because
loadability-type divergence is not decidable structurally. The per-scenario convergence masks,
the likely-cause heuristic, the Jacobian criticality analysis and the `loadability_limit`
continuation are documented with the solver API.

## Switch states as admittance scaling

Varying which switches are open across a batch changes the sparsity of `Y`, so it cannot be an
operating-point override. Two implementations were considered. One assembles a system per
configuration, which is exact but costs an assembly each and forbids batching. The other
assembles the superset once and scales each switchable branch's primitive stamp by a
per-scenario state in `[0, 1]`. pgml uses the scaling, `branch_states`, for three reasons.

1. The stamps are already built vectorised and scatter-added, so a per-branch factor is one
   broadcast multiply and one assembly yields the whole batch of matrices.
2. A state of exactly 0 reproduces the unstamped matrix bit for bit and 1 the closed branch, so
   the mask is not an approximation.
3. A continuous state is a differentiable parameter. Gradients reach the state itself, which
   turns reconfiguration into something gradient-based search can optimise. None of the
   reference tools expose that.

The state overrides the branch's static in-service and closed flags, so one superset grid
describes every configuration. Adding a branch that is absent from the superset still needs a
new assembly. Because an open state can disconnect part of a grid, the per-scenario topologies
are connectivity-checked up front, vectorised over the batch on a condensed component graph,
and a disconnecting scenario is reported by index instead of surfacing as a singular batch
element.

## Switch-state sweeps as a low-rank update

Admittance scaling turns a sweep into a single assembly, but a batched `branch_states` still
factors every state independently, which is `O(S·N³)` work and an `[S, N, N]` matrix. A
switched branch only ever touches its own terminal rows, since a branch enters `Y` exclusively
as `Y[rows, rows] += block`. Scaling it by `s` therefore changes `Y` by `(s − 1)` times a term
of rank at most `2P`, with `P` the branch's phase count. That is the shape the
Sherman-Morrison-Woodbury identity exploits. Factor the network once and reach every state
through an update of that one factorization, at `O(N²k + k³)` per state with `k` the sum of
`2P` over the switched branches, instead of a fresh assembly and factorization per state.

`solve_power_flow(..., branch_states_method="woodbury")` implements it. The update never
inverts the per-state correction directly. The arrangement
$A^{-1} - A^{-1}U(I + CV^HA^{-1}U)^{-1}CV^HA^{-1}$ stays invertible even at `s = 0`, where the
naive stamp is singular, and a branch sitting at its base state contributes a zero correction,
which reproduces the base solve bit for bit.

The base state matters. The correction reads values off the base solution and amplifies them by
`‖(I + CZ)⁻¹CZ‖`. Adding admittance to the base keeps that factor of order one. Removing a
near-ideal closed switch from the base pushes it toward `|y · z_thevenin|`, because the quantity
the downdate multiplies is that branch's own voltage drop, the difference of two nearly equal
node voltages, which floating point resolves poorly. A 1e-4 Ω switch, the default contact
resistance a converter assigns, already costs about four digits this way, which is enough to
stall the fixed point. The sweep's base therefore omits every switched branch it can, opened one
at a time while the grid stays connected without it, so an update only ever adds admittance.

Measured with the shipped `benchmark_woodbury.py` example on an i7-12700 CPU with the automatic
backend, which resolves to SuperLU, over 8 switch-configuration states: for 1 to 4 switched
three-phase branches, `k` from 6 to 24, the low-rank path is 3.2 to 3.5 times faster at 600
rows, 4.3 to 6.8 times at 1200, about 5.8 times at 2100 and about 5.4 times at 3000 rows. It is
still 1.7 to 4.4 times faster at `k = 96`. The crossover sits around `k ≈ N/3`, measured at 600
rows as 1.1 times at `k = 192` and 0.2 times at `k = 384`, beyond which the correction's own
`O(k³)` factoring dominates and assembling per state wins. Voltages agree with the assembling
path to about 1e-12 relative. Only the forward solve changes, because the backward pass always
rebuilds the per-state admittance differentiably from the parameter and state leaves, so
gradients including those with respect to the switch states are identical between the two
methods.

The method is an explicit opt-in with no automatic heuristic, because the win depends on `k/N`,
and only the caller knows how many branches its sweep switches.

The rank `k` against that crossover is the single criterion for whether an update pays.

- A switch-state sweep on one grid keeps `k` far below `N` at any grid size, since each
  switched `P`-phase branch is a term of rank at most `2P`.
- An out-of-service line is an open switch for this purpose, and `s = 0` is exact, so one fixed
  candidate grid whose states open and close member lines sweeps topologies over a shared node
  set. The `k` budget still applies. Neighbouring topologies, a few reconfiguration moves around
  a base, are the winning regime, while arbitrary pairs of radial trees drawn from a dense
  candidate mesh differ in a large fraction of their edges, which pushes `k` toward `N`. Every
  state must also pass its own connectivity check.
- Different operating points on one grid need no update at all. Constant-power and ZIP loads
  live on the right-hand side, so the factor-once path already solves every scenario as one
  back-substitution.
- The same node set with moved or resized edges is an update of rank at most `2P` per changed
  branch against the parent factorization. {func}`pgml.assembly.branch_stamp_blocks` is the
  seam for that.
- Different node counts cannot be reached. The base and target matrices have different
  dimensions, which no low-rank identity bridges. Growing a system is a bordered factorization
  update, which is a different technique and is not implemented.

The identity is due to Sherman and Morrison (1950, *Ann. Math. Stat.* 21:124) for rank one and
Woodbury (1950, Statistical Research Group Memorandum Report 42, Princeton) for the block form.
Hager (1989, *SIAM Review* 31:221) surveys the numerics, including the conditioning of
downdates that motivates the base-state rule above.

## Ensembles of grids, the disjoint union

Solving many different grids at once needs no dedicated machinery, which is the design. A
disjoint union of grids, ids remapped and nothing else changed, is itself a valid `Grid`. No
branch crosses the members, so the assembled admittance is exactly block diagonal, each member
keeps its own source and slack rows, and the sparse backend factors the union in roughly the sum
of the members' costs. {func}`pgml.multigrid.merge_grids` builds the union and translates
member-local identifiers, operating points and switch states, and the solved state splits back
into per-member views.

Padding every member to the largest size and batching dense was rejected for the reason padding
always fails on heterogeneous data. A few large members dominate the padded size for everyone,
and a dense factorization's cost cubes with that size. Concatenating and remembering the
offsets is the same trick PyTorch-Geometric's `Batch` uses for variable-size graphs, applied to
the physics. Because the union shares the members' parameter objects, gradients from an ensemble
solve land on the original grids' own tensors.

On CUDA that leaves a gap, because the only union factorization there is dense. A dense LU of
the union costs $O((\sum_k n_k)^3)$ where the members cost $O(\sum_k n_k^3)$, a factor $G^2$ for
$G$ equal-sized members. The block-diagonal backend closes it by factoring the diagonal blocks
instead of the union.

```python
res = solve_power_flow(
    merged.grid, linear_solver="block", block_rows=merged.block_rows()
)
```

{meth}`pgml.multigrid.MergedGrid.block_rows` hands over the row partition. Blocks of equal size
are stacked and factored by a single batched `torch.linalg.lu_factor`, so the number of LU calls
is the number of distinct member sizes rather than the number of members, and the stored factor
shrinks from $O((\sum_k n_k)^2)$ to $O(\sum_k n_k^2)$. The right-hand side is gathered per
block, back-substituted with the whole scenario batch folded into the multiple-RHS axis, and
scattered back. Ideal slack works as it does elsewhere, where a block's free rows are its own
rows minus its slack rows. Everything is plain torch, so gradients and the adjoint solve come
from the same factors.

This backend is an explicit opt-in that the automatic selection never chooses. The partition is
taken on trust, since admittance outside the listed blocks would be ignored, and on CPU the
sparse union backend already exploits the block structure plus the sparsity inside each block.

## What stays sequential

Newton's forward pass for a batched operating point solves scenario by scenario. A batch-native
alternative, with a block-diagonal Jacobian assembled column by column and a per-element line
search, was implemented and measured about 4 times slower at 64 scenarios on a mid-size feeder.
The full batched Jacobian needs $O(B^2 (2N)^2)$ memory, and building its diagonal blocks column
wise costs $2N$ full residual sweeps per Newton step, which loses to one vectorised
per-scenario Jacobian evaluation. The division of labour stands. The current-injection fixed
point is the bulk-batch solver and Newton is the solver for hard grids near the loadability
nose.

## Precision policy

`complex128` is the accuracy and gradient-checking dtype. `complex64` is the throughput dtype
for bulk data generation, and the reason a GPU is attractive there given workstation-class FP64
rates.

One caveat is policy. A physical feeder's admittance matrix in SI units mixes values across
many decades, line shunts around $10^{-8}\,\mathrm{S}$ against source stamps around
$10^{5}\,\mathrm{S}$, which produces condition numbers of $10^6$ to $10^9$. At single precision
the error bound $\kappa\varepsilon$ can reach order one. This is observed rather than
theoretical. On such a grid a `complex64` representation of the operator already perturbs the
solved voltages at the percent level, and CPU and CUDA factorizations then disagree visibly
with each other. Two rules follow.

- Solve in `complex128` wherever the result feeds anything precision-sensitive and cast the
  outputs down, so storage in `complex64` costs only the final rounding.
- Generate datasets at `complex128` and store them as `complex64`, rather than generating at
  `complex64`, whenever the grid's conditioning is unknown.
