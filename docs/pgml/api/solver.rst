pgml.solver
===========

Complex batched differentiable solve of ``Y(f) V(f) = I(f)``.

The solver package provides the following entry points:

- :func:`~pgml.solver.solve_harmonic` — batched complex linear solve for a
  single assembled Y-bus.  Supports Norton mode (default) and ideal-slack
  Schur-partition mode, both fully differentiable.
- :func:`~pgml.solver.solve_power_flow` — nonlinear const-P / ZIP fundamental
  power flow using a current-injection fixed-point forward pass and an
  implicit-function-theorem (IFT) backward pass (real-coordinate adjoint).
- :func:`~pgml.solver.solve_harmonic_flow` — full harmonic flow orchestration:
  runs the fundamental power flow to convergence (``method=`` selects the
  fundamental solver: ``"current_injection"`` or ``"newton"`` for stiff
  inverter control loops), then solves each harmonic in one batched pass.
- :func:`~pgml.solver.assemble_harmonic_system` — assembles both the harmonic
  admittance matrix ``Y(h)`` and the realized nodal injection vector ``I(h)``
  for orders ``h > 1`` (used by the physics-consistency layer).
- :func:`~pgml.solver.assemble_harmonic_ybus` — assembles the harmonic
  admittance matrix ``Y(h)`` for orders ``h > 1`` WITHOUT the injection RHS.
  This is the entry point for a physics-informed decoder that predicts the nodal
  injection current ``I_pred`` and reconstructs the voltage state via
  ``V(h) = solve(Y(h), I_pred)``, using only the differentiable grid description and no
  ground-truth injection.  ``Y`` is
  grid-constant (assembled once, reused across the batch) and differentiable
  w.r.t. the network parameters, so the same call powers a learned
  grid-parameter calibration.
- :func:`~pgml.solver.check_connectivity` — the pre-solve structural gate:
  raises :class:`~pgml.errors.ConnectivityError` when part of the grid has no
  galvanic path to an in-service source, before any factorization is attempted.
- :func:`~pgml.solver.check_branch_impedances` — the second pre-solve gate: reports
  an in-service branch whose series impedance is exactly zero.  Under the default
  ``branch.zero_impedance: fuse`` those branches are collapsed instead
  (:func:`pgml.assembly.fusion_map`); under ``error`` the solve is refused by name.
- :func:`~pgml.solver.harmonic_injections` — the realized nodal injection vector
  ``I(h)`` on its own, given a converged fundamental voltage.  The counterpart to
  :func:`~pgml.solver.assemble_harmonic_ybus` for a caller that assembles the matrix
  and the right-hand side separately.
- :func:`~pgml.solver.loadability_limit` — walks the loading parameter λ from a
  feasible base to the largest value the Newton corrector still solves, and reports
  the margin, the critical bus and the limiting load.
- :func:`~pgml.solver.prepare_power_flow` — assembles and factors the
  operating-point-independent part of a nonlinear solve once, returning a
  :class:`~pgml.solver.PowerFlowSystem` that repeated
  :func:`~pgml.solver.solve_power_flow` calls on the same grid can reuse.
  Reuse validates stored network values, voltage bases, modeling defaults, and
  detached snapshots of ``param_overrides`` and ``branch_states``. Changed values
  (including in-place tensor edits) raise ``InputError``; re-prepare the system.
  Equal-valued replacement tensors can reuse factors and receive their own gradients.
- :mod:`pgml.solver.equilibration` — the diagonal scaling used by the power-flow and
  harmonic factorization paths, and the factored handle
  :class:`~pgml.solver.equilibration.EquilibratedLU` for a caller that wants it
  directly.

.. rubric:: Differentiability

When assembling harmonics separately after a fundamental solve, use
``operating_point=pf.resolved_operating_point(original_operating_point)``. This
replaces configured PV reactive powers with the solved output, including binding
limits and unequal phase allocation. ``solve_harmonic_flow`` and ``simulate``
perform this handoff automatically. The input dictionary is not mutated and the
resolved powers remain differentiable.

Repeated harmonic studies
-------------------------

Use explicit preparation for repeated calls on one network::

    from pgml.solver import HarmonicFlowSystem, solve_harmonic_flow

    system = HarmonicFlowSystem()
    first = solve_harmonic_flow(grid, [1, 5, 7], system=system)
    next_state = solve_harmonic_flow(
        grid, [1, 5, 7], operating_point=new_op, system=system,
    )
    print(system.stats)

Alternatively, ``prepare_harmonic_flow(grid, orders, **solve_kwargs)`` warms the
same preparation with a complete initial solve and returns only the preparation.
``simulate(..., harmonic_system=system)`` accepts it too. Without a preparation,
the solver retains its uncached behavior. No process-global harmonic cache exists.

Unlike ``PowerFlowSystem`` (which rejects changes), harmonic preparation
automatically rebuilds affected entries. Three separate checks govern reuse:

* Fundamental preparation: current grid data, defaults, parameter overrides,
  branch states and numerical settings must match their value snapshots. A new
  fundamental operating point is always solved. The explicit ``ignore``
  connectivity policy and zeroed-subgrid recursion bypass fundamental preparation
  because its constructor requires connectivity; harmonic reuse remains available.
* Network assembly: current grid data (including DER impedances and connection
  flags), defaults, frequencies, overrides, branch states, dtype/device and fused
  row layout must match. This is deliberately more conservative than the
  fundamental-only network fingerprint: even irrelevant stored device changes
  can cause a rebuild. Load shunts and node-source shunts are evaluated anew.
* Numerical harmonic factors: the **evaluated matrix**, backend, block partition,
  precision, equilibration and defaults must match exactly. A new RHS alone does
  not require new factors. An operating-point shunt change does; a Woodbury
  device-shunt update instead rebuilds its small correction and reuses a matching
  base. Sparse numerical factorization still includes symbolic analysis; there
  is no separate symbolic-analysis cache.

Snapshots detect replacement, addition, removal and in-place edits. Gradients
through a harmonic matrix bypass factor caching and network-parameter gradients
bypass assembly caching, so an equal-valued replacement tensor uses the current
autograd graph. RHS-only gradients can reuse constant factors. Cached numerical
entries own their storage; they do not retain the harmonic autograd graph.

Preparation retains at most one entry at each level, but a scenario matrix and
its factors can be large. ``HarmonicFlowSystem(cache_batched_factors=False)``
avoids retaining scenario-dependent factors; chunked ``run_scenarios`` uses this
mode automatically. Use ``system.clear()`` to release entries. Preparation is
not thread-safe and exact tensor comparisons can synchronize CUDA. Benchmarks
should distinguish uncached calls, the first prepared call, repeated identical
batches and genuinely changed operating points. Preparation is not guaranteed
to improve small-system latency. ``load_shunt_basis="nameplate"`` is a physical
model choice, not a cache optimization interchangeable with operating-point shunts.

For threaded execution, allocate one ``HarmonicFlowSystem`` per worker and reuse
it sequentially within that worker. Shared-instance lookups, rebuilds, eviction,
statistics and backend-factor use are not synchronized. If sharing is necessary,
hold an external lock for the entire solve and for ``clear()``. Per-thread
preparations do not make concurrent mutation of a shared grid, input tensors or
defaults safe; keep those inputs read-only or give each worker its own copies.

``solve_harmonic`` differentiates cleanly via the linear solve adjoint.
``solve_power_flow`` uses an explicit IFT adjoint (the only sanctioned
``.detach()`` in the codebase) so gradients flow through the converged solution.

Convergence criteria and working precision
-------------------------------------------

Both nonlinear entry points judge convergence in PER UNIT on two criteria, and both must
hold:

- ``tol`` — the largest nodal apparent-power mismatch in per unit of ``s_base_va``
  (defaults ``solver.convergence.mismatch_pu`` = 1e-8 pu on a 1e6 VA base).  This is the
  quantity pandapower's ``tolerance_mva`` and power-grid-model's ``error_tolerance``
  report, so iteration counts are comparable across the three tools.
- ``tol_update_pu`` — the largest per-row voltage update in per unit of that node's
  line-to-neutral rated voltage (default ``solver.convergence.update_pu`` = 1e-8 pu).
  Per-row normalisation makes one tolerance mean the same thing on a 400 V node and a
  20 kV node, and makes it independent of the row count, so a multi-voltage grid or a
  merged ensemble is judged exactly like a single feeder.

``tol`` is a power tolerance, not a voltage tolerance.  On the default base it accepts 0.01 VA
of mismatch per row, which leaves a voltage error of about ``tol * s_base_va / S_k`` per unit
at a node of short-circuit power ``S_k``.  The update is not the remaining error either.  The
fixed point's error is ``ρ / (1 - ρ)`` times its last update, with ``ρ`` the contraction
factor, roughly 0.1 to 0.5 on a distribution feeder, and Newton's error is far below its last
update.  With the defaults the voltages are good to about 1e-8 pu, a few microvolts at 230 V.
For a tighter answer, such as a comparison at 1e-12 pu, lower both ``tol`` and
``tol_update_pu`` at ``complex128``, or lower ``s_base_va`` for a small grid.

Each is capped by what the working precision can resolve.  A tighter request logs a warning
naming the floor, and the floor governs.
:class:`~pgml.solver.ConvergenceDiagnostics` reports both achieved values.

``precision`` selects the working precision of the factorization independently of ``dtype``:

- ``"full"`` (default) factors and iterates at ``dtype``.
- ``"mixed"`` factors at ``complex64`` while the iteration, the residual and the
  convergence test stay at ``complex128``, refining against the double-precision residual
  (``solver.precision.refine_steps``).  It reaches the ``complex128`` solution to better
  than 1e-12 pu on every grid measured, at single-precision factorization cost.  Requires
  ``dtype=torch.complex128``.

A plain ``complex64`` solve logs a one-time warning when the condition estimate of the
matrix it factored exceeds ``solver.precision.complex64_cond_warn``.  A
:class:`~pgml.solver.PowerFlowSystem` records its ``precision``, and a solve that reuses it
must request the same one.  See :doc:`/pgml/modeling/solver-performance` for the measured
accuracy and cost.

Equilibration (``equilibrate``)
---------------------------------

An SI-unit nodal matrix is badly SCALED: a stiff source row carries an admittance near
1e5 S where a low-voltage cable row carries 1e-2 S, and harmonic-frequency series and
shunt terms can widen that spread.  By default, the power-flow and harmonic factorization
paths use the equilibrated matrix ``D_r A D_c``. The scaling is undone on the solution:
the right-hand side you pass and the voltages you read are SI, the residuals and tolerances
are unchanged, and the gradients are unchanged. ``equilibrate="off"`` keeps these paths
unscaled. :func:`~pgml.solver.solve_anchored` and
:class:`~pgml.solver.AnchoredSystem` do not yet apply this equilibration.

``equilibrate`` is accepted by :func:`~pgml.solver.solve_harmonic`,
:func:`~pgml.solver.solve_power_flow`, :func:`~pgml.solver.prepare_power_flow`,
:func:`~pgml.solver.solve_harmonic_flow`, :func:`~pgml.solver.loadability_limit` and
:func:`pgml.simulation.simulate`:

- ``None`` — the documented default ``solver.equilibration.mode``.
- ``"symmetric"`` (shipped) — the van der Sluis scaling ``d_i = |A_ii|**-0.5``, with the
  factors rounded to powers of two (``solver.equilibration.power_of_two``) so the scaled
  matrix is exact in binary floating point.
- ``"off"`` — factor the matrix as assembled.

:mod:`pgml.solver.equilibration`, documented at the bottom of this page, exposes the pieces
for a caller that wants the scaling directly, including
:class:`~pgml.solver.equilibration.EquilibratedLU`, which answers right-hand sides at the
input matrix's own dtype and handles the adjoint solve.  Every factored handle records which
mode it was factored under, and :class:`~pgml.solver.PowerFlowSystem` rejects a reuse under a
different one.

Bus fusion: ideal branches in a solved result
-----------------------------------------------

A branch whose series impedance is exactly zero has no primitive admittance, because the
nodal formulation inverts it.  Under the default ``branch.zero_impedance: fuse`` the solve
collapses such a branch's terminal node-phase rows into ONE row of the system it factors,
solves the reduced system, and reports the result on the grid's OWN row layout — so
``PowerFlowResult.v`` stays ``[*batch, N]`` and ``HarmonicFlowResult.v`` stays
``[*batch, H, N]``, with every node of a fused group carrying the group's voltage.

:class:`~pgml.solver.PowerFlowResult`, :class:`~pgml.solver.HarmonicFlowResult`,
:class:`~pgml.solver.PowerFlowSystem` and :class:`pgml.simulation.SolvedState` carry the
:class:`~pgml.assembly.FusionMap` in a ``fusion`` field (``None`` on a grid without such a
branch).  :func:`~pgml.assembly.branch_currents` takes it and recovers the current through a
fused branch from Kirchhoff's law at the fused node.  See the "Bus fusion" section of
:doc:`assembly` for the map itself and its rules.

``branch_states`` and fusion are mutually exclusive on the same branch: a swept branch is
reached by scaling its stamped admittance, and a fused branch has none.  A zero-impedance
branch listed in ``branch_states`` is refused by name, with both ways out.

Voltage-regulating generators (PV terminals)
----------------------------------------------

A :class:`~pgml.schemas.grid_schema.Generator` carrying a
:class:`~pgml.schemas.grid_schema.VoltageRegulation` block is a PV terminal: its active
power is given, its terminal voltage magnitude is held at ``v_set_pu``, and its reactive
power is whatever that takes within ``q_min_var`` / ``q_max_var``.  The solver substitutes
the imaginary half of that row's residual with the setpoint condition and recovers the
reactive power from the converged solution, so the state stays ``[Re V; Im V]`` and the
``[2N, 2N]`` IFT Jacobian, the adjoint and the batching are unchanged.

Such a grid is always solved by Newton — a current-injection update has no injection to
form at a regulated row — and ``method="current_injection"`` logs the switch.
``enforce_q_limits`` (default from ``appliance.generator.enforce_q_limits``) decides whether
the reactive limits bound the output; when they do, a unit outside its band is re-solved as
a PQ injection pinned at the limit and released when its voltage crosses the setpoint from
the other side.  :class:`~pgml.solver.VoltageRegulationResult`, on
``PowerFlowResult.regulation``, reports per unit the resolved reactive power (differentiable),
the active set and which terminals are pinned, and per scenario whether the switching settled
within its round cap.  A scenario that did not settle is reported as not converged.  The model, its scope and the reference comparisons are
in :doc:`/pgml/modeling/der-pv-storage`.

Loadability (``loadability_limit``)
-------------------------------------

:func:`~pgml.solver.loadability_limit` scales the injections by λ from a feasible base,
Newton-corrects at each step and bisects onto the first λ the corrector cannot solve.  The
reported ``breaking_lambda`` is therefore the largest λ at which the corrector still
converges, which is a LOWER BOUND on the true P-V nose: a plain corrector fails before the
singularity because the Jacobian becomes ill-conditioned first (measured about 4 % below the
closed-form nose of a two-bus feeder).  It is a step-and-bisect on feasibility, not an
arc-length predictor-corrector continuation, so it cannot turn the nose, and the Jacobian
figures it reports describe the last converged point.

``ramp`` chooses what λ multiplies.  The default is ``solver.loadability.ramp`` = ``"load"``:
the loads scale and generation stays at nameplate, which is the textbook
continuation-power-flow ramp.  ``"all"`` scales every injecting device together.  On a feeder
with generation the two differ materially (1.625 against 2.0 on a two-bus example with
generation at 0.3 of the nose power), so :class:`~pgml.solver.LoadabilityResult` records
which ramp it measured.

Harmonic solver options
-------------------------

:func:`~pgml.solver.solve_harmonic_flow` takes the same factorization options as the
fundamental solve.  ``linear_solver`` and ``block_rows`` select the backend of the
fundamental system AND of every harmonic order, since each order is one direct solve of a
system with the same sparsity and the same row partition.  ``equilibrate`` sets the
equilibration of all of them, ``criticality`` the Jacobian diagnostic of the fundamental
solve, and ``branch_states_method`` how a switch-state sweep reaches each state at the
fundamental — the harmonic orders always assemble their own per-state admittance, because a
low-rank update is built from one frequency's stamps.  ``load_shunt`` selects the harmonic
device Norton shunt.  The pre-solve connectivity and zero-impedance checks run once for the
whole study.

A per-scenario operating point ``[B]`` may be combined with a deeper harmonic-injection
sequence ``[B, T]``.  The result is ``[B, T, H, N]``: the fundamental voltage of each
scenario is broadcast across its ``T`` injection steps.  When the device shunt makes the
harmonic matrix scenario-dependent, its layout is ``[B, 1, H, N, N]``.  The singleton
step axis records that each ``(B, H)`` factorization serves all ``T`` right-hand sides, so
the dense, sparse and block backends do not tile or refactor the matrix per step.

Symmetry kwarg
--------------

Both :func:`~pgml.solver.solve_power_flow` and
:func:`~pgml.solver.solve_harmonic_flow` accept a ``symmetry`` keyword argument
(``None`` / ``"auto"`` / ``"symmetric"`` / ``"asymmetric"``).  It is resolved
**once** at the start of each top-level call:

- ``solve_power_flow`` resolves and logs the modeling summary once, then
  passes the resolved string into every :func:`~pgml.assembly.device_current_injections`
  call of the fixed-point iteration (no per-iteration logging).
- ``solve_harmonic_flow`` resolves once, threads the result into the fundamental
  ``solve_power_flow`` call (which logs once), and also into the harmonic
  injection power resolution.

``None`` reads the config key ``calculation.symmetry`` (default ``"auto"``).
For the semantics of each mode, see the "Symmetric vs asymmetric calculation"
section on the :doc:`assembly` page.

Connectivity checking
---------------------

Every solve entry point runs a pre-solve structural check by default: a
``(node, phase)`` row with no galvanic path to an in-service
:class:`~pgml.schemas.grid_schema.Source` — an open switch or an
out-of-service line/transformer on the only path, or no source at all — makes
the nodal system singular there, so the solve is refused up front with the
concrete disconnected nodes and fixes named, instead of surfacing later as an
opaque singular-matrix or non-convergence failure. The ``on_disconnected``
keyword of :func:`~pgml.solver.solve_power_flow` and
:func:`~pgml.solver.solve_harmonic_flow` controls the response:

- ``"raise"`` (default) — raise :class:`~pgml.errors.ConnectivityError`, whose
  message names the disconnected nodes, the separating branches, and how to
  reconnect them.
- ``"zero"`` — solve the energized sub-grid
  (:func:`pgml.topology.energized_subgrid`) and report 0 V on the disconnected
  rows; the result keeps the full grid's row layout. A node with only some
  phases unenergized cannot be split this way and still raises
  :class:`~pgml.errors.ConnectivityError`.
- ``"ignore"`` — skip the check (the historical behavior: a disconnected area
  surfaces as a singular factorization or a non-convergence).

:func:`~pgml.solver.check_connectivity` exposes the same gate directly (raises
:class:`~pgml.errors.ConnectivityError` or returns ``None``), and
:func:`pgml.topology.connectivity_report` returns the full structured report
(islands, reconnect hints) for callers that want to inspect or display it
before deciding what to do — see :doc:`topology`.

With ``branch_states`` (below), ``"raise"`` checks every scenario's effective
topology in one vectorized pass; ``"zero"`` is unsupported there, since a
per-scenario topology has no single energized sub-grid.

Topology / switch-state batching (``branch_states``)
------------------------------------------------------

``branch_states={branch_id: state}`` lets a solve treat a branch's in-service
status as a continuous, batchable, differentiable quantity rather than a fixed
schema flag. Every listed branch is always stamped and its primitive
admittance block scaled by ``state`` — a python float, a 0-d tensor, or a
``[*batch]`` scenario tensor in ``[0, 1]``: ``0`` opens the branch, ``1`` puts
it fully in service, and intermediate values scale the admittance
continuously. The state OVERRIDES the branch's static ``in_service`` /
``closed`` flags.

Two things follow from this:

- **Switch-state batching.** A ``[*batch]`` state solves every switch
  configuration of interest in ONE batched call — one assembly, one
  per-scenario ``Y`` — instead of looping python-side over configurations. It
  broadcasts against a batched ``operating_point`` by the usual rules.
- **Differentiable topology.** Because the state is a tensor on the autograd
  tape, gradients flow to it through the same implicit-function-theorem
  adjoint that differentiates network parameters — a continuous relaxation of
  a switch state is a valid gradient-descent variable for topology search or
  reconfiguration studies, not just a discrete flag.

``branch_states`` is accepted by :func:`~pgml.solver.solve_power_flow`,
:func:`~pgml.solver.solve_harmonic_flow`,
:func:`~pgml.solver.assemble_harmonic_system`,
:func:`~pgml.solver.assemble_harmonic_ybus`, and the underlying assemblers
(:func:`~pgml.assembly.assemble_ybus`, :func:`~pgml.assembly.assemble_network_ybus`,
:func:`~pgml.assembly.branch_currents`) — see the "Topology / switch-state
masking" section of :doc:`assembly` for the assembly-level mechanics.
:func:`~pgml.grids.synthetic_feeder`'s ``tie_switches`` argument builds the
canonical normally-open tie-switch scenario for this study (see :doc:`grids`).

``method="newton"`` accepts either a batched ``operating_point`` or batched
``branch_states``, not both at once (its per-scenario slicing covers the
operating point only); ``method="current_injection"`` batches both freely.

Switch-state sweeps: the low-rank update (``branch_states_method``)
------------------------------------------------------------------------

``branch_states_method`` (accepted by :func:`~pgml.solver.solve_power_flow` and
:func:`~pgml.solver.prepare_power_flow`) selects **how** a batched ``branch_states``
sweep reaches each state's linear system:

- ``"assemble"`` (default) — assemble and factor the admittance of every state:
  ``O(S·N³)`` work and an ``[S, N, N]`` matrix in memory.
- ``"woodbury"`` — assemble and factor the BASE network once, then reach every state
  through a Sherman-Morrison-Woodbury low-rank update of that single factorization
  (:mod:`pgml.solver.lowrank`). A switched branch enters ``Y`` only on its own terminal
  rows (:func:`~pgml.assembly.branch_stamp_blocks`), so scaling it is a rank-``≤ 2P``
  modification (``P`` = its phase count); with ``k = Σ 2P`` summed over every switched
  branch, a state then costs ``O(N²k + k³)`` instead of a fresh assembly and
  factorization.

Explicit opt-in only — there is no ``"auto"`` heuristic, because the win depends on
``k / N``, which only the caller (who knows how many branches its sweep switches) can
judge in advance. Requires ``branch_states`` and ``method="current_injection"``; a
reused ``system=`` must have been :func:`~pgml.solver.prepare_power_flow`-d with the
SAME ``branch_states_method``. FORWARD-only: the IFT backward always rebuilds the
per-state admittance differentiably, so gradients — including with respect to the
switch states themselves — are identical between the two methods.

The sweep's BASE omits every switched branch it can: the downdate that would REMOVE a
near-ideal closed switch from the base is ill-conditioned (it amplifies the base
solution's rounding by the switch's Thévenin-impedance ratio), while ADDING admittance
to reach a closed state is numerically benign, so the base is built without the
switched branches wherever the grid stays connected without them. See the
"Switch-state sweeps as a low-rank update (Woodbury)" section of
:doc:`/pgml/modeling/solver-performance` for the measured speedup, the crossover in
``k``, and the conditioning argument in full, and
``run/examples/pgml/benchmark_woodbury.py`` for the reproducible benchmark.

Solve performance: factorization backend and system reuse
-------------------------------------------------------------

Two independent knobs speed up repeated or large-scale solves without
changing any result:

**``linear_solver``** selects the inner linear-solve backend of
:func:`~pgml.solver.solve_power_flow` (and, via ``system=``, of
:func:`~pgml.solver.prepare_power_flow`). For
``method="current_injection"`` this picks the factorization of the constant
``Y_eff``: ``"auto"`` (default) uses a SciPy SuperLU SPARSE factorization on
CPU systems of roughly 500 or more rows — a power-grid ``Y`` has ``O(N)``
nonzeros, so sparse factorization is close to ``O(N)`` where dense LU is
``O(N³)`` — and falls back to the batched dense ``torch`` LU everywhere else
(CUDA is always dense). ``"dense"`` / ``"sparse"`` force the choice. The
sparse backend is fully differentiable via the linear-solve adjoint, so
switching backends never changes which quantities carry gradients. For
``method="newton"``, ``"dense"`` builds the explicit Jacobian and solves it
directly (what ``"auto"`` resolves to); ``"matrix_free"`` runs a
Jacobian-free Newton-Krylov solve (GMRES on finite-difference Jacobian-vector
products), trading iteration count for ``O(N)`` memory on very large grids.
``run/examples/pgml/benchmark_sparse.py`` sweeps :func:`~pgml.grids.synthetic_feeder`
across sizes and reports the sparse/dense crossover on CPU (and the dense-GPU
baseline it must be checked against) — see :doc:`/pgml/examples`.

A fourth choice, ``"block"``, factors a system that is *structurally* block-diagonal —
an ensemble of independent grids disjoint-unioned by :func:`pgml.multigrid.merge_grids`
— one member's diagonal block at a time (equal-sized members sharing one batched LU)
instead of factoring the union as a whole. Pass the member row partition via
``block_rows=merged.block_rows()`` (:meth:`~pgml.multigrid.MergedGrid.block_rows`); it
costs ``O(Σ n³)`` / ``O(Σ n²)`` where a dense factorization of the union costs
``O((Σ N)³)`` / ``O((Σ N)²)`` — the backend that makes a many-grid ensemble viable on
CUDA, where dense is otherwise the only union option. It is never selected by
``"auto"``: the partition is taken on trust (admittance outside the listed blocks would
be silently ignored), and on CPU the sparse union backend already exploits the same
block structure and stays the better choice. Unsupported with ``on_disconnected="zero"``
(dropping dead rows re-indexes the system, invalidating a fixed partition). See
:doc:`multigrid` and the "Ensembles of grids" section of
:doc:`/pgml/modeling/solver-performance` for the design rationale.

**``system``** lets repeated solves of the SAME grid skip the
operating-point-independent work entirely. Everything about the network side
of a nonlinear power flow — the node-phase index, ``Y_eff``, the slack rows
and reference, the factorization, and the grid-parameter leaves — does not
depend on the operating point, so :func:`~pgml.solver.prepare_power_flow`
computes it once into a :class:`~pgml.solver.PowerFlowSystem`, and passing
``solve_power_flow(..., system=that_system)`` reuses it across every call that
only varies the operating point (the pattern
:func:`pgml.scenarios.run_scenarios` uses internally for its chunked scenario
loop)::

    from pgml.solver import prepare_power_flow, solve_power_flow

    system = prepare_power_flow(grid, slack="ideal", linear_solver="auto")
    for op in operating_points:
        result = solve_power_flow(grid, slack="ideal", operating_point=op,
                                   system=system)

The reused system must come from the SAME grid, modeling defaults, ``slack``, ``dtype``,
``device``, ``param_overrides``, and ``branch_states`` as the solve that
consumes it. Modeling defaults are captured during preparation and compared on reuse, so
a system prepared under one :func:`pgml.defaults.use_preset` context is rejected under a
different preset. ``slack`` / ``dtype`` / ``device`` / size are validated cheaply on
every call; :func:`~pgml.solver.prepare_power_flow` additionally records the
grid's :func:`~pgml.topology.network_fingerprint` (every node, branch, source,
shunt and their parameter values) at prepare time, and
:func:`~pgml.solver.solve_power_flow` recomputes and compares it on each reuse —
a same-size grid whose topology or impedances have since changed is REJECTED
with :class:`~pgml.errors.InputError` instead of silently solving with the
stale factorization. ``param_overrides`` / ``branch_states`` equality remains
the caller's own contract (not fingerprinted). Reuse is a FORWARD-only optimization: the
IFT backward always rebuilds its differentiable system from the parameter leaves. The
autograd node retains an immutable snapshot of the resolved forward defaults, so a delayed
backward keeps the same physical model after a preset context exits or
:func:`pgml.defaults.reload` changes the process-wide defaults source.

Per-phase / connection-aware harmonic injection
-----------------------------------------------

The harmonic solver is **connection-aware**: it uses the
same terminal incidence matrix ``M`` that the fundamental power flow uses, so
DELTA (line-to-line) and 4-wire WYE (phase minus neutral) terminal voltages
are correctly applied when normalising each device's per-element harmonic
current.

The solver sources per-element spectra from three places, in priority order:

1. **Runtime override** — ``harmonic_injection`` kwarg to
   :func:`~pgml.solver.solve_harmonic_flow`:
   ``{appliance_id: {order: (magnitude_pu, phase_deg)}}``.
   A Python ``list``/``tuple`` of length ``n_elem`` is **per-element**; a
   scalar or bare tensor **broadcasts** to all elements (backward-compatible).
2. **Per-phase schema field** — ``Load.spectrum_per_phase`` /
   ``Generator.spectrum_per_phase``: ``dict[Phase, Spectrum]`` mapping each
   connected phase (or DELTA branch starting phase) to its own
   :class:`~pgml.schemas.grid_schema.Spectrum`.  Phases with no entry inject
   no harmonics.
3. **All-phases schema field** — ``Load.spectrum`` /
   ``Generator.spectrum``: one :class:`~pgml.schemas.grid_schema.Spectrum`
   broadcast to every element (OpenDSS multi-phase semantics).

``spectrum`` and ``spectrum_per_phase`` are **mutually exclusive** on any
single appliance.  See the "Per-phase / connection-aware harmonic injection"
section of the :doc:`/pgml/concepts` page for usage examples and the full override
convention.

Fundamental-current anchor (``S_eff``)
-----------------------------------------

Every device's harmonic spectrum scales from the ELEMENT (terminal) current it actually
draws at the converged fundamental voltage, not from its nameplate power. Per element,
``I1_elem = sign * conj(S_eff) / conj(V_term)`` (load convention, ``sign`` +1 load / -1
generator), where ``S_eff`` is the model-consistent power the device draws at that
voltage — identical to what the nonlinear fundamental solve itself resolves
(``device_current_injections``):

- **Inverter-controlled device** (``Load``/``Generator``/``Storage`` with a ``control``)
  — the control-resolved ``(P, Q)`` at the converged terminal voltage.
- **Voltage-dependent load model** (``load_model`` other than the const-power default)
  — the ZIP-scaled power ``S_eff = S0 * (z*r^2 + i*r + p)`` at ``r = |V_term| / V0``, the
  same law :func:`~pgml.assembly.device_current_injections` applies.
- **Const-power default** — the base operating point, unscaled.

Every order's magnitude and angle then follow the usual spectrum convention relative to
this ``I1_elem`` (see "Per-node harmonic source" below). Anchoring to the ACTUAL drawn
current rather than the nameplate power is what makes a ``CONST_IMPEDANCE`` /
``CONST_CURRENT`` / ``ZIP`` load's harmonic spectrum agree with an independent reference
engine's per-model fundamental-current scaling — see the OpenDSS scenario oracle's
matched-mode parity figures in :doc:`evaluation`.

Per-node harmonic source (``node_sources``)
--------------------------------------------

:class:`~pgml.solver.NodeHarmonicSource` models a harmonic disturbance at
**any** node of the grid — independent of any attached load or generator.  It
is injected **only at orders** ``h > 1``, so the nonlinear fundamental power
flow is preserved exactly (no damping reactor is needed; pgml solves each
harmonic as its own independent linear system).

The full physics derivation and OpenDSS equivalence are in
the per-node harmonic disturbance source (:doc:`/pgml/modeling/error-injection`).

Two source kinds are supported:

- ``kind="voltage"`` — **Thévenin** model: a finite-strength EMF ``E_h`` behind
  a resistive source impedance ``Z_s = V_base² / S_sc``.  Both ``Y_s`` (added
  to the diagonal of ``Y(h)``) and the Norton current ``I_N = E_h * Y_s`` are
  stamped.  Large ``S_sc`` (stiff source) → the node voltage converges to ``E_h``.
- ``kind="current"`` — **Norton** model: only ``I_N`` is added to ``I(h)``; no
  shunt is added.  The injected current is independent of the network impedance.

The EMF magnitude and phase follow the same convention used for device harmonic
injection::

    |E_h| = (mag_h / mag_1) * |V1|
    arg(E_h) = ang_h + h * (arg(V1) - ang_1)

where ``V1`` is the converged fundamental voltage at the injection row.

**Usage example** — inject a stiff 5th-harmonic voltage source at node 12::

    from pgml.solver import NodeHarmonicSource, solve_harmonic_flow

    src = NodeHarmonicSource(
        node_id=12,
        spectrum={5: (0.04, 0.0), 7: (0.03, 0.0)},   # 4 % / 3 % of fundamental
        source_power_va=1e6,                            # 1 MVAsc (stiff)
        kind="voltage",
    )
    hres = solve_harmonic_flow(
        grid, [1, 5, 7], node_sources=[src], slack="norton",
    )
    # hres.v  complex [H, N],  H = 3

Multiple simultaneous sources are allowed (pass a list); they superpose.  Both
``source_power_va`` and the spectrum coefficients may be 0-d / ``[*batch]``
tensors — gradients flow to them and (via ``V1``) to grid parameters.

**Per-node sweep.** To sweep the source over every node in the grid (one node
per scenario) use :func:`~pgml.scenarios.run_node_injection_sweep` from
:mod:`pgml.scenarios` (see :doc:`scenarios`).

.. automodule:: pgml.solver
   :members:
   :show-inheritance:

pgml.solver.equilibration
---------------------------

.. automodule:: pgml.solver.equilibration
   :members:
   :show-inheritance:
