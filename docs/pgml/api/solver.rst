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
  Used by the physics-informed injection decoder in ``pgl``: the model predicts
  the nodal injection current ``I_pred`` and reconstructs the full voltage state
  via ``V(h) = solve(Y(h), I_pred)`` — a self-consistency that uses only the
  (differentiable) grid description, not the ground-truth injection.  ``Y`` is
  grid-constant (assembled once, reused across the batch) and differentiable
  w.r.t. the network parameters, so the same call powers a learned
  grid-parameter calibration.
- :func:`~pgml.solver.check_connectivity` — the pre-solve structural gate:
  raises :class:`~pgml.errors.ConnectivityError` when part of the grid has no
  galvanic path to an in-service source, before any factorization is attempted.
- :func:`~pgml.solver.prepare_power_flow` — assembles and factors the
  operating-point-independent part of a nonlinear solve once, returning a
  :class:`~pgml.solver.PowerFlowSystem` that repeated
  :func:`~pgml.solver.solve_power_flow` calls on the same grid can reuse.

.. rubric:: Differentiability

``solve_harmonic`` differentiates cleanly via the linear solve adjoint.
``solve_power_flow`` uses an explicit IFT adjoint (the only sanctioned
``.detach()`` in the codebase) so gradients flow through the converged solution.

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

The reused system must come from the SAME grid, ``slack``, ``dtype``,
``device``, ``param_overrides``, and ``branch_states`` as the solve that
consumes it. ``slack`` / ``dtype`` / ``device`` / size are validated cheaply on
every call; :func:`~pgml.solver.prepare_power_flow` additionally records the
grid's :func:`~pgml.topology.network_fingerprint` (every node, branch, source,
shunt and their parameter values) at prepare time, and
:func:`~pgml.solver.solve_power_flow` recomputes and compares it on each reuse —
a same-size grid whose topology or impedances have since changed is REJECTED
with :class:`~pgml.errors.InputError` instead of silently solving with the
stale factorization. ``param_overrides`` / ``branch_states`` equality remains
the caller's own contract (not fingerprinted). Reuse is a FORWARD-only
optimization: the IFT backward always rebuilds its differentiable system from
the parameter leaves, so gradients are byte-identical to a solve without
``system``.

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
