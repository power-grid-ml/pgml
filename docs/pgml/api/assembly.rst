pgml.assembly
=============

Differentiable, batched, per-frequency Y-bus assembly.

The assembly package turns a :class:`~pgml.schemas.Grid` (physical parameters)
into a per-frequency complex nodal admittance matrix ``Y(f)``
**[\\*batch, H, N, N]** and a Norton injection vector ``I(f)``
**[\\*batch, H, N]**.

.. rubric:: Key properties

- Every stamp operation is torch-native and GPU-ready.
- Gradients flow from every physical parameter (``R``, ``L``, ``G``, ``C``,
  transformer tap) through to the assembled ``Y``.
- Compact node-phase indexing (:class:`NodePhaseIndex`) assigns one row per
  existing ``(node, phase)`` pair rather than a padded A/B/C/N grid.
- Lines with ``conductor_geometry`` use the :mod:`pgml.geometry` Carson/Deri
  path; otherwise explicit R/L/G/C per metre are used.
- A branch with exactly zero series impedance has no stamp; its terminal rows are
  collapsed instead (see "Bus fusion" below), so the assembled system can be
  smaller than the grid's own row count.

Symmetric vs asymmetric calculation
------------------------------------

Both :func:`~pgml.assembly.assemble_ybus` and
:func:`~pgml.assembly.device_current_injections` accept a ``symmetry`` keyword
argument that controls how each Load/Generator's P/Q is distributed across its
phases:

- ``"symmetric"`` — the total P/Q of every appliance is split equally over its
  connected phases, regardless of any per-phase nameplate data.  Matches
  power-grid-model's ``symmetric=True``.
- ``"asymmetric"`` — per-phase values (``p_nom_per_phase_w`` /
  ``q_nom_per_phase_var``, or per-phase operating-point keys) are honored.
  Where no per-phase split is given, the total is still divided equally.
- ``"auto"`` (default in config) — selects asymmetric iff *any* appliance or
  operating-point entry carries per-phase data; otherwise symmetric.
- ``None`` — reads the config key ``calculation.symmetry`` (default ``"auto"``).

Pass the same string to both functions to keep a multi-call workflow consistent:

.. code-block:: python

   yb = assemble_ybus(grid, freqs, symmetry="asymmetric")
   i_dev = device_current_injections(grid, v, yb.index, freqs, symmetry="asymmetric")

WYE/DELTA connection and neutral modeling
------------------------------------------

Each Load/Generator/Storage can carry an explicit ``connection`` field
(:class:`~pgml.schemas.grid_schema.WindingConnection`).  When omitted, the
config keys ``appliance.load.default_connection`` (multi-phase) and
``appliance.load.single_phase_connection`` (1-phase) are used (default
``"wye"``).

**WYE loads** return into the node's ``Phase.N`` row when one is present (4-wire
network); otherwise the return path is ground.  The assembled nodal block is
``M^T diag(y_elem) M`` with ``M = [I_n | -1]`` for the 4-wire case, so the
neutral row accumulates the phase return currents automatically.

**DELTA-3 loads** use the circulant difference incidence
``M = [[1,-1,0],[0,1,-1],[-1,0,1]]`` (phase-to-phase elements).  The base
voltage ``V_0`` is the line-to-line rated voltage.

.. note::

   DELTA, 4-wire WYE (a node carrying ``Phase.N``), and the corresponding
   connection-aware **harmonic injection** are fully modelled, both in the Y-bus
   assembly and in :func:`~pgml.solver.solve_harmonic_flow`. A DELTA connection
   requires at least two phases (a single-phase DELTA raises ``ModelingError``).

**Per-appliance return-path override.** The node-level WYE return rule above is the
*default* (``return_path="auto"``); each
:class:`~pgml.schemas.grid_schema.InjectionAppliance` may override it individually via
:attr:`~pgml.schemas.grid_schema.InjectionAppliance.return_path`
(``"auto"`` / ``"neutral"`` / ``"ground"``). ``"ground"`` forces ``M = I_n`` even on a
neutral-carrying node; ``"neutral"`` forces the ``[I_n | -1]`` 4-wire form and raises
``ModelingError`` if the node has no ``Phase.N``; a non-``"auto"`` value on a DELTA
appliance also raises. The internal ``group_appliances`` helper
(:mod:`pgml.assembly`'s underscore ``_incidence`` module) folds ``return_path`` into its
grouping key (``(connection, n_phases, has_neutral_return)``), so a grounded and a
neutral-returning WYE appliance sharing one node land in two different groups with
different incidence matrices — see the schema field for the full decision table
(:doc:`schemas`).

Shunt appliances (WYE / DELTA)
---------------------------------

:class:`~pgml.schemas.grid_schema.ShuntAppliance` (a fixed linear ``G + jB`` shunt) is
stamped per its :attr:`~pgml.schemas.grid_schema.ShuntAppliance.connection`:

- **WYE** (default) — each phase's admittance connects to ground; the historical diagonal
  stamp.
- **DELTA** — element ``k`` connects phase ``k`` to phase ``(k + 1) mod n`` (cyclic,
  ``n >= 2``), stamped ``Mᵀ·diag(y)·M`` with the same cyclic incidence
  (``cyclic_delta_incidence``, an internal ``_incidence`` helper) the DELTA
  load/generator path uses.

Both are frequency-correct at every harmonic order (``B(h) = 2πh f0 C``) and differentiable
w.r.t. ``G`` / ``C`` — the incidence ``M`` is a constant topology matrix, never on the
autograd tape.

Topology / switch-state masking
---------------------------------

Every public assembler — :func:`~pgml.assembly.assemble_ybus`,
:func:`~pgml.assembly.assemble_network_ybus`, and
:func:`~pgml.assembly.branch_currents` — accepts an optional
``branch_states={branch_id: state}``. A branch listed there is ALWAYS stamped
(overriding its static ``in_service`` / ``closed`` flags) and its primitive
admittance block is multiplied by ``state``: a python float, a 0-d tensor, or
a ``[*batch]`` scenario tensor in ``[0, 1]`` (``0`` = open, ``1`` = in
service, intermediate values scale the admittance continuously and stay on
the autograd tape). A batched state promotes ``Y`` to ``[*batch, H, N, N]``,
so one assembly covers a whole batch of switch/topology configurations, and
:func:`~pgml.assembly.branch_currents` scales each branch's terminal current
by the same state (an open branch reports exactly 0 A).

This is the assembly-level mechanic behind the solver's topology / switch-state
batching — see the ``branch_states`` section of :doc:`solver` for the
solve-level behaviour (differentiable topology search, per-scenario
connectivity checking) and :func:`~pgml.grids.synthetic_feeder`'s
``tie_switches`` for a ready-made grid to exercise it on.

Bus fusion: the ideal (zero-impedance) branch
-----------------------------------------------

A branch whose series impedance is exactly zero — a closed switch with no impedance data, a
bus coupler or jumper modelled as a zero-impedance line, a zero-length line — has no
primitive admittance, because the nodal formulation inverts every branch's series impedance.
What the element states is an equality: its two terminals carry the same voltage.  Under the
documented default ``branch.zero_impedance: fuse`` the assembly imposes exactly that by
collapsing the branch's terminal node-phase rows into ONE row of the matrix it builds.

:func:`~pgml.assembly.fusion_map` returns the :class:`~pgml.assembly.FusionMap` for a grid
(``None`` when the grid has no such branch), and
:func:`~pgml.assembly.zero_impedance_branches` lists the candidates with the reason each one
is or is not fusable.  :func:`~pgml.assembly.assemble_ybus` and
:func:`~pgml.assembly.assemble_network_ybus` take ``fusion=`` as a map, ``True`` to build
one, ``False`` to refuse, or ``None`` for the documented default, and the resulting
:class:`~pgml.assembly.YBus` carries it.  A fused ``Y`` is ``[*batch, H, M, M]`` with ``M``
the reduced row count.

The map is a many-to-one row relabelling, so it is cheap and exact:

- ``prolong(v_red)`` spreads an ``[..., M]`` reduced solution back to ``[..., N]`` by gather,
  which is what every public result reports.
- ``restrict(x_full)`` sums an ``[..., N]`` vector into ``[..., M]``, the adjoint of
  ``prolong``, which is what the injections and the current balance use.
- ``sample(x_full)`` takes the representative row of each group instead of summing, for a
  quantity that is equal across a group rather than additive, such as the converged
  fundamental voltage a harmonic assembly reads.
- ``reduce_rows`` maps a row SET (the slack rows, a block partition) to the reduced layout,
  and ``duplicate_rows`` reports which full rows share a reduced one.

The reduced system satisfies ``Y_red = Pᵀ Y P`` with ``P`` the prolongation, so the fused
solve is the exactly constrained one and not an approximation.  A zero-impedance branch that
can be neither stamped nor fused is refused by name, with the fix stated: a
:class:`~pgml.schemas.grid_schema.Transformer`, whose ratio and vector group relate its
terminals by more than equality; a zero-series branch that still carries a shunt admittance; a
branch the caller put under a ``branch_states`` sweep; and a branch between two nodes of
different rated voltage, which is a data error.  A solve additionally refuses a fused group
that would collapse two slack terminals or two regulating terminals into one row, which
``duplicate_rows`` detects.  Two ideal branches in parallel leave a circulating current
undetermined; they land in one group, the reported current split is the minimum-norm one, and
the branch ids appear in ``indeterminate_branch_ids`` and in a warning.

``param_overrides`` decides the question, because fusion reads the EFFECTIVE impedance.  A
branch whose zero impedance is overridden by a finite leaf stays stamped and keeps its
gradient; a substitution of zero fuses.

:func:`~pgml.assembly.branch_currents` takes the same ``fusion=`` map and recovers the
current through a fused branch from Kirchhoff's law at the fused node, optionally given the
nodal injection ``i_inj=`` so that a device shunt contributes its share.  The gradient of a
fused branch's own series parameters with respect to any output is structurally zero, since
those parameters no longer enter the system; every other gradient is unchanged.  Tensors are
int64 and follow ``FusionMap.to(device)``.

Performance: precomputed injection plans
-------------------------------------------

:func:`~pgml.assembly.device_current_injections` resolves the operating point
(walking appliances, pydantic fields, and config defaults) and then evaluates
the voltage-dependent ZIP current law — but the nonlinear solvers call it
every fixed-point / Newton iteration, where the python-side resolution work
dominates the solve time even though it never changes between iterations.
:func:`~pgml.assembly.build_injection_plan` splits it into its two halves: it
precomputes the voltage-INDEPENDENT tensors once per solve into an
:class:`~pgml.assembly.InjectionPlan`, and
:func:`~pgml.assembly.injections_from_plan` evaluates that plan against the
current voltage on every iteration using pure tensor ops only. The nonlinear
solvers reuse one plan across all their iterations; a plan built under
``torch.no_grad()`` is the detached fast path, while one built with gradients
enabled stays fully differentiable — the composition
``injections_from_plan(build_injection_plan(...), v)`` is byte-identical to
:func:`~pgml.assembly.device_current_injections`.

Primitive stamp blocks (``branch_stamp_blocks``)
----------------------------------------------------

:func:`~pgml.assembly.branch_stamp_blocks` hands out one named branch's primitive
admittance block and the global Y rows it occupies
(:class:`~pgml.assembly.BranchStampBlock`) — the SAME branch-stamp registry walk that
the assembly and :func:`~pgml.assembly.branch_currents` use, so no stamp physics is
re-derived here. Because a branch only ever enters ``Y`` as ``Y[rows, rows] += block``,
this pair is the *complete* description of what scaling that branch's admittance by
``s`` changes: ``(s − 1) × block`` on ``rows``, a rank-``≤ M`` modification (``M = 2P``
for a two-terminal branch, ``P`` for a single-terminal shunt). Every requested branch is
stamped regardless of its ``in_service`` / ``closed`` flags, and the returned block is
always the UNSCALED (state-1) primitive; the block is differentiable w.r.t. the branch's
parameters, and device/dtype follow the arguments.

This is the structural input a LOW-RANK admittance update needs — the seam
:mod:`pgml.solver.lowrank` builds the Woodbury switch-state sweep on (see the
"Switch-state sweeps" section of :doc:`solver`), and more generally the seam for
solving a mutated grid from its parent's factorization.

.. automodule:: pgml.assembly
   :members:
   :show-inheritance:
