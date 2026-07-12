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

Each Load/Generator can carry an explicit ``connection`` field
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

.. automodule:: pgml.assembly
   :members:
   :show-inheritance:
