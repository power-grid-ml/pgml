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

.. automodule:: pgml.assembly
   :members:
   :show-inheritance:
