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
