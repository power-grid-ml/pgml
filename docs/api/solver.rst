pgml.solver
===========

Complex batched differentiable solve of ``Y(f) V(f) = I(f)``.

The solver package provides three entry points:

- :func:`~pgml.solver.solve_harmonic` — batched complex linear solve for a
  single assembled Y-bus.  Supports Norton mode (default) and ideal-slack
  Schur-partition mode, both fully differentiable.
- :func:`~pgml.solver.solve_power_flow` — nonlinear const-P / ZIP fundamental
  power flow using a current-injection fixed-point forward pass and an
  implicit-function-theorem (IFT) backward pass (real-coordinate adjoint).
- :func:`~pgml.solver.solve_harmonic_flow` — full harmonic flow orchestration:
  runs the fundamental power flow to convergence, then solves each harmonic
  in one batched pass.

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

As of Increment 2 the harmonic solver is **connection-aware**: it uses the
same terminal incidence matrix ``M`` that the fundamental power flow uses, so
DELTA (line-to-line) and 4-wire WYE (phase minus neutral) terminal voltages
are correctly applied when normalising each device's per-element harmonic
current.  The ``NotImplementedError`` guard for DELTA and 4-wire WYE harmonic
injection that existed in Increment 1 is removed.

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
section of the :doc:`/concepts` page for usage examples and the full override
convention.

.. automodule:: pgml.solver
   :members:
   :show-inheritance:
