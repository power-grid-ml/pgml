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

.. automodule:: pgml.solver
   :members:
   :show-inheritance:
