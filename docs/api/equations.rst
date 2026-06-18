pgml.equations
==============

Residual-form scalar physical-law registry.

The equations package stores the physical laws as SymPy expressions in
residual form ``0 = a - b``, then compiles them to both torch and numpy
evaluators.  This separation keeps the physics symbolic and the compute
layer replaceable.

Importing the package registers the M1 load-flow laws as a side effect
(see ``laws.py``).  The module-level ``registry`` singleton is then
available for lookups, evaluation, and LaTeX rendering.

.. automodule:: pgml.equations
   :members:
   :show-inheritance:
