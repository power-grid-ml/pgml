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

.. automodule:: pgml.assembly
   :members:
   :show-inheritance:
