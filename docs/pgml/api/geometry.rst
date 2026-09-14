pgml.geometry
=============

Differentiable conductor geometry → line constants (Carson/Deri).

The geometry package implements the full Carson/Deri earth-return model for
per-unit-length series impedance ``Z(h)`` and shunt admittance ``Yc(h)``
(Maxwell capacitance).  On the same conductor geometry it agrees with a running
OpenDSS to 4.8e-8 relative on ``Z`` and 2.1e-5 on ``C`` below 1 kHz — those two
residuals are the ``mu0`` and ``e0`` constants OpenDSS truncates, not the model.
``line.geometry.internal_inductance`` selects how the conductor's internal
inductance enters above power frequency, and its ``"gmr_power_frequency"`` option
reproduces OpenDSS's own 1 kHz rule at every frequency (4.6e-8 relative, 20 Hz to
3 kHz).  See :doc:`/pgml/modeling/harmonic-line-model`.

.. rubric:: Sub-modules

- **carson** — full earth-return model (Deri approximation): series impedance
  from conductor geometry, internal impedance with skin effect, Maxwell
  capacitance, and Kron reduction.  Results are differentiable and batched.
- **sequence** — positive-sequence harmonic model (no earth-return floor):
  suitable for R/X-defined overhead feeders in balanced operation.
  A sequence-aware variant correctly models zero-sequence via the Carson earth
  resistance for unbalanced / 4-wire studies.
- **synthesis** — R/X → geometry: synthesise a
  :class:`~pgml.schemas.grid_schema.LineGeometry` that reproduces a line's
  R/X at the fundamental frequency, enabling the Carson path for lines whose
  geometry is not explicitly known.

.. rubric:: Device and dtype

All functions honour ``device`` and ``dtype`` of their inputs.  Complex outputs
use ``complex64`` or ``complex128`` depending on the input float dtype.
Gradients flow from all geometry parameters (conductor positions, GMR, Rdc).

.. automodule:: pgml.geometry
   :members:
   :show-inheritance:
