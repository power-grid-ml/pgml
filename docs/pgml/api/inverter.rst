pgml.inverter
=============

Harmonic emission models of grid-following inverters. A unit is described by its physical
blocks (LCL and EMC filter, current control, PLL, PWM modulator, dead time, DC-link control
and MPPT) and its operating state, and evaluates to a frequency-domain Norton equivalent
``i = I_N - Y v`` at its terminals through
:meth:`~pgml.inverter.GridFollowingInverter.norton`. The network solve with these equivalents
is :func:`~pgml.studies.emission_flow`; the user guide is :doc:`../studies/emission-flow` and
the model derivation is :doc:`/pgml/modeling/inverter-emission`.

The models are per-solve inputs keyed by device id, not grid data, so the
:class:`~pgml.schemas.grid_schema.Grid` contract is unchanged.

.. automodule:: pgml.inverter
   :members:
   :show-inheritance:
