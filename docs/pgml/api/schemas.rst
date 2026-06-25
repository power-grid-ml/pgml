pgml.schemas
============

The three schema modules are the **frozen data contracts** for `pgml`.
They define the complete input (``Grid``), output (``ResultSet``), and
scenario (``Scenario``) representation.  Import canonical types from
here — do **not** modify these files.

See also: :doc:`/pgml/concepts` for the modelling conventions (phase-domain,
SI units, float/tensor duality).

Appliance types
---------------

The grid input carries five appliance kinds (all subclasses of
:class:`~pgml.schemas.grid_schema.ApplianceBase`):

- :class:`~pgml.schemas.grid_schema.Source` — Thevenin slack / external network.
- :class:`~pgml.schemas.grid_schema.Load` — passive demand (const-P / ZIP / harmonic).
- :class:`~pgml.schemas.grid_schema.Generator` — active injection (PV, wind, CHP).
- :class:`~pgml.schemas.grid_schema.Storage` — bidirectional inverter (battery);
  ``p_nom_w > 0`` = discharging / injecting, ``< 0`` = charging.  At a power-flow
  snapshot a storage is treated identically to a generator; the state-of-charge
  integration lives in :mod:`pgml.scenarios` (see :doc:`scenarios`).
- :class:`~pgml.schemas.grid_schema.ShuntAppliance` — passive shunt (G + jB).

:class:`~pgml.schemas.grid_schema.Load`, :class:`~pgml.schemas.grid_schema.Generator`,
and :class:`~pgml.schemas.grid_schema.Storage` all inherit
:class:`~pgml.schemas.grid_schema.InjectionAppliance`.

DER inverter control
--------------------

:class:`~pgml.schemas.grid_schema.Generator` and
:class:`~pgml.schemas.grid_schema.Storage` carry an optional ``control`` field
(type :data:`~pgml.schemas.grid_schema.InverterControl`) that attaches one of six
operating-point control laws:

- :class:`~pgml.schemas.grid_schema.ConstantPowerFactorControl` — fixed
  ``cos(phi)``.
- :class:`~pgml.schemas.grid_schema.ConstantReactivePowerControl` — fixed Q
  setpoint.
- :class:`~pgml.schemas.grid_schema.PowerFactorWattControl` — ``cosphi(P)``
  characteristic (VDE-AR-N 4105).
- :class:`~pgml.schemas.grid_schema.VoltVarControl` — ``Q(V)`` Volt-VAr curve.
- :class:`~pgml.schemas.grid_schema.VoltWattControl` — ``P(V)`` Volt-Watt curve.
- :class:`~pgml.schemas.grid_schema.VoltVarVoltWattControl` — combined Volt-VAr
  and Volt-Watt (OpenDSS ``InvControl CombiMode=VV_VW``).

All six share :class:`~pgml.schemas.grid_schema.InverterControlBase` (rating cap
``s_rated_va``, soft-saturation ``smoothing`` for C\ :sup:`1` gradients).  Curves
are represented by :class:`~pgml.schemas.grid_schema.Characteristic` (piecewise
``y = f(x)``); breakpoints and levels are tensor-capable, so curve parameters are
differentiable leaves.  Use ``method="newton"`` in
:func:`~pgml.solver.solve_harmonic_flow` when stiff Volt-VAr / Volt-Watt loops
cause the current-injection fixed point to oscillate.

.. automodule:: pgml.schemas
   :members:
   :show-inheritance:

pgml.schemas.grid_schema
------------------------

.. automodule:: pgml.schemas.grid_schema
   :members:
   :show-inheritance:
   :no-index:
   :member-order: bysource

pgml.schemas.result_schema
--------------------------

.. automodule:: pgml.schemas.result_schema
   :members:
   :show-inheritance:
   :no-index:
   :member-order: bysource

pgml.schemas.scenario_schema
-----------------------------

.. automodule:: pgml.schemas.scenario_schema
   :members:
   :show-inheritance:
   :no-index:
   :member-order: bysource
