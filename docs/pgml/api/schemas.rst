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

WYE return-path override and delta shunts
-------------------------------------------

Every :class:`~pgml.schemas.grid_schema.InjectionAppliance`
(:class:`~pgml.schemas.grid_schema.Load`, :class:`~pgml.schemas.grid_schema.Generator`,
:class:`~pgml.schemas.grid_schema.Storage`) carries
:attr:`~pgml.schemas.grid_schema.InjectionAppliance.return_path`
(``"auto"`` / ``"neutral"`` / ``"ground"``, default ``"auto"``) — a per-appliance override
of the WYE return-conductor decision that is otherwise made at the NODE level (return
through ``Phase.N`` whenever the node carries one, else ground). ``"ground"`` pins a WYE
appliance's return to true ground even on a neutral-carrying node — the OpenDSS
``bus1=b1.1.2.3`` idiom on a 4-wire bus, where one element ties to ground while a sibling
element on the same bus explicitly returns through ``.4``. ``"neutral"`` requires the host
node to carry ``Phase.N`` (assembly raises otherwise). ``return_path`` is meaningful for WYE
only — a non-``"auto"`` value on a DELTA-connected appliance raises. Because a grounded and
a neutral-returning WYE appliance on the SAME node need different incidence matrices,
:mod:`pgml.assembly` groups appliances by ``(connection, phase count, effective neutral
return)``, folding ``return_path`` into that grouping key — see the "WYE/DELTA connection
and neutral modeling" section of :doc:`assembly`. The default ``"auto"`` reproduces the
historical node-level rule byte-for-byte.

:class:`~pgml.schemas.grid_schema.ShuntAppliance` carries a
:attr:`~pgml.schemas.grid_schema.ShuntAppliance.connection`
(:class:`~pgml.schemas.grid_schema.WindingConnection`, default ``WYE``): the default WYE
bank connects each phase's ``G + jB`` to ground (the historical stamp); ``DELTA`` connects
element ``k`` between phase ``k`` and phase ``k + 1`` (cyclic over the appliance's own
phases, at least two phases required) — a delta capacitor or reactor bank. Zigzag is
rejected. This is the schema surface behind the OpenDSS converter's DELTA
``Capacitor`` / ``Reactor`` conversion (:doc:`convert`) and the assembly delta-shunt stamp
(:doc:`assembly`).

Measurement instrumentation
----------------------------

:attr:`~pgml.schemas.grid_schema.Grid.measurement_devices` carries the installed
metering hardware as INERT metadata — plain floats, never on the autograd tape and
never consumed by assembly or the solver. A
:class:`~pgml.schemas.grid_schema.MeasurementDevice` is node-anchored (a meter cabinet
at a bus): it measures voltage at its ``node`` and, per
:class:`~pgml.schemas.grid_schema.CurrentChannel`, current on branches incident to
that node — the physical picture is a CT clamped onto a feeder, with the metered
terminal inferred from the device's own node. ``measured_quantities``
(:class:`~pgml.schemas.grid_schema.MeasuredQuantity`: voltage / current / power)
states what the device records; ``max_harmonic_order`` / ``max_current_channels``
state its capability; ``manufacturer`` / ``model`` its identity;
``supported_averaging_intervals_s`` / ``averaging_interval_s`` its acquisition
timing.

``accuracy_class`` (e.g. ``"0.2S"`` per IEC 62053, ``"A"`` / ``"S"`` per
IEC 61000-4-30) is categorical — pgml attaches no numeric interpretation to it; it
is the intended key for a measurement-noise model outside the engine.
``connection`` is free-form JSON describing how to reach the physical
instrument (e.g. a Modbus TCP host/port); it is interpreted by the external
acquisition service, never by pgml.

The intended authoring flow attaches devices to an already-converted grid (e.g. a
DSO network plan) via
:meth:`~pgml.schemas.grid_schema.Grid.attach_measurement_devices`, which re-runs the
grid's cross-reference validation (node/branch existence, incidence, phase subsets)
atomically — a rejected attach restores the previous device list rather than
leaving the grid half-updated::

    from pgml.schemas import CurrentChannel, MeasurementDevice

    grid = grid.attach_measurement_devices([
        MeasurementDevice(
            id=0, node=4, accuracy_class="0.5S",
            measured_quantities=("voltage", "current"),
            current_channels=[CurrentChannel(branch=7)],
        ),
    ])

Slack designation is unrelated to instrumentation and stays ``Source``-based
(:func:`pgml.topology.slack_node_ids`). A consumer derives its sensor set, and whether
the slack is metered, from the attached devices.

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
