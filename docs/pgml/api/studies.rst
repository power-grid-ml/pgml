pgml.studies
============

Grid-planning studies built on the harmonic power flow: each takes a
:class:`~pgml.schemas.grid_schema.Grid` and returns a frozen result object with a JSON-ready
``to_dict()`` and a ``from_dict()``. The user guide is :doc:`../studies/index`.

Impedance scan
--------------

:func:`~pgml.studies.impedance_scan` evaluates the driving-point and transfer impedance
seen from a set of buses over frequency; :meth:`~pgml.studies.ImpedanceScan.resonances`
locates the parallel and series resonances. User guide: :doc:`../studies/impedance-scan`.

Voltage stability
-----------------

:func:`~pgml.studies.pv_curve` traces the P-V curve through its nose by continuation power
flow; :func:`~pgml.studies.qv_curve` gives the Q-V curve and the reactive power margin of a
bus. User guide: :doc:`../studies/continuation`.

Hosting capacity
----------------

:func:`~pgml.studies.hosting_capacity` searches, for every candidate bus at once, the
largest new generation or demand that keeps the :class:`~pgml.studies.HostingLimits`;
:func:`~pgml.studies.branch_ratings` resolves the line ampacities and transformer ratings
those limits are checked against. User guide: :doc:`../studies/hosting-capacity`.

Reference
---------

.. automodule:: pgml.studies
   :members:
   :show-inheritance:

Default frequency grid of :func:`~pgml.studies.impedance_scan`:

.. autodata:: pgml.studies.impedance.DEFAULT_MAX_ORDER

.. autodata:: pgml.studies.impedance.DEFAULT_ORDER_STEP

Default relative prominence of :meth:`~pgml.studies.ImpedanceScan.resonances`:

.. autodata:: pgml.studies.impedance.DEFAULT_MIN_PROMINENCE
