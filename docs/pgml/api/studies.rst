pgml.studies
============

Grid-planning studies built on the harmonic power flow: each takes a
:class:`~pgml.schemas.grid_schema.Grid` and returns a frozen result object with a JSON-ready
``to_dict()`` and a ``from_dict()``. The user guide is :doc:`../studies/index`.

Hosting capacity
----------------

:func:`~pgml.studies.hosting_capacity` searches, for every candidate bus at once, the
largest new generation or demand that keeps the :class:`~pgml.studies.HostingLimits`;
:func:`~pgml.studies.branch_ratings` resolves the line ampacities and transformer ratings
those limits are checked against.

.. automodule:: pgml.studies
   :members:
   :show-inheritance:
