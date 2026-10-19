pgml.studies
============

Grid-planning studies that answer a question about a grid rather than solve one operating
point. Each study takes a :class:`~pgml.schemas.grid_schema.Grid` and returns a result
object with a JSON-ready ``to_dict()`` and a ``from_dict()``. User guide:
:doc:`../studies/impedance-scan`.

.. automodule:: pgml.studies
   :members:
   :show-inheritance:

Default frequency grid of :func:`~pgml.studies.impedance_scan`:

.. autodata:: pgml.studies.impedance.DEFAULT_MAX_ORDER

.. autodata:: pgml.studies.impedance.DEFAULT_ORDER_STEP
