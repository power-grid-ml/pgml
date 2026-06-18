pgml.convert
============

Converters from external formats to the `pgml` schema.

Each sub-package provides a ``to_grid()`` function that converts a
source-library object to a :class:`~pgml.schemas.Grid`.

.. automodule:: pgml.convert
   :members:
   :show-inheritance:

pgml.convert.pandapower
------------------------

Convert a `pandapower <https://www.pandapower.org/>`_ network to a
:class:`~pgml.schemas.Grid`.

.. note::

   ``pandapower`` is mocked in the docs build (numpy 2.x incompatibility).
   The public API (``to_grid``) is documented from the source directly.

.. automodule:: pgml.convert.pandapower
   :members:
   :show-inheritance:

pgml.convert.pgm
-----------------

Convert a `power-grid-model <https://power-grid-model.readthedocs.io/>`_
``input_data`` dict to a :class:`~pgml.schemas.Grid`.

.. automodule:: pgml.convert.pgm
   :members:
   :show-inheritance:

pgml.convert.opendss
---------------------

Convert an OpenDSS circuit (via `opendssdirect
<https://opendssdirect.readthedocs.io/>`_) to a :class:`~pgml.schemas.Grid`.

.. automodule:: pgml.convert.opendss
   :members:
   :show-inheritance:
