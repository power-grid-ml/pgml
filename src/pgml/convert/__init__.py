"""pgml.convert — converters from external formats to our schema.

One subpackage per source:
- ``pandapower/``  — pandapower net -> Grid
- ``pgm/``         — power-grid-model input_data -> Grid  (future)
- ``opendss/``     — OpenDSS dss handle -> Grid           (future)

See each subpackage for the public ``to_grid`` signature.
"""
