"""pgml.convert.pandapower — converter from a pandapower net to our Grid schema.

Public API
----------
- ``to_grid(net) -> (Grid, id_map)``
  Convert a materialised pandapower network to a :class:`~pgml.schemas.grid_schema.Grid`
  and an ``id_map`` that maps source identifiers back to our schema ids.

Usage example::

    import numpy as np
    np.Inf = np.inf          # numpy 2.x compat shim (pandapower 2.14 uses removed alias)
    np.in1d = np.isin        # same
    import pandapower.networks as pn
    from pgml.convert.pandapower import to_grid

    net = pn.case33bw()
    grid, id_map = to_grid(net)

The converter handles the element types present in typical radial distribution
feeders (bus, line, load, ext_grid).  Trafo/shunt/gen/sgen conversion is
structured to extend but not over-built.

id_map format
-------------
A dict with string keys for each element table that was converted::

    {
        "bus":             {pp_bus_index: Node.id, ...},
        "line":            {pp_line_index: Line.id, ...},
        "trafo":           {pp_trafo_index: Transformer.id, ...},
        "switch":          {pp_switch_index: Switch.id, ...},
        "load":            {pp_load_index: Load.id, ...},
        "asymmetric_load": {pp_asym_index: Load.id, ...},  # THREE_PHASE only
        "ext_grid":        {pp_extgrid_index: Source.id, ...},
        "slack_v_complex": complex,  # ideal-slack phasor (V, line-to-line)
    }

Only entries that were actually converted are included.
"""

from pgml.convert._common import PhaseMode

from .converter import to_grid

__all__ = ["to_grid", "PhaseMode"]
