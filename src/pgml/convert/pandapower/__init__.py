"""pgml.convert.pandapower — converter from a pandapower net to our Grid schema.

Public API
----------
- ``to_grid(net) -> (Grid, id_map)``
  Convert a materialised pandapower network to a :class:`~pgml.schemas.grid_schema.Grid`
  and an ``id_map`` that maps source identifiers back to our schema ids.
- :class:`~pgml.convert.pandapower.converter.GenMode` — how ``net.gen`` (the PV bus)
  is treated: exact voltage regulation by default, dropped, or approximated by a
  Volt-VAr droop.
- ``DEFAULT_GEN_VOLT_VAR_SLOPE_PU`` — the default droop steepness of that
  approximation.

Usage example::

    from pgml.convert.pandapower import GenMode, to_grid

    import pandapower.networks as pn

    net = pn.case33bw()
    grid, id_map = to_grid(net)

    # A transmission benchmark carries its generators in `net.gen` (PV buses):
    net118 = pn.case118()
    grid118, id_map118 = to_grid(net118)

The converter handles the element types present in typical radial distribution
feeders (bus, line, load, ext_grid) plus two-winding transformers, switches,
static generators, voltage-regulating generators, fixed shunts and storage.
Unsupported non-empty tables are reported rather than dropped silently.

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
        "sgen":            {pp_sgen_index: Generator.id, ...},
        "gen":             {pp_gen_index: Generator.id, ...},
        "storage":         {pp_storage_index: Storage.id, ...},
        "shunt":           {pp_shunt_index: ShuntAppliance.id, ...},
        "ext_grid":        {pp_extgrid_index: Source.id, ...},
        "open_terminal":   {pp_switch_index: Node.id, ...},
        "slack_v_complex": complex,  # ideal-slack phasor (V, line-to-line)
    }

Only entries that were actually converted are included.
"""

from pgml.convert._common import PhaseMode

from .converter import DEFAULT_GEN_VOLT_VAR_SLOPE_PU, GenMode, to_grid

__all__ = [
    "to_grid",
    "PhaseMode",
    "GenMode",
    "DEFAULT_GEN_VOLT_VAR_SLOPE_PU",
]
