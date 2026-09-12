"""pgml.convert.opendss — converter from a live OpenDSS circuit to our Grid schema.

Public API
----------
- ``to_grid(dss_handle, *, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV) -> (Grid, id_map)``
  Convert the currently-loaded OpenDSS circuit (via opendssdirect) to a
  :class:`~pgml.schemas.grid_schema.Grid` and an ``id_map`` that maps DSS
  element names back to our schema ids.

Usage example::

    import opendssdirect as dss
    from pgml.convert.opendss import to_grid

    dss.Text.Command("Redirect myfeeder.dss")
    dss.Text.Command("Solve")
    grid, id_map = to_grid(dss)

    # Or with explicit three-phase mode:
    from pgml.convert._common import PhaseMode
    grid_3ph, id_map_3ph = to_grid(dss, phase_mode=PhaseMode.THREE_PHASE)

Supported element types
-----------------------
- Line: single-phase or multi-phase series branch (R/X/C per-length n×n matrices)
- Transformer (two-winding): vector-group aware -> :class:`~pgml.schemas.grid_schema.Transformer`
- Vsource: external grid equivalent -> :class:`~pgml.schemas.grid_schema.Source`
- Load: shunt injection (single-phase or balanced multi-phase) -> :class:`~pgml.schemas.grid_schema.Load`
- Capacitor, Reactor: shunt elements -> :class:`~pgml.schemas.grid_schema.ShuntAppliance`
- Generator, PVSystem: PQ injection -> :class:`~pgml.schemas.grid_schema.Generator`
- Storage: PQ injection with energy-state fields -> :class:`~pgml.schemas.grid_schema.Storage`

Every other non-empty element class present in the circuit (``Isource``, controls,
monitors/meters, three-winding transformers, ...) is not converted; a WARNING names
the kind and count so nothing is dropped silently.

id_map format
-------------
A dict with string keys for each element type that was converted::

    {
        "bus":           {dss_bus_name: Node.id, ...},       # lowercase bus names
        "line":          {dss_line_name: Line.id, ...},      # lowercase element names
        "trafo":         {dss_trafo_name: Transformer.id, ...},
        "load":          {dss_load_name: Load.id, ...},
        "vsource":       {dss_vsrc_name: Source.id, ...},
        "capacitor":     {dss_cap_name: ShuntAppliance.id, ...},
        "reactor":       {dss_reactor_name: ShuntAppliance.id, ...},
        "generator":     {dss_gen_name: Generator.id, ...},
        "pvsystem":      {dss_pv_name: Generator.id, ...},
        "storage":       {dss_storage_name: Storage.id, ...},
        "slack_v_complex": complex,  # slack voltage phasor (V, L-L) from first Vsource
    }

DSS element names are normalised to lowercase. Node ids are assigned in the
order buses are first encountered in the YNodeOrder list (which is the order
OpenDSS uses for its Y matrix rows/columns).
"""

from pgml.convert._common import PhaseMode

from .converter import to_grid

__all__ = ["to_grid", "PhaseMode"]
