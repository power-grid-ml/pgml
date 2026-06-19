"""pgml.convert.pgm — power-grid-model input_data -> Grid converter.

Public API
----------
- ``to_grid(input_data, *, base_frequency_hz=50.0, load_model=LoadModel.CONST_IMPEDANCE, phase_mode=PhaseMode.SINGLE_PHASE_EQUIV) -> (Grid, id_map)``
  Convert a power-grid-model ``input_data`` dict to a
  :class:`~pgml.schemas.grid_schema.Grid` and an ``id_map``.

Usage example::

    from power_grid_model import initialize_array, ComponentType
    from pgml.convert.pgm import to_grid

    input_data = { ... }   # dict of numpy structured arrays
    grid, id_map = to_grid(input_data, base_frequency_hz=50.0)

    # Three-phase mode — genuine abc grid:
    from pgml.convert.pgm import PhaseMode
    grid_3ph, id_map_3ph = to_grid(input_data, phase_mode=PhaseMode.THREE_PHASE)

Phase mode
----------
``phase_mode=PhaseMode.SINGLE_PHASE_EQUIV`` (default) mirrors pgm's symmetric
calculation: one positive-sequence equivalent per node, ``phases=(Phase.A,)``,
``u_rated_v = node.u_rated`` (line-to-line, V), 1x1 line matrices.

``phase_mode=PhaseMode.THREE_PHASE`` produces a genuine abc grid: nodes/branches
become ``(A, B, C)``; lines are expanded from sequence quantities via the
symmetric-component identity (zero-sequence from line ``r0/x0/c0`` fields when
present, else from ``pgml.config`` defaults); sources become balanced 3-phase
Thevenins (angles offset by 0 / −120 / +120 deg); ``asym_load`` entries are
captured with per-phase P/Q and connection ``WYE`` (power-grid-model models
all loads wye; no connection field is available).

id_map format
-------------
A dict with string keys for each component type that was converted::

    {
        "node":            {pgm_id: Node.id, ...},
        "line":            {pgm_id: Line.id, ...},
        "sym_load":        {pgm_id: Load.id, ...},
        "asym_load":       {pgm_id: Load.id, ...},  # THREE_PHASE only
        "source":          {pgm_id: Source.id, ...},
        "load_types":      {pgm_id: LoadGenType value, ...},
        "slack_v_complex": complex,  # slack voltage phasor (V, L-L)
    }

Only in-service elements are converted.
"""

from pgml.convert._common import PhaseMode

from .converter import to_grid

__all__ = ["to_grid", "PhaseMode"]
