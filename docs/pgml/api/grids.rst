pgml.grids
==========

Reference grid builders: pandapower → pgml :class:`~pgml.schemas.grid_schema.Grid`, with
optional conductor geometry and converter harmonic spectra.
Also :func:`~pgml.grids.synthetic_feeder`, a schema-only synthetic feeder of arbitrary size
for solver scaling and topology-batching studies.

Most builders convert a well-known benchmark network (IEEE 33-bus, CIGRE LV) from
pandapower.  They are the canonical input grids for the examples and the oracle
comparison tests, so they live in the core package rather than the evaluation oracles.  :mod:`pgml.evaluation.oracles.grids`
re-exports the builders unchanged for existing importers.

Importing :mod:`pgml.grids` itself has no heavy dependency; each pandapower-backed builder
function defers its ``pandapower`` import to function scope (the optional ``convert``
extra). :func:`~pgml.grids.synthetic_feeder` has no such dependency at all.

Grid builders
-------------

- :func:`~pgml.grids.ieee33_geometry_grid` — the IEEE 33-bus feeder with synthesized
  Carson conductor geometry and converter harmonic spectra on the farthest loads.
- :func:`~pgml.grids.cigre_lv_geometry_grid` — the residential CIGRE LV feeder (one
  feeder, fed by a Thévenin source at its LV busbar) with synthesized Carson geometry.
- :func:`~pgml.grids.cigre_lv_full_grid` — the **full** CIGRE LV benchmark (all three
  feeders, three 20/0.4 kV transformers, MV ext-grid source), with standard R/X lines::

      from pgml.grids import cigre_lv_full_grid
      from pgml.convert.pandapower import PhaseMode

      grid, id_map = cigre_lv_full_grid()
      # or with an explicit phase mode and source impedance:
      grid, id_map = cigre_lv_full_grid(
          phase_mode=PhaseMode.THREE_PHASE,
          source_impedance_ohm=5.0,   # None -> config default source.series_impedance_ohm
      )

  The stock pandapower ext-grid converts to a near-ideal source that short-circuits the
  bus at harmonics; ``source_impedance_ohm`` applies a finite upstream-grid impedance so
  harmonics are not fully absorbed at the slack. Passing ``0`` keeps the converted
  near-ideal source; the X/R split is controlled by the config key ``source.xr_ratio``.

Synthetic feeder (solver scaling benchmarks)
------------------------------------------------

- :func:`~pgml.grids.synthetic_feeder` — a parameterizable 3-phase radial MV
  feeder of *arbitrary* size, built directly on the schema with **no external
  dependency** (unlike the pandapower-backed builders above). Node 0 is the
  station bus carrying the :class:`~pgml.schemas.grid_schema.Source`; the
  remaining ``n_nodes - 1`` nodes are spread over ``n_feeders`` radial chains
  of typical 20 kV overhead-line segments, each carrying an equal share of
  ``total_load_w`` as a constant-power WYE load. The node-phase system has
  ``3 * n_nodes`` rows, so ``n_nodes`` is a direct scaling knob for solver
  benchmarks::

      from pgml.grids import synthetic_feeder
      from pgml.solver import solve_power_flow

      grid = synthetic_feeder(300, n_feeders=8)
      result = solve_power_flow(grid)   # 900-row system

  Passing ``tie_switches=k`` adds ``k`` normally-open
  :class:`~pgml.schemas.grid_schema.Switch` elements between the far ends of
  adjacent feeders — open by default (the base grid stays radial and fully
  energized) and the canonical grid for exercising the solver's
  ``branch_states`` switch-state batching (see the "Topology / switch-state
  batching" section of :doc:`solver`).

Distributed generation and harmonic emission
----------------------------------------------

- :func:`~pgml.grids.add_pv_systems` — attaches a unity-power-factor PV
  :class:`~pgml.schemas.grid_schema.Generator` (tagged ``consumer_type="pv"``) to a
  fraction of a grid's load nodes, mirroring each host load's node/phases and rated at
  half its nameplate active power.
- :data:`~pgml.grids.LOAD_HARMONIC_ORDERS` and :data:`~pgml.grids.PV_HARMONIC_ORDERS` —
  the order sets a converter load and a PV inverter emit on, and
  :data:`~pgml.grids.CONVERTER_SPECTRUM` the six-pulse spectrum the builders attach.

The builders here produce input GRIDS.  How a BATCH over such a grid is drawn — which
quantities vary, over what ranges — is a study's decision rather than the engine's: build one
from :class:`~pgml.scenarios.ScenarioConfig`, or hand explicit values to
:func:`~pgml.scenarios.batch_from_values`.  See :doc:`scenarios`.

.. automodule:: pgml.grids
   :members:
   :show-inheritance:
