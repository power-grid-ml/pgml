pgml.grids
==========

Reference grid builders: pandapower → pgml :class:`~pgml.schemas.grid_schema.Grid`, with
optional geometry, converter harmonic spectra, and a state-estimation benchmark recipe.

Each grid builder converts a well-known benchmark network (IEEE 33-bus, CIGRE LV) from
pandapower.  These are the suite's canonical INPUT grids — training-data generation,
examples, and the oracle comparison tests all build on them — so they live in the core
package rather than the evaluation oracles.  :mod:`pgml.evaluation.oracles.grids`
re-exports the builders unchanged for existing importers.

Importing :mod:`pgml.grids` itself has no heavy dependency; each builder function defers
its ``pandapower`` import to function scope (the optional ``convert`` extra).

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
  near-ideal source; the R/X split is controlled by the config key ``source.rx_ratio``.

State-estimation benchmark
---------------------------

- :func:`~pgml.grids.add_pv_systems` — attaches a unity-power-factor PV
  :class:`~pgml.schemas.grid_schema.Generator` (tagged ``consumer_type="pv"``) to a
  fraction of a grid's load nodes, mirroring each host load's node/phases and rated at
  half its nameplate active power.
- :func:`~pgml.grids.se_benchmark_scenario_config` — the canonical randomized
  state-estimation sampling recipe: a per-phase-independent load apparent-power scale, a
  per-load harmonic spectrum (fraction of the EN 50160 limit, orders
  :data:`~pgml.grids.LOAD_HARMONIC_ORDERS`), and — when the grid carries PV — one shared
  irradiance scale for all PV plus a per-inverter harmonic signature (orders
  :data:`~pgml.grids.PV_HARMONIC_ORDERS`)::

      from pgml.grids import (
          add_pv_systems, se_benchmark_scenario_config, cigre_lv_full_grid,
      )

      grid, _ = cigre_lv_full_grid()
      add_pv_systems(grid, fraction=0.5)
      cfg = se_benchmark_scenario_config(grid, n_samples=512, seed=0)

  This single recipe is the source of truth: the dataset-generation examples and the
  ``pgl`` test fixtures both build from it, so what a state-estimation model trains on
  cannot silently drift from what the documented benchmark generates.

.. automodule:: pgml.grids
   :members:
   :show-inheritance:
