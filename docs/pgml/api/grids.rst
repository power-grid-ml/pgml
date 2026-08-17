pgml.grids
==========

Reference grid builders: pandapower → pgml :class:`~pgml.schemas.grid_schema.Grid`, with
optional geometry, converter harmonic spectra, and a state-estimation benchmark recipe.
Also :func:`~pgml.grids.synthetic_feeder`, a schema-only synthetic feeder of arbitrary size
for solver scaling and topology-batching studies.

Most builders convert a well-known benchmark network (IEEE 33-bus, CIGRE LV) from
pandapower.  These are the suite's canonical INPUT grids — training-data generation,
examples, and the oracle comparison tests all build on them — so they live in the core
package rather than the evaluation oracles.  :mod:`pgml.evaluation.oracles.grids`
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
  near-ideal source; the R/X split is controlled by the config key ``source.rx_ratio``.

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

State-estimation benchmark
---------------------------

- :func:`~pgml.grids.add_pv_systems` — attaches a unity-power-factor PV
  :class:`~pgml.schemas.grid_schema.Generator` (tagged ``consumer_type="pv"``) to a
  fraction of a grid's load nodes, mirroring each host load's node/phases and rated at
  half its nameplate active power.
- :func:`~pgml.grids.se_benchmark_scenario_config` — a thin front door onto the ONE
  calibrated recipe every state-estimation generator in the suite shares
  (:func:`pgml.scenarios.se_random_scenario_config`, see :doc:`scenarios`'s "Calibrated
  state-estimation presets" section): correlated load levels through a shared latent, a
  small per-phase unbalance, a slack-voltage draw, and a per-device IEC 61000-3-2-referenced
  harmonic spectrum with per-order emission-phase diversity (orders
  :data:`~pgml.grids.LOAD_HARMONIC_ORDERS`), plus — when the grid carries PV — one shared
  irradiance scale for all PV plus a per-inverter harmonic signature (orders
  :data:`~pgml.grids.PV_HARMONIC_ORDERS`)::

      from pgml.grids import (
          add_pv_systems, se_benchmark_scenario_config, cigre_lv_full_grid,
      )

      grid, _ = cigre_lv_full_grid()
      add_pv_systems(grid, fraction=0.5)
      cfg = se_benchmark_scenario_config(grid, n_samples=512, seed=0)

  Sharing the ONE builder (rather than each caller assembling its own specs) is what keeps
  this benchmark, the training-workflow datasets, and the multi-grid corpus from drifting
  apart: the dataset-generation examples and the ``pgl`` test fixtures all build from it, so
  what a state-estimation model trains on cannot silently diverge from what the documented
  benchmark generates.

.. automodule:: pgml.grids
   :members:
   :show-inheritance:
