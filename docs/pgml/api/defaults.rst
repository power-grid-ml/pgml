pgml.defaults
=============

Explicit, documented modelling defaults.

All default constants and default model selections live in ``data/defaults.yaml``
inside the package, ordered by component — each leaf carries a value, units, and a short
description.  The file is read via :mod:`importlib.resources` and resolves identically
from a source checkout and from an installed wheel.

Resolution precedence
---------------------

.. code-block:: text

    explicit (user) > defaults (this module) > converter (source library)

A user-supplied value always wins; a modeling default overrides whatever a converter would
infer.  :func:`~pgml.defaults.use_preset` temporarily selects the supported choices of a
reference library while preserving that precedence::

    import pgml.defaults as defaults
    from pgml.solver import solve_harmonic_flow

    with defaults.use_preset("opendss"):
        result = solve_harmonic_flow(grid, [1, 3, 5])

The selection is local to the context, including concurrent threads and tasks, and the
previous selection is restored on exit.  Keep conversion, direct geometry calculations,
preparation and solving in the same context.  See :doc:`/pgml/modeling/presets` for the
available presets and their scope.

Overriding the defaults file
-----------------------------

Set the ``PGML_DEFAULTS`` environment variable to an absolute or relative path to
substitute a different YAML file at process start, or call :func:`~pgml.defaults.reload`
at runtime (a test or project-level override hook).

Selected default keys
---------------------

The full table lives in ``pgml/data/defaults.yaml`` inside the installed package; the keys
most relevant to typical usage are listed below.  Read a key at runtime with
:func:`~pgml.defaults.get`::

    import pgml.defaults as defaults

    z_ohm = defaults.get("source.series_impedance_ohm")   # 5.0 (default)
    xr    = defaults.get("source.xr_ratio")                # 10.0 (default)

Upstream-grid (slack Source) modelling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``source.series_impedance_ohm`` (default 5.0 Ω)
    Magnitude of the upstream-grid Thévenin/Norton series impedance, applied at the
    source's rated voltage.  Used by builders like
    :func:`~pgml.grids.cigre_lv_full_grid` when the converted source is
    near-ideal (e.g. the stock pandapower ext-grid gives R ~1e-6 Ω, which short-circuits
    the bus at harmonics).  Larger values produce a weaker upstream grid with more
    cross-feeder harmonic coupling; the user/grid value always wins via
    :func:`~pgml.defaults.resolve`.

``source.xr_ratio`` (default 10.0)
    X/R ratio of the source series impedance
    (R = ``|Z|`` / sqrt(1 + xr\ :sup:`2`), X = xr · R).  A value of 10 is typical for a
    stiff MV grid.

``source.zero_sequence.r0_over_r1`` / ``.x0_over_x1`` (default 1.0)
    Fallback ratios of a source Thévenin's zero-sequence impedance to its
    positive-sequence one, used only when the source library supplies no native
    zero-sequence data.  ``1.0`` means ``Z0 = Z1``, which is what pandapower and
    power-grid-model themselves default to, and the solve warns when the fallback is used.

Where a model is chosen rather than a value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several keys select a MODEL, with a shipped choice and documented alternatives for
reference comparisons:

``line.harmonic_model.three_phase`` / ``.single_phase``
    Which frequency-dependent line model a converter writes into
    ``Line.harmonic_line_model`` for an R/X line — ``sequence_aware`` and
    ``positive_sequence`` respectively.  See :doc:`/pgml/modeling/harmonic-line-model`.

``line.geometry.internal_inductance``
    How the conductor's internal inductance enters the geometry path above power frequency:
    ``gmr`` (shipped), ``gmr_skin``, ``gmr_power_frequency`` (OpenDSS's own 1 kHz rule) or
    ``bessel``.

``line.earth_return.x0_frequency``
    ``linear`` (shipped) or ``carson_sublinear``, the zero-sequence reactance law of the
    lumped sequence-aware model. ``linear`` suits cables and zero-sequence data derived
    from a ratio; ``carson_sublinear`` suits an overhead line whose stored ``X0`` contains
    the earth return, and ``line.earth_return.x0_nonnegative`` guards its extrapolation.
    See :doc:`/pgml/modeling/harmonic-line-model` and :doc:`/pgml/modeling/presets`.

``branch.zero_impedance``
    ``fuse`` (shipped) collapses an ideal branch's terminal rows exactly; ``error`` refuses
    such a grid and points at ``branch.near_ideal_series_resistance_ohm``.
    ``branch.switch_model`` is the companion choice a CONVERTER makes for a closed switch
    with no impedance data.

``transformer.magnetizing_placement``
    ``split`` (shipped, power-grid-model's placement), ``to_terminal`` (OpenDSS's placement)
    or ``from_terminal``. ``transformer.harmonic_resistance.law`` and
    ``transformer.zero_sequence.*`` are the other two transformer model choices; see
    :doc:`/pgml/modeling/transformer`.

``appliance.harmonic_shunt.model`` / ``.generation_model``
    The harmonic device Norton shunt, ``opendss`` (shipped) / ``motor`` / ``none``, and the
    separate policy for a generation-sign device, shipped as ``none``.

``appliance.generator.enforce_q_limits``
    Whether a voltage-regulating generator's reactive limits bound its output; shipped
    ``true``, where pandapower's ``runpp`` defaults to the unbounded solve.

``solver.convergence.*``, ``solver.precision.*``, ``solver.equilibration.*``, ``solver.ift.*`` and ``solver.loadability.ramp``
    The per-unit tolerances and their power base, the mixed-precision refinement, the
    diagonal equilibration, the gradient Jacobian's memory budget and adjoint cache, and
    what the loadability λ multiplies.  See :doc:`/pgml/modeling/solver-performance`.

.. automodule:: pgml.defaults
   :members:
   :show-inheritance:
