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
infer.

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

.. automodule:: pgml.defaults
   :members:
   :show-inheritance:
