pgml.config
===========

Explicit, documented modelling defaults.

All default constants and default model selections live in ``defaults.yaml``,
ordered by component with a value, units, and a short description.

Resolution precedence:

    **explicit (user)** > **config (this package)** > **converter (source library)**

This means a user-supplied value always wins; a config default overrides
whatever a converter would infer.

Selected default keys
---------------------

The full table lives in ``src/pgml/config/defaults.yaml``; the keys most
relevant to typical usage are listed below.  Read a key at runtime with
:func:`~pgml.config.get`::

    import pgml.config as cfg
    z_ohm = cfg.get("source.series_impedance_ohm")   # 5.0 (default)
    rx    = cfg.get("source.rx_ratio")                # 10.0 (default)

Upstream-grid (slack Source) modelling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``source.series_impedance_ohm`` (default 5.0 Ohm)
    Magnitude ``|Z|`` [Ohm] of the upstream-grid Thevenin/Norton series impedance,
    applied at the source's rated voltage.  Used by builders like
    :func:`~pgml.evaluation.references.cigre_lv_full_grid` when the converted
    source is near-ideal (e.g. the stock pandapower ext-grid gives R ~1e-6 Ohm,
    which short-circuits the bus at harmonics).  Larger values produce a weaker
    upstream grid with more cross-feeder harmonic coupling; the user/grid value
    always wins via :func:`~pgml.config.resolve`.

``source.rx_ratio`` (default 10.0)
    X/R ratio of the source series impedance
    (``R = |Z| / sqrt(1 + rx^2)``, ``X = rx * R``).  A value of 10 is typical for a
    stiff MV grid.

.. automodule:: pgml.config
   :members:
   :show-inheritance:
