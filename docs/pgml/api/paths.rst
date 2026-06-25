pgml.paths
==========

Filesystem conventions for experiment inputs and outputs.

pgml, pgl, and pgg never write experiment data into the library tree.  A run's
datasets, checkpoints, run configs, and tracking go under the **experiments root** — a
single directory the user owns and keeps out of version control.  Keeping reproducible-but-bulky
experiment artifacts separate from the source also lets a run be relocated (e.g. to fast
cluster scratch) by pointing one environment variable elsewhere.

The root is ``$PGML_EXPERIMENTS`` when set, else ``./experiments`` relative to the current
working directory.

Usage::

    from pgml import experiments_root   # re-exported from pgml top-level

    run_dir = experiments_root() / "ieee33_harmonic_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Or access the module directly for the full surface:
    from pgml.paths import experiments_root

.. automodule:: pgml.paths
   :members:
   :show-inheritance:
